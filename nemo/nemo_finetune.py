"""NeMo SFT fine-tuning using MegatronGPTSFTModel — NVIDIA-native multi-GPU training."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# NeMo conditional import
try:
    import pytorch_lightning as pl
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import (
        MegatronGPTSFTModel,
    )
    from nemo.collections.nlp.parts.nlp_overrides import (
        NLPDDPStrategy,
        NLPSaveRestoreConnector,
    )
    from omegaconf import DictConfig, OmegaConf

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo not available. Install with: pip install nemo_toolkit[nlp]")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def validate_config(cfg: dict) -> None:
    """Validate required fields in the NeMo SFT config.

    Raises
    ------
    ValueError
        If any required field is missing or invalid.
    """
    # Trainer section
    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is None:
        raise ValueError("Config missing required section: 'trainer'")

    devices = trainer_cfg.get("devices")
    if devices is None or int(devices) < 1:
        raise ValueError(
            f"trainer.devices must be an integer >= 1, got: {devices!r}"
        )

    max_steps = trainer_cfg.get("max_steps")
    if max_steps is None or int(max_steps) < 1:
        raise ValueError(
            f"trainer.max_steps must be an integer >= 1, got: {max_steps!r}"
        )

    # Model / data section
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Config missing required section: 'model'")

    data_cfg = model_cfg.get("data", {})
    train_ds = data_cfg.get("train_ds", {})
    file_names = train_ds.get("file_names")
    if not isinstance(file_names, list) or len(file_names) < 1:
        raise ValueError(
            "model.data.train_ds.file_names must be a non-empty list of file paths, "
            f"got: {file_names!r}"
        )

    # exp_manager section
    exp_manager_cfg = cfg.get("exp_manager")
    if exp_manager_cfg is None:
        raise ValueError("Config missing required section: 'exp_manager'")

    exp_dir = exp_manager_cfg.get("exp_dir")
    if not exp_dir:
        raise ValueError(
            "exp_manager.exp_dir must be set to a non-empty string path"
        )

    # Optional: validate restore_from_path if present
    restore_path = model_cfg.get("restore_from_path")
    if restore_path:
        is_registry = str(restore_path).startswith("nemo://")
        path_exists = Path(str(restore_path)).exists()
        if not is_registry and not path_exists:
            raise ValueError(
                f"model.restore_from_path '{restore_path}' does not exist on disk "
                "and does not start with 'nemo://' (registry prefix). "
                "Either provide a valid local path or a registry URI like 'nemo://model-name'."
            )


def load_config(config_path: str) -> dict:
    """Load and validate a NeMo SFT YAML config.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Validated configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    validate_config(cfg)
    logger.info("Config loaded and validated from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Trainer / Model builders
# ---------------------------------------------------------------------------


def build_trainer(cfg: dict) -> "pl.Trainer":
    """Construct a PyTorch Lightning Trainer with NLPDDPStrategy.

    Parameters
    ----------
    cfg:
        Full validated config dict.

    Returns
    -------
    pl.Trainer
    """
    if not NEMO_AVAILABLE:
        raise RuntimeError("NeMo is not available; cannot build trainer.")

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )

    trainer_cfg = cfg["trainer"]
    trainer = pl.Trainer(
        devices=trainer_cfg["devices"],
        num_nodes=trainer_cfg.get("num_nodes", 1),
        accelerator="gpu",
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        max_steps=trainer_cfg["max_steps"],
        val_check_interval=trainer_cfg.get("val_check_interval", 100),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        strategy=strategy,
        callbacks=[],
        enable_progress_bar=True,
    )

    logger.info(
        "Trainer built: devices=%s, num_nodes=%s, precision=%s, max_steps=%s",
        trainer_cfg["devices"],
        trainer_cfg.get("num_nodes", 1),
        trainer_cfg.get("precision", "bf16-mixed"),
        trainer_cfg["max_steps"],
    )
    return trainer


def build_model(
    cfg: dict, trainer: "pl.Trainer"
) -> "MegatronGPTSFTModel":
    """Instantiate or restore a MegatronGPTSFTModel.

    Parameters
    ----------
    cfg:
        Full validated config dict.
    trainer:
        PyTorch Lightning Trainer instance.

    Returns
    -------
    MegatronGPTSFTModel
    """
    if not NEMO_AVAILABLE:
        raise RuntimeError("NeMo is not available; cannot build model.")

    omega_cfg: DictConfig = OmegaConf.create(cfg)
    restore_path: Optional[str] = cfg["model"].get("restore_from_path")

    if restore_path:
        logger.info("Restoring MegatronGPTSFTModel from: %s", restore_path)
        model = MegatronGPTSFTModel.restore_from(
            restore_path=restore_path,
            override_config_path=omega_cfg.model,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    else:
        logger.info("Instantiating MegatronGPTSFTModel from scratch (no restore_from_path).")
        model = MegatronGPTSFTModel(cfg=omega_cfg.model, trainer=trainer)

    return model


# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------


def _setup_wandb(cfg: dict) -> None:
    """Initialize Weights & Biases if configured in exp_manager."""
    exp_manager_cfg = cfg.get("exp_manager", {})
    wandb_cfg = exp_manager_cfg.get("wandb_logger_kwargs", {})

    if not wandb_cfg or not wandb_cfg.get("project"):
        logger.info("W&B not configured in exp_manager.wandb_logger_kwargs — skipping.")
        return

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning(
            "WANDB_API_KEY not set in environment; W&B run will be anonymous."
        )

    try:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "llm-finetuning-lab"),
            name=wandb_cfg.get("name", cfg.get("run_name", "nemo-sft-run")),
            config=cfg,
        )
        logger.info(
            "W&B initialized: project=%s, run=%s",
            wandb_cfg.get("project"),
            wandb_cfg.get("name"),
        )
    except ImportError:
        logger.warning("wandb package not installed; skipping W&B init.")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def run_training(config_path: str) -> None:
    """End-to-end NeMo SFT training pipeline.

    Parameters
    ----------
    config_path:
        Path to the NeMo SFT YAML config file.
    """
    if not NEMO_AVAILABLE:
        print(
            "\n[ERROR] NeMo Toolkit is not installed.\n"
            "Install it with:\n\n"
            "  pip install nemo_toolkit[nlp]\n\n"
            "For GPU-optimised installs see:\n"
            "  https://github.com/NVIDIA/NeMo#installation\n"
        )
        sys.exit(1)

    # Load + validate config
    cfg = load_config(config_path)

    # Print resolved config
    logger.info("Resolved training config:\n%s", yaml.dump(cfg, default_flow_style=False))
    print("=== NeMo SFT Training Config ===")
    print(yaml.dump(cfg, default_flow_style=False))
    print("=================================")

    # Ensure experiment output directory exists
    exp_dir = Path(cfg["exp_manager"]["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory: %s", exp_dir)

    # W&B setup (best-effort)
    _setup_wandb(cfg)

    # Build components
    trainer = build_trainer(cfg)
    model = build_model(cfg, trainer)

    # Train
    logger.info("Starting NeMo SFT training...")
    trainer.fit(model)

    # Log completion
    checkpoint_dir = exp_dir / cfg["exp_manager"].get("name", "megatron_gpt_sft")
    logger.info("Training complete. Checkpoints saved to: %s", checkpoint_dir)
    print(f"\n[INFO] Training complete. Checkpoints saved to: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NeMo SFT fine-tuning using MegatronGPTSFTModel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="nemo/sft_config.yaml",
        help="Path to the NeMo SFT YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config, print it, then exit without training.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if not NEMO_AVAILABLE:
        print(
            "\n[ERROR] NeMo Toolkit is not installed.\n"
            "Install it with:\n\n"
            "  pip install nemo_toolkit[nlp]\n\n"
            "For GPU-optimised installs and Docker images see:\n"
            "  https://github.com/NVIDIA/NeMo#installation\n"
            "  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo\n"
        )
        sys.exit(1)

    parser = _build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] Loading and validating config: {args.config}")
        try:
            cfg = load_config(args.config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[DRY RUN] Config validation FAILED: {exc}")
            sys.exit(1)

        print("[DRY RUN] Config is valid. Resolved contents:")
        print(yaml.dump(cfg, default_flow_style=False))
        print("[DRY RUN] Exiting without training.")
        sys.exit(0)

    run_training(args.config)


if __name__ == "__main__":
    main()
