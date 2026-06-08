"""DPO (Direct Preference Optimization) fine-tuning via TRL DPOTrainer."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
import yaml
from dataclasses import dataclass
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load training config from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    required = ["base_model", "output_dir"]
    for key in required:
        if not cfg.get(key):
            raise ValueError(f"Config is missing required key: '{key}'")


# ---------------------------------------------------------------------------
# Model / tokenizer helpers
# ---------------------------------------------------------------------------


def _build_bnb_config(cfg: dict) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytesConfig for QLoRA if quantization is configured."""
    quant_cfg = cfg.get("quantization")
    if not quant_cfg:
        return None

    if not quant_cfg.get("load_in_4bit", False):
        return None

    logger.info("Enabling QLoRA (4-bit quantization via BitsAndBytes).")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _load_tokenizer(base_model: str) -> AutoTokenizer:
    """Load tokenizer and ensure pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        # Use eos_token as pad_token when not explicitly defined
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token not found; set to eos_token: %s", tokenizer.eos_token)
    return tokenizer


def _load_model(
    base_model: str,
    bnb_config: BitsAndBytesConfig | None,
    requires_grad: bool = True,
) -> AutoModelForCausalLM:
    """Load a causal language model, optionally with 4-bit quantization.

    Parameters
    ----------
    base_model:
        HuggingFace model name or path.
    bnb_config:
        Optional BitsAndBytesConfig for QLoRA; pass None for full precision.
    requires_grad:
        If False, freeze all parameters (used for the reference model).

    Returns
    -------
    AutoModelForCausalLM
    """
    load_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model.config.use_cache = False  # disable KV cache during training

    if not requires_grad:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        logger.info("Reference model loaded and frozen.")

    return model


def _build_lora_config(cfg: dict) -> LoraConfig:
    """Build a PEFT LoraConfig from the training config."""
    lora_cfg = cfg.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config_path: str) -> None:
    """Full DPO training pipeline.

    Parameters
    ----------
    config_path:
        Path to DPO YAML config file.
    """
    cfg = load_config(config_path)
    logger.info("Config loaded:\n%s", yaml.dump(cfg, default_flow_style=False))

    # ---- W&B init ----
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "llm-finetuning-lab"),
        name=cfg.get("run_name", "dpo-run"),
        config=cfg,
    )

    base_model: str = cfg["base_model"]
    output_dir: str = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    logger.info("Loading tokenizer from: %s", base_model)
    tokenizer = _load_tokenizer(base_model)

    # ---- Quantization config (optional QLoRA) ----
    bnb_config = _build_bnb_config(cfg)

    # ---- Policy model ----
    logger.info("Loading policy model from: %s", base_model)
    model = _load_model(base_model, bnb_config, requires_grad=True)

    # ---- Reference model (frozen copy) ----
    # For QLoRA the reference model uses the same quant config but is frozen.
    logger.info("Loading reference model from: %s (frozen)", base_model)
    model_ref = _load_model(base_model, bnb_config, requires_grad=False)

    # ---- LoRA ----
    use_lora: bool = cfg.get("use_lora", True)
    lora_config: LoraConfig | None = None
    if use_lora:
        lora_config = _build_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA applied to policy model.")

    # ---- Dataset ----
    dpo_data_path = os.getenv("DPO_DATA_PATH", "./data/dpo_train.jsonl")
    logger.info("Loading DPO dataset from: %s", dpo_data_path)
    dataset = load_dataset(
        "json",
        data_files=dpo_data_path,
        split="train",
    )

    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"DPO dataset is missing required columns: {missing}. "
            f"Found columns: {dataset.column_names}"
        )

    logger.info(
        "Dataset loaded: %d examples, columns: %s",
        len(dataset),
        dataset.column_names,
    )

    # ---- DPO config ----
    training_cfg = cfg.get("training", {})
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 1024),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 100),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        report_to="wandb",
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    logger.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config if use_lora else None,
    )

    # ---- Train ----
    logger.info("Starting DPO training...")
    trainer_output = trainer.train()

    # ---- Save ----
    final_model_dir = Path(output_dir) / "final"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to: %s", final_model_dir)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # ---- Log final metrics ----
    metrics = trainer_output.metrics
    logger.info("Final metrics: %s", metrics)
    wandb.log({f"final/{k}": v for k, v in metrics.items()})
    wandb.finish()

    print(f"\n[INFO] DPO training complete. Model saved to: {final_model_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPO fine-tuning via TRL DPOTrainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo.yaml",
        help="Path to the DPO YAML configuration file.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args.config)
