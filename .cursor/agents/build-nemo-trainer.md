---
name: build-nemo-trainer
description: Invoke when building or modifying the NeMo SFT fine-tuning script. Use when the user asks to implement nemo_finetune.py, wire NeMo SFT config, run multi-GPU training with Megatron-LM, or troubleshoot NeMo training setup.
model: inherit
readonly: false
is_background: false
---

# Build NeMo SFT Fine-Tuning Script

## Objective

Create `nemo/nemo_finetune.py` — the **NVIDIA differentiator** of this repo. A production-quality Python trainer that runs NeMo SFT using `MegatronGPTSFTModel` with the existing `nemo/sft_config.yaml`. This script demonstrates enterprise-grade multi-GPU fine-tuning using the NVIDIA NeMo framework, which is fundamentally different from HuggingFace PEFT/TRL.

---

## Files to Create / Modify

### Create: `nemo/nemo_finetune.py`

Full production implementation. Every function documented, fully typed, no placeholders.

**Imports:**
```python
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf

# NeMo imports — wrapped in try/except for import-time graceful degradation
try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import (
        MegatronGPTSFTModel,
    )
    from nemo.collections.nlp.parts.nlp_overrides import (
        NLPDDPStrategy,
        NLPSaveRestoreConnector,
    )
    from nemo.utils import logging as nemo_logging
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

logger = logging.getLogger(__name__)
```

**`validate_config(cfg: DictConfig) -> None`:**
- Validate required fields: `cfg.model.restore_from_path`, `cfg.trainer.devices`, `cfg.model.data.train_ds.file_path`
- Verify `cfg.model.restore_from_path` file exists (or is a valid NeMo registry path like `nemo://llama3-8b`)
- Validate `cfg.trainer.devices > 0`
- Raise `TrainingConfigError` on validation failure

**`setup_wandb(cfg: DictConfig) -> wandb.run | None`:**
- Initialize wandb if `cfg.exp_manager.create_wandb_logger` is True
- Project: `cfg.exp_manager.wandb_logger_kwargs.project`
- Run name: `cfg.exp_manager.wandb_logger_kwargs.name`
- Config: log full OmegaConf config as wandb config
- Return wandb run object or None if wandb not configured

**`build_trainer(cfg: DictConfig) -> pl.Trainer`:**
```python
def build_trainer(cfg: DictConfig) -> pl.Trainer:
    """Build PyTorch Lightning Trainer with NeMo DDP strategy.
    
    Uses NLPDDPStrategy for NeMo's Megatron-LM compatible distributed training.
    Configures gradient clipping, precision, and checkpoint callbacks from config.
    
    Args:
        cfg: OmegaConf config with trainer section.
    Returns:
        Configured pl.Trainer instance.
    """
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )
    
    callbacks = []
    
    # Learning rate monitor
    from pytorch_lightning.callbacks import LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Model checkpoint
    from pytorch_lightning.callbacks import ModelCheckpoint
    callbacks.append(ModelCheckpoint(
        dirpath=cfg.exp_manager.explicit_log_dir,
        filename="nemo-sft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.trainer.get("save_top_k", 3),
    ))
    
    return pl.Trainer(
        strategy=strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.get("num_nodes", 1),
        accelerator="gpu",
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.get("max_steps", -1),
        val_check_interval=cfg.trainer.get("val_check_interval", 0.5),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        callbacks=callbacks,
        enable_checkpointing=True,
    )
```

**`build_model(cfg: DictConfig, trainer: pl.Trainer) -> MegatronGPTSFTModel`:**
```python
def build_model(cfg: DictConfig, trainer: pl.Trainer) -> MegatronGPTSFTModel:
    """Load and configure MegatronGPTSFTModel from checkpoint or pretrained.
    
    Handles both .nemo checkpoint restoration and fresh initialization.
    
    Args:
        cfg: OmegaConf config with model section.
        trainer: Configured pl.Trainer instance.
    Returns:
        MegatronGPTSFTModel ready for training.
    Raises:
        FileNotFoundError: If restore_from_path doesn't exist.
        TrainingConfigError: If model config is invalid.
    """
    restore_path = cfg.model.restore_from_path
    logger.info(f"Restoring model from: {restore_path}")
    
    model = MegatronGPTSFTModel.restore_from(
        restore_path=restore_path,
        override_config_path=cfg.model,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        strict=True,
    )
    
    logger.info(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")
    return model
```

**`train(cfg: DictConfig) -> dict[str, Any]`:**
```python
def train(cfg: DictConfig) -> dict[str, Any]:
    """Run full NeMo SFT training pipeline.
    
    Validates config, builds trainer + model, runs training, logs to wandb.
    
    Args:
        cfg: OmegaConf DictConfig (loaded from sft_config.yaml via Hydra).
    Returns:
        Dict with training results: best_val_loss, checkpoint_path, wandb_run_url.
    Raises:
        TrainingConfigError: If config validation fails.
    """
    if not NEMO_AVAILABLE:
        raise ImportError("NeMo is not installed. Install with: pip install nemo_toolkit[nlp]")
    
    validate_config(cfg)
    run = setup_wandb(cfg)
    
    logger.info("Building trainer...")
    trainer = build_trainer(cfg)
    
    logger.info("Loading model...")
    model = build_model(cfg, trainer)
    
    logger.info("Starting NeMo SFT training...")
    trainer.fit(model)
    
    results = {
        "best_val_loss": trainer.callback_metrics.get("val_loss", float("inf")),
        "checkpoint_path": str(cfg.exp_manager.explicit_log_dir),
        "wandb_run_url": run.get_url() if run else None,
        "epochs_completed": trainer.current_epoch,
    }
    
    if run:
        run.log(results)
        run.finish()
    
    logger.info(f"Training complete: {results}")
    return results
```

**CLI entrypoint with Hydra:**
```python
import hydra

@hydra.main(config_path=".", config_name="sft_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """NeMo SFT fine-tuning entrypoint.
    
    Usage:
        python nemo/nemo_finetune.py
        python nemo/nemo_finetune.py trainer.devices=4
        python nemo/nemo_finetune.py model.restore_from_path=/path/to/llama3.nemo
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("NeMo SFT Fine-Tuning Lab")
    logger.info(OmegaConf.to_yaml(cfg))
    
    results = train(cfg)
    logger.info(f"Final results: {results}")

if __name__ == "__main__":
    main()
```

---

### Modify: `nemo/sft_config.yaml`

Ensure config has all required sections. Add missing sections if absent:

```yaml
# NeMo SFT Config — used by nemo_finetune.py

trainer:
  devices: 1              # override with trainer.devices=N for multi-GPU
  num_nodes: 1
  accelerator: gpu
  precision: bf16-mixed
  max_epochs: 3
  max_steps: -1
  val_check_interval: 0.5
  limit_val_batches: 50
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  log_every_n_steps: 10
  save_top_k: 3

exp_manager:
  explicit_log_dir: results/nemo_sft
  exp_dir: null
  name: nemo-sft-finetune
  create_wandb_logger: true
  wandb_logger_kwargs:
    project: llm-finetuning-lab
    name: nemo-sft-run
  resume_if_exists: true
  resume_ignore_no_checkpoint: true

model:
  restore_from_path: null   # set to .nemo checkpoint path
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1

  data:
    train_ds:
      file_path: data/train.jsonl
      max_seq_length: 2048
      micro_batch_size: 2
      global_batch_size: 16
      shuffle: true
      num_workers: 4
      pin_memory: true
    validation_ds:
      file_path: data/val.jsonl
      max_seq_length: 2048
      micro_batch_size: 2
      global_batch_size: 16
      shuffle: false
      num_workers: 2

  peft:
    peft_scheme: lora
    lora_tuning:
      target_modules: [q_proj, v_proj]
      adapter_dim: 32
      alpha: 64
      dropout: 0.05

  optim:
    name: distributed_fused_adam
    lr: 1.0e-4
    weight_decay: 0.01
    betas: [0.9, 0.98]
    sched:
      name: CosineAnnealing
      warmup_steps: 50
      constant_steps: 0
      min_lr: 1.0e-5
```

---

### Create: `tests/test_nemo_finetune.py`

```python
def test_validate_config_passes_valid_config(mock_sft_config): ...
def test_validate_config_raises_on_missing_restore_path(mock_sft_config): ...
def test_validate_config_raises_on_zero_devices(mock_sft_config): ...
def test_build_trainer_returns_pl_trainer(mock_sft_config, mock_nemo): ...
def test_setup_wandb_returns_none_when_disabled(mock_sft_config): ...
def test_train_raises_import_error_when_nemo_unavailable(monkeypatch): ...
def test_train_calls_trainer_fit(mock_sft_config, mock_nemo_model, mock_trainer): ...
```

---

## Acceptance Criteria

- [ ] `python nemo/nemo_finetune.py --help` prints Hydra help without errors
- [ ] `python nemo/nemo_finetune.py trainer.devices=1 model.restore_from_path=test.nemo` (with mock) runs without error
- [ ] `pytest tests/test_nemo_finetune.py` passes (mock all NeMo + GPU calls)
- [ ] `mypy --strict nemo/nemo_finetune.py` exits 0
- [ ] `ruff check nemo/nemo_finetune.py` exits 0
- [ ] `validate_config()` raises `TrainingConfigError` (not ValueError/KeyError)
- [ ] wandb logging is conditional — no crash if wandb not configured
- [ ] Multi-GPU is configurable via `trainer.devices=N` Hydra override (not hardcoded)
