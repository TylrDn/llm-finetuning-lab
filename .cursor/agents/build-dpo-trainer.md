---
name: build-dpo-trainer
description: Invoke when building or modifying the DPO (Direct Preference Optimization) trainer. Use when the user asks to implement dpo_train.py, create dpo.yaml config, set up preference datasets, or run DPO fine-tuning.
model: inherit
readonly: false
is_background: false
---

# Build DPO Trainer

## Objective

Create `training/dpo_train.py` and `configs/dpo.yaml` — a production-quality DPO (Direct Preference Optimization) trainer using TRL's `DPOTrainer`. Config-driven, same architecture as `lora_train.py`. Supports optional LoRA on top of DPO for memory efficiency. Logs to wandb.

---

## Files to Create

### Create: `configs/dpo.yaml`

Full DPO training configuration:

```yaml
# DPO Training Configuration
# Usage: python training/dpo_train.py --config configs/dpo.yaml

run_name: dpo-llama3-8b-instruct
output_dir: results/dpo

# Model
base_model: meta-llama/Meta-Llama-3-8B-Instruct
model_dtype: bfloat16        # bfloat16 | float16 | float32

# DPO Hyperparameters
beta: 0.1                    # KL penalty coefficient (higher = less deviation from ref model)
loss_type: sigmoid            # sigmoid | hinge | ipo | kto_pair
label_smoothing: 0.0
reference_free: false         # if true, no reference model needed

# Dataset
train_dataset_path: data/dpo_train.jsonl    # JSONL with: prompt, chosen, rejected
eval_dataset_path: data/dpo_eval.jsonl
dataset_format: alpaca_dpo                  # alpaca_dpo | chatml_dpo | raw

# Training hyperparameters
num_train_epochs: 3
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-7          # DPO uses much lower LR than SFT
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
max_length: 1024               # max total sequence length
max_prompt_length: 512         # max prompt length (rest for response)

# LoRA (optional — reduces memory, often used with DPO)
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

# Quantization (QLoRA for DPO)
use_4bit: false                # true for QLoRA
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Logging & Checkpoints
logging_steps: 10
eval_steps: 100
save_steps: 100
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: eval_loss

# Weights & Biases
wandb_project: llm-finetuning-lab
wandb_run_name: ${run_name}
report_to: wandb
```

---

### Create: `training/dpo_train.py`

Full production implementation.

**Imports:**
```python
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)
```

**Config dataclass (pydantic-validated from YAML):**

```python
from pydantic import BaseModel, Field

class DPOTrainingConfig(BaseModel):
    """Validated DPO training configuration."""
    run_name: str
    output_dir: Path
    base_model: str
    model_dtype: str = "bfloat16"
    beta: float = 0.1
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    reference_free: bool = False
    train_dataset_path: Path
    eval_dataset_path: Path
    dataset_format: str = "alpaca_dpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 1024
    max_prompt_length: int = 512
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    wandb_project: str = "llm-finetuning-lab"
    wandb_run_name: str = ""
    report_to: str = "wandb"
```

**`load_config(config_path: Path) -> DPOTrainingConfig`:**
- Load YAML, validate with pydantic `DPOTrainingConfig`
- Raise `TrainingConfigError` on validation failure

**`load_model_and_tokenizer(config: DPOTrainingConfig) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]`:**
- Returns `(model, ref_model, tokenizer)` — DPO requires a reference model copy
- If `config.use_4bit`: apply `BitsAndBytesConfig`
- If `config.use_lora`: wrap model with `get_peft_model()`, apply `LoraConfig`
- Load ref_model separately (no LoRA, no quantization) — used for KL penalty
- If `config.reference_free=True`, ref_model is `None`
- Set `tokenizer.pad_token = tokenizer.eos_token` if pad_token is None

**`load_datasets(config: DPOTrainingConfig, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset]`:**
- Load JSONL files with `datasets.load_dataset("json", data_files=...)`
- Expected JSONL columns: `prompt`, `chosen`, `rejected`
- Validate all 3 columns present
- Return `(train_dataset, eval_dataset)`

**`build_dpo_config(config: DPOTrainingConfig) -> DPOConfig`:**
```python
def build_dpo_config(config: DPOTrainingConfig) -> DPOConfig:
    return DPOConfig(
        output_dir=str(config.output_dir),
        run_name=config.run_name,
        beta=config.beta,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
        reference_free=config.reference_free,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        report_to=[config.report_to],
        bf16=config.model_dtype == "bfloat16",
        fp16=config.model_dtype == "float16",
    )
```

**`train(config: DPOTrainingConfig) -> dict[str, Any]`:**
Full pipeline: load → build → train → save → return metrics.

**CLI entrypoint:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO fine-tuning")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))
    config = load_config(args.config)
    results = train(config)
    logger.info(f"DPO training complete: {results}")
```

---

### Create: `tests/test_dpo_train.py`

```python
def test_load_config_valid(tmp_path): ...
def test_load_config_raises_on_missing_model(): ...
def test_build_dpo_config_sets_beta(): ...
def test_load_datasets_validates_columns(mock_dataset): ...

@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_load_model_sets_correct_dtype(dtype, mock_auto_model): ...

def test_train_logs_to_wandb(mock_dpo_trainer, mock_wandb): ...
def test_train_saves_to_output_dir(mock_dpo_trainer, tmp_path): ...
```

---

## Acceptance Criteria

- [ ] `python training/dpo_train.py --config configs/dpo.yaml` runs (with mocked model load)
- [ ] `pytest tests/test_dpo_train.py` passes (mock all GPU/model calls)
- [ ] `mypy --strict training/dpo_train.py` exits 0
- [ ] `ruff check training/dpo_train.py` exits 0
- [ ] `DPOTrainingConfig` validates all fields and raises `TrainingConfigError` on bad config
- [ ] Reference model is loaded separately from training model
- [ ] LoRA is optional (controlled by `use_lora` config flag)
- [ ] wandb logging uses `config.wandb_project` and `config.wandb_run_name`
- [ ] `configs/dpo.yaml` validates against `DPOTrainingConfig` pydantic model without errors
