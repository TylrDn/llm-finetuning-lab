---
name: build-export-tools
description: Invoke when building or modifying model export tools. Use when the user asks to convert models to GGUF format, push models to HuggingFace Hub, create model cards, or build docker-compose for the training stack.
model: inherit
readonly: false
is_background: false
---

# Build Model Export Tools

## Objective

Create three export utilities and the root `docker-compose.yml`:
1. `export/convert_gguf.py` — convert merged model to GGUF using llama.cpp
2. `export/push_to_hub.py` — push fine-tuned model to HuggingFace Hub with model card
3. `docker-compose.yml` (repo root) — training + wandb services

---

## Files to Create

### Create: `export/convert_gguf.py`

Full production implementation.

**Imports:**
```python
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
```

**`GGUFConvertConfig` dataclass:**
```python
@dataclass
class GGUFConvertConfig:
    model_path: Path                        # merged HF model directory
    output_path: Path                       # output GGUF file path
    quantization_type: str = "q4_k_m"     # q4_k_m | q5_k_m | q8_0 | f16 | f32
    llamacpp_dir: Path = Path("llama.cpp") # path to llama.cpp repo
    context_length: int = 4096
    outtype: str = "f16"                   # intermediate conversion type before quantization
```

**Valid quantization types:**
```python
VALID_QUANT_TYPES: set[str] = {
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k", "q8_0", "f16", "f32",
}
```

**`find_llamacpp_scripts(llamacpp_dir: Path) -> tuple[Path, Path]`:**
- Find `convert_hf_to_gguf.py` and `quantize` binary in llamacpp_dir
- Try common locations: `{llamacpp_dir}/convert_hf_to_gguf.py`, `{llamacpp_dir}/build/bin/llama-quantize`
- Raise `ExportError` if not found with helpful error message pointing to llama.cpp build instructions

**`convert_to_gguf_f16(config: GGUFConvertConfig) -> Path`:**
```python
def convert_to_gguf_f16(config: GGUFConvertConfig) -> Path:
    """Convert HF model to GGUF (F16) using llama.cpp convert_hf_to_gguf.py.
    
    This is step 1 of a 2-step process (convert → quantize).
    
    Args:
        config: GGUFConvertConfig instance.
    Returns:
        Path to the F16 GGUF file.
    Raises:
        ExportError: If conversion subprocess fails.
    """
    convert_script, _ = find_llamacpp_scripts(config.llamacpp_dir)
    f16_output = config.output_path.parent / f"{config.output_path.stem}-f16.gguf"
    
    cmd = [
        sys.executable,
        str(convert_script),
        str(config.model_path),
        "--outfile", str(f16_output),
        "--outtype", config.outtype,
        "--ctx", str(config.context_length),
    ]
    
    logger.info(f"Converting to GGUF F16: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        logger.error(f"GGUF conversion failed:\n{result.stderr}")
        raise ExportError(f"convert_hf_to_gguf.py failed with code {result.returncode}")
    
    logger.info(f"F16 GGUF written to: {f16_output}")
    return f16_output
```

**`quantize_gguf(f16_path: Path, config: GGUFConvertConfig) -> Path`:**
```python
def quantize_gguf(f16_path: Path, config: GGUFConvertConfig) -> Path:
    """Quantize F16 GGUF to target quantization type using llama-quantize.
    
    Args:
        f16_path: Path to the F16 GGUF file from convert_to_gguf_f16.
        config: GGUFConvertConfig with quantization_type and output_path.
    Returns:
        Path to quantized GGUF file.
    Raises:
        ExportError: If quantize subprocess fails.
    """
    _, quantize_bin = find_llamacpp_scripts(config.llamacpp_dir)
    
    cmd = [
        str(quantize_bin),
        str(f16_path),
        str(config.output_path),
        config.quantization_type.upper(),
    ]
    
    logger.info(f"Quantizing to {config.quantization_type}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        raise ExportError(f"llama-quantize failed:\n{result.stderr}")
    
    # Clean up intermediate F16 file
    f16_path.unlink(missing_ok=True)
    
    logger.info(f"Quantized GGUF: {config.output_path} ({config.output_path.stat().st_size / 1e9:.2f} GB)")
    return config.output_path
```

**`convert_to_gguf(config: GGUFConvertConfig) -> Path`:** Orchestrate → validate → convert_to_gguf_f16 → quantize_gguf → return.

**CLI:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert merged HF model to GGUF")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantization", default="q4_k_m", choices=sorted(VALID_QUANT_TYPES))
    parser.add_argument("--llamacpp-dir", type=Path, default=Path("llama.cpp"))
    parser.add_argument("--context-length", type=int, default=4096)
```

---

### Create: `export/push_to_hub.py`

Full production implementation.

**Imports:**
```python
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, upload_folder, ModelCard, ModelCardData
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
```

**`HubPushConfig` dataclass:**
```python
@dataclass
class HubPushConfig:
    model_path: Path                    # local merged model directory
    repo_id: str                        # "username/model-name"
    base_model: str                     # e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    training_method: str = "lora"       # lora | qlora | dpo | nemo_sft
    dataset_name: str = ""
    mmlu_score: float | None = None
    hellaswag_score: float | None = None
    license: str = "llama3"
    private: bool = False
    push_model_weights: bool = True
    push_tokenizer: bool = True
    hf_token_env: str = "HUGGINGFACE_TOKEN"
```

**`generate_model_card(config: HubPushConfig) -> str`:**

Generate a complete HuggingFace model card (README.md content):
```markdown
---
license: {config.license}
base_model: {config.base_model}
tags:
  - fine-tuned
  - {config.training_method}
  - nvidia-nim
datasets:
  - {config.dataset_name}
language:
  - en
---

# {config.repo_id}

Fine-tuned {config.base_model} using {config.training_method.upper()} training.

## Training

- **Base Model:** {config.base_model}
- **Training Method:** {config.training_method.upper()}
- **Dataset:** {config.dataset_name or 'Custom dataset'}
- **Fine-tuning Framework:** {framework description based on training_method}

## Evaluation

| Benchmark | Score |
|-----------|-------|
| MMLU (5-shot) | {config.mmlu_score:.1%} or N/A |
| HellaSwag | {config.hellaswag_score:.1%} or N/A |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{config.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{config.repo_id}")
```

## License

{license description based on base model}
```

**`push_to_hub(config: HubPushConfig) -> str`:**
1. Validate `model_path` exists and has `config.json`
2. Read HF token from `os.environ[config.hf_token_env]` — raise `ExportError` if not set
3. `api = HfApi(token=hf_token)`
4. `create_repo(repo_id=config.repo_id, private=config.private, exist_ok=True)`
5. Write `model_card_text` to `model_path/README.md`
6. If `config.push_model_weights`: `upload_folder(folder_path=config.model_path, repo_id=config.repo_id, ignore_patterns=["*.pyc", "__pycache__"])`
7. Log INFO with repo URL: `https://huggingface.co/{config.repo_id}`
8. Return repo URL

**CLI:**
```python
parser.add_argument("--model-path", type=Path, required=True)
parser.add_argument("--repo-id", required=True, help="e.g. myuser/llama3-finetuned")
parser.add_argument("--base-model", required=True)
parser.add_argument("--training-method", default="lora", choices=["lora", "qlora", "dpo", "nemo_sft"])
parser.add_argument("--mmlu-score", type=float, default=None)
parser.add_argument("--hellaswag-score", type=float, default=None)
parser.add_argument("--private", action="store_true")
parser.add_argument("--no-push-weights", action="store_true")
```

---

### Create: `docker-compose.yml` (repo root)

Full production docker-compose:

```yaml
version: "3.9"

services:
  training:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    image: llm-finetuning-lab:latest
    container_name: finetuning-training
    runtime: nvidia
    ipc: host                          # required for multi-GPU training
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - WANDB_API_KEY=${WANDB_API_KEY}
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
      - NIM_API_KEY=${NIM_API_KEY}
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/workspace/data
      - ./results:/workspace/results
      - ./configs:/workspace/configs
      - ./nemo:/workspace/nemo
      - model-cache:/root/.cache/huggingface
    working_dir: /workspace
    command: ["python", "training/lora_train.py", "--config", "configs/lora.yaml"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  wandb-local:
    image: wandb/local:latest
    container_name: wandb-local
    ports:
      - "8080:8080"
    environment:
      - LOCAL_WANDB_SERVER=true
    volumes:
      - wandb-data:/vol
    profiles:
      - wandb                          # only start with: docker-compose --profile wandb up

  jupyter:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    image: llm-finetuning-lab:latest
    container_name: finetuning-jupyter
    runtime: nvidia
    ports:
      - "8888:8888"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
    volumes:
      - ./data:/workspace/data
      - ./results:/workspace/results
      - ./notebooks:/workspace/notebooks
      - model-cache:/root/.cache/huggingface
    working_dir: /workspace
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
    profiles:
      - jupyter

volumes:
  model-cache:
    driver: local
  wandb-data:
    driver: local
```

---

### Create: `tests/test_export.py`

```python
def test_convert_to_gguf_finds_scripts(tmp_path, mock_llamacpp_dir): ...
def test_convert_to_gguf_raises_on_missing_llamacpp(tmp_path): ...
def test_quantize_gguf_calls_correct_binary(mock_subprocess, tmp_path): ...
def test_generate_model_card_includes_metrics(): ...
def test_generate_model_card_has_yaml_frontmatter(): ...
def test_push_to_hub_raises_without_token(monkeypatch, tmp_path): ...
def test_push_to_hub_creates_model_card(mock_hf_api, tmp_path): ...
def test_valid_quant_types_are_complete(): ...
```

---

## Acceptance Criteria

- [ ] `python export/convert_gguf.py --model-path results/lora_merged/ --output results/model.gguf --quantization q4_k_m` runs (with mock subprocess)
- [ ] `python export/push_to_hub.py --model-path results/lora_merged/ --repo-id myuser/test-model --base-model meta-llama/Meta-Llama-3-8B-Instruct` runs (with mock HfApi)
- [ ] Model card includes YAML frontmatter with `license`, `base_model`, `tags`
- [ ] Model card includes eval scores when provided
- [ ] `push_to_hub` raises `ExportError` (not `KeyError`) when `HUGGINGFACE_TOKEN` not set
- [ ] GGUF intermediate F16 file is cleaned up after quantization
- [ ] `docker-compose up training` starts without errors (with valid `.env`)
- [ ] `pytest tests/test_export.py` passes
- [ ] `mypy --strict export/convert_gguf.py export/push_to_hub.py` exits 0
- [ ] `ruff check export/` exits 0
