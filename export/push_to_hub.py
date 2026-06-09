"""Push fine-tuned or merged model to HuggingFace Hub with auto-generated model card.

Handles:
  - Loading model + tokenizer from a local directory
  - Auto-generating a model card with training details and eval results
  - Uploading model, tokenizer, and model card to the Hub
  - Optional public/private repo toggle
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import ModelCard
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval results helpers
# ---------------------------------------------------------------------------


def _find_latest_eval_file(results_dir: str, prefix: str) -> str | None:
    """Find the most recently modified JSON file matching prefix."""
    import glob

    pattern = str(Path(results_dir) / f"{prefix}*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_eval_results(results_dir: str | None) -> dict[str, Any]:
    """Load latest MMLU and HellaSwag results from evals/results/ if present."""
    if results_dir is None:
        # Try default relative path
        default = Path("evals") / "results"
        if not default.exists():
            return {}
        results_dir = str(default)

    mmlu_path = _find_latest_eval_file(results_dir, "mmlu_")
    hellaswag_path = _find_latest_eval_file(results_dir, "hellaswag_")

    evals: dict[str, Any] = {}

    if mmlu_path:
        try:
            with open(mmlu_path) as fh:
                data = json.load(fh)
            evals["mmlu"] = {
                "overall": data.get("overall"),
                "by_group": data.get("by_group", {}),
                "n_shots": data.get("n_shots", 5),
                "source": mmlu_path,
            }
            logger.info("Loaded MMLU results from: %s", mmlu_path)
        except Exception as exc:
            logger.warning("Could not load MMLU results: %s", exc)

    if hellaswag_path:
        try:
            with open(hellaswag_path) as fh:
                data = json.load(fh)
            evals["hellaswag"] = {
                "acc_norm": data.get("acc_norm"),
                "n_samples": data.get("n_samples"),
                "source": hellaswag_path,
            }
            logger.info("Loaded HellaSwag results from: %s", hellaswag_path)
        except Exception as exc:
            logger.warning("Could not load HellaSwag results: %s", exc)

    return evals


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------


def _build_eval_section(evals: dict[str, Any]) -> str:
    """Build the evaluation results section of the model card."""
    if not evals:
        return "_No evaluation results found. Run `evals/mmlu_eval.py` and `evals/hellaswag_eval.py` to populate._"  # noqa: E501

    lines: list[str] = []

    if "mmlu" in evals:
        mmlu = evals["mmlu"]
        overall = mmlu.get("overall")
        n_shots = mmlu.get("n_shots", 5)
        lines.append(f"### MMLU ({n_shots}-shot)")
        if overall is not None:
            lines.append(f"- **Overall accuracy:** {overall * 100:.2f}%")
        by_group = mmlu.get("by_group", {})
        if by_group:
            lines.append("")
            lines.append("| Group | Accuracy |")
            lines.append("|-------|----------|")
            for group, stats in sorted(by_group.items()):
                acc = (stats or {}).get("accuracy")
                acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
                lines.append(f"| {group} | {acc_str} |")

    if "hellaswag" in evals:
        hs = evals["hellaswag"]
        acc_norm = hs.get("acc_norm")
        n_samples = hs.get("n_samples")
        lines.append("")
        lines.append("### HellaSwag")
        if acc_norm is not None:
            lines.append(f"- **acc_norm:** {acc_norm * 100:.2f}%")
        if n_samples:
            lines.append(f"- Evaluated on {n_samples} examples")

    return "\n".join(lines)


def _detect_finetuning_method(config: dict | None) -> str:
    """Detect fine-tuning method from config."""
    if config is None:
        return "LoRA"
    if config.get("beta") is not None:
        return "DPO + LoRA" if config.get("use_lora", True) else "DPO"
    quant = config.get("quantization", {})
    if quant and quant.get("load_in_4bit"):
        return "QLoRA"
    if config.get("lora") or config.get("use_lora", True):
        return "LoRA"
    return "Full Fine-tuning"


def _build_model_card(
    repo_id: str,
    base_model: str,
    config: dict | None,
    evals: dict[str, Any],
    config_path: str | None,
) -> str:
    """Generate model card markdown content.

    Parameters
    ----------
    repo_id:
        HuggingFace Hub repo ID (e.g. username/model-name).
    base_model:
        Base model identifier.
    config:
        Training config dict (optional).
    evals:
        Evaluation results dict.
    config_path:
        Path to config YAML for linking.
    """
    method = _detect_finetuning_method(config)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lora_r = (config or {}).get("lora", {}).get("r", "N/A") if config else "N/A"
    lora_alpha = (config or {}).get("lora", {}).get("lora_alpha", "N/A") if config else "N/A"
    epochs = (config or {}).get("training", {}).get("num_train_epochs", "N/A") if config else "N/A"
    lr = (config or {}).get("training", {}).get("learning_rate", "N/A") if config else "N/A"

    config_link = f"[config]({config_path})" if config_path else "N/A"
    eval_section = _build_eval_section(evals)

    card = f"""---
base_model: {base_model}
library_name: transformers
tags:
  - fine-tuned
  - lora
  - llm-finetuning-lab
license: apache-2.0
---

# {repo_id}

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) using **{method}** via [llm-finetuning-lab](https://github.com/llm-finetuning-lab/llm-finetuning-lab).

> Generated on {now}

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `{base_model}` |
| Fine-tuning method | {method} |
| LoRA rank (r) | {lora_r} |
| LoRA alpha | {lora_alpha} |
| Training epochs | {epochs} |
| Learning rate | {lr} |
| Training config | {config_link} |

---

## Evaluation Results

{eval_section}

---

## Usage

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="{repo_id}", device_map="auto")
output = pipe("Hello, I am", max_new_tokens=128)
print(output[0]["generated_text"])
```

With LoRA adapter (if not merged):
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

---

## About llm-finetuning-lab

This model was trained with the [llm-finetuning-lab](https://github.com/TylrDn/llm-finetuning-lab)
pipeline.
"""
    return card


# ---------------------------------------------------------------------------
# Core push function
# ---------------------------------------------------------------------------


def push_to_hub(
    model_dir: str,
    repo_id: str,
    config_path: str | None = None,
    private: bool = True,
    results_dir: str | None = None,
) -> str:
    """Load and push a fine-tuned model to HuggingFace Hub.

    Parameters
    ----------
    model_dir:
        Path to the local model directory (safetensors / PEFT adapter).
    repo_id:
        HuggingFace Hub repo ID, e.g. ``username/my-llama3-dpo``.
    config_path:
        Optional path to the training YAML config for model card metadata.
    private:
        If True (default), create a private repo.
    results_dir:
        Optional path to eval results directory. Defaults to ``evals/results/``.

    Returns
    -------
    str
        URL of the uploaded model on the Hub.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN / HUGGINGFACE_TOKEN not set. "
            "You may be unable to push to private repos."
        )

    # Load training config if provided
    config: dict | None = None
    base_model = "unknown"
    if config_path:
        try:
            with open(config_path) as fh:
                config = yaml.safe_load(fh)
            base_model = config.get("base_model", "unknown")
            logger.info("Training config loaded from: %s", config_path)
        except Exception as exc:
            logger.warning("Could not load training config: %s", exc)
    else:
        # Try to infer base_model from adapter_config.json
        adapter_cfg_path = Path(model_dir) / "adapter_config.json"
        if adapter_cfg_path.exists():
            try:
                with open(adapter_cfg_path) as fh:
                    adapter_cfg = json.load(fh)
                base_model = adapter_cfg.get("base_model_name_or_path", "unknown")
            except Exception:
                pass

    # Load eval results
    evals = _load_eval_results(results_dir)

    # Load model + tokenizer
    logger.info("Loading model from: %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True
    )

    # Push tokenizer
    logger.info("Pushing tokenizer to Hub: %s", repo_id)
    tokenizer.push_to_hub(repo_id, private=private, token=hf_token)

    # Push model
    logger.info("Pushing model to Hub: %s (this may take a while...)", repo_id)
    model.push_to_hub(repo_id, private=private, token=hf_token)

    # Generate + push model card
    card_content = _build_model_card(
        repo_id=repo_id,
        base_model=base_model,
        config=config,
        evals=evals,
        config_path=config_path,
    )
    model_card = ModelCard(card_content)
    logger.info("Pushing model card to Hub: %s", repo_id)
    model_card.push_to_hub(repo_id, token=hf_token)

    hub_url = f"https://huggingface.co/{repo_id}"
    logger.info("Model successfully pushed to: %s", hub_url)
    print(f"\n[INFO] Model pushed to Hub: {hub_url}")
    return hub_url


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push a fine-tuned model to HuggingFace Hub with auto-generated model card.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the local model directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repo ID (e.g. username/model-name).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to the training YAML config for model card metadata.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Push to a public repo (default: private).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing eval result JSON files. Defaults to evals/results/.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    push_to_hub(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        config_path=args.config_path,
        private=not args.public,
        results_dir=args.results_dir,
    )
