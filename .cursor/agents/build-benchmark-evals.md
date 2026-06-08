---
name: build-benchmark-evals
description: Invoke when building or modifying benchmark evaluation scripts. Use when the user asks to implement MMLU eval, HellaSwag eval, comparison reports, or set up lm-eval-harness benchmarking for fine-tuned models.
model: inherit
readonly: false
is_background: false
---

# Build Benchmark Evaluation Scripts

## Objective

Create three evaluation scripts:
1. `evals/mmlu_eval.py` — 5-shot MMLU benchmark
2. `evals/hellaswag_eval.py` — HellaSwag completion benchmark
3. `evals/compare_report.py` — base vs fine-tuned comparison report with wandb artifacts

All use `lm-eval-harness` (`lm_eval`) and produce JSON results written to `results/evals/`. The compare_report provides the key "before/after" metrics that demonstrate fine-tuning value.

---

## Files to Create

### Create: `evals/mmlu_eval.py`

Full production implementation.

**Imports:**
```python
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)
```

**`MMLUEvalConfig` dataclass:**
```python
@dataclass
class MMLUEvalConfig:
    model_path: str                     # HF model path or local dir
    num_fewshot: int = 5                # 5-shot is standard MMLU
    batch_size: int = 8
    device: str = "cuda"
    dtype: str = "float16"
    output_dir: Path = Path("results/evals")
    run_name: str = ""
    wandb_project: str = "llm-finetuning-lab"
    log_to_wandb: bool = True
    mmlu_subcategories: list[str] = field(default_factory=list)  # empty = all
```

**`run_mmlu_eval(config: MMLUEvalConfig) -> MMLUEvalResults`:**

```python
@dataclass
class MMLUEvalResults:
    model_path: str
    timestamp: str
    num_fewshot: int
    overall_accuracy: float
    per_category_accuracy: dict[str, float]   # e.g. {"mmlu_abstract_algebra": 0.72}
    num_tasks: int
    results_path: Path
```

Implementation:
1. Load model: `HFLM(pretrained=config.model_path, dtype=config.dtype, device=config.device, batch_size=config.batch_size)`
2. Determine tasks: if `config.mmlu_subcategories` is empty, use `"mmlu"` (all 57 subcategories); otherwise use specified list
3. Run eval: `evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=config.num_fewshot, verbosity="ERROR")`
4. Extract `overall_accuracy` from `results["results"]["mmlu"]["acc,none"]` (or average of subcategories)
5. Extract per-category: iterate `results["results"]` for keys starting with `"mmlu_"`
6. Save JSON to `results/evals/mmlu_{model_name}_{timestamp}.json`
7. Log to wandb if configured: `wandb.log({"mmlu_accuracy": overall_accuracy, **per_category_accuracy})`
8. Return `MMLUEvalResults`

**MMLU subcategory groupings (for reporting):**
```python
MMLU_CATEGORY_GROUPS: dict[str, list[str]] = {
    "STEM": ["abstract_algebra", "astronomy", "college_biology", "college_chemistry",
             "college_computer_science", "college_mathematics", "college_physics",
             "computer_security", "conceptual_physics", "electrical_engineering",
             "elementary_mathematics", "high_school_biology", "high_school_chemistry",
             "high_school_computer_science", "high_school_mathematics", "high_school_physics",
             "high_school_statistics", "machine_learning"],
    "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                   "high_school_world_history", "international_law", "jurisprudence",
                   "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
                   "prehistory", "professional_law", "world_religions"],
    "Social Sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                        "high_school_macroeconomics", "high_school_microeconomics",
                        "high_school_psychology", "human_sexuality", "political_science",
                        "psychology", "public_relations", "security_studies", "sociology"],
    "Other": ["anatomy", "clinical_knowledge", "college_medicine", "global_facts",
              "human_aging", "management", "marketing", "medical_genetics",
              "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
              "virology"],
}
```

**CLI:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-shot MMLU evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/evals"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--subcategories", nargs="+", default=[])
```

---

### Create: `evals/hellaswag_eval.py`

Same pattern as `mmlu_eval.py`. Key differences:

**`HellaSwagEvalConfig` and `HellaSwagEvalResults`:**
- Task: `"hellaswag"` (single task, not subcategories)
- Metric: `acc_norm` (normalized accuracy — standard HellaSwag metric)
- `num_fewshot: int = 0` (HellaSwag is typically 0-shot)

**`run_hellaswag_eval(config: HellaSwagEvalConfig) -> HellaSwagEvalResults`:**
1. Load HFLM model
2. `evaluator.simple_evaluate(model=lm, tasks=["hellaswag"], num_fewshot=0)`
3. Extract `acc_norm` from results
4. Save JSON to `results/evals/hellaswag_{model_name}_{timestamp}.json`
5. Log to wandb: `{"hellaswag_acc_norm": acc_norm}`
6. Return `HellaSwagEvalResults`

---

### Create: `evals/compare_report.py`

Full implementation.

**Purpose:** Load MMLU and HellaSwag results for base model and fine-tuned model, produce a markdown comparison table and log it as a wandb artifact.

**`CompareConfig` dataclass:**
```python
@dataclass
class CompareConfig:
    base_model_path: str
    finetuned_model_path: str
    output_dir: Path = Path("results/evals")
    wandb_project: str = "llm-finetuning-lab"
    run_name: str = "base-vs-finetuned"
    run_mmlu: bool = True
    run_hellaswag: bool = True
    num_fewshot_mmlu: int = 5
    batch_size: int = 8
    dtype: str = "float16"
```

**`CompareReport` dataclass:**
```python
@dataclass  
class CompareReport:
    base_model: str
    finetuned_model: str
    timestamp: str
    mmlu_base: float | None
    mmlu_finetuned: float | None
    mmlu_delta: float | None
    hellaswag_base: float | None
    hellaswag_finetuned: float | None
    hellaswag_delta: float | None
    mmlu_category_comparison: dict[str, dict[str, float]]  # {category: {base, ft, delta}}
    markdown_table: str
    report_path: Path
```

**`run_comparison(config: CompareConfig) -> CompareReport`:**
1. Run MMLU eval on base model → `base_mmlu`
2. Run MMLU eval on fine-tuned model → `ft_mmlu`
3. Run HellaSwag on base → `base_hs`
4. Run HellaSwag on fine-tuned → `ft_hs`
5. Compute deltas: `delta = finetuned - base`
6. Generate markdown table

**`generate_markdown_table(report: CompareReport) -> str`** — full markdown:

```markdown
# Model Comparison: Base vs Fine-Tuned

| Benchmark | Base Model | Fine-Tuned | Delta |
|-----------|-----------|------------|-------|
| MMLU (5-shot) | {base_mmlu:.1%} | {ft_mmlu:.1%} | {delta:+.1%} |
| HellaSwag | {base_hs:.1%} | {ft_hs:.1%} | {delta:+.1%} |

## MMLU by Category Group

| Group | Base | Fine-Tuned | Delta |
|-------|------|-----------|-------|
| STEM | ... | ... | ... |
| Humanities | ... | ... | ... |
| Social Sciences | ... | ... | ... |
| Other | ... | ... | ... |
```

**wandb artifact upload:**
```python
artifact = wandb.Artifact("eval_comparison", type="evaluation")
artifact.add_file(str(report.report_path))
run.log_artifact(artifact)
run.log({
    "mmlu_delta": report.mmlu_delta,
    "hellaswag_delta": report.hellaswag_delta,
})
```

---

### Create: `tests/test_evals.py`

```python
def test_mmlu_eval_config_defaults(): ...
def test_hellaswag_eval_config_defaults(): ...
def test_compare_config_validates(): ...
def test_generate_markdown_table_format(mock_compare_report): ...
def test_mmlu_category_grouping_covers_all_57(): ...

@pytest.mark.parametrize("accuracy", [0.0, 0.5, 1.0])
def test_compare_report_delta_calculation(accuracy): ...

def test_run_mmlu_eval_saves_json(mock_lm_eval, tmp_path): ...
def test_run_hellaswag_eval_saves_json(mock_lm_eval, tmp_path): ...
def test_run_comparison_logs_to_wandb(mock_lm_eval, mock_wandb, tmp_path): ...
```

---

## Acceptance Criteria

- [ ] `python evals/mmlu_eval.py --model-path results/lora_merged/` runs (mock lm_eval)
- [ ] `python evals/hellaswag_eval.py --model-path results/lora_merged/` runs (mock lm_eval)
- [ ] `python evals/compare_report.py --base-model meta-llama/Meta-Llama-3-8B-Instruct --finetuned-model results/lora_merged/` runs
- [ ] `MMLU_CATEGORY_GROUPS` covers all 57 standard MMLU subjects
- [ ] Markdown table includes delta column with `+/-` sign
- [ ] wandb artifact is uploaded when `WANDB_API_KEY` is set
- [ ] All JSON results written to `results/evals/`
- [ ] `pytest tests/test_evals.py` passes (mock lm_eval + GPU)
- [ ] `mypy --strict evals/mmlu_eval.py evals/hellaswag_eval.py evals/compare_report.py` exits 0
- [ ] `ruff check evals/` exits 0
