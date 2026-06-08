# llm-finetuning-lab — Task Board

**Repo:** llm-finetuning-lab
**Completion:** 50%
**Last Audit:** 2026-06-08
**Open Tasks:** 6 critical / 2 polish / 2 enhancement

---

## Priority 1 — CRITICAL

### [ ] 1.1 NeMo SFT Fine-Tuning Script (NVIDIA Differentiator)
**File:** `nemo/nemo_finetune.py`
**What:** Python trainer using `MegatronGPTSFTModel` + PyTorch Lightning. Functions: `validate_config()`, `setup_wandb()`, `build_trainer()` (NLPDDPStrategy), `build_model()` (restore from .nemo checkpoint), `train()`. Hydra entrypoint with `@hydra.main(config_path=".", config_name="sft_config")`. Multi-GPU via `trainer.devices=N` Hydra override.
**Acceptance Criteria:**
- `python nemo/nemo_finetune.py --help` prints Hydra help without import errors
- `pytest tests/test_nemo_finetune.py` passes (mock NeMo + GPU)
- `validate_config()` raises `TrainingConfigError` on missing `restore_from_path`
- wandb logging conditional on config flag (no crash if not configured)
- `mypy --strict nemo/nemo_finetune.py` exits 0

---

### [ ] 1.2 DPO Trainer + Config
**Files:** `training/dpo_train.py`, `configs/dpo.yaml`
**What:** `DPOTrainer` (TRL) wrapper with pydantic `DPOTrainingConfig`. Key params: `beta=0.1`, `loss_type=sigmoid`, reference model loaded separately. Optional LoRA (`use_lora: true`). Config: `configs/dpo.yaml` with full DPO hyperparameters, wandb settings. Loads `(prompt, chosen, rejected)` JSONL datasets.
**Acceptance Criteria:**
- `python training/dpo_train.py --config configs/dpo.yaml` runs (mock model load)
- Reference model loaded separately from training model
- `DPOTrainingConfig` validates all fields via pydantic
- `pytest tests/test_dpo_train.py` passes
- `mypy --strict training/dpo_train.py` exits 0

---

### [ ] 1.3 MMLU Benchmark Evaluation
**File:** `evals/mmlu_eval.py`
**What:** 5-shot MMLU eval using `lm-eval-harness`. `HFLM` model loading, `evaluator.simple_evaluate(tasks=["mmlu"], num_fewshot=5)`. Per-category accuracy across all 57 MMLU subjects, grouped by STEM/Humanities/Social Sciences/Other. Results JSON to `results/evals/mmlu_{model}_{timestamp}.json`. wandb logging.
**Acceptance Criteria:**
- All 57 MMLU subcategories covered in `MMLU_CATEGORY_GROUPS`
- `pytest tests/test_evals.py::test_run_mmlu_eval_saves_json` passes
- JSON output contains `overall_accuracy` and `per_category_accuracy`
- `mypy --strict evals/mmlu_eval.py` exits 0

---

### [ ] 1.4 HellaSwag Benchmark Evaluation
**File:** `evals/hellaswag_eval.py`
**What:** HellaSwag completion benchmark using `lm-eval-harness`. 0-shot, metric `acc_norm`. Same pattern as mmlu_eval.py. Results JSON to `results/evals/hellaswag_{model}_{timestamp}.json`.
**Acceptance Criteria:**
- Uses `acc_norm` (not `acc`) as primary metric
- `num_fewshot=0` (HellaSwag standard)
- `pytest tests/test_evals.py::test_run_hellaswag_eval_saves_json` passes
- `mypy --strict evals/hellaswag_eval.py` exits 0

---

### [ ] 1.5 Model Comparison Report
**File:** `evals/compare_report.py`
**What:** Run MMLU + HellaSwag on both base model and fine-tuned model. Compute deltas. Generate markdown comparison table (base | fine-tuned | delta columns). Upload as wandb artifact. `CompareReport` dataclass with all metrics and `markdown_table` field.
**Acceptance Criteria:**
- Markdown table includes `+/-` delta column
- wandb artifact uploaded when `WANDB_API_KEY` set
- `pytest tests/test_evals.py::test_run_comparison_logs_to_wandb` passes
- `generate_markdown_table()` is independently testable

---

### [ ] 1.6 Export Tools + docker-compose
**Files:** `export/convert_gguf.py`, `export/push_to_hub.py`, `docker-compose.yml` (root)
**What:**
- `convert_gguf.py`: Convert merged HF model to GGUF via llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`. Support all standard quant types (q4_k_m, q5_k_m, q8_0, f16, f32). 2-step process: convert → quantize → cleanup F16.
- `push_to_hub.py`: Push to HuggingFace Hub with auto-generated model card (YAML frontmatter, eval scores, usage example). Requires `HUGGINGFACE_TOKEN` env var.
- `docker-compose.yml` (root): `training` service (nvidia runtime, GPU deployment), `wandb-local` service (profile: wandb), `jupyter` service (profile: jupyter). `model-cache` volume for HF cache.
**Acceptance Criteria:**
- GGUF intermediate F16 cleaned up after quantization
- Model card has valid YAML frontmatter with `license`, `base_model`, `tags`
- `push_to_hub` raises `ExportError` (not `KeyError`) when token missing
- `docker-compose up training` starts without errors
- `pytest tests/test_export.py` passes

---

## Priority 2 — POLISH

### [ ] 2.1 Existing eval_runner.py Integration
**File:** `evals/eval_runner.py`
**What:** Update generic eval runner to call `mmlu_eval.py` and `hellaswag_eval.py` as sub-evals. Add `--compare` flag to trigger `compare_report.py`. Ensure LangSmith logging works alongside wandb.
**Acceptance Criteria:**
- `python evals/eval_runner.py --model results/lora_merged/ --compare` runs full eval pipeline
- Both LangSmith and wandb are logged simultaneously
- No circular imports between eval modules

### [ ] 2.2 LoRA Merge → GGUF Pipeline Script
**File:** `export/merge_and_export.py`
**What:** Update existing `merge_and_export.py` to optionally call `convert_gguf.py` after merging. Add `--to-gguf` flag and `--quant-type` option. One-command: merge LoRA → convert to GGUF.
**Acceptance Criteria:**
- `python export/merge_and_export.py --to-gguf --quant-type q4_k_m` runs end-to-end
- Existing merge-only behavior preserved (no breaking changes)

---

## Priority 3 — ENHANCEMENT

### [ ] 3.1 Perplexity Evaluation
**File:** `evals/perplexity_eval.py`
**What:** Compute perplexity on held-out data using `lm-eval-harness`. Run on WikiText-2 and custom domain test set. Compare base vs fine-tuned perplexity — lower is better for in-domain data.
**Acceptance Criteria:**
- `perplexity_eval.py` runs on WikiText-2 without GPU in mock mode
- Results appended to `compare_report.py` table
- Perplexity delta logged to wandb

### [ ] 3.2 Multi-GPU Training CI Test
**File:** `.github/workflows/ci.yml`
**What:** Add CI job that dry-runs NeMo config validation without GPU. Validates that `nemo/sft_config.yaml` + all `configs/*.yaml` load without pydantic errors. Runs `test_nemo_finetune.py` and `test_dpo_train.py` in CPU mock mode.
**Acceptance Criteria:**
- CI job completes in <3 minutes
- All config YAML files validated in CI
- No GPU required for CI config validation
