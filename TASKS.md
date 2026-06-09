# llm-finetuning-lab — Task Board

**Repo:** llm-finetuning-lab
**Completion:** 100% COMPLETE
**Last Audit:** 2026-06-09
**Status:** Priority 1 production bar met — NeMo/DPO/export/eval tests, CI `--cov-fail-under=80`, eval_runner integration tests.

---

## Priority 1 — CRITICAL

### [x] 1.1 NeMo SFT Fine-Tuning Script (NVIDIA Differentiator)
### [x] 1.2 DPO Trainer + Config
### [x] 1.3 MMLU Benchmark Evaluation
### [x] 1.4 HellaSwag Benchmark Evaluation
### [x] 1.5 Model Comparison Report
### [x] 1.6 Export Tools + docker-compose

**Verification:** `pytest tests/` green with `--cov-fail-under=80`; `tests/test_nemo_finetune.py`, `test_dpo_train.py`, `test_export.py`, `test_evals.py`, `test_mmlu_helpers.py` cover config validation, export helpers, LangSmith eval_runner, and comparison reports.

---

## Priority 2 — POLISH

### [ ] 2.1 Live GPU integration tests (`@pytest.mark.gpu`)
### [ ] 2.2 Full lm-eval-harness CI job with cached models

---

## Priority 3 — ENHANCEMENT

### [ ] 3.1 NeMo checkpoint resume from interrupted runs
### [ ] 3.2 Multi-node DDP launch scripts
