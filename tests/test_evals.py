"""Tests for evaluation scripts and LangSmith integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from evals.compare_report import (
    _build_summary_table_md,
    _fmt_acc,
    _fmt_delta,
    _interpret_results,
    _load_results,
    generate_comparison,
)
from evals.eval_runner import exact_match_evaluator, run_eval


def test_exact_match_evaluator_scores_match() -> None:
    run = MagicMock(outputs={"response": "Paris"})
    example = MagicMock(outputs={"expected": "paris"})
    result = exact_match_evaluator(run, example)
    assert result["score"] == 1


def test_run_eval_invokes_langsmith_evaluate() -> None:
    with patch("evals.eval_runner.evaluate") as mock_evaluate:
        with patch("evals.eval_runner.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="answer"))]
            )
            run_eval(dataset_name="test-dataset", model_endpoint="http://localhost:8000/v1")
    mock_evaluate.assert_called_once()



def test_compare_report_formatters() -> None:
    assert _fmt_acc(0.5123) == "51.23%"
    assert "+1.00%" in _fmt_delta(0.5, 0.51)


def test_load_results_finds_mmlu_files(tmp_path) -> None:
    (tmp_path / "mmlu_run.json").write_text('{"overall": 0.5}', encoding="utf-8")
    loaded = _load_results(str(tmp_path))
    assert loaded["mmlu"]["overall"] == 0.5


def test_build_summary_table_includes_benchmark_rows() -> None:
    table = _build_summary_table_md(
        base_mmlu={"overall": 0.4, "by_group": {"STEM": {"accuracy": 0.4}}},
        ft_mmlu={"overall": 0.45, "by_group": {"STEM": {"accuracy": 0.45}}},
        base_hs=None,
        ft_hs=None,
    )
    assert "MMLU" in table
    assert "STEM" in table


def test_interpret_results_describes_improvement() -> None:
    text = _interpret_results(
        base_mmlu={"overall": 0.4, "by_group": {"STEM": {"accuracy": 0.4}}},
        ft_mmlu={"overall": 0.45, "by_group": {"STEM": {"accuracy": 0.45}}},
        base_hs=None,
        ft_hs=None,
    )
    assert "improved" in text


def test_generate_comparison_writes_markdown(tmp_path) -> None:
    base = tmp_path / "base"
    finetuned = tmp_path / "finetuned"
    out = tmp_path / "reports"
    base.mkdir()
    finetuned.mkdir()
    (base / "mmlu_base.json").write_text(
        json.dumps({"overall": 0.40, "by_group": {"STEM": {"accuracy": 0.40}}}),
        encoding="utf-8",
    )
    (finetuned / "mmlu_finetuned.json").write_text(
        json.dumps({"overall": 0.45, "by_group": {"STEM": {"accuracy": 0.45}}}),
        encoding="utf-8",
    )
    with patch("evals.compare_report._upload_wandb_artifact"):
        generate_comparison(
            base_results_dir=str(base),
            finetuned_results_dir=str(finetuned),
            output_dir=str(out),
        )
    reports = list(out.glob("comparison_report_*.md"))
    assert reports
    assert "Comparison Report" in reports[0].read_text(encoding="utf-8")
