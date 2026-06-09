"""Tests for model export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from export.convert_gguf import SUPPORTED_QUANT_TYPES, _check_llama_cpp
from export.push_to_hub import _find_latest_eval_file, _load_eval_results


def test_supported_quant_types_include_common_values() -> None:
    assert "Q4_K_M" in SUPPORTED_QUANT_TYPES
    assert "Q8_0" in SUPPORTED_QUANT_TYPES


def test_check_llama_cpp_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _check_llama_cpp(str(tmp_path / "missing"))


def test_load_eval_results_reads_latest_json(tmp_path: Path) -> None:
    older = tmp_path / "mmlu_old.json"
    newer = tmp_path / "mmlu_new.json"
    older.write_text(json.dumps({"overall": 0.5}), encoding="utf-8")
    newer.write_text(json.dumps({"overall": 0.7}), encoding="utf-8")
    assert _find_latest_eval_file(str(tmp_path), "mmlu_") == str(newer)

    results = _load_eval_results(str(tmp_path))
    assert results["mmlu"]["overall"] == 0.7
