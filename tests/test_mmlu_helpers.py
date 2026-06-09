"""Tests for MMLU helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import sys

sys.modules.setdefault("datasets", MagicMock())

from evals.mmlu_eval import SUBJECT_GROUPS, _extract_answer_letter, _subject_to_group


def test_mmlu_subject_group_mapping() -> None:
    assert _subject_to_group("high_school_physics") == "STEM"
    assert len(SUBJECT_GROUPS) >= 4


def test_mmlu_extract_answer_letter() -> None:
    assert _extract_answer_letter("B") == "B"
    assert _extract_answer_letter("Answer: C") == "C"
    assert _extract_answer_letter("no letter here") is None
