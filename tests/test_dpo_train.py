"""Tests for DPO training configuration helpers."""

from __future__ import annotations

import pytest

from training.dpo_train import _build_bnb_config, load_config


def test_load_config_reads_dpo_yaml() -> None:
    cfg = load_config("configs/dpo.yaml")
    assert cfg["base_model"]
    assert cfg["output_dir"]


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yaml")


def test_build_bnb_config_disabled_without_quantization() -> None:
    assert _build_bnb_config({}) is None


def test_build_bnb_config_enabled_for_4bit() -> None:
    cfg = {"quantization": {"load_in_4bit": True}}
    bnb = _build_bnb_config(cfg)
    assert bnb is not None
