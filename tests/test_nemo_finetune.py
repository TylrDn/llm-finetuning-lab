"""Tests for NeMo SFT fine-tuning helpers."""

from __future__ import annotations

import copy

import pytest

from nemo.nemo_finetune import load_config, validate_config

VALID_CONFIG = {
    "trainer": {"devices": 1, "max_steps": 100},
    "model": {"data": {"train_ds": {"file_names": ["data/train.jsonl"]}}},
    "exp_manager": {"exp_dir": "./outputs"},
}


def test_validate_config_accepts_minimal_valid_config() -> None:
    validate_config(copy.deepcopy(VALID_CONFIG))


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda cfg: cfg.pop("trainer"), "trainer"),
        (lambda cfg: cfg["trainer"].update({"devices": 0}), "trainer.devices"),
        (lambda cfg: cfg["model"]["data"]["train_ds"].update({"file_names": []}), "file_names"),
        (lambda cfg: cfg.pop("exp_manager"), "exp_manager"),
    ],
)
def test_validate_config_rejects_invalid_config(mutator, match: str) -> None:
    cfg = copy.deepcopy(VALID_CONFIG)
    mutator(cfg)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_load_config_reads_sft_yaml() -> None:
    cfg = load_config("nemo/sft_config.yaml")
    assert cfg["trainer"]["devices"] >= 1
    assert cfg["model"]["data"]["train_ds"]["file_names"]


def test_validate_config_rejects_missing_restore_path(tmp_path) -> None:
    cfg = copy.deepcopy(VALID_CONFIG)
    cfg["model"]["restore_from_path"] = str(tmp_path / "missing.nemo")
    with pytest.raises(ValueError, match="restore_from_path"):
        validate_config(cfg)


def test_build_trainer_requires_nemo_install() -> None:
    from nemo.nemo_finetune import build_trainer

    with pytest.raises(RuntimeError, match="NeMo is not available"):
        build_trainer(copy.deepcopy(VALID_CONFIG))


def test_setup_wandb_skips_without_project() -> None:
    from nemo.nemo_finetune import _setup_wandb

    _setup_wandb({"exp_manager": {"wandb_logger_kwargs": {}}})
