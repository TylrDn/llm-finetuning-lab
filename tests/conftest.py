"""Shared pytest stubs for heavy ML dependencies."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from unittest.mock import MagicMock


def _make_module(name: str, *, is_package: bool = False) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    if is_package:
        mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def _install_stubs() -> None:
    package_modules = [
        "torch",
        "transformers",
        "trl",
        "peft",
        "nemo.collections",
        "nemo.collections.nlp",
        "nemo.collections.nlp.models",
        "nemo.collections.nlp.models.language_modeling",
        "nemo.collections.nlp.parts",
    ]
    for name in package_modules:
        sys.modules.setdefault(name, _make_module(name, is_package=True))

    for name in ("wandb", "bitsandbytes", "pytorch_lightning", "omegaconf"):
        sys.modules.setdefault(name, _make_module(name))

    peft = sys.modules["peft"]
    peft.LoraConfig = MagicMock()
    peft.TaskType = MagicMock()
    peft.get_peft_model = MagicMock()

    transformers = sys.modules["transformers"]
    transformers.AutoModelForCausalLM = MagicMock()
    transformers.AutoTokenizer = MagicMock()
    transformers.BitsAndBytesConfig = MagicMock()

    trl = sys.modules["trl"]
    trl.DPOConfig = MagicMock()
    trl.DPOTrainer = MagicMock()

    torch = sys.modules["torch"]
    torch.bfloat16 = MagicMock()


_install_stubs()
