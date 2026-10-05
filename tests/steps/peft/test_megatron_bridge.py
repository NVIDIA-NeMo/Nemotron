# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks and PEFT import fallbacks for ``steps/peft/megatron_bridge``."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nemotron.steps._runners import megatron_bridge as runner

from .._step_helpers import assert_step_static, step_dir


def test_peft_megatron_bridge_static() -> None:
    assert_step_static(
        step_dir(__file__, "peft", "megatron_bridge"),
        expected_name="steps/peft/megatron_bridge",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_lightning35_overlay_inherits_default_yaml() -> None:
    from nemo_runspec.config.loader import load_config

    step = step_dir(__file__, "peft", "megatron_bridge")
    cfg = load_config(step / "config" / "lightning35.yaml")
    assert "defaults" not in cfg
    assert cfg.recipe._target_ == (
        "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning_peft_config"
    )
    assert cfg.recipe.packed_sequence is None
    assert cfg.recipe.peft is None
    assert cfg.peft.type == "lora"
    assert cfg.peft.dim == 32
    assert "in_proj" in list(cfg.peft.target_modules)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.train.global_batch_size == 128
    assert cfg.hf_load.inherit_save_as_load is False
    assert cfg.checkpoint.fully_parallel_save is False
    assert cfg.run.env.container_image == "nvcr.io/nvidia/nemo:26.08"


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def _peft_yaml(**overrides: object) -> dict:
    block = {
        "type": "lora",
        "dim": 32,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["linear_qkv", "in_proj"],
    }
    block.update(overrides)
    return {"peft": block}


def test_maybe_apply_peft_skips_when_block_missing() -> None:
    cfg = SimpleNamespace(peft=None)
    runner._maybe_apply_peft(cfg, {})
    assert cfg.peft is None


def test_maybe_apply_peft_uses_dataset_utils_on_nemo_26_08(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nemo:26.08 / Megatron-Bridge 0c565c9 moved default_peft_config here."""
    calls: list[tuple[object, dict]] = []

    def fake_helper(*, peft_scheme, **kwargs):
        calls.append((peft_scheme, kwargs))
        return SimpleNamespace(scheme=peft_scheme, **kwargs)

    dataset_utils = ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    dataset_utils.default_peft_config = fake_helper
    _install_module(monkeypatch, "megatron.bridge.recipes.utils.dataset_utils", dataset_utils)

    import importlib

    original_import = importlib.import_module

    def boom(name: str, *args, **kwargs):
        if name.endswith("finetune_utils"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", boom)

    cfg = SimpleNamespace(peft=None)
    runner._maybe_apply_peft(cfg, _peft_yaml())

    assert calls == [
        (
            "lora",
            {
                "dim": 32,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": ["linear_qkv", "in_proj"],
            },
        )
    ]
    assert cfg.peft.scheme == "lora"
    assert cfg.peft.dim == 32


def test_maybe_apply_peft_constructs_lora_when_helpers_are_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLoRA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    lora_mod = ModuleType("megatron.bridge.peft.lora")
    lora_mod.LoRA = FakeLoRA
    _install_module(monkeypatch, "megatron.bridge.peft.lora", lora_mod)

    import importlib

    original_import = importlib.import_module

    def boom(name: str, *args, **kwargs):
        if name.endswith(("finetune_utils", "dataset_utils")):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", boom)

    cfg = SimpleNamespace(peft=None)
    runner._maybe_apply_peft(cfg, _peft_yaml(type="lora", dim=16))

    assert isinstance(cfg.peft, FakeLoRA)
    assert cfg.peft.kwargs["dim"] == 16
    assert cfg.peft.kwargs["target_modules"] == ["linear_qkv", "in_proj"]
