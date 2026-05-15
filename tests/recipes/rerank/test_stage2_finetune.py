from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron.recipes.rerank.stage2_finetune.train import (
    FinetuneConfig,
    _assert_optimizer_metadata_fp32,
    _install_fp32_restore_hook,
    _optimizer_metadata_dtype_counts,
)


def _metadata_entry(dtype: str) -> SimpleNamespace:
    return SimpleNamespace(properties=SimpleNamespace(dtype=dtype))


def test_finetune_config_keeps_default_checkpoint_frequency():
    assert FinetuneConfig().checkpoint_every_steps == 100


def test_optimizer_metadata_dtype_counts_only_adam_state():
    metadata = SimpleNamespace(
        state_dict_metadata={
            "optim.state.model.score.weight.step": _metadata_entry("torch.float32"),
            "optim.state.model.score.weight.exp_avg": _metadata_entry("torch.float32"),
            "optim.state.model.score.weight.exp_avg_sq": _metadata_entry("torch.float32"),
            "optim.param_groups.0.lr": _metadata_entry("torch.float64"),
            "model.score.weight": _metadata_entry("torch.bfloat16"),
        }
    )

    assert _optimizer_metadata_dtype_counts(metadata) == {
        "step": {"torch.float32": 1},
        "exp_avg": {"torch.float32": 1},
        "exp_avg_sq": {"torch.float32": 1},
    }


def test_optimizer_metadata_assertion_rejects_bf16_resume_checkpoint(tmp_path, monkeypatch):
    metadata = SimpleNamespace(
        state_dict_metadata={
            "optim.state.model.score.weight.step": _metadata_entry("torch.float32"),
            "optim.state.model.score.weight.exp_avg": _metadata_entry("torch.bfloat16"),
            "optim.state.model.score.weight.exp_avg_sq": _metadata_entry("torch.float32"),
        }
    )

    monkeypatch.setattr(
        "nemotron.recipes.rerank.stage2_finetune.train._read_optimizer_metadata_dtype_counts",
        lambda checkpoint_dir: _optimizer_metadata_dtype_counts(metadata),
    )
    with pytest.raises(AssertionError, match="Expected resumed optimizer checkpoint metadata to be fp32"):
        _assert_optimizer_metadata_fp32(tmp_path)


def test_fp32_restore_hook_casts_model_before_optimizer_load(monkeypatch, tmp_path):
    events: list[str] = []

    class ModelPart:
        dtype = "torch.bfloat16"

        def float(self):
            events.append("cast_model_to_fp32")
            self.dtype = "torch.float32"

    model_part = ModelPart()

    class Checkpointer:
        config = SimpleNamespace(enabled=True, checkpoint_dir=str(tmp_path))

        def load_model(self, model, model_dir):
            events.append("load_model")

        def load_optimizer(self, optimizer, model, ckpt_dir, scheduler):
            events.append(f"load_optimizer_with_{model_part.dtype}")

    class Recipe:
        def __init__(self):
            self.checkpointer = Checkpointer()
            self.model_parts = [model_part]

        def load_checkpoint(self, restore_from=None):
            events.append(f"original_load_checkpoint:{restore_from}")

        def _validate_checkpoint_dir_exists(self, ckpt_dir, restore_from, is_rank_0):
            events.append(f"validate:{Path(ckpt_dir).name}")

        def _load_checkpoint_tracked_state(self, ckpt_dir):
            events.append(f"load_tracked_state:{Path(ckpt_dir).name}")
            return object(), object(), object()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            distributed=SimpleNamespace(
                is_initialized=lambda: False,
                get_rank=lambda: 0,
            )
        ),
    )

    recipe = Recipe()
    _install_fp32_restore_hook(recipe)
    recipe.load_checkpoint(str(tmp_path / "epoch_0_step_1"))

    assert events == [
        "validate:epoch_0_step_1",
        "load_tracked_state:epoch_0_step_1",
        "load_model",
        "cast_model_to_fp32",
        "load_optimizer_with_torch.float32",
    ]
