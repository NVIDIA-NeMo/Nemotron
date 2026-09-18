# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression coverage for the explicit native FP32 optimizer path."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from nemotron.recipes.embed.stage2_finetune import train
from torch.distributed.checkpoint import FileSystemReader


@pytest.mark.parametrize("model_family", ["text", "mistral3_vl"])
def test_native_fp32_does_not_require_optional_optimizers(monkeypatch, model_family):
    def unavailable():
        raise AssertionError("Native AdamW must not import optional GPU extensions")

    monkeypatch.setattr(train, "_can_import_fused_adam", unavailable)
    monkeypatch.setattr(train, "_can_import_flash_adamw", unavailable)
    cfg = train.FinetuneConfig(model_family=model_family, optimizer_backend="torch_adamw")
    raw, backend = train._load_automodel_config(cfg, dict)
    assert backend == "torch_adamw"
    assert raw["optimizer"]["_target_"] == "torch.optim.AdamW"
    assert raw["optimizer"]["fused"] is True
    assert raw["model"]["torch_dtype"] == "float32"
    assert {raw["distributed"]["mp_policy"][key] for key in ("param_dtype", "reduce_dtype", "output_dtype")} == {
        "float32"
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_moment_dtype_is_checked(monkeypatch, dtype):
    metadata = SimpleNamespace(
        state_dict_metadata={
            f"optim.state.weight.{name}": SimpleNamespace(properties=SimpleNamespace(dtype=dtype))
            for name in ("exp_avg", "exp_avg_sq")
        }
    )
    monkeypatch.setattr(FileSystemReader, "read_metadata", lambda self: metadata)
    if dtype == torch.float32:
        assert train._assert_optimizer_metadata_fp32(Path("checkpoint")) == {"exp_avg": 1, "exp_avg_sq": 1}
    else:
        with pytest.raises(AssertionError, match="FP32"):
            train._assert_optimizer_metadata_fp32(Path("checkpoint"))


def test_checkpoint_missing_moment_is_rejected(monkeypatch):
    monkeypatch.setattr(FileSystemReader, "read_metadata", lambda self: SimpleNamespace(state_dict_metadata={}))
    with pytest.raises(AssertionError, match="Missing"):
        train._assert_optimizer_metadata_fp32(Path("checkpoint"))
