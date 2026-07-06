"""Tests for embed finetune optimizer backend selection."""

from __future__ import annotations

from typing import Any

import pytest

from nemotron.recipes.embed.stage2_finetune import train


def _as_dict(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Stand in for Automodel ConfigNode while preserving raw config for assertions."""
    return raw_config


def test_auto_optimizer_uses_fused_adam_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "_can_import_fused_adam", lambda: (True, None))
    monkeypatch.setattr(train, "_can_import_flash_adamw", lambda: (True, None))

    raw_config, optimizer_backend = train._load_automodel_config(train.FinetuneConfig(), _as_dict)

    assert optimizer_backend == "fused_adam"
    assert raw_config["recipe"] == "TrainBiEncoderRecipe"
    assert raw_config["model"]["_target_"] == "nemo_automodel.NeMoAutoModelBiEncoder.from_pretrained"
    assert raw_config["tokenizer"]["_target_"] == "nemo_automodel.NeMoAutoTokenizer.from_pretrained"
    assert raw_config["tokenizer"]["add_eos_token"] is False
    assert raw_config["dataloader"]["dataset"]["n_passages"] == 5
    assert raw_config["dataloader"]["collate_fn"]["_target_"] == (
        "nemo_automodel.components.datasets.llm.BiEncoderCollator"
    )
    assert raw_config["dataloader"]["collate_fn"]["query_prefix"] == "query: "
    assert raw_config["dataloader"]["collate_fn"]["passage_prefix"] == "passage: "
    assert raw_config["distributed"]["strategy"] == "fsdp2"
    assert raw_config["optimizer"]["_target_"] == (
        "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
    )
    assert raw_config["optimizer"]["adam_w_mode"] is True
    assert raw_config["optimizer"]["master_weights"] is True


def test_flash_adamw_backend_rewrites_optimizer_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "_can_import_fused_adam", lambda: (False, "missing TE"))
    monkeypatch.setattr(train, "_can_import_flash_adamw", lambda: (True, None))

    cfg = train.FinetuneConfig(
        optimizer_backend="flash_adamw",
        flash_adamw_master_weight_bits=24,
    )
    raw_config, optimizer_backend = train._load_automodel_config(cfg, _as_dict)

    assert optimizer_backend == "flash_adamw"
    assert raw_config["optimizer"] == {
        "_target_": "flashoptim.FlashAdamW",
        "lr": 5.0e-6,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "quantize": False,
        "compress_state_dict": False,
        "master_weight_bits": 24,
        "fused": True,
    }
    assert raw_config["model"]["torch_dtype"] == "bfloat16"


def test_flash_adamw_disables_master_weights_for_fp32_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "_can_import_fused_adam", lambda: (False, "missing TE"))
    monkeypatch.setattr(train, "_can_import_flash_adamw", lambda: (True, None))

    cfg = train.FinetuneConfig(
        optimizer_backend="flash_adamw",
        flash_adamw_master_weight_bits=None,
    )
    raw_config, optimizer_backend = train._load_automodel_config(cfg, _as_dict)

    assert optimizer_backend == "flash_adamw"
    assert raw_config["optimizer"]["master_weight_bits"] is None


@pytest.mark.parametrize(
    ("configured_prefix", "expected_collator_prefix", "expected_text"),
    [
        ("query: ", "query:", "query: example"),
        ("passage: ", "passage:", "passage: example"),
        ("custom:  ", "custom: ", "custom:  example"),
        ("", "", "example"),
    ],
)
def test_automodel_collator_preserves_configured_separator(
    configured_prefix: str, expected_collator_prefix: str, expected_text: str
) -> None:
    collator_prefix = train._automodel_collator_prefix(configured_prefix)
    effective_text = f"{collator_prefix} example" if collator_prefix else "example"

    assert collator_prefix == expected_collator_prefix
    assert effective_text == expected_text


def test_checkpoint_interval_auto_scaling_can_be_disabled() -> None:
    cfg = train.FinetuneConfig(
        checkpoint_every_steps=1000,
        val_every_steps=1000,
        auto_scale_checkpoint_intervals=False,
    )

    _, _, checkpoint_every, val_every = train._auto_scale_hyperparams(cfg, num_examples=1145)

    assert checkpoint_every == 1000
    assert val_every == 1000
