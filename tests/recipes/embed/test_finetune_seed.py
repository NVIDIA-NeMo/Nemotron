"""Seed contract tests for retrieval training configuration."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.recipes.embed.stage2_finetune import train


def _as_dict(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Expose the configuration passed to the training engine."""
    return raw_config


@pytest.mark.parametrize("model_family", ["text", "mistral3_vl"])
@pytest.mark.parametrize("seed", [None, 0, 173, 4294967295])
def test_seed_reaches_training_and_dataset_configuration(
    monkeypatch: pytest.MonkeyPatch, model_family: str, seed: int | None
) -> None:
    """Default and selected seeds reach both consumers, not just the CLI model."""
    monkeypatch.setattr(train, "_can_import_fused_adam", lambda: (True, None))
    monkeypatch.setattr(train, "_can_import_flash_adamw", lambda: (True, None))
    overrides = {} if seed is None else {"seed": seed}
    cfg = train.FinetuneConfig(model_family=model_family, **overrides)

    raw_config, _ = train._load_automodel_config(cfg, _as_dict)

    expected_seed = 42 if seed is None else seed
    dataset = raw_config["dataset"] if model_family == "mistral3_vl" else raw_config["dataloader"]["dataset"]
    assert cfg.seed == expected_seed
    assert raw_config["seed"] == expected_seed
    assert dataset["seed"] == expected_seed


@pytest.mark.parametrize("seed", [-1, 4294967296, 1.5, True])
def test_invalid_seed_rejected_before_training(seed: object) -> None:
    """Invalid inputs fail public configuration validation."""
    with pytest.raises(ValidationError, match="seed"):
        train.FinetuneConfig(seed=seed)


@pytest.mark.parametrize("profile", ["default", "llama", "mistral3-vl"])
def test_dotlist_seed_override_reaches_training_and_dataset(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    """The public CLI loader preserves the selected seed in every profile."""
    monkeypatch.setattr(train, "_can_import_fused_adam", lambda: (True, None))
    monkeypatch.setattr(train, "_can_import_flash_adamw", lambda: (True, None))
    cfg = train.load_config(train.STAGE_PATH / "config" / f"{profile}.yaml", ["seed=173"], train.FinetuneConfig)

    raw_config, _ = train._load_automodel_config(cfg, _as_dict)

    dataset = raw_config["dataset"] if cfg.model_family == "mistral3_vl" else raw_config["dataloader"]["dataset"]
    assert (cfg.seed, raw_config["seed"], dataset["seed"]) == (173, 173, 173)
