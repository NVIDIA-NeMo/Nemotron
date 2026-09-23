"""Public exact-step configuration and artifact regressions."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.kit.artifacts.embed import EmbedModelArtifact
from nemotron.recipes.embed.stage2_finetune import train


def test_default_training_budget_remains_epoch_based() -> None:
    """Existing callers retain the default three-epoch budget."""
    cfg = train.FinetuneConfig()
    assert cfg.num_epochs == 3
    assert cfg.max_steps is None


def test_exact_step_budget() -> None:
    """A step-based configuration explicitly clears the epoch limit."""
    cfg = train.FinetuneConfig(num_epochs=None, max_steps=96)
    assert cfg.num_epochs is None
    assert cfg.max_steps == 96


@pytest.mark.parametrize(
    "options",
    [
        {"num_epochs": 3, "max_steps": 96},
        {"num_epochs": None, "max_steps": None},
        {"num_epochs": None, "max_steps": 0},
        {"num_epochs": None, "max_steps": True},
        {"num_epochs": None, "max_steps": 1.5},
    ],
)
def test_invalid_training_budget(options: dict[str, Any]) -> None:
    """Ambiguous or non-positive/non-integer budgets fail before execution."""
    with pytest.raises(ValidationError):
        train.FinetuneConfig(**options)


@pytest.mark.parametrize("profile", ["default", "llama", "mistral3-vl"])
def test_public_dotlist_parses_strict_exact_step_budget(profile: str) -> None:
    """Profile defaults do not replace an explicit null epoch override."""
    cfg = train.load_config(
        train.STAGE_PATH / "config" / f"{profile}.yaml",
        ["num_epochs=null", "max_steps=7"],
        train.FinetuneConfig,
    )

    assert cfg.num_epochs is None
    assert cfg.max_steps == 7


@pytest.mark.parametrize("value", ["true", "1.5", "0", "-1"])
def test_public_dotlist_rejects_non_positive_strict_max_steps(value: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        train.load_config(
            train.STAGE_PATH / "config/default.yaml",
            ["num_epochs=null", f"max_steps={value}"],
            train.FinetuneConfig,
        )


def test_epoch_artifact_retains_epoch_metadata(tmp_path) -> None:
    artifact = EmbedModelArtifact(
        path=tmp_path,
        base_model="base",
        training_examples=8,
        num_epochs=3,
        global_batch_size=8,
        learning_rate=1.0e-5,
        temperature=0.02,
    )

    serialized = artifact.model_dump(mode="json")
    assert serialized["num_epochs"] == 3
    assert serialized["max_steps"] is None
    assert artifact.metadata["num_epochs"] == 3
    assert "max_steps" not in artifact.metadata


def test_step_artifact_records_exact_budget(tmp_path) -> None:
    artifact = EmbedModelArtifact(
        path=tmp_path,
        base_model="base",
        training_examples=8,
        num_epochs=None,
        max_steps=96,
        global_batch_size=8,
        learning_rate=1.0e-5,
        temperature=0.02,
    )

    serialized = artifact.model_dump(mode="json")
    assert serialized["num_epochs"] is None
    assert serialized["max_steps"] == 96
    assert "num_epochs" not in artifact.metadata
    assert artifact.metadata["max_steps"] == 96
