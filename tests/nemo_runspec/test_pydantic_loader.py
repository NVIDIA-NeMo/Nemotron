"""Tests for recipe script Pydantic config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ConfigDict, Field

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config


class _PathConfig(RecipeSettings):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(default=Path("output"))


class _BoolConfig(RecipeSettings):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    name: str = "default"


class _ArtifactPathConfig(RecipeSettings):
    model_config = ConfigDict(extra="forbid")

    artifact_root: Path
    output_dir: Path


def test_load_config_resolves_oc_env_with_default(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("output_dir: ${oc.env:NEMO_RUN_DIR,.}/output/rerank/stage0_sdg\n")
    monkeypatch.delenv("NEMO_RUN_DIR", raising=False)

    cfg = load_config(config, [], _PathConfig)

    assert cfg.output_dir == Path("output/rerank/stage0_sdg")


def test_load_config_resolves_oc_env_from_environment(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("output_dir: ${oc.env:NEMO_RUN_DIR,.}/output/rerank/stage0_sdg\n")
    monkeypatch.setenv("NEMO_RUN_DIR", "/shared/nemotron")

    cfg = load_config(config, [], _PathConfig)

    assert cfg.output_dir == Path("/shared/nemotron/output/rerank/stage0_sdg")


def test_load_config_resolves_top_level_path_reference(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("artifact_root: ./output/embed/nemotron-3-1b\noutput_dir: ${artifact_root}/stage0_sdg\n")

    cfg = load_config(config, [], _ArtifactPathConfig)

    assert cfg.output_dir == Path("output/embed/nemotron-3-1b/stage0_sdg")


def test_cli_root_override_updates_derived_path(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("artifact_root: ./output/embed/nemotron-3-1b\noutput_dir: ${artifact_root}/stage0_sdg\n")

    cfg = load_config(
        config,
        ["artifact_root=/tmp/embed-8b"],
        _ArtifactPathConfig,
    )

    assert cfg.artifact_root == Path("/tmp/embed-8b")
    assert cfg.output_dir == Path("/tmp/embed-8b/stage0_sdg")


def test_top_level_reference_resolves_inside_oc_env_default(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "artifact_root: ./output/embed/nemotron-3-1b\noutput_dir: ${oc.env:MODEL_OUTPUT,${artifact_root}/stage0_sdg}\n"
    )

    monkeypatch.delenv("MODEL_OUTPUT", raising=False)
    default_cfg = load_config(config, [], _ArtifactPathConfig)
    assert default_cfg.output_dir == Path("output/embed/nemotron-3-1b/stage0_sdg")

    monkeypatch.setenv("MODEL_OUTPUT", "/tmp/explicit-output")
    overridden_cfg = load_config(config, [], _ArtifactPathConfig)
    assert overridden_cfg.output_dir == Path("/tmp/explicit-output")


def test_load_config_rejects_unknown_cli_override(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("enabled: false\n")

    with pytest.raises(SystemExit):
        load_config(config, ["typo=true"], _BoolConfig)


def test_load_config_accepts_bare_bool_flag(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("enabled: false\n")

    cfg = load_config(config, ["--enabled"], _BoolConfig)

    assert cfg.enabled is True


def test_load_config_preserves_key_value_pair_after_flag(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("enabled: true\nname: default\n")

    cfg = load_config(config, ["--enabled", "false", "--name", "custom"], _BoolConfig)

    assert cfg.enabled is False
    assert cfg.name == "custom"


def test_load_config_rejects_missing_required_oc_env(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("output_dir: ${oc.env:REQUIRED_MODEL_PATH}\n")
    monkeypatch.delenv("REQUIRED_MODEL_PATH", raising=False)

    with pytest.raises(ValueError, match="REQUIRED_MODEL_PATH"):
        load_config(config, [], _PathConfig)
