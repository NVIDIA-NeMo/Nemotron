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
