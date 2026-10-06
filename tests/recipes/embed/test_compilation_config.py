"""Boundary tests for optional embedding deployment compilation settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_runspec.config import load_pydantic_config
from nemotron.recipes.embed.stage5_deploy import deploy


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path) -> None:
    """Keep command-building filesystem effects in this test's cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def command_for(model_dir: Path, **overrides) -> tuple[deploy.DeployConfig, list[str]]:
    """Exercise the real public settings and deployment command builder."""
    cfg = deploy.DeployConfig(nim_image="example.invalid/nim:test", model_dir=model_dir, **overrides)
    return cfg, deploy.build_docker_command(cfg)


@pytest.mark.parametrize("model_family", ["nemotron3_text", "mistral3_vl"])
def test_compilation_is_opt_in_for_text_and_image_profiles(tmp_path, model_family) -> None:
    cfg, default = command_for(tmp_path, backend="vllm", model_family=model_family)
    _, explicit_null = command_for(tmp_path, backend="vllm", vllm_compilation_config=None, model_family=model_family)
    assert cfg.vllm_compilation_config is None
    assert "--compilation-config" not in default
    assert default == explicit_null


@pytest.mark.parametrize(
    "settings",
    [{}, {"custom_ops": ["all", "-conv2d"], "debug_dump_path": "/tmp/compile artifacts", "mode": 0}],
)
def test_compilation_mapping_is_one_json_argv_item(tmp_path, settings) -> None:
    hf_overrides = {"vision_config": {"image_size": 1120}}
    _, baseline = command_for(tmp_path, backend="vllm", vllm_hf_overrides=hf_overrides)
    cfg, command = command_for(
        tmp_path, backend="vllm", vllm_hf_overrides=hf_overrides, vllm_compilation_config=settings
    )
    assert cfg.vllm_compilation_config == settings
    assert command.count("--compilation-config") == 1
    index = command.index("--compilation-config")
    assert json.loads(command[index + 1]) == settings
    assert command[:index] + command[index + 2 :] == baseline
    assert json.loads(command[command.index("--hf-overrides") + 1]) == hf_overrides


def test_nim_command_ignores_vllm_compilation_settings(tmp_path) -> None:
    _, baseline = command_for(tmp_path, backend="nim")
    _, command = command_for(tmp_path, backend="nim", vllm_compilation_config={"custom_ops": ["all", "-conv2d"]})
    assert command == baseline
    assert "--compilation-config" not in command


@pytest.mark.parametrize("invalid", ['{"custom_ops":["-conv2d"]}', ["-conv2d"], 1])
def test_compilation_rejects_non_mapping_settings(tmp_path, invalid) -> None:
    with pytest.raises(ValidationError, match="vllm_compilation_config") as error:
        deploy.DeployConfig(model_dir=tmp_path, vllm_compilation_config=invalid)
    assert error.value.errors()[0]["type"] == "dict_type"


@pytest.mark.parametrize("profile", ["default", "llama", "mistral3-vl"])
def test_profile_defaults_and_explicit_dotlist_override(profile) -> None:
    path = deploy.STAGE_PATH / "config" / f"{profile}.yaml"
    default = load_pydantic_config(path, [], deploy.DeployConfig)
    assert default.vllm_compilation_config is None
    configured = load_pydantic_config(
        path,
        ["backend=vllm", 'vllm_compilation_config={"custom_ops":["all","-conv2d"]}'],
        deploy.DeployConfig,
    )
    command = deploy.build_docker_command(configured)
    assert json.loads(command[command.index("--compilation-config") + 1]) == {"custom_ops": ["all", "-conv2d"]}
