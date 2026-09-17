"""Dependency selection uses the resolved model family, not profile names."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import tomllib

from nemotron.recipes.embed.runtime import require_remote_runtime, runtime_project

EMBED = Path(__file__).resolve().parents[3] / "src/nemotron/recipes/embed"
STAGES = ("stage1_data_prep", "stage2_finetune", "stage3_eval")


@pytest.mark.parametrize("stage", STAGES)
def test_vl_stages_share_native_runtime(stage: str) -> None:
    assert runtime_project(EMBED / stage, "mistral3_vl") == EMBED / "runtimes/native"


@pytest.mark.parametrize("stage", ("stage1_data_prep", "stage3_eval"))
def test_text_prep_and_eval_keep_existing_projects(stage: str) -> None:
    assert runtime_project(EMBED / stage, "text") == EMBED / stage


def test_text_training_keeps_pre_vl_dependencies() -> None:
    project = runtime_project(EMBED / "stage2_finetune", "text")
    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert "transformers==5.12.1" in config["project"]["dependencies"]
    assert "a9f4423819c513fd08083324fe1f738746ac6e54" in config["tool"]["uv"]["sources"]["nemo-automodel"]["url"]


def test_native_runtime_pins_wheels_and_scopes_torch_index() -> None:
    project = runtime_project(EMBED / "stage2_finetune", "mistral3_vl")
    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert "transformers==5.15.1" in config["project"]["dependencies"]
    sources = config["tool"]["uv"]["sources"]
    assert sources["torch"]["index"] == sources["torchvision"]["index"] == "pytorch-cu129"
    assert sources["nemo-automodel"]["path"].endswith("0.7.0+4c50ab3c-py3-none-any.whl")
    assert "plugin-aa6c9652" in sources["data-designer-retrieval-sdg"]["path"]
    assert sources["data-designer-engine"]["path"].endswith("0.9.1+barejson.1-py3-none-any.whl")
    assert config["tool"]["uv"]["index"][0]["explicit"] is True


def test_generation_and_native_stages_use_the_same_reviewed_sdg_wheels() -> None:
    sdg = tomllib.loads((EMBED / "stage0_sdg/pyproject.toml").read_text())
    native = tomllib.loads((EMBED / "runtimes/native/pyproject.toml").read_text())
    for package in ("data-designer", "data-designer-config", "data-designer-engine", "data-designer-retrieval-sdg"):
        sdg_wheel = (EMBED / "stage0_sdg" / sdg["tool"]["uv"]["sources"][package]["path"]).resolve()
        native_wheel = (EMBED / "runtimes/native" / native["tool"]["uv"]["sources"][package]["path"]).resolve()
        assert sdg_wheel == native_wheel
        assert any(item.startswith(package + "==") for item in sdg["project"]["dependencies"])
    assert "torch" not in sdg["tool"]["uv"]["sources"]


@pytest.mark.parametrize("command", ("prep", "finetune", "eval"))
@pytest.mark.parametrize("family", ("text", "mistral3_vl"))
def test_cli_selects_runtime_after_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, family: str
) -> None:
    module = importlib.import_module(f"nemotron.cli.commands.embed.{command}")
    config = tmp_path / "resolved.yaml"
    config.write_text(
        "model_family: text\nlocal_backend: automodel\n"
        "query_max_length: 512\npassage_max_length: 8192\nimage_longest_edge: 1120\n"
        if command == "eval"
        else "model_family: text\n"
    )
    execute = Mock()
    monkeypatch.setattr("nemo_runspec.execution.execute_uv_local_from_spec", execute)
    overrides = [f"model_family={family}"]
    if command == "finetune":
        module._execute_uv_local(config, overrides, {"WANDB_ENABLED": "false"})
    else:
        module._execute_uv_local(config, overrides)
    kwargs = execute.call_args.kwargs
    assert kwargs["project_dir"] == runtime_project(Path(module.SPEC.script_path).parent, family)
    assert kwargs["train_path"] == config
    assert kwargs["passthrough"] == overrides


def test_unknown_family_fails() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding model family"):
        runtime_project(EMBED / "stage1_data_prep", "unknown")


def test_reviewed_wheel_lanes_refuse_remote_execution() -> None:
    with pytest.raises(RuntimeError, match="Stage 0 uses reviewed local"):
        require_remote_runtime("stage0_sdg")
    for stage in STAGES:
        with pytest.raises(RuntimeError, match="not included by the container packager"):
            require_remote_runtime(stage, "mistral3_vl")
        require_remote_runtime(stage, "text")


@pytest.mark.parametrize("command", ("sdg", "prep", "finetune", "eval"))
def test_remote_invocation_refuses_unpacked_reviewed_wheels(tmp_path: Path, command: str) -> None:
    module = importlib.import_module(f"nemotron.cli.commands.embed.{command}")
    train_path = tmp_path / "resolved.yaml"
    if command == "eval":
        train_path.write_text(
            "model_family: mistral3_vl\nlocal_backend: automodel\n"
            "query_max_length: 512\npassage_max_length: 8192\nimage_longest_edge: 1120\n"
        )
    elif command == "sdg":
        train_path.write_text("")
    else:
        train_path.write_text("model_family: mistral3_vl\n")

    with pytest.raises(RuntimeError, match="reviewed|multimodal runtime"):
        module._execute_remote(
            train_path=train_path,
            env=None,
            passthrough=[],
            attached=True,
            env_vars={},
            force_squash=False,
        )
