"""Stage-local manifests preserve text compatibility without local wheel projects."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import tomllib

from nemo_runspec._pyproject import _write_temp_pyproject

EMBED = Path(__file__).resolve().parents[3] / "src/nemotron/recipes/embed"
STAGES = ("stage1_data_prep", "stage2_finetune", "stage3_eval")
AUTOMODEL_REV = "0e02c4274d09e7e09159009916f7348fcb7dc9bb"
AUTOMODEL_URL = f"https://github.com/NVIDIA-NeMo/Automodel/archive/{AUTOMODEL_REV}.tar.gz"


@pytest.mark.parametrize("stage", STAGES)
def test_model_dependencies_are_stage_local_exclusive_extras(stage: str) -> None:
    config = tomllib.loads((EMBED / stage / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    assert "transformers==5.15.1" in extras["vl"]
    assert "torchvision>=0.25,<0.26" in extras["vl"]
    expected_text = "transformers==5.12.1" if stage == "stage2_finetune" else "transformers>=5.1,<5.6"
    assert expected_text in extras["text"]
    assert config["tool"]["uv"]["conflicts"] == [[{"extra": "text"}, {"extra": "vl"}]]
    sources = config["tool"]["uv"]["sources"]
    model_sources = sources["nemo-automodel"]
    if isinstance(model_sources, dict):
        model_sources = [model_sources]
    assert next(source for source in model_sources if source["extra"] == "vl") == {
        "url": AUTOMODEL_URL,
        "extra": "vl",
    }
    assert sources["torch"]["index"] == sources["torchvision"]["index"] == "pytorch-cu129"
    assert config["tool"]["uv"]["index"][0]["explicit"] is True
    assert config["tool"]["nemotron"]["container-extras"] == ["text"]
    if stage == "stage2_finetune":
        assert "wandb>=0.21,<1" in config["project"]["dependencies"]
        assert next(source for source in model_sources if source["extra"] == "text")["url"] == (
            "https://github.com/NVIDIA-NeMo/Automodel/archive/a9f4423819c513fd08083324fe1f738746ac6e54.tar.gz"
        )
    elif stage == "stage1_data_prep":
        assert "nemo-automodel==0.4.0" in extras["text"]


@pytest.mark.parametrize("stage", ("stage0_sdg", *STAGES))
def test_no_stage_depends_on_local_wheels(stage: str) -> None:
    text = (EMBED / stage / "pyproject.toml").read_text()
    assert ".whl" not in text
    assert "barejson" not in text
    assert "runtimes/" not in text


@pytest.mark.parametrize("stage", STAGES)
def test_container_manifest_preserves_conflicts_and_git_sources(stage: str) -> None:
    config = tomllib.loads((EMBED / stage / "pyproject.toml").read_text())
    generated = _write_temp_pyproject(config, EMBED / stage, ["torch", "torchvision"])
    actual = tomllib.loads((generated / "pyproject.toml").read_text())
    assert actual["project"]["optional-dependencies"] == config["project"]["optional-dependencies"]
    assert actual["tool"]["uv"]["sources"] == config["tool"]["uv"]["sources"]
    assert actual["tool"]["uv"]["conflicts"] == config["tool"]["uv"]["conflicts"]


@pytest.mark.parametrize("command", ("prep", "finetune", "eval"))
@pytest.mark.parametrize("family", ("text", "mistral3_vl"))
def test_cli_selects_stage_extra_after_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, family: str
) -> None:
    module = importlib.import_module(f"nemotron.cli.commands.embed.{command}")
    config = tmp_path / "resolved.yaml"
    config.write_text(
        "model_family: text\nlocal_backend: automodel\nquery_max_length: 512\npassage_max_length: 8192\n"
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
    assert "project_dir" not in kwargs
    assert kwargs["extras"] == ["vl" if family == "mistral3_vl" else "text"]
    assert kwargs["train_path"] == config
    assert kwargs["passthrough"] == overrides


@pytest.mark.parametrize("command", ("prep", "finetune", "eval"))
def test_unvalidated_vl_container_lane_still_fails_before_submission(tmp_path: Path, command: str) -> None:
    module = importlib.import_module(f"nemotron.cli.commands.embed.{command}")
    train_path = tmp_path / "resolved.yaml"
    train_path.write_text(
        "model_family: mistral3_vl\nlocal_backend: automodel\nquery_max_length: 512\npassage_max_length: 8192\n"
        if command == "eval"
        else "model_family: mistral3_vl\n"
    )
    with pytest.raises(RuntimeError, match="VL container dependency selection"):
        module._execute_remote(
            train_path=train_path,
            env=None,
            passthrough=[],
            attached=True,
            env_vars={},
            force_squash=False,
        )
