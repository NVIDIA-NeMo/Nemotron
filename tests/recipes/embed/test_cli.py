"""Integration tests for embed CLI structure and dry-run mode.

Uses ``typer.testing.CliRunner`` for in-process, fast CLI testing.
"""

from __future__ import annotations

import sys
from importlib import import_module

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from nemo_runspec.config import load_config
from nemotron.cli.bin.nemotron import app

from .conftest import STAGES

runner = CliRunner()

# All embed subcommands (including 'info')
ALL_COMMANDS = ["sdg", "prep", "finetune", "eval", "export", "deploy", "info"]

# Stage commands only (for dry-run tests)
STAGE_COMMANDS = [s["command"] for s in STAGES]

# Recipe names per stage for dry-run output verification
RECIPE_NAMES = {
    "sdg": "embed/sdg",
    "prep": "embed/prep",
    "finetune": "embed/finetune",
    "eval": "embed/eval",
    "export": "embed/export",
    "deploy": "embed/deploy",
}


class TestEmbedAppStructure:
    def test_help_succeeds(self):
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("command", ALL_COMMANDS)
    def test_command_exists(self, command):
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert command in result.output

    @pytest.mark.parametrize("command", ALL_COMMANDS)
    def test_command_help_succeeds(self, command):
        result = runner.invoke(app, ["embed", command, "--help"])
        assert result.exit_code == 0

    def test_info_command_output(self):
        result = runner.invoke(app, ["embed", "info"])
        assert result.exit_code == 0
        assert "sdg" in result.output
        assert "prep" in result.output
        assert "finetune" in result.output
        assert "nemotron-3: nvidia/Nemotron-3-Embed-1B-BF16" in result.output
        assert "llama: nvidia/llama-nemotron-embed-1b-v2" in result.output
        assert "with NIM or vLLM" in result.output

    def test_deploy_help_shows_direct_checkpoint_defaults(self):
        result = runner.invoke(app, ["embed", "deploy", "--help"])
        assert result.exit_code == 0
        assert "NIM_ENGINE_MODE" in result.output
        assert "16gb" in result.output

    def test_export_help_explains_default_no_op(self):
        result = runner.invoke(app, ["embed", "export", "--help"])
        assert result.exit_code == 0
        assert "direct-checkpoint profiles skip this stage" in result.output


class TestDryRun:
    @pytest.mark.parametrize("command", STAGE_COMMANDS)
    def test_dry_run_nemotron_3_config(self, command, monkeypatch):
        # Monkeypatch sys.argv so ConfigBuilder doesn't pick up pytest's argv
        monkeypatch.setattr(sys, "argv", ["nemotron", "embed", command, "-c", "nemotron-3", "-d"])
        result = runner.invoke(app, ["embed", command, "-c", "nemotron-3", "-d"])
        assert result.exit_code == 0, (
            f"dry-run failed for '{command}': {result.output}\n{result.exception}"
        )
        assert len(result.output.strip()) > 0

    @pytest.mark.parametrize("command", STAGE_COMMANDS)
    def test_dry_run_shows_recipe_name(self, command, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "embed", command, "-c", "nemotron-3", "-d"])
        result = runner.invoke(app, ["embed", command, "-c", "nemotron-3", "-d"])
        assert result.exit_code == 0, (
            f"dry-run failed for '{command}': {result.output}\n{result.exception}"
        )
        assert RECIPE_NAMES[command] in result.output


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage["command"])
def test_omitted_selector_uses_nemotron_3_profile(stage, monkeypatch):
    module = import_module(stage["cli_module"])
    captured = []
    monkeypatch.setattr(module, "display_job_config", lambda job, **_: captured.append(job))
    args = ["embed", stage["command"], "-d"]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert module.SPEC.config.default == module.META.default_config == "nemotron-3"
    config_module = import_module(stage["config_model_path"])
    assert config_module.DEFAULT_CONFIG_PATH == stage["yaml_path"]
    expected = OmegaConf.to_container(load_config(stage["yaml_path"]), resolve=True)
    actual = OmegaConf.to_container(captured[0], resolve=True)
    for key, value in expected.items():
        if key != "run":
            assert actual[key] == value, f"{stage['command']}: {key}"


@pytest.mark.parametrize("command", [*STAGE_COMMANDS, "run"])
def test_removed_default_profile_is_rejected(command, monkeypatch):
    args = ["embed", command, "-c", "default", "-d"]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "Config 'default' not found" in f"{result.output}\n{result.exception}"


@pytest.mark.parametrize("selector", [[], ["-c", "nemotron-3"], ["-c", "llama"], ["-c", "nemotron-3.5"]])
def test_named_pipeline_profiles_preview_supported_stages(selector, monkeypatch):
    last_stage = "eval" if "nemotron-3.5" in selector else "deploy"
    args = ["embed", "run", *selector, "-d", "--to", last_stage]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert "Dry run" in result.output


def test_existing_custom_config_path_remains_usable(tmp_path, monkeypatch):
    config_path = tmp_path / "default.yaml"
    config_path.write_text(STAGES[0]["yaml_path"].read_text())
    args = ["embed", "sdg", "-c", str(config_path), "-d", "corpus_id=custom-config"]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert "corpus_id: custom-config" in result.output


def test_missing_stage_profile_is_rejected_before_pipeline_execution(monkeypatch):
    from nemotron.cli.commands.embed import run as run_module

    monkeypatch.setattr(
        run_module,
        "_run_pipeline_local",
        lambda *_: pytest.fail("A stage must not launch when a later stage has no selected profile"),
    )
    args = ["embed", "run", "-c", "nemotron-3.5", "--to", "export"]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "export:" in result.output


@pytest.mark.parametrize("command", STAGE_COMMANDS)
def test_llama_profile_dry_run(command, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nemotron", "embed", command, "-c", "llama", "-d"])
    result = runner.invoke(app, ["embed", command, "-c", "llama", "-d"])
    assert result.exit_code == 0, f"Llama dry-run failed for {command!r}: {result.output}\n{result.exception}"
    assert RECIPE_NAMES[command] in result.output
