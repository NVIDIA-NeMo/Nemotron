"""Fast CLI smoke tests for the rerank recipe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app
from nemotron.cli.commands.rerank import finetune as finetune_module

runner = CliRunner()
STAGE_COMMANDS = ["sdg", "prep", "finetune", "eval", "export", "deploy"]


class TestRerankAppStructure:
    def test_help_succeeds(self):
        result = runner.invoke(app, ["rerank", "--help"])
        assert result.exit_code == 0
        assert "deploy" in result.output
        assert "Reranking" in result.output

    @pytest.mark.parametrize("command", [*STAGE_COMMANDS, "info", "run"])
    def test_command_help_succeeds(self, command):
        result = runner.invoke(app, ["rerank", command, "--help"])
        assert result.exit_code == 0

    def test_command_help_uses_existing_default_config_example(self):
        result = runner.invoke(app, ["rerank", "finetune", "--help"])
        assert result.exit_code == 0
        assert "-c default" in result.output
        assert "-c tiny" not in result.output


class TestRerankDryRun:
    @pytest.mark.parametrize("command", STAGE_COMMANDS)
    def test_stage_dry_run_default_config(self, command, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "rerank", command, "-c", "default", "-d"])
        result = runner.invoke(app, ["rerank", command, "-c", "default", "-d"])
        assert result.exit_code == 0, result.output
        assert f"rerank/{command}" in result.output

    def test_run_dry_run_accepts_local_deploy(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "rerank", "run", "-c", "default", "-d", "--to", "deploy"])
        result = runner.invoke(app, ["rerank", "run", "-c", "default", "-d", "--to", "deploy"])
        assert result.exit_code == 0, result.output
        assert "sdg -> prep -> finetune -> eval -> export -> deploy" in result.output

    def test_remote_run_rejects_local_only_deploy_before_env_lookup(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["nemotron", "rerank", "run", "--run", "missing", "--dry-run", "--to", "deploy"]
        )
        result = runner.invoke(app, ["rerank", "run", "--run", "missing", "--dry-run", "--to", "deploy"])
        assert result.exit_code == 1
        assert "deploy do not support remote execution" in result.output

    def test_remote_run_rejects_stage_flag_before_submission(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["nemotron", "rerank", "run", "--run", "missing", "--stage", "--from", "sdg", "--to", "eval"]
        )
        result = runner.invoke(app, ["rerank", "run", "--run", "missing", "--stage", "--from", "sdg", "--to", "eval"])
        assert result.exit_code == 1
        assert "--stage is not supported for rerank run" in result.output


def test_finetune_local_uses_runspec_gpu_worker_default(monkeypatch):
    captured = {}

    def fake_execute_uv_local_from_spec(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "nemo_runspec.execution.execute_uv_local_from_spec",
        fake_execute_uv_local_from_spec,
    )

    finetune_module._execute_uv_local(Path("/tmp/train.yaml"), [])

    assert "torchrun_nproc_per_node" not in captured
    assert captured["spec"].resources.gpus_per_node == "gpu"
