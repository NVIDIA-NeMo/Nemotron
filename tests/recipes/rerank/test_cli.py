"""Fast CLI smoke tests for the rerank recipe."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

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
        monkeypatch.setattr(sys, "argv", ["nemotron", "rerank", "run", "--run", "missing", "--dry-run", "--to", "deploy"])
        result = runner.invoke(app, ["rerank", "run", "--run", "missing", "--dry-run", "--to", "deploy"])
        assert result.exit_code == 1
        assert "deploy do not support remote execution" in result.output

    def test_remote_run_rejects_stage_flag_before_submission(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "rerank", "run", "--run", "missing", "--stage", "--from", "sdg", "--to", "eval"])
        result = runner.invoke(app, ["rerank", "run", "--run", "missing", "--stage", "--from", "sdg", "--to", "eval"])
        assert result.exit_code == 1
        assert "--stage is not supported for rerank run" in result.output
