"""Integration tests for embed CLI structure and dry-run mode.

Uses ``typer.testing.CliRunner`` for in-process, fast CLI testing.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

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


class TestDryRun:
    @pytest.mark.parametrize("command", STAGE_COMMANDS)
    def test_dry_run_default_config(self, command, monkeypatch):
        # Monkeypatch sys.argv so ConfigBuilder doesn't pick up pytest's argv
        monkeypatch.setattr(sys, "argv", ["nemotron", "embed", command, "-c", "default", "-d"])
        result = runner.invoke(app, ["embed", command, "-c", "default", "-d"])
        assert result.exit_code == 0, (
            f"dry-run failed for '{command}': {result.output}\n{result.exception}"
        )
        assert len(result.output.strip()) > 0

    @pytest.mark.parametrize("command", STAGE_COMMANDS)
    def test_dry_run_shows_recipe_name(self, command, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "embed", command, "-c", "default", "-d"])
        result = runner.invoke(app, ["embed", command, "-c", "default", "-d"])
        assert result.exit_code == 0, (
            f"dry-run failed for '{command}': {result.output}\n{result.exception}"
        )
        assert RECIPE_NAMES[command] in result.output
