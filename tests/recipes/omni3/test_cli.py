"""Integration tests for omni3 CLI structure and dry-run mode."""

from __future__ import annotations

import sys

from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()


class TestOmni3AppStructure:
    def test_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "--help"])
        assert result.exit_code == 0

    def test_build_command_exists(self):
        result = runner.invoke(app, ["omni3", "--help"])
        assert result.exit_code == 0
        assert "build" in result.output

    def test_build_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "build", "--help"])
        assert result.exit_code == 0


class TestDryRun:
    def test_build_dry_run_smoke_stage(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "omni3", "build", "_smoke_test_stage", "-d"])
        result = runner.invoke(app, ["omni3", "build", "_smoke_test_stage", "-d"])
        assert result.exit_code == 0, f"dry-run failed: {result.output}\n{result.exception}"
        assert "omni3/_smoke_test_stage/build" in result.output
        assert "_smoke_test_stage" in result.output
