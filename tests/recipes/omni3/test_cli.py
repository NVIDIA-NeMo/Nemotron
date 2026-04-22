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

    def test_commit4_commands_exist(self):
        result = runner.invoke(app, ["omni3", "--help"])
        assert result.exit_code == 0
        assert "build" in result.output
        assert "sft" in result.output
        assert "data" in result.output
        assert "model" in result.output
        assert "rl" in result.output

    def test_build_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "build", "--help"])
        assert result.exit_code == 0

    def test_sft_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "sft", "--help"])
        assert result.exit_code == 0

    def test_data_prep_sft_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "data", "prep", "sft", "--help"])
        assert result.exit_code == 0

    def test_data_prep_rl_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "data", "prep", "rl", "--help"])
        assert result.exit_code == 0

    def test_build_rl_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "build", "rl", "--help"])
        assert result.exit_code == 0

    def test_rl_group_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "rl", "--help"])
        assert result.exit_code == 0
        assert "mpo" in result.output
        assert "text" in result.output
        assert "vision" in result.output

    def test_rl_mpo_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "rl", "mpo", "--help"])
        assert result.exit_code == 0

    def test_rl_text_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "rl", "text", "--help"])
        assert result.exit_code == 0

    def test_rl_vision_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "rl", "vision", "--help"])
        assert result.exit_code == 0

    def test_model_import_pretrain_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "model", "import", "pretrain", "--help"])
        assert result.exit_code == 0

    def test_model_import_roundtrip_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "model", "import", "roundtrip", "--help"])
        assert result.exit_code == 0

    def test_model_export_pretrain_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "model", "export", "pretrain", "--help"])
        assert result.exit_code == 0

    def test_model_eval_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "model", "eval", "--help"])
        assert result.exit_code == 0

    def test_model_lora_merge_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "model", "lora-merge", "--help"])
        assert result.exit_code == 0

    def test_model_adapter_export_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "model", "adapter-export", "--help"])
        assert result.exit_code == 0

    def test_eval_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "eval", "--help"])
        assert result.exit_code == 0


class TestDryRun:
    def test_build_dry_run_sft_stage(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "omni3", "build", "sft", "-d"])
        result = runner.invoke(app, ["omni3", "build", "sft", "-d"])
        assert result.exit_code == 0, f"dry-run failed: {result.output}\n{result.exception}"
        assert "omni3/stage0_sft/build" in result.output
        assert "stage0_sft" in result.output
