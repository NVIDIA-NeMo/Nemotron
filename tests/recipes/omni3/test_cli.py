"""Integration tests for omni3 CLI structure and dry-run mode."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app
from nemotron.cli.commands.omni3.data.prep.rl import _make_job_name

runner = CliRunner()


class TestRLJobNameUniqueness:
    """Regression tests for the parallel-batch race that produced
    ``Config file not found: config.yaml`` when sibling RL configs (mpo,
    text, vision) were launched in the same wall-clock second. The local
    ``repo_config`` filename and the remote Ray code dir both derive
    from the job name, so collisions corrupted the per-job config
    upload."""

    def test_consecutive_invocations_unique(self):
        names = [_make_job_name("omni3-data-prep-rl") for _ in range(8)]
        assert len(set(names)) == len(names), names

    def test_parallel_invocations_unique(self):
        # Simulate the exact failure mode: 4+ submissions in the same
        # wall-clock second from the same dispatcher process.
        with ThreadPoolExecutor(max_workers=8) as pool:
            names = list(pool.map(_make_job_name, ["omni3-data-prep-rl"] * 32))
        assert len(set(names)) == len(names), names

    def test_name_starts_with_prefix(self):
        name = _make_job_name("omni3-data-prep-rl")
        assert name.startswith("omni3-data-prep-rl_"), name


class TestOmni3AppStructure:
    def test_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "--help"])
        assert result.exit_code == 0

    def test_commit4_commands_exist(self):
        result = runner.invoke(app, ["omni3", "--help"])
        assert result.exit_code == 0
        assert "sft" in result.output
        assert "data" in result.output
        assert "model" in result.output
        assert "rl" in result.output

    def test_sft_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "sft", "--help"])
        assert result.exit_code == 0

    def test_data_prep_sft_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "data", "prep", "sft", "--help"])
        assert result.exit_code == 0

    def test_data_prep_rl_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "data", "prep", "rl", "--help"])
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

    def test_pipe_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "pipe", "--help"])
        assert result.exit_code == 0


class TestDryRun:
    def test_pipe_dry_run_succeeds(self, monkeypatch):
        # Vision RL launcher landed upstream — pipe runs all 4 stages by
        # default and surfaces ``omni3-vision-rl-model:latest`` as the
        # final artifact.
        monkeypatch.setattr(sys, "argv", ["nemotron", "omni3", "pipe", "-d"])
        result = runner.invoke(app, ["omni3", "pipe", "-d"])
        assert result.exit_code == 0, f"pipe dry-run failed: {result.output}\n{result.exception}"
        assert "sft -> rl mpo -> rl text -> rl vision" in result.output
        assert "omni3-vision-rl-model:latest" in result.output
