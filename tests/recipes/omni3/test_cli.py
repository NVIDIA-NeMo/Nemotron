"""Integration tests for omni3 CLI structure and dry-run mode."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app
from nemotron.cli.commands.omni3.build import (
    BUILD_IMAGE,
    REMOTE_CODE_ROOT,
    STAGES,
    STAGE_ALIASES,
    _make_build_script,
)
from nemotron.cli.commands.omni3.data.prep.rl import _make_job_name

runner = CliRunner()


class TestOmni3BuildScript:
    """Unit tests for the inline bash script the build dispatcher
    submits to slurm via ``run.Script(inline=...)``.

    These pin the build pipeline's three phases (enroot install →
    podman build → enroot import → squashfs) and the path conventions
    that downstream stages depend on.
    """

    def test_renders_three_phases(self):
        script = _make_build_script(
            stage_dir=Path("src/nemotron/recipes/omni3/stage0_sft"),
            image_tag="nemotron/omni3-sft:latest",
            sqsh_name="omni3-sft.sqsh",
            enroot_version="3.5.0",
        )
        # Phase 1: enroot installs from GitHub release RPMs.
        assert "enroot-3.5.0-1.el8.x86_64.rpm" in script
        assert "enroot+caps-3.5.0-1.el8.x86_64.rpm" in script
        # Phase 2: podman build against the in-container Dockerfile path.
        assert (
            f"{REMOTE_CODE_ROOT}/src/nemotron/recipes/omni3/stage0_sft/Dockerfile"
            in script
        )
        assert "-t nemotron/omni3-sft:latest" in script
        # Phase 3: enroot import from podman storage to squashfs.
        assert "enroot import --output" in script
        assert "podman://nemotron/omni3-sft:latest" in script
        assert "/nemotron-cache/containers/omni3-sft.sqsh" in script

    def test_strict_bash_mode(self):
        script = _make_build_script(
            stage_dir=Path("src/nemotron/recipes/omni3/stage0_sft"),
            image_tag="t:latest",
            sqsh_name="t.sqsh",
            enroot_version="3.5.0",
        )
        assert "set -euo pipefail" in script

    def test_extra_podman_args_are_quoted_safely(self):
        script = _make_build_script(
            stage_dir=Path("src/nemotron/recipes/omni3/stage0_sft"),
            image_tag="t:latest",
            sqsh_name="t.sqsh",
            enroot_version="3.5.0",
            extra_podman_args=["--build-arg", "FOO=bar baz", "--squash-all"],
        )
        # ``shlex.quote`` should wrap the multi-word arg in single quotes
        # so the shell sees one token, not three.
        assert "'FOO=bar baz'" in script
        assert "--squash-all" in script

    def test_enroot_version_pin_propagates(self):
        script = _make_build_script(
            stage_dir=Path("src/nemotron/recipes/omni3/stage0_sft"),
            image_tag="t:latest",
            sqsh_name="t.sqsh",
            enroot_version="4.1.2",
        )
        assert "enroot-4.1.2-1.el8.x86_64.rpm" in script
        assert "v4.1.2" in script


class TestOmni3BuildStageRegistry:
    """Adding/removing stages should be a single-row change in
    ``STAGES``; this test guards the public surface."""

    def test_known_stages_have_image_and_sqsh_names(self):
        for canonical, spec in STAGES.items():
            assert spec["image_tag"], canonical
            assert spec["sqsh_name"].endswith(".sqsh"), canonical

    def test_aliases_resolve_to_known_stages(self):
        for alias, canonical in STAGE_ALIASES.items():
            assert canonical in STAGES, (alias, canonical)

    def test_build_image_uses_enroot_uri_form(self):
        # Pyxis on dlw misroutes bare ``quay.io/...`` strings to
        # docker.io; the ``#`` separator is required.
        assert BUILD_IMAGE.startswith("docker://"), BUILD_IMAGE
        assert "#" in BUILD_IMAGE, BUILD_IMAGE


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
            names = list(
                pool.map(_make_job_name, ["omni3-data-prep-rl"] * 32)
            )
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

    def test_pipe_help_succeeds(self):
        result = runner.invoke(app, ["omni3", "pipe", "--help"])
        assert result.exit_code == 0


class TestDryRun:
    def test_build_dry_run_sft_stage(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "omni3", "build", "sft", "-d"])
        result = runner.invoke(app, ["omni3", "build", "sft", "-d"])
        assert result.exit_code == 0, f"dry-run failed: {result.output}\n{result.exception}"
        # The dispatcher resolves the alias to the canonical stage name
        # and shows the produced sqsh + image tag.
        assert "stage0_sft" in result.output
        assert "omni3-sft.sqsh" in result.output
        assert "nemotron/omni3-sft:latest" in result.output

    def test_pipe_dry_run_succeeds(self, monkeypatch):
        # Vision RL launcher landed upstream — pipe runs all 4 stages by
        # default and surfaces ``omni3-vision-rl-model:latest`` as the
        # final artifact.
        monkeypatch.setattr(sys, "argv", ["nemotron", "omni3", "pipe", "-d"])
        result = runner.invoke(app, ["omni3", "pipe", "-d"])
        assert result.exit_code == 0, f"pipe dry-run failed: {result.output}\n{result.exception}"
        assert "sft -> rl mpo -> rl text -> rl vision" in result.output
        assert "omni3-vision-rl-model:latest" in result.output
