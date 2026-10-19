"""Integration tests for super35 CLI structure and dry-run mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()

TEACHERS = ("reasoning", "vision", "video", "vision-reasoning", "rlhf", "agentic")


def _invoke(*args: str):
    result = runner.invoke(app, ["super35", *args])
    assert result.exit_code == 0, f"{' '.join(args)} failed: {result.output}\n{result.exception}"
    return result


class TestSuper35AppStructure:
    def test_help_lists_stages(self):
        result = _invoke("--help")
        for command in ("pretrain", "sft", "rl", "eval", "data", "model"):
            assert command in result.output

    @pytest.mark.parametrize(
        "args",
        [
            ("pretrain", "--help"),
            ("sft", "--help"),
            ("eval", "--help"),
            ("rl", "--help"),
            ("rl", "rlvr", "--help"),
            ("rl", "teacher", "--help"),
            ("rl", "mopd", "--help"),
            ("model", "import", "--help"),
            ("data", "import", "--help"),
        ],
    )
    def test_subcommand_help_succeeds(self, args):
        _invoke(*args)

    def test_rl_teacher_lists_every_teacher(self):
        result = _invoke("rl", "teacher", "--help")
        for teacher in TEACHERS:
            assert teacher in result.output

    @pytest.mark.parametrize("teacher", TEACHERS)
    def test_rl_teacher_help_succeeds(self, teacher):
        _invoke("rl", "teacher", teacher, "--help")


RECIPES = Path(__file__).resolve().parents[3] / "src/nemotron/recipes/super35"

# Stage directory -> CLI words that select it.
STAGE_ARGS = {
    "stage0_pretrain": ("pretrain",),
    "stage1_sft": ("sft",),
    "stage2_rl/stage1_rlvr": ("rl", "rlvr"),
    "stage2_rl/stage2_teachers/reasoning": ("rl", "teacher", "reasoning"),
    "stage2_rl/stage2_teachers/vision": ("rl", "teacher", "vision"),
    "stage2_rl/stage2_teachers/video": ("rl", "teacher", "video"),
    "stage2_rl/stage2_teachers/vision_reasoning": ("rl", "teacher", "vision-reasoning"),
    "stage2_rl/stage2_teachers/rlhf": ("rl", "teacher", "rlhf"),
    "stage2_rl/stage2_teachers/agentic": ("rl", "teacher", "agentic"),
    "stage2_rl/stage3_mopd": ("rl", "mopd"),
}
STAGE_CONFIGS = sorted(
    (stage, config.stem)
    for stage in STAGE_ARGS
    for config in (RECIPES / stage / "config").glob("*.yaml")
)


def test_every_stage_with_configs_is_reachable_from_the_cli():
    stages = {str(c.parent.parent.relative_to(RECIPES)) for c in RECIPES.glob("**/config/*.yaml")}
    assert stages - {"stage3_eval"} == set(STAGE_ARGS)


class TestDryRun:
    @pytest.mark.parametrize(("stage", "config"), STAGE_CONFIGS, ids=["/".join(x) for x in STAGE_CONFIGS])
    def test_every_config_dry_runs(self, monkeypatch, stage, config):
        args = (*STAGE_ARGS[stage], "-c", config, "-d")
        monkeypatch.setattr(sys, "argv", ["nemotron", "super35", *args])
        result = _invoke(*args)
        # The dry run shows the selected config, so its output artifact (if it names one) appears.
        raw = OmegaConf.load(RECIPES / stage / "config" / f"{config}.yaml")
        expected = OmegaConf.select(raw, "run.output_artifact")
        if expected:
            assert expected in result.output

    def test_eval_dry_run(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nemotron", "super35", "eval", "-d"])
        _invoke("eval", "-d")


class TestImport:
    """Import commands register the artifact names the stages read by default.

    W&B is stubbed at ``wandb.Artifact`` so nothing is uploaded; the kit
    normalizes ``/`` in names to ``-`` before that point.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        import wandb

        names: list[str] = []

        class _StopError(Exception):
            pass

        class _Artifact(wandb.Artifact):
            def __init__(self, name, *args, **kwargs):
                names.append(name)
                raise _StopError

        monkeypatch.setattr(wandb, "Artifact", _Artifact)
        monkeypatch.setenv("WANDB_MODE", "disabled")
        # A wandb session started by an earlier test ignores WANDB_MODE, so
        # tear it down; the import must never reach the W&B servers.
        wandb.teardown()
        return names

    @pytest.fixture
    def inputs(self, tmp_path):
        (tmp_path / "train.jsonl").write_text('{"responses_create_params": {"input": []}}\n')
        (tmp_path / "energon" / ".nv-meta").mkdir(parents=True)
        (tmp_path / "ckpt").mkdir()
        return tmp_path

    @pytest.mark.parametrize(
        ("args", "consumer_key", "consumer_config"),
        [
            (("data", "import", "rl", "{d}/train.jsonl"), "run.data", "stage2_rl/stage1_rlvr/config/default.yaml"),
            (("data", "import", "sft", "{d}/energon"), "run.data", "stage1_sft/config/default.yaml"),
            (("model", "import", "pretrain", "{d}/ckpt"), "run.model", "stage1_sft/config/default.yaml"),
            (("model", "import", "sft", "{d}/ckpt"), "run.model", "stage2_rl/stage1_rlvr/config/default.yaml"),
            (
                ("model", "import", "rl", "{d}/ckpt"),
                "run.model",
                "stage2_rl/stage2_teachers/reasoning/config/default.yaml",
            ),
        ],
    )
    def test_default_name_matches_consumer(self, captured, inputs, args, consumer_key, consumer_config):
        args = tuple(a.format(d=inputs) for a in args)
        runner.invoke(app, ["super35", *args, "-p", "test-project", "-e", "test-entity"])
        assert captured, f"{' '.join(args)} never reached W&B registration"
        expected = OmegaConf.select(OmegaConf.load(RECIPES / consumer_config), consumer_key).split(":")[0]
        assert captured[-1] == expected

    def test_data_import_rl_writes_manifest(self, captured, inputs):
        runner.invoke(
            app,
            ["super35", "data", "import", "rl", f"{inputs}/train.jsonl", "--name", "super35-rl-teacher-video-data",
             "-p", "test-project", "-e", "test-entity"],
        )
        manifest = json.loads((inputs / "super35-rl-teacher-video-data" / "manifest.json").read_text())
        # Without --val the validation split defaults to the training file.
        assert manifest == {"train": str(inputs / "train.jsonl"), "val": str(inputs / "train.jsonl")}
        assert captured[-1] == "super35-rl-teacher-video-data"
