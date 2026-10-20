# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Keep rerank training, evaluation, and serving on the same run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nemo_runspec.config.pydantic_loader import load_config
from nemotron.recipes.rerank.stage2_finetune import train
from nemotron.recipes.rerank.stage3_eval import eval as evaluate
from nemotron.recipes.rerank.stage5_deploy import deploy


def _profiles(profile: str, overrides: list[str]):
    return (
        load_config(train.STAGE_PATH / f"config/{profile}.yaml", overrides, train.FinetuneConfig),
        load_config(evaluate.STAGE_PATH / f"config/{profile}.yaml", overrides, evaluate.EvalConfig),
        load_config(deploy.STAGE_PATH / f"config/{profile}.yaml", overrides, deploy.DeployConfig),
    )


@pytest.mark.parametrize("run_base", [None, "/tmp/remote-rerank-run"])
@pytest.mark.parametrize("profile", ["default", "nemotron-3.5"])
def test_profile_defaults_preserve_paths(monkeypatch, run_base, profile):
    if run_base is None:
        monkeypatch.delenv("NEMO_RUN_DIR", raising=False)
    else:
        monkeypatch.setenv("NEMO_RUN_DIR", run_base)
    common_root = Path(run_base or ".") / "output/rerank"
    artifact_root = common_root / "nemotron-3.5" if profile == "nemotron-3.5" else common_root

    finetune, evaluation, serving = _profiles(profile, [])

    assert finetune.train_data_path == common_root / "stage1_prep/train_mined.automodel_unrolled.json"
    assert evaluation.eval_data_path == common_root / "stage1_prep/eval_beir"
    assert finetune.checkpoint_dir == artifact_root / "stage2_finetune/checkpoints"
    assert evaluation.finetuned_model_path == finetune.checkpoint_dir / "LATEST/model/consolidated"
    assert evaluation.output_dir == artifact_root / "stage3_eval"
    if profile == "nemotron-3.5":
        assert serving.model_dir == evaluation.finetuned_model_path
    else:
        assert serving.model_dir == artifact_root / "stage4_export/onnx"


def test_multimodal_root_connects_stages_without_moving_shared_data(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_RUN_DIR", raising=False)
    root = tmp_path / "rerank-run"

    finetune, evaluation, serving = _profiles("nemotron-3.5", [f"artifact_root={root}"])

    assert finetune.artifact_root == evaluation.artifact_root == serving.artifact_root == root
    assert finetune.checkpoint_dir == root / "stage2_finetune/checkpoints"
    assert evaluation.finetuned_model_path == finetune.checkpoint_dir / "LATEST/model/consolidated"
    assert evaluation.output_dir == root / "stage3_eval"
    assert serving.model_dir == evaluation.finetuned_model_path
    assert finetune.train_data_path == Path("output/rerank/stage1_prep/train_mined.automodel_unrolled.json")
    assert evaluation.eval_data_path == Path("output/rerank/stage1_prep/eval_beir")


def test_explicit_paths_take_priority_over_multimodal_root(tmp_path):
    overrides = [f"artifact_root={tmp_path / 'rerank-run'}"]
    checkpoint = tmp_path / "selected-checkpoint"
    training_data = tmp_path / "embedding-run/train.json"
    evaluation_data = tmp_path / "independent-beir"
    evaluation_output = tmp_path / "endpoint-comparison"
    finetune = load_config(
        train.STAGE_PATH / "config/nemotron-3.5.yaml",
        [*overrides, f"checkpoint_dir={checkpoint}", f"train_data_path={training_data}"],
        train.FinetuneConfig,
    )
    evaluation = load_config(
        evaluate.STAGE_PATH / "config/nemotron-3.5.yaml",
        [
            *overrides,
            f"finetuned_model_path={checkpoint}",
            f"eval_data_path={evaluation_data}",
            f"output_dir={evaluation_output}",
        ],
        evaluate.EvalConfig,
    )
    serving = load_config(
        deploy.STAGE_PATH / "config/nemotron-3.5.yaml",
        [*overrides, f"model_dir={checkpoint}"],
        deploy.DeployConfig,
    )
    base_serving = load_config(
        deploy.STAGE_PATH / "config/nemotron-3.5.yaml", [*overrides, "model_dir=null"], deploy.DeployConfig
    )

    assert finetune.checkpoint_dir == evaluation.finetuned_model_path == serving.model_dir == checkpoint
    assert finetune.train_data_path == training_data
    assert evaluation.eval_data_path == evaluation_data
    assert evaluation.output_dir == evaluation_output
    assert base_serving.model_dir is None


@pytest.mark.parametrize("stage", ["finetune", "eval", "deploy"])
def test_multimodal_root_cli_preview_does_not_launch(monkeypatch, stage):
    from typer.testing import CliRunner

    from nemotron.cli.bin.nemotron import app
    from nemotron.cli.commands.rerank import eval as eval_command
    from nemotron.cli.commands.rerank import finetune as finetune_command

    def unexpected_launch(*args, **kwargs):
        pytest.fail("configuration preview launched a workload")

    monkeypatch.setattr(finetune_command, "_execute_uv_local", unexpected_launch)
    monkeypatch.setattr(eval_command, "_execute_uv_local", unexpected_launch)
    monkeypatch.setattr(deploy.subprocess, "run", unexpected_launch)
    args = ["rerank", stage, "-c", "nemotron-3.5", "-d", "artifact_root=/tmp/rerank-run"]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "/tmp/rerank-run" in result.output
