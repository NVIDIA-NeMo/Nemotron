"""The execution smoke must reject completed but ineffective optimizer steps."""

import json
from pathlib import Path

import pytest
from run_smoke import check_training_steps
from smoke_task import serialization_similarity


def histories(output: Path, *, rl_gradients=(0.1, 0.2), rl_rates=(1e-6, 1e-6)) -> None:
    for stage, gradients in (("sft", (0.3, 0.4)), ("rlvr", rl_gradients)):
        directory = output / stage
        directory.mkdir()
        rows = [
            dict(
                step=step, phase="train",
                metrics=dict(grad_norm=gradient, lr=rl_rates[step - 1] if stage == "rlvr" else 1e-6),
            )
            for step, gradient in enumerate(gradients, 1)
        ]
        (directory / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_smoke_accepts_finite_nonzero_gradients_and_two_steps(tmp_path):
    histories(tmp_path)
    assert check_training_steps(tmp_path)["rlvr"] == {"completed_steps": [1, 2], "grad_norms": [0.1, 0.2]}


@pytest.mark.parametrize("gradients", [(0.0, 0.0), (float("nan"), 0.2), (float("inf"), 0.2)])
def test_smoke_rejects_zero_or_nonfinite_rl_gradients(tmp_path, gradients):
    histories(tmp_path, rl_gradients=gradients)
    with pytest.raises(AssertionError, match="RLVR did not report finite gradients"):
        check_training_steps(tmp_path)


def test_smoke_rejects_missing_second_rl_step(tmp_path):
    histories(tmp_path, rl_gradients=(0.2,))
    with pytest.raises(AssertionError, match="two completed rlvr updates"):
        check_training_steps(tmp_path)


@pytest.mark.parametrize("rates", [(0.0, 0.0), (float("nan"), 0.0), (float("inf"), 0.0)])
def test_smoke_rejects_rl_gradients_without_positive_finite_learning_rate(tmp_path, rates):
    histories(tmp_path, rl_rates=rates)
    with pytest.raises(AssertionError, match="RLVR.*positive learning rate"):
        check_training_steps(tmp_path)


def test_smoke_requires_learning_rate_on_the_nonzero_gradient_step(tmp_path):
    histories(tmp_path, rl_gradients=(0.1, 0.0), rl_rates=(0.0, 1e-6))
    with pytest.raises(AssertionError, match="RLVR.*positive learning rate"):
        check_training_steps(tmp_path)


@pytest.mark.parametrize("profile_path", ["", "/customer", "/nemo_run/code/tests/recipes/parse"])
def test_smoke_cli_import_path_wins_over_profile(profile_path, monkeypatch):
    from omegaconf import OmegaConf

    from nemo_runspec.cli_context import GlobalContext
    from nemo_runspec.config import loader

    monkeypatch.setattr(loader, "get_wandb_config", lambda: None)
    monkeypatch.setattr(loader, "get_artifacts_config", lambda: None)
    expected = "/nemo_run/code/tests/recipes/parse:/nemo_run/code/src"
    context = GlobalContext(run="smoke", dotlist=[f"run.env.env_vars.PYTHONPATH={expected}"])
    profile = OmegaConf.create({"executor": "docker", "env_vars": {"PYTHONPATH": profile_path}})
    job = loader.build_job_config(
        OmegaConf.create({}), context, "parse/rlvr", "src/nemotron/kit/run_uv.py", [], env_profile=profile
    )
    assert job.run.env.env_vars.PYTHONPATH == expected


def test_smoke_similarity_is_dense_bounded_and_preserves_internal_markers():
    assert serialization_similarity("</s>Total: 17<pad>", "Total: 17") == 1
    partial = serialization_similarity("Total: 71", "Total: 17")
    wrong = serialization_similarity("wrong", "Total: 17")
    assert 0 <= wrong < partial < 1
    assert serialization_similarity("left</s>right", "leftright") < 1
