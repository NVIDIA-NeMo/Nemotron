from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nemotron.cli.commands.steps.backends import JobContext
from nemotron.cli.commands.steps.backends.slurm import SlurmBackend


def _ctx(*, env: dict[str, object], cmd: str | None = None) -> JobContext:
    spec = SimpleNamespace(
        run=SimpleNamespace(cmd=cmd, launch="python"),
        image=None,
        resources=None,
    )
    return JobContext(
        step_id="byob/mcq",
        script_path=Path("/repo/src/nemotron/steps/byob/mcq/step.py"),
        train_path=Path("/repo/.nemotron/train.yaml"),
        spec=spec,
        env=env,
        env_vars={},
        passthrough=[],
        startup_commands=[],
        attached=True,
        force_squash=False,
    )


def test_slurm_backend_honors_env_run_command() -> None:
    command = (
        "python -m nemotron.steps._bootstrap.curator_runtime --profile byob "
        "-- python -m nemotron.steps.byob.mcq.step --config {config}"
    )

    assert SlurmBackend._build_cmd(_ctx(env={"run_command": command})) == (
        "export PYTHONPATH=/nemo_run/code/src${PYTHONPATH:+:$PYTHONPATH}; "
        "python -m nemotron.steps._bootstrap.curator_runtime --profile byob "
        "-- python -m nemotron.steps.byob.mcq.step --config config.yaml"
    )


def test_slurm_backend_uses_code_packager_for_curator_runtime() -> None:
    command = (
        "python -m nemotron.steps._bootstrap.curator_runtime --profile byob "
        "-- python -m nemotron.steps.byob.mcq.step --config {config}"
    )

    assert SlurmBackend._uses_code_packager(_ctx(env={"run_command": command}))


def test_slurm_backend_prefers_runspec_cmd_over_env_run_command() -> None:
    ctx = _ctx(
        env={"run_command": "python -m env.wrapper --config {config}"},
        cmd="python -m spec.wrapper --config {config}",
    )

    assert SlurmBackend._build_cmd(ctx) == "python -m spec.wrapper --config config.yaml"
