# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare and transfer NeMo Run experiments without submitting them."""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from nemo_runspec.cli_context import GlobalContext
from nemo_runspec.env import parse_env
from nemo_runspec.execution import get_executor_type


def parse_env_for_stage(ctx: GlobalContext, *, stage: bool) -> Any:
    """Load an execution profile and make staging lookup failures CLI-friendly."""
    try:
        return parse_env(ctx)
    except ValueError as error:
        if not stage:
            raise
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


def validate_staging_target(env: Any) -> None:
    """Require the remote profile shape supported by file staging."""
    if get_executor_type(env, default="local") != "slurm":
        _fail("--stage currently supports only Slurm execution profiles")

    required = {
        "tunnel": "ssh",
        "host": None,
        "user": None,
        "remote_job_dir": None,
    }
    missing = [
        key
        for key, expected in required.items()
        if not _env_value(env, key) or (expected is not None and _env_value(env, key) != expected)
    ]
    if missing:
        fields = ", ".join(missing)
        _fail(
            "--stage requires a Slurm SSH profile with: "
            f'tunnel = "ssh", host, user, remote_job_dir (invalid: {fields})'
        )


def stage_experiment(experiment: Any) -> list[str]:
    """Package and transfer an experiment, stopping before job submission.

    NeMo Run's dry-run lifecycle materializes the packaged code, configuration,
    and scheduler script. This function transfers those prepared artifacts by
    following the same SSH/rsync path used immediately before normal scheduling.
    """
    ssh_tunnel_type, rsync = _load_nemo_run_staging_api()

    experiment.dryrun(log=False, delete_exp_dir=False)

    jobs = list(experiment.jobs)
    if not jobs:
        _fail("cannot stage an experiment without jobs")

    tunnels = list(experiment.tunnels.values())
    if not tunnels:
        _fail("--stage did not produce an SSH staging target")

    local_experiment_dir = Path(jobs[0].executor.experiment_dir)
    for job in jobs:
        _write_task_launcher(job)

    staged_dirs: list[str] = []

    for tunnel in tunnels:
        if not isinstance(tunnel, ssh_tunnel_type):
            _fail("--stage currently supports only Slurm profiles using an SSH tunnel")

        tunnel.connect()
        if tunnel.session is None:
            _fail(f"failed to connect to staging target {tunnel.key}")

        remote_experiment_dir = str(tunnel.job_dir)
        rsync(
            tunnel.session,
            str(local_experiment_dir),
            posixpath.dirname(remote_experiment_dir),
        )

        symlink_commands = [job.symlink_cmd() for job in tunnel.packaging_jobs.values() if job.symlink]
        if symlink_commands:
            tunnel.run(" && ".join(symlink_commands))

        staged_dirs.append(remote_experiment_dir)
        typer.echo(f"Staged experiment files to {tunnel.key}:{remote_experiment_dir}")
        typer.echo("Run a staged task from a compatible Slurm allocation:")
        for job in jobs:
            task_dir = Path(job.executor.job_dir).name
            code_dir = posixpath.join(remote_experiment_dir, task_dir, "code")
            typer.echo(f"  cd {shlex.quote(code_dir)}")
            typer.echo("  ./run.sh")

    return staged_dirs


def run_or_stage_experiment(experiment: Any, *, stage: bool, attached: bool) -> list[str] | None:
    """Either stage prepared artifacts or follow the ordinary submission path."""
    if stage:
        return stage_experiment(experiment)

    experiment.run(detach=not attached, tail_logs=attached)
    return None


def _load_nemo_run_staging_api() -> tuple[type[Any], Callable[..., Any]]:
    """Load NeMo Run lazily so local-only CLI use does not require it."""
    from nemo_run.core.tunnel.client import SSHTunnel
    from nemo_run.core.tunnel.rsync import rsync

    return SSHTunnel, rsync


def _env_value(env: Any, key: str) -> Any:
    if env is None:
        return None
    return env.get(key) if hasattr(env, "get") else getattr(env, key, None)


def _write_task_launcher(job: Any) -> None:
    """Write a manual launcher for the prepared Slurm execution script."""
    executor = job.executor
    code_dir = Path(executor.job_dir) / "code"
    if not code_dir.is_dir():
        _fail(f"prepared task directory not found: {code_dir}")

    scheduler_script = Path(executor.experiment_dir) / f"{executor.job_name}_sbatch.sh"
    if not scheduler_script.is_file():
        _fail(f"prepared Slurm script not found: {scheduler_script}")

    scheduler_name = shlex.quote(scheduler_script.name)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f'exec bash "$script_dir"/../../{scheduler_name} "$@"',
        "",
    ]

    launcher = code_dir / "run.sh"
    launcher.write_text("\n".join(lines), encoding="utf-8")
    launcher.chmod(0o755)


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)
