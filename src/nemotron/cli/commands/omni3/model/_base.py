# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared execution helpers for omni3 model commands."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nemo_runspec.execution import (
    build_env_vars,
    clone_git_repos_via_tunnel,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.env import parse_env
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.squash import ensure_squashed_image

console = Console()

DEFAULT_CONTAINER_IMAGE = "oci-archive:///home/$USER/.cache/nemotron/containers/omni3-sft.tar"
DEFAULT_WORKDIR = "/workspace/Megatron-Bridge"


def _partition_for_mode(env, attached: bool) -> str | None:
    """Resolve the Slurm partition for attached vs detached execution."""
    if env is None:
        return None
    if attached:
        return env.get("run_partition") or env.get("partition")
    return env.get("batch_partition") or env.get("partition")


def _show_plan(
    *,
    job_name: str,
    cfg,
    env,
    container_image: str,
    workdir: str,
    command: list[str],
    nodes: int,
    gpus_per_node: int,
    time_limit: str,
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Job", job_name)
    table.add_row("Mode", cfg.mode)
    table.add_row("Profile", cfg.profile or "")
    table.add_row("Container", container_image)
    table.add_row("Workdir", workdir)
    table.add_row("Nodes", str(nodes))
    table.add_row("GPUs/node", str(gpus_per_node))
    table.add_row("Time", time_limit)
    table.add_row("Command", shlex.join(command))
    if env is not None:
        partition = _partition_for_mode(env, cfg.attached)
        table.add_row("Partition", partition or "")
    console.print(table)
    console.print()


def _execute_local(
    *,
    command: list[str],
    workdir: str,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
) -> None:
    effective_cwd = Path(workdir)
    if not effective_cwd.exists():
        typer.echo(
            f"[warning] Workdir {workdir!r} does not exist locally; running from {Path.cwd()} instead.",
            err=True,
        )
        effective_cwd = Path.cwd()

    child_env = dict(os.environ)
    child_env.update(env_vars)

    if startup_commands:
        command_line = prepend_startup_to_cmd(startup_commands, shlex.join(command))
        result = subprocess.run(["bash", "-lc", command_line], cwd=effective_cwd, env=child_env, check=False)
    else:
        result = subprocess.run(command, cwd=effective_cwd, env=child_env, check=False)
    raise typer.Exit(result.returncode)


def _execute_remote(
    *,
    job_name: str,
    command: list[str],
    workdir: str,
    env,
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
    nodes: int,
    gpus_per_node: int,
    time_limit: str,
) -> None:
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.packaging import CodePackager
    from nemo_runspec.run import patch_nemo_run_rsync_accept_new_host_keys

    patch_nemo_run_rsync_accept_new_host_keys()

    def _get(key: str, default=None):
        if env is None:
            return default
        return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)

    tunnel = None
    remote_job_dir = _get("remote_job_dir")
    if _get("tunnel") == "ssh":
        tunnel = run.SSHTunnel(
            host=_get("host", "localhost"),
            user=_get("user"),
            job_dir=remote_job_dir,
        )

    container_image = _get("container_image") or _get("container") or DEFAULT_CONTAINER_IMAGE
    if container_image and tunnel and remote_job_dir:
        tunnel.connect()
        container_image = ensure_squashed_image(
            tunnel, container_image, remote_job_dir, env, force=force_squash
        )

    tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    try:
        tmp_config.write("{}\n")
        tmp_config.close()

        packager = CodePackager(
            script_path=__file__,
            train_path=tmp_config.name,
        )

        git_mounts = []
        if tunnel and remote_job_dir:
            tunnel.connect()
            git_mounts = clone_git_repos_via_tunnel(tunnel, remote_job_dir)

        raw_mounts = list(_get("mounts") or [])
        mounts = [mount for mount in raw_mounts if not mount.startswith("__auto_mount__:")]
        mounts.extend(git_mounts)
        mounts.append("/lustre:/lustre")

        partition = _partition_for_mode(env, attached)

        executor = run.SlurmExecutor(
            account=_get("account"),
            partition=partition,
            nodes=nodes,
            ntasks_per_node=1,
            gpus_per_node=gpus_per_node,
            cpus_per_task=_get("cpus_per_task"),
            time=_get("time", time_limit),
            container_image=container_image,
            container_mounts=mounts,
            tunnel=tunnel,
            packager=packager,
            mem=_get("mem"),
            env_vars=env_vars,
            launcher=None,
        )

        remote_cmd = shlex.join(command)
        if workdir:
            remote_cmd = f"cd {workdir} && {remote_cmd}"
        if startup_commands:
            remote_cmd = prepend_startup_to_cmd(startup_commands, remote_cmd)

        script_task = run.Script(path="bash", args=["-lc", remote_cmd])
        recipe_name = job_name.replace("/", "-")
        with run.Experiment(recipe_name) as exp:
            exp.add(script_task, executor=executor, name=recipe_name)
            exp.run(detach=not attached)
    finally:
        Path(tmp_config.name).unlink(missing_ok=True)


def execute_model_command(
    ctx: typer.Context,
    *,
    job_name: str,
    command: list[str],
    workdir: str = DEFAULT_WORKDIR,
    nodes: int = 1,
    gpus_per_node: int = 0,
    time_limit: str = "04:00:00",
    extra_env: dict[str, str] | None = None,
) -> None:
    """Execute an omni3 model lifecycle command."""
    cfg = parse_recipe_config(ctx)
    env = parse_env(cfg.ctx)
    env_vars = build_env_vars({}, env)
    if extra_env:
        env_vars.update(extra_env)
    startup_commands = get_startup_commands(env)

    full_command = [*command, *cfg.passthrough]
    container_image = (
        env.get("container_image") or env.get("container") if env else None
    ) or DEFAULT_CONTAINER_IMAGE

    _show_plan(
        job_name=job_name,
        cfg=cfg,
        env=env,
        container_image=container_image,
        workdir=workdir,
        command=full_command,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        time_limit=time_limit,
    )

    if cfg.dry_run:
        return

    if cfg.mode == "local":
        _execute_local(
            command=full_command,
            workdir=workdir,
            env_vars=env_vars,
            startup_commands=startup_commands,
        )
    else:
        _execute_remote(
            job_name=job_name,
            command=full_command,
            workdir=workdir,
            env=env,
            attached=cfg.attached,
            env_vars=env_vars,
            startup_commands=startup_commands,
            force_squash=cfg.force_squash,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
        )
