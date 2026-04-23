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

"""Build container images for Omni3 stages."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nemo_runspec import parse as parse_runspec
from nemo_runspec.env import parse_env
from nemo_runspec.execution import clone_git_repos_via_tunnel, get_startup_commands, prepend_startup_to_cmd
from nemo_runspec.packaging import CodePackager
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.squash import resolve_build_image, resolve_build_partition, resolve_build_time

console = Console()

RECIPE_ROOT = Path("src/nemotron/recipes/omni3")
BUILD_IMAGE = "quay.io/podman/stable:v5.3"
BUILD_CACHE_MOUNT = f"{Path.home() / '.cache' / 'nemotron'}:/nemotron-cache"
STAGE_ALIASES = {
    "sft": "stage0_sft",
    "rl": "stage1_rl",
}


def _resolve_stage_dir(stage: str) -> Path:
    stage_dir = RECIPE_ROOT / STAGE_ALIASES.get(stage, stage)
    if not stage_dir.is_dir():
        raise typer.BadParameter(
            f"Unknown omni3 build stage '{stage}'. Expected one of: {', '.join(sorted(STAGE_ALIASES))}, or an existing stage directory."
        )
    build_script = stage_dir / "build.py"
    if not build_script.is_file():
        raise typer.BadParameter(f"Stage '{stage}' does not contain a build.py script")
    return stage_dir


def _show_build_plan(stage: str, stage_dir: Path, build_script: Path, spec, cfg, env) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Stage", stage)
    table.add_row("Stage dir", str(stage_dir))
    table.add_row("Recipe", spec.name)
    table.add_row("Script", str(build_script))
    table.add_row("Mode", cfg.mode)
    if env is not None:
        table.add_row("Profile", cfg.profile or "")
        table.add_row("Build image", resolve_build_image(env, BUILD_IMAGE))
        table.add_row("Partition", resolve_build_partition(env) or "")
        table.add_row("Time", resolve_build_time(env, "02:00:00"))
    else:
        table.add_row("Build image", "local host execution")
    table.add_row("Cache mount", BUILD_CACHE_MOUNT)
    if cfg.passthrough:
        table.add_row("Extra args", " ".join(cfg.passthrough))
    console.print(table)
    console.print()


def _execute_local(build_script: Path, passthrough: list[str]) -> None:
    result = subprocess.run(["python", str(build_script), *passthrough], check=False)
    raise typer.Exit(result.returncode)


def _execute_remote(build_script: Path, spec, passthrough: list[str], attached: bool, env) -> None:
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

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

    tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    try:
        tmp_config.write("{}\n")
        tmp_config.close()

        packager = CodePackager(
            script_path=str(build_script),
            train_path=tmp_config.name,
        )

        git_mounts = []
        if tunnel and remote_job_dir:
            tunnel.connect()
            git_mounts = clone_git_repos_via_tunnel(tunnel, remote_job_dir)

        raw_mounts = list(_get("mounts") or [])
        mounts = [m for m in raw_mounts if not m.startswith("__auto_mount__:")]
        mounts.extend(git_mounts)
        mounts.append("/lustre:/lustre")
        mounts.append(BUILD_CACHE_MOUNT)

        executor = run.SlurmExecutor(
            account=_get("account"),
            partition=resolve_build_partition(env),
            nodes=spec.resources.nodes,
            ntasks_per_node=1,
            gpus_per_node=spec.resources.gpus_per_node,
            cpus_per_task=_get("build_cpus", 16),
            time=resolve_build_time(env, "02:00:00"),
            container_image=resolve_build_image(env, BUILD_IMAGE),
            container_mounts=mounts,
            tunnel=tunnel,
            packager=packager,
            mem=_get("build_mem") or _get("mem"),
            env_vars={"NEMOTRON_CACHE_DIR": "/nemotron-cache"},
            launcher=None,
        )

        script_path = str(build_script)
        startup_commands = get_startup_commands(env)
        if startup_commands:
            build_cmd = shlex.join(["python", script_path, *passthrough])
            full_cmd = prepend_startup_to_cmd(startup_commands, build_cmd)
            script_task = run.Script(path="bash", args=["-lc", full_cmd])
        else:
            script_task = run.Script(path=script_path, args=passthrough, entrypoint="python")

        recipe_name = spec.name.replace("/", "-")
        with run.Experiment(recipe_name) as exp:
            exp.add(script_task, executor=executor, name=recipe_name)
            exp.run(detach=not attached)
    finally:
        Path(tmp_config.name).unlink(missing_ok=True)


def build(
    ctx: typer.Context,
    stage: str = typer.Argument(..., help="Stage name or directory to build (for example: sft, rl)."),
) -> None:
    """Build an Omni3 stage container via podman on the cluster."""
    cfg = parse_recipe_config(ctx)
    stage_dir = _resolve_stage_dir(stage)
    build_script = stage_dir / "build.py"
    spec = parse_runspec(build_script)
    env = parse_env(cfg.ctx)

    _show_build_plan(stage, stage_dir, build_script, spec, cfg, env)

    if cfg.dry_run:
        return

    if cfg.mode == "local":
        _execute_local(build_script, cfg.passthrough)
    else:
        _execute_remote(build_script, spec, cfg.passthrough, cfg.attached, env)
