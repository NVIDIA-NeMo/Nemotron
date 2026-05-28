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

"""`nemotron steps run` — generic step execution.

Thin dispatcher. The job of this command is to:
  1. Resolve a step id → step.py + runspec.
  2. Compile the YAML config + CLI overrides + env profile.
  3. Pick a backend by name (``local`` / ``slurm`` / ``lepton`` / ``dgxcloud``).
  4. Hand off to ``backend.submit(ctx)``.

All execution-mechanics live in the per-backend modules under
``nemotron.cli.commands.steps.backends.*``. To add a new submission target,
write one Backend subclass and ``register()`` it — no edits here.

Overrides are passed as bare ``key=value`` positionals at the end of the
command, e.g. ``nemotron steps run peft/automodel -c default train.train_iters=5000``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.cli_context import GlobalContext, split_unknown_args
from nemo_runspec.config import (
    build_job_config,
    extract_train_config,
    generate_job_dir,
    parse_config,
    save_configs,
)
from nemo_runspec.display import display_job_config, display_job_submission
from nemo_runspec.env import parse_env
from nemo_runspec.execution import build_env_vars, get_startup_commands
from nemotron.cli.commands.steps._resolve import resolve_step
from nemotron.cli.commands.steps.backends import JobContext, get_backend

_CURATOR_RUNTIME_MODULE = "nemotron.steps._bootstrap.curator_runtime"


def run_step(
    ctx: typer.Context,
    step_id: Annotated[str, typer.Argument(help="Step id, e.g. peft/automodel.")],
    config: Annotated[
        str | None,
        typer.Option("-c", "--config", help="Config name (in step's config/) or path."),
    ] = None,
    run: Annotated[str | None, typer.Option("-r", "--run", help="Attached execution via env profile.")] = None,
    batch: Annotated[str | None, typer.Option("-b", "--batch", help="Detached execution via env profile.")] = None,
    dry_run: Annotated[bool, typer.Option("-d", "--dry-run", help="Compile config and exit.")] = False,
    force_squash: Annotated[bool, typer.Option("--force-squash", help="Force re-squash of container image.")] = False,
) -> None:
    step = resolve_step(step_id)
    script_path = step.path / "step.py"
    if not script_path.exists():
        typer.echo(f"step.py missing for {step.id} at {script_path}", err=True)
        raise typer.Exit(1)

    spec = parse_runspec(str(script_path))
    config_name = config or spec.config.default

    global_ctx = GlobalContext(
        config=config_name,
        run=run,
        batch=batch,
        dry_run=dry_run,
        force_squash=force_squash,
    )
    dotlist, passthrough, global_ctx = split_unknown_args(ctx.args, global_ctx)
    global_ctx.dotlist = dotlist
    global_ctx.passthrough = passthrough

    train_config = parse_config(global_ctx, spec.config_dir, spec.config.default)
    env = parse_env(global_ctx)
    job_config = build_job_config(
        train_config,
        global_ctx,
        spec.name,
        str(script_path),
        ctx.args,
        env_profile=env,
    )

    for_remote = global_ctx.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)
    if global_ctx.dry_run:
        return

    job_dir = generate_job_dir(spec.name)
    train_for_script = extract_train_config(job_config, for_remote=for_remote)
    job_path, train_path = save_configs(job_config, train_for_script, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    if env_for_executor is not None and not isinstance(env_for_executor, dict):
        # OmegaConf DictConfig → plain dict (validators downstream type-check on dict / list).
        try:
            from omegaconf import OmegaConf

            env_for_executor = OmegaConf.to_container(env_for_executor, resolve=True)
        except Exception:
            pass

    env_vars = build_env_vars(job_config, env_for_executor)
    startup_commands = list(get_startup_commands(env_for_executor) or [])
    curator_runtime_env = _build_curator_runtime_env_vars(
        script_path=script_path,
        env=env_for_executor,
        mode=global_ctx.mode,
    )

    display_job_submission(
        job_path,
        train_path,
        env_vars,
        global_ctx.mode,
        artifacts=job_config.get("artifacts"),
    )
    env_vars.update(curator_runtime_env)

    executor_type = _executor_type(env_for_executor, default="local" if global_ctx.mode == "local" else None)
    if executor_type is None:
        typer.echo(
            "No executor selected. Pass --run <profile> / --batch <profile> "
            "or set executor in env.toml.",
            err=True,
        )
        raise typer.Exit(1)

    backend = get_backend(executor_type)
    backend.submit(
        JobContext(
            step_id=step.id,
            script_path=script_path,
            train_path=train_path,
            spec=spec,
            env=env_for_executor,
            env_vars=env_vars,
            passthrough=global_ctx.passthrough,
            startup_commands=startup_commands,
            attached=(global_ctx.mode == "run"),
            force_squash=global_ctx.force_squash,
        )
    )


def _executor_type(env: object, *, default: str | None) -> str | None:
    """Read ``executor`` from a plain dict or OmegaConf DictConfig, with a fallback."""
    if env is None:
        return default
    if hasattr(env, "get"):
        return env.get("executor", default)
    return getattr(env, "executor", default)


def _build_curator_runtime_env_vars(*, script_path: Path, env: object, mode: str) -> dict[str, str]:
    """Build env-encoded Curator runtime requirements for remote submission."""
    if mode not in {"run", "batch"} or not _uses_curator_runtime(env):
        return {}

    from nemotron.steps._bootstrap import runtime_payloads

    source_checkout = _find_source_checkout_root(script_path)
    if source_checkout is not None:
        payloads = runtime_payloads.build_runtime_payloads(source_checkout)
        source_description = str(source_checkout)
    else:
        payloads = runtime_payloads.read_runtime_payloads()
        source_description = str(runtime_payloads.DEFAULT_OUTPUT_DIR)

    if not payloads:
        typer.echo(
            "Curator runtime metadata is required for this remote profile, but none was found. "
            "Run this command from a source checkout containing pyproject.toml and src/nemotron, "
            "or install a wheel that includes nemotron.steps._bootstrap.runtime package data.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Prepared Curator runtime requirements from {source_description}")
    return runtime_payloads.encode_runtime_payload_env(payloads)


def _uses_curator_runtime(env: object) -> bool:
    run_command = _env_get(env, "run_command")
    return isinstance(run_command, str) and _CURATOR_RUNTIME_MODULE in run_command


def _env_get(env: object, key: str, default: object = None) -> object:
    if env is None:
        return default
    if hasattr(env, "get"):
        return env.get(key, default)
    return getattr(env, key, default)


def _find_source_checkout_root(path: Path) -> Path | None:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "nemotron").is_dir():
            return candidate
    return None
