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

"""Pipeline orchestration for the rerank recipe."""

from __future__ import annotations

import shutil
import subprocess
import sys

import typer

from nemo_runspec.cli_context import GlobalContext
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config

# Stage order defines the DAG
STAGE_ORDER = ["sdg", "prep", "finetune", "eval", "export", "deploy"]
DEFAULT_TO = "eval"  # export/deploy are opt-in

# Map stage names to their modules (lazy import to avoid circular deps)
STAGE_MODULES = {
    "sdg": "nemotron.cli.commands.rerank.sdg",
    "prep": "nemotron.cli.commands.rerank.prep",
    "finetune": "nemotron.cli.commands.rerank.finetune",
    "eval": "nemotron.cli.commands.rerank.eval",
    "export": "nemotron.cli.commands.rerank.export",
    # deploy has no remote execution support
}

# Map stage names to their execution function names
STAGE_RUN_FNS = {
    "sdg": "_execute_sdg",
    "prep": "_execute_prep",
    "finetune": "_execute_finetune",
    "eval": "_execute_eval",
    "export": "_execute_export",
    # deploy has no remote execution support
}


def _resolve_stages(from_stage: str, to_stage: str) -> list[str]:
    """Return the ordered list of stages to run."""
    if from_stage not in STAGE_ORDER:
        raise typer.BadParameter(f"Unknown stage: {from_stage}")
    if to_stage not in STAGE_ORDER:
        raise typer.BadParameter(f"Unknown stage: {to_stage}")
    start = STAGE_ORDER.index(from_stage)
    end = STAGE_ORDER.index(to_stage)
    if start > end:
        raise typer.BadParameter(f"--from {from_stage} is after --to {to_stage}")
    return STAGE_ORDER[start : end + 1]


def _split_stage_overrides(dotlist: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split dotlist into per-stage overrides and global overrides.

    'finetune.learning_rate=1e-5' -> stage='finetune', override='learning_rate=1e-5'
    'num_epochs=5'                -> global override (applied to all stages)
    """
    stage_overrides: dict[str, list[str]] = {}
    global_overrides: list[str] = []

    for item in dotlist:
        key = item.split("=", 1)[0]
        parts = key.split(".", 1)
        if len(parts) == 2 and parts[0] in STAGE_ORDER:
            stage_name = parts[0]
            rest = parts[1] + "=" + item.split("=", 1)[1] if "=" in item else parts[1]
            stage_overrides.setdefault(stage_name, []).append(rest)
        else:
            global_overrides.append(item)

    return stage_overrides, global_overrides


def _run_pipeline_local(
    stages: list[str],
    base_options: RecipeConfig,
    stage_overrides: dict[str, list[str]],
    global_overrides: list[str],
) -> None:
    """Run stages sequentially as local subprocesses."""
    for stage_name in stages:
        print(f"\n{'=' * 60}")
        print(f"  Stage: {stage_name}")
        print(f"{'=' * 60}\n")

        stage_dotlist = global_overrides + stage_overrides.get(stage_name, [])

        uv_cmd = shutil.which("uv")
        cmd = [uv_cmd or "uv", "run", "nemotron", "rerank", stage_name]
        if base_options.config:
            cmd.extend(["-c", base_options.config])
        cmd.extend(stage_dotlist)
        cmd.extend(base_options.passthrough)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nError: Stage '{stage_name}' failed with exit code {result.returncode}")
            raise typer.Exit(result.returncode)


def _env_value(env, key: str):
    if env is None:
        return None
    getter = getattr(env, "get", None)
    if getter is not None:
        return getter(key)
    return getattr(env, key, None)


def _executor_type(env) -> str:
    value = _env_value(env, "executor")
    return str(value) if value is not None else "local"


def _has_shared_run_dir(env) -> bool:
    if _env_value(env, "remote_job_dir"):
        return True
    env_vars = _env_value(env, "env_vars")
    return bool(_env_value(env_vars, "NEMO_RUN_DIR"))


def _validate_remote_stages(stages: list[str]) -> None:
    """Fail fast when a pipeline includes stages that cannot run remotely."""
    unsupported = [stage for stage in stages if stage not in STAGE_RUN_FNS]
    if unsupported:
        names = ", ".join(unsupported)
        print(
            f"Error: stage(s) {names} do not support remote execution; "
            "run them locally or stop the remote pipeline before them.",
            file=sys.stderr,
        )
        raise typer.Exit(1)


def _validate_remote_pipeline(stages: list[str], base_options: RecipeConfig) -> None:
    _validate_remote_stages(stages)
    if base_options.stage:
        print(
            "Error: --stage is not supported for rerank run; "
            "run a single stage command with --dry-run for inspection.",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    if base_options.dry_run or len(stages) <= 1:
        return

    from nemo_runspec.env import parse_env

    env = parse_env(base_options.ctx)
    # Local and Docker executors run on this machine. For Docker, nemo-run stages
    # execute from the packaged code directory inside the container, so the
    # default ${oc.env:NEMO_RUN_DIR,.} paths still share the mounted worktree.
    if _executor_type(env) in {"local", "docker"}:
        return

    if not _has_shared_run_dir(env):
        print(
            "Error: remote rerank pipelines with multiple stages require env.remote_job_dir "
            "or env.env_vars.NEMO_RUN_DIR so stages share outputs.",
            file=sys.stderr,
        )
        raise typer.Exit(1)


def _run_pipeline_remote(
    stages: list[str],
    base_options: RecipeConfig,
    stage_overrides: dict[str, list[str]],
    global_overrides: list[str],
) -> None:
    """Run stages as tasks in a single nemo-run Experiment."""
    _validate_remote_pipeline(stages, base_options)

    from importlib import import_module

    import nemo_run as run

    experiment_name = "rerank-" + "-".join(stages)

    with run.Experiment(experiment_name) as exp:
        for stage_name in stages:
            stage_dotlist = global_overrides + stage_overrides.get(stage_name, [])

            ctx = GlobalContext(
                config=base_options.ctx.config,
                run=base_options.ctx.run,
                batch=base_options.ctx.batch,
                dry_run=base_options.ctx.dry_run,
                stage=base_options.ctx.stage,
                force_squash=base_options.ctx.force_squash,
            )
            ctx.dotlist = stage_dotlist
            ctx.passthrough = base_options.passthrough

            options = RecipeConfig(
                ctx=ctx,
                argv=sys.argv,
                dotlist=stage_dotlist,
                passthrough=base_options.passthrough,
            )

            mod = import_module(STAGE_MODULES[stage_name])
            run_fn = getattr(mod, STAGE_RUN_FNS[stage_name])
            run_fn(options, experiment=exp)

        exp.run(
            sequential=True,
            detach=not base_options.attached,
            tail_logs=base_options.attached,
        )


def run(
    ctx: typer.Context,
    from_stage: str = typer.Option("sdg", "--from", help="Stage to start from"),
    to_stage: str = typer.Option(DEFAULT_TO, "--to", help="Stage to stop at (inclusive)"),
) -> None:
    """Run the rerank pipeline end-to-end.

    Executes stages sequentially from --from to --to (default: sdg through eval).
    Supports local, Docker, and Slurm execution via --run/--batch.

    Per-stage overrides use stage prefix: sdg.num_pairs=20, finetune.learning_rate=1e-5
    """
    options = parse_recipe_config(ctx)
    stages = _resolve_stages(from_stage, to_stage)
    stage_overrides, global_overrides = _split_stage_overrides(options.dotlist)

    if options.mode != "local":
        _validate_remote_pipeline(stages, options)

    print(f"Rerank pipeline: {' -> '.join(stages)}")
    if options.config:
        print(f"Config: {options.config}")
    print()

    if options.dry_run:
        print("Dry run — would execute these stages:")
        for s in stages:
            overrides = global_overrides + stage_overrides.get(s, [])
            print(f"  {s}: {overrides or '(no overrides)'}")
        return

    if options.mode == "local":
        _run_pipeline_local(stages, options, stage_overrides, global_overrides)
    else:
        _run_pipeline_remote(stages, options, stage_overrides, global_overrides)
