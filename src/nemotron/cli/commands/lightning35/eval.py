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

"""Eval command implementation for lightning35 recipe (stage3, NeMo Gym).

NeMo Evaluator is deprecated for the instruct-model benchmark suite in
favor of NeMo Gym. Unlike the old evaluator command, this stage has a
regular recipe script (stage3_eval/eval.py) that serves the checkpoint
with vLLM in-job and drives `gym eval` against it — so submission follows
the standard runspec pattern (same as data prep / sft).

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Fork this file to change how evaluations are submitted
"""

from __future__ import annotations

from pathlib import Path

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import (
    build_job_config,
    extract_train_config,
    generate_job_dir,
    parse_config,
    save_configs,
)
from nemo_runspec.display import display_job_config, display_job_submission
from nemo_runspec.env import parse_env
from nemo_runspec.evaluator import get_non_task_args, parse_task_flags
from nemo_runspec.execution import (
    build_env_vars,
    create_executor,
    execute_cloud,
    execute_local,
    get_executor_type,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.packaging import REMOTE_CONFIG, REMOTE_SCRIPT
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

# =============================================================================
# Recipe Metadata (read from [tool.runspec] in script)
# =============================================================================

SCRIPT_PATH = "src/nemotron/recipes/lightning35/stage3_eval/eval.py"
SPEC = parse_runspec(SCRIPT_PATH)

# For help panels
META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"model": "HF model checkpoint to evaluate (default: RL stage output)"},
    output_artifacts={},
)


def _tasks_to_override(passthrough: list[str]) -> list[str]:
    """Translate -t/--task flags into a gym.benchmarks override.

    ``-t gpqa -t scicode`` becomes ``gym.benchmarks=[gpqa,scicode]``; all
    other passthrough args are forwarded to the script as hydra overrides.
    """
    tasks = parse_task_flags(passthrough)
    remaining = get_non_task_args(passthrough)
    if tasks:
        remaining.append(f"gym.benchmarks=[{','.join(tasks)}]")
    return remaining


# =============================================================================
# Execution Logic
# =============================================================================


def _execute_eval(cfg: RecipeConfig):
    """Execute Gym evaluation via the standard runspec submission path."""
    # --stage is not supported for eval
    if cfg.stage:
        typer.echo("Error: --stage is not supported for eval commands", err=True)
        raise typer.Exit(1)

    passthrough = _tasks_to_override(cfg.passthrough)

    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        SCRIPT_PATH,
        cfg.argv,
        env_profile=env,
    )

    for_remote = cfg.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)

    if cfg.dry_run:
        return

    # =========================================================================
    # 2. Save configs and prepare execution
    # =========================================================================
    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=for_remote)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    env_vars = build_env_vars(job_config, env_for_executor)

    display_job_submission(job_path, train_path, env_vars, cfg.mode, artifacts=job_config.get("artifacts"))

    startup_commands = get_startup_commands(env_for_executor)

    # =========================================================================
    # 3. Execute based on mode
    # =========================================================================
    if cfg.mode == "local":
        execute_local(
            SCRIPT_PATH,
            train_path,
            passthrough,
            torchrun=False,  # single-process driver: vLLM + gym CLI subprocesses
            env_vars=env_vars,
            startup_commands=startup_commands,
        )
    elif get_executor_type(env_for_executor) in ("dgxcloud", "lepton"):
        execute_cloud(
            SCRIPT_PATH,
            train_path,
            env=env_for_executor,
            env_vars=env_vars,
            passthrough=passthrough,
            attached=cfg.attached,
            default_image=SPEC.image,
            script_resources=SPEC.resources,
            startup_commands=startup_commands,
        )
    else:
        _execute_remote(
            train_path=train_path,
            env=env_for_executor,
            passthrough=passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            startup_commands=startup_commands,
            force_squash=cfg.force_squash,
        )


def _execute_remote(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
):
    """Execute via nemo-run with Slurm backend.

    FORK POINT: Replace this function with SkyPilot, custom submission, etc.
    """
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.packaging import SelfContainedPackager
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    packager = SelfContainedPackager(
        script_path=SCRIPT_PATH,
        train_path=str(train_path),
    )

    executor = create_executor(
        env=env,
        env_vars=env_vars,
        packager=packager,
        attached=attached,
        force_squash=force_squash,
        default_image=SPEC.image,
    )

    recipe_name = SPEC.name.replace("/", "-")
    script_args = ["--config", REMOTE_CONFIG, *passthrough]

    if startup_commands:
        import shlex

        eval_cmd = shlex.join(["python", REMOTE_SCRIPT, *script_args])
        full_cmd = prepend_startup_to_cmd(startup_commands, eval_cmd)
        script_task = run.Script(path="bash", args=["-lc", full_cmd])
    else:
        script_task = run.Script(path=REMOTE_SCRIPT, args=script_args, entrypoint="python")

    with run.Experiment(recipe_name) as exp:
        exp.add(script_task, executor=executor, name=recipe_name)
        exp.run(detach=not attached)


# =============================================================================
# CLI Entry Point
# =============================================================================


def eval(ctx: typer.Context) -> None:
    """Run evaluation with NeMo Gym (stage3).

    Serves the trained checkpoint with vLLM in-job and runs the Gym-native
    benchmark suite against it. By default, evaluates the RL stage output
    (run.model=lightning35-rl-model:latest).

    Examples:
        # Eval on cluster (loads env.toml profile)
        nemotron lightning35 eval --run MY-CLUSTER

        # Override model artifact
        nemotron lightning35 eval --run MY-CLUSTER run.model=sft-model:v2

        # Filter specific benchmarks
        nemotron lightning35 eval --run MY-CLUSTER -t gpqa -t scicode

        # Smoke run (5 rows per benchmark)
        nemotron lightning35 eval --run MY-CLUSTER gym.limit=5

        # Dry run (show resolved config without executing)
        nemotron lightning35 eval --run MY-CLUSTER --dry-run
    """
    cfg = parse_recipe_config(ctx)
    _execute_eval(cfg)
