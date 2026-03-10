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

"""SFT command implementation.

This module defines the `sft` command for the super3 recipe with
**visible execution logic**. All nemo-run setup is explicit in this file.

To change the execution backend (e.g., swap nemo-run for SkyPilot),
modify _execute_sft() in this file. The runtime script (train.py)
and config builder stay unchanged.

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Fork this file to change how jobs are submitted
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
from nemo_runspec.execution import (
    build_env_vars,
    create_executor,
    execute_local,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.packaging import REMOTE_CONFIG, REMOTE_SCRIPT
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

# =============================================================================
# Recipe Metadata (read from [tool.runspec] in script)
# =============================================================================

SCRIPT_PATH = "src/nemotron/recipes/super3/stage1_sft/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

# For help panels
META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={
        "model": "Base model checkpoint (from pretrain)",
        "data": "SFT data artifact (packed .npy)",
    },
    output_artifacts={"model": "Fine-tuned model checkpoint"},
)


# =============================================================================
# Execution Logic
# =============================================================================


def _execute_sft(cfg: RecipeConfig, *, experiment=None):
    """Execute SFT with nemo-run.

    This function contains the VISIBLE execution logic. To swap nemo-run
    for SkyPilot or another backend, modify this function.

    Args:
        cfg: Parsed recipe configuration
        experiment: Optional nemo-run Experiment for pipeline composition.
                   If provided, adds task and returns. If None, creates
                   experiment and runs immediately.

    Returns:
        For pipeline composition, returns the added task.
    """
    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    # Build full job config with provenance
    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        SCRIPT_PATH,
        cfg.argv,
        env_profile=env,
    )

    # Display compiled configuration
    for_remote = cfg.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)

    # Handle dry-run mode
    if cfg.dry_run:
        return

    # =========================================================================
    # 2. Save configs and prepare execution
    # =========================================================================
    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=for_remote)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    # Get env config from job_config.run.env (merged YAML + env.toml)
    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None

    env_vars = build_env_vars(job_config, env_for_executor)

    # Display job submission summary
    display_job_submission(job_path, train_path, env_vars, cfg.mode)

    # Get startup commands from env config
    startup_commands = get_startup_commands(env_for_executor)

    # =========================================================================
    # 3. Execute based on mode
    # =========================================================================
    if cfg.mode == "local":
        execute_local(
            SCRIPT_PATH,
            train_path,
            cfg.passthrough,
            torchrun=(SPEC.run.launch == "torchrun"),
            env_vars=env_vars,
            startup_commands=startup_commands,
        )
    else:
        # Remote execution via nemo-run
        _execute_remote(
            train_path=train_path,
            env=env_for_executor,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            startup_commands=startup_commands,
            force_squash=cfg.force_squash,
            experiment=experiment,
        )


def _execute_remote(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
    experiment=None,
):
    """Execute via nemo-run with Slurm backend.

    This is the VISIBLE nemo-run execution logic. To understand how SFT
    jobs are submitted, read this function.

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

    # Apply nemo-run patches
    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    # Build packager - explicit choice of how code is bundled
    packager = SelfContainedPackager(
        script_path=SCRIPT_PATH,
        train_path=str(train_path),
    )

    # Build Executor - for SkyPilot or other backends, replace create_executor
    executor = create_executor(
        env=env,
        env_vars=env_vars,
        packager=packager,
        attached=attached,
        force_squash=force_squash,
        default_image=SPEC.image,
    )

    # Build Script and Run
    recipe_name = SPEC.name.replace("/", "-")
    script_args = ["--config", REMOTE_CONFIG, *passthrough]

    if startup_commands:
        import shlex

        train_cmd = shlex.join(["python", REMOTE_SCRIPT, *script_args])
        full_cmd = prepend_startup_to_cmd(startup_commands, train_cmd)
        script_task = run.Script(
            path="bash",
            args=["-lc", full_cmd],
        )
    else:
        script_task = run.Script(
            path=REMOTE_SCRIPT,
            args=script_args,
            entrypoint="python",
        )

    if experiment is not None:
        return experiment.add(script_task, executor=executor, name=recipe_name)

    with run.Experiment(recipe_name) as exp:
        exp.add(script_task, executor=executor, name=recipe_name)
        exp.run(detach=not attached)


# =============================================================================
# CLI Entry Point
# =============================================================================


def sft(ctx: typer.Context) -> None:
    """Run supervised fine-tuning with Megatron-Bridge (stage1).

    This command fine-tunes a Nemotron Super3 model using Megatron-Bridge.
    The execution logic is visible in this file - see _execute_sft()
    for nemo-run setup.
    """
    cfg = parse_recipe_config(ctx)
    _execute_sft(cfg)
