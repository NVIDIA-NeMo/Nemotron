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

"""Finetune command implementation.

This module defines the `finetune` command for the embed recipe.
Fine-tunes an embedding model using contrastive learning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from nemotron.kit.cli.config import ConfigBuilder
from nemotron.kit.cli.display import display_job_config, display_job_submission
from nemotron.kit.cli.globals import GlobalContext
from nemotron.kit.cli.recipe_config import RecipeConfig, parse_recipe_config

# Module-level constants (visible paths and config)
# Resolve paths relative to Nemotron library root using importlib
import importlib.util
import nemotron

# Get Nemotron package root (libraries/Nemotron/src/nemotron)
_NEMOTRON_PKG_ROOT = Path(nemotron.__file__).parent
_NEMOTRON_ROOT = _NEMOTRON_PKG_ROOT.parent.parent  # Go up from src/nemotron to libraries/Nemotron

# Script paths for different execution modes
# Local: use train.py with PEP 723 header (UV resolves deps in isolated env)
# Docker/Slurm: use run_uv.py (creates venv with system-site-packages + exclude-dependencies)
SCRIPT_LOCAL = _NEMOTRON_ROOT / "src/nemotron/recipes/embed/stage2_finetune/train.py"
SCRIPT_REMOTE = _NEMOTRON_ROOT / "src/nemotron/recipes/embed/stage2_finetune/run_uv.py"
CONFIG_DIR = _NEMOTRON_ROOT / "src/nemotron/recipes/embed/stage2_finetune/config"
# Config model for help display
from nemotron.recipes.embed.stage2_finetune.train import FinetuneConfig
CONFIG_MODEL = FinetuneConfig
# Dependencies managed differently per mode:
# - Local: PEP 723 in train.py (UV handles everything)
# - Remote: pyproject.toml in stage2_finetune (exclude-dependencies for container packages)
DEPENDENCIES = []  # No pip dependencies - UV/pyproject.toml handle everything

# Artifact metadata for resolution (deferred to future PR)
ARTIFACTS = {
    "data": {
        "default": "EmbedTrainingData-default",
        "mappings": {"path": "dataloader.dataset.data_dir_list"},
    },
}


def _finetune_nemo_run(options: RecipeConfig, experiment=None):
    """Build nemo-run objects for finetune execution.

    Called directly by pipe for composition, or by finetune() for standalone execution.

    Args:
        options: Parsed recipe configuration
        experiment: Optional nemo-run Experiment for composition

    Returns:
        If experiment provided, returns the configured task for composition.
        Otherwise, executes immediately and does not return.
    """
    # 1. Build GlobalContext from RecipeConfig for ConfigBuilder
    global_ctx = GlobalContext(
        config=options.config,
        run=options.run,
        batch=options.batch,
        dry_run=options.dry_run,
        stage=options.stage,
        force_squash=options.force_squash,
        dotlist=options.dotlist,
        passthrough=options.passthrough,
    )

    # 2. Build configuration
    # Use appropriate script based on execution mode
    script_path = SCRIPT_LOCAL if options.mode == "local" else SCRIPT_REMOTE
    builder = ConfigBuilder(
        recipe_name="embed/finetune",
        script_path=str(script_path),
        config_dir=CONFIG_DIR,
        default_config="default",
        ctx=global_ctx,
        argv=sys.argv,
    )

    # Load and merge config
    builder.load_and_merge()

    # TODO: Resolve artifacts if run.data specified
    # Implementation deferred to future PR
    # This would:
    # 1. Check if run.data in dotlist overrides
    # 2. Parse artifact reference (name:version or name:latest)
    # 3. Download artifact from W&B
    # 4. Apply mappings from ARTIFACTS["data"]["mappings"] to config
    # 5. For example: set dataloader.dataset.data_dir_list to artifact path

    # Build full job config
    builder.build_job_config()

    # 3. Display compiled configuration
    # Show resolved paths for remote execution (--run/--batch), but not for "code" packager
    for_remote = options.mode in ("run", "batch") and False  # code packager: always False
    display_job_config(builder.job_config, for_remote=for_remote)

    # 4. Handle dry-run mode
    if options.dry_run:
        return

    # 5. Save configs
    job_path, train_path = builder.save(packager="code")

    # 6. Handle stage-only mode
    if options.stage:
        from nemotron.kit.cli.recipe import _execute_stage_only

        _execute_stage_only(
            script_path=str(script_path),
            train_path=train_path,
            job_dir=builder.job_dir,
            job_config=builder.job_config,
            packager="code",
            torchrun=False,  # Finetune doesn't use torchrun (Automodel handles distributed)
        )
        return

    # 7. Extract env config for building env vars
    env_config = None
    if hasattr(builder.job_config, "run") and hasattr(builder.job_config.run, "env"):
        from omegaconf import OmegaConf

        env_config = OmegaConf.to_container(builder.job_config.run.env, resolve=True)

    # 8. Build env vars for display (needs job_config for wandb settings)
    from nemotron.kit.cli.recipe import _build_env_vars

    env_vars = _build_env_vars(builder.job_config, env_config)

    # 9. Display job submission summary
    display_job_submission(job_path, train_path, env_vars, options.mode)

    # 10. Execute based on mode
    if options.mode == "local":
        import subprocess
        import shutil
        from pathlib import Path

        # Use uv run to execute the script in an isolated environment
        # The script has PEP 723 metadata that specifies its dependencies
        uv_cmd = shutil.which("uv")
        if not uv_cmd:
            typer.echo("Error: 'uv' command not found. Please install uv.", err=True)
            raise typer.Exit(1)

        # Use uv run with pyproject.toml from stage2_finetune
        # This reads dependencies from pyproject.toml (single source of truth)
        stage_dir = SCRIPT_LOCAL.parent  # stage2_finetune directory
        cmd = [
            uv_cmd,
            "run",
            "--with",
            str(_NEMOTRON_ROOT),  # Add Nemotron library (has pyproject.toml)
            "--project",
            str(stage_dir),  # Use pyproject.toml from stage2_finetune/
            "python",
            str(SCRIPT_LOCAL),  # train.py
            "--config",
            str(train_path),
            *options.passthrough,
        ]

        # Unset VIRTUAL_ENV to avoid conflicts with UV's project environment
        import os
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)

        typer.echo(f"Executing with uv isolated environment: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        raise typer.Exit(result.returncode)
    else:
        # Build nemo-run executor and execute
        from nemotron.kit.cli.recipe import _build_executor

        import nemo_run as run

        # Apply patches before building executor
        from nemotron.kit.run import (
            patch_nemo_run_ray_template_for_cpu,
            patch_nemo_run_rsync_accept_new_host_keys,
        )

        patch_nemo_run_rsync_accept_new_host_keys()
        patch_nemo_run_ray_template_for_cpu()

        # Build executor (VISIBLE - no longer hidden in decorator)
        # Use run_uv.py for Docker/Slurm execution (handles venv + exclude-dependencies)
        executor = _build_executor(
            env_config,
            builder.job_config,
            str(SCRIPT_REMOTE),  # run_uv.py for Docker/Slurm
            train_path,
            builder.job_dir,
            env_vars,
            torchrun=False,  # Finetune doesn't use torchrun
            ray=False,  # Finetune doesn't use Ray
            attached=options.attached,
            packager="code",  # Full repo sync
            force_squash=options.force_squash,
            dependencies=DEPENDENCIES,  # EXPLICIT dependency list
        )

        # Use packager's main.py launcher (which calls run_uv.py → UV → train.py)
        # The packager's main.py already adds --config config.yaml, so only
        # pass through extra args here to avoid duplicate --config flags.
        script_args = [*options.passthrough]

        # Get experiment name from recipe
        recipe_name = "embed-finetune"

        # Composable: if experiment provided (by pipe), add and return
        if experiment:
            return experiment.add(
                run.Script(
                    path="main.py",  # Packager's generated launcher
                    args=script_args,
                    entrypoint="python3",
                ),
                executor=executor,
                name=recipe_name,
            )

        # Otherwise run immediately
        with run.Experiment(recipe_name) as exp:
            exp.add(
                run.Script(
                    path="main.py",  # Packager's generated launcher
                    args=script_args,
                    entrypoint="python3",
                ),
                executor=executor,
                name=recipe_name,
            )
            exp.run(detach=not options.attached, tail_logs=options.attached)


def finetune(ctx: typer.Context) -> None:
    """Fine-tune embedding model with contrastive learning."""
    options = parse_recipe_config(ctx)
    _finetune_nemo_run(options)
