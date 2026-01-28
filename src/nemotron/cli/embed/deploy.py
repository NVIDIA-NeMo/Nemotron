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

"""Deploy command implementation.

This module defines the `deploy` command for the embed recipe.
Launches NIM container with custom fine-tuned model for inference.
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

# Deploy only needs local execution (it's just a Docker wrapper)
# No need for dual-mode since it doesn't run inside containers
SCRIPT = _NEMOTRON_ROOT / "src/nemotron/recipes/embed/stage5_deploy/deploy.py"
CONFIG_DIR = _NEMOTRON_ROOT / "src/nemotron/recipes/embed/stage5_deploy/config"
# Config model for help display
from nemotron.recipes.embed.stage5_deploy.deploy import DeployConfig
CONFIG_MODEL = DeployConfig


def _deploy_local(options: RecipeConfig):
    """Execute deploy script locally.

    Deploy is a simple Docker wrapper that doesn't need nemo-run or remote execution.

    Args:
        options: Parsed recipe configuration
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
    builder = ConfigBuilder(
        recipe_name="embed/deploy",
        script_path=str(SCRIPT),
        config_dir=CONFIG_DIR,
        default_config="default",
        ctx=global_ctx,
        argv=sys.argv,
    )

    # Load and merge config
    builder.load_and_merge()

    # Build full job config
    builder.build_job_config()

    # 3. Display compiled configuration
    display_job_config(builder.job_config, for_remote=False)

    # 4. Handle dry-run mode
    if options.dry_run:
        return

    # 5. Save configs
    job_path, train_path = builder.save(packager="code")

    # 6. Execute the deploy script directly with current Python
    import subprocess

    cmd = [
        sys.executable,  # Use current Python environment
        str(SCRIPT),
        "--config",
        str(train_path),
        *options.passthrough,
    ]

    typer.echo(f"Executing deploy script: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


def deploy(ctx: typer.Context) -> None:
    """Deploy NIM container with custom fine-tuned model for inference."""
    options = parse_recipe_config(ctx)
    _deploy_local(options)
