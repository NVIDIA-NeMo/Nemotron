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

Launches NIM container with custom fine-tuned model for inference.
Deploy is local-only (it's just a Docker wrapper).
"""

from __future__ import annotations

import subprocess
import sys

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import (
    build_job_config,
    extract_train_config,
    generate_job_dir,
    parse_config,
    save_configs,
)
from nemo_runspec.display import display_job_config
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.recipes.embed.stage5_deploy.deploy import DeployConfig

SCRIPT_PATH = "src/nemotron/recipes/embed/stage5_deploy/deploy.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    config_model=DeployConfig,
    default_config=SPEC.config.default,
    input_artifacts={"model": "Exported model directory (ONNX/TensorRT)"},
)


def _execute_deploy(cfg: RecipeConfig):
    """Execute deploy locally."""
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        SCRIPT_PATH,
        cfg.argv,
    )

    display_job_config(job_config, for_remote=False)

    if cfg.dry_run:
        return

    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=False)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    cmd = [
        sys.executable,
        str(SPEC.script_path),
        "--config", str(train_path),
        *cfg.passthrough,
    ]

    typer.echo(f"Executing deploy script: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


def deploy(ctx: typer.Context) -> None:
    """Deploy NIM container with custom fine-tuned model for inference."""
    cfg = parse_recipe_config(ctx)
    _execute_deploy(cfg)
