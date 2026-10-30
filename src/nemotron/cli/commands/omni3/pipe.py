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

"""Pipeline command for the Omni3 family."""

from __future__ import annotations

import typer

from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

META = RecipeMeta(
    name="omni3/pipe",
    script_path="",
    config_dir="",
    default_config="",
    input_artifacts={
        "model": "Imported GA checkpoint consumed by the SFT stage",
        "data": "Prepared SFT and RL data artifacts",
    },
    output_artifacts={"model": "Final Omni checkpoint after the vision RL stage"},
)


def _print_plan(cfg) -> None:
    typer.echo("Omni3 pipeline: sft -> rl mpo -> rl text -> rl vision")
    typer.echo("Artifact chain:")
    typer.echo("  sft         -> omni3-sft-model:latest")
    typer.echo("  rl mpo      <- omni3-sft-model:latest")
    typer.echo("  rl text     <- omni3-rl-mpo-model:latest")
    typer.echo("  rl vision   <- omni3-rl-text-model:latest")
    typer.echo("  final       -> omni3-vision-rl-model:latest")
    if cfg.dotlist:
        typer.echo(f"Dotlist overrides: {' '.join(cfg.dotlist)}")
    if cfg.passthrough:
        typer.echo(f"Passthrough args: {' '.join(cfg.passthrough)}")


def _execute_pipe(cfg):
    """Run the Omni SFT -> RL pipeline serially."""
    from nemotron.cli.commands.omni3.rl._base import _execute_rl
    from nemotron.cli.commands.omni3.rl.mpo import SCRIPT_PATH as MPO_SCRIPT_PATH
    from nemotron.cli.commands.omni3.rl.mpo import SPEC as MPO_SPEC
    from nemotron.cli.commands.omni3.rl.text import SCRIPT_PATH as TEXT_SCRIPT_PATH
    from nemotron.cli.commands.omni3.rl.text import SPEC as TEXT_SPEC
    from nemotron.cli.commands.omni3.rl.vision import SCRIPT_PATH as VISION_SCRIPT_PATH
    from nemotron.cli.commands.omni3.rl.vision import SPEC as VISION_SPEC
    from nemotron.cli.commands.omni3.sft import _execute_sft

    _print_plan(cfg)

    if cfg.dry_run:
        return

    if cfg.mode != "run":
        typer.echo(
            "Error: omni3 pipe requires --run for execution so stages complete serially through artifact lineage. "
            "Use --dry-run to preview the plan.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("\n=== Stage 1/4: sft ===\n")
    _execute_sft(cfg)

    typer.echo("\n=== Stage 2/4: rl mpo ===\n")
    _execute_rl(cfg, script_path=MPO_SCRIPT_PATH, spec=MPO_SPEC)

    typer.echo("\n=== Stage 3/4: rl text ===\n")
    _execute_rl(cfg, script_path=TEXT_SCRIPT_PATH, spec=TEXT_SPEC)

    typer.echo("\n=== Stage 4/4: rl vision ===\n")
    _execute_rl(cfg, script_path=VISION_SCRIPT_PATH, spec=VISION_SPEC)


def pipe(ctx: typer.Context) -> None:
    """Run the Omni SFT -> MPO -> text RL -> vision RL pipeline.

    `--dry-run` previews the planned stage order anywhere. Actual execution
    requires `--run` so each stage can finish before the next consumes its
    checkpoint artifact.
    """
    cfg = parse_recipe_config(ctx)
    _execute_pipe(cfg)
