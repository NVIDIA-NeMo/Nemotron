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


def _is_vision_stubbed(vision_spec) -> bool:
    """Detect the vision stub by its degenerate resource footprint.

    stage3_vision_rl/train.py declares nodes=1, gpus_per_node=0 while its
    main() raises NotImplementedError; the real launcher (TBD upstream) will
    bump these back to the production footprint. Treat that signal as "skip
    vision" in pipe runs so we don't burn 3 successful stages only to crash
    at stage 4.
    """
    return (vision_spec.resources.gpus_per_node or 0) == 0


def _force_vision(cfg) -> bool:
    """User override to include vision even when it looks stubbed.

    Pass `pipe.force_vision=true` via the standard dotlist mechanism once the
    upstream launcher has landed and the runspec has been bumped back.
    """
    return any("pipe.force_vision=true" in item for item in (cfg.dotlist or []))


def _print_plan(cfg, *, include_vision: bool) -> None:
    typer.echo("Omni3 pipeline: sft -> rl mpo -> rl text" + (" -> rl vision" if include_vision else ""))
    typer.echo("Artifact chain:")
    typer.echo("  sft         -> omni3-sft-model:latest")
    typer.echo("  rl mpo      <- omni3-sft-model:latest")
    typer.echo("  rl text     <- omni3-rl-mpo-model:latest")
    if include_vision:
        typer.echo("  rl vision   <- omni3-rl-text-model:latest")
        typer.echo("  final       -> omni3-vision-rl-model:latest")
    else:
        typer.echo("  rl vision   SKIPPED (launcher stubbed; pass pipe.force_vision=true to include)")
        typer.echo("  final       -> omni3-rl-text-model:latest")
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

    include_vision = not _is_vision_stubbed(VISION_SPEC) or _force_vision(cfg)

    _print_plan(cfg, include_vision=include_vision)

    if cfg.dry_run:
        return

    if cfg.mode != "run":
        typer.echo(
            "Error: omni3 pipe requires --run for execution so stages complete serially through artifact lineage. "
            "Use --dry-run to preview the plan.",
            err=True,
        )
        raise typer.Exit(1)

    total_stages = 4 if include_vision else 3

    typer.echo(f"\n=== Stage 1/{total_stages}: sft ===\n")
    _execute_sft(cfg)

    typer.echo(f"\n=== Stage 2/{total_stages}: rl mpo ===\n")
    _execute_rl(cfg, script_path=MPO_SCRIPT_PATH, spec=MPO_SPEC)

    typer.echo(f"\n=== Stage 3/{total_stages}: rl text ===\n")
    _execute_rl(cfg, script_path=TEXT_SCRIPT_PATH, spec=TEXT_SPEC)

    if include_vision:
        typer.echo(f"\n=== Stage 4/{total_stages}: rl vision ===\n")
        _execute_rl(cfg, script_path=VISION_SCRIPT_PATH, spec=VISION_SPEC)
    else:
        typer.echo(
            "\n=== Stage 4 (rl vision) skipped: launcher is stubbed. ===\n"
            "Final usable artifact: omni3-rl-text-model:latest.\n"
            "Once the upstream vision RL launcher lands, bump stage3_vision_rl/train.py's "
            "runspec footprint back to production (nodes=16, gpus_per_node=8) or pass "
            "pipe.force_vision=true to override today.\n"
        )


def pipe(ctx: typer.Context) -> None:
    """Run the Omni SFT -> MPO -> text RL -> vision RL pipeline.

    `--dry-run` previews the planned stage order anywhere. Actual execution
    requires `--run` so each stage can finish before the next consumes its
    checkpoint artifact.
    """
    cfg = parse_recipe_config(ctx)
    _execute_pipe(cfg)
