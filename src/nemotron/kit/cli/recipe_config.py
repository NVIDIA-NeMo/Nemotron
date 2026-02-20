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

"""RecipeConfig - parsed CLI options for recipe execution.

This module normalizes CLI options into a typed object, handling:
- Late global options (--run after subcommand)
- Dotlist overrides (key=value)
- Passthrough args (--mock, etc.)

Design principle: make what each invocation means explicit and testable.
The RecipeConfig maps 1:1 with what --help shows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import typer

from nemotron.kit.cli.globals import GlobalContext, split_unknown_args


@dataclass
class RecipeConfig:
    """Parsed CLI options for a recipe invocation.

    This dataclass normalizes the result of parsing CLI arguments,
    including late global options that appear after the subcommand.

    Attributes:
        ctx: Underlying GlobalContext with parsed options
        argv: Original command line arguments (for debugging)
        dotlist: List of key=value overrides
        passthrough: List of args to pass through to script

    Properties:
        mode: Execution mode ('run', 'batch', or 'local')
        attached: Whether running attached (--run) vs detached (--batch)
        profile: The env profile name (from --run or --batch)
        config: The config name or path (-c/--config)
        dry_run: Whether to preview config without execution
        stage: Whether to stage files for interactive debugging
        force_squash: Whether to force re-squash container
    """

    ctx: GlobalContext
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    dotlist: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        """Get execution mode: 'run', 'batch', or 'local'."""
        return self.ctx.mode

    @property
    def attached(self) -> bool:
        """Whether running attached (--run) vs detached (--batch)."""
        return self.ctx.run is not None

    @property
    def profile(self) -> str | None:
        """Get the env profile name (from --run or --batch)."""
        return self.ctx.profile

    @property
    def config(self) -> str | None:
        """Get the config name or path (-c/--config)."""
        return self.ctx.config

    @property
    def dry_run(self) -> bool:
        """Whether to preview config without execution."""
        return self.ctx.dry_run

    @property
    def stage(self) -> bool:
        """Whether to stage files for interactive debugging."""
        return self.ctx.stage

    @property
    def force_squash(self) -> bool:
        """Whether to force re-squash container."""
        return self.ctx.force_squash


def parse_recipe_config(typer_ctx: typer.Context) -> RecipeConfig:
    """Parse typer context into RecipeConfig.

    Handles late placement of global options (--run after subcommand)
    by calling split_unknown_args() to extract them from ctx.args.

    Args:
        typer_ctx: Typer context with obj containing GlobalContext
                   and args containing unknown arguments

    Returns:
        Fully parsed RecipeConfig

    Raises:
        typer.Exit: If validation fails (e.g., both --run and --batch set)
    """
    # Start from global callback state
    base: GlobalContext | None = typer_ctx.obj
    if base is None:
        base = GlobalContext()

    # Split ctx.args into dotlist overrides, passthrough args, and late globals
    # This handles cases like: nemotron nano3 pretrain --run dgx train.train_iters=1000
    dotlist, passthrough, updated_ctx = split_unknown_args(typer_ctx.args or [], base)

    # Store dotlist and passthrough in ctx for later use
    updated_ctx.dotlist = dotlist
    updated_ctx.passthrough = passthrough

    # Validate after late-global extraction
    if updated_ctx.run and updated_ctx.batch:
        typer.echo("Error: --run and --batch cannot both be set", err=True)
        raise typer.Exit(1)

    if updated_ctx.stage and not updated_ctx.profile:
        typer.echo(
            "Error: --stage requires --run or --batch to specify target cluster", err=True
        )
        raise typer.Exit(1)

    return RecipeConfig(
        ctx=updated_ctx,
        argv=sys.argv,
        dotlist=dotlist,
        passthrough=passthrough,
    )
