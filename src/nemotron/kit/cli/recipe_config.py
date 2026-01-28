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

"""Recipe configuration dataclass and parser.

Provides RecipeConfig dataclass representing CLI options that maps 1:1 with
--help output, and parse_recipe_config() for extracting options from typer Context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import typer

from nemotron.kit.cli.globals import GlobalContext, split_unknown_args


@dataclass
class RecipeConfig:
    """Recipe command options. Mirrors --help output.

    Attributes:
        dry_run: If True, print config and exit
        run: Profile name for attached execution (--run)
        batch: Profile name for detached execution (--batch)
        stage: If True, stage files to remote without execution
        force_squash: If True, re-squash container image
        config: Config name or path (-c/--config)
        dotlist: Hydra-style dotlist overrides (key.sub=value)
        passthrough: Other args to pass to script (--mock, etc.)
    """

    dry_run: bool = False
    run: str | None = None
    batch: str | None = None
    stage: bool = False
    force_squash: bool = False
    config: str | None = None
    dotlist: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)

    @property
    def profile(self) -> str | None:
        """Get the env profile name (from --run or --batch)."""
        return self.run or self.batch

    @property
    def attached(self) -> bool:
        """Whether running in attached mode (--run vs --batch)."""
        return self.run is not None

    @property
    def mode(self) -> str:
        """Get execution mode: 'run', 'batch', or 'local'."""
        if self.run:
            return "run"
        elif self.batch:
            return "batch"
        return "local"


def parse_recipe_config(ctx: typer.Context) -> RecipeConfig:
    """Parse typer context into RecipeConfig.

    Extracts options from ctx.obj (GlobalContext) and ctx.args (unknown args),
    handling late global options that appear after the subcommand.

    Args:
        ctx: Typer context with global options in ctx.obj and unknown args in ctx.args

    Returns:
        RecipeConfig with all options populated
    """
    # Get global context (populated by global callback)
    global_ctx: GlobalContext = ctx.obj
    if global_ctx is None:
        global_ctx = GlobalContext()

    # Split unknown args into dotlist and passthrough
    # Also extract any global options that appear after the subcommand
    dotlist, passthrough, global_ctx = split_unknown_args(ctx.args or [], global_ctx)

    # Validate options after split_unknown_args has extracted all global options
    if global_ctx.run and global_ctx.batch:
        typer.echo("Error: --run and --batch cannot both be set", err=True)
        raise typer.Exit(1)

    if global_ctx.stage and not global_ctx.profile:
        typer.echo(
            "Error: --stage requires --run or --batch to specify target cluster", err=True
        )
        raise typer.Exit(1)

    return RecipeConfig(
        dry_run=global_ctx.dry_run,
        run=global_ctx.run,
        batch=global_ctx.batch,
        stage=global_ctx.stage,
        force_squash=global_ctx.force_squash,
        config=global_ctx.config,
        dotlist=dotlist,
        passthrough=passthrough,
    )
