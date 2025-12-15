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

"""Global CLI options and context management.

Provides the GlobalContext dataclass and typer callback for handling
global options like --config, --run, --batch, --dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import typer


@dataclass
class GlobalContext:
    """Global CLI options shared across all commands.

    Attributes:
        config: Config name or path (-c/--config)
        run: Env profile name for attached execution (-r/--run)
        batch: Env profile name for detached execution (-b/--batch)
        dry_run: If True, print config and exit (-d/--dry-run)
        dotlist: Hydra-style dotlist overrides (key.sub=value)
        passthrough: Other args to pass to script (--mock, etc.)
    """

    config: str | None = None
    run: str | None = None
    batch: str | None = None
    dry_run: bool = False
    stage: bool = False
    dotlist: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        """Get execution mode: 'run', 'batch', or 'local'."""
        if self.run:
            return "run"
        elif self.batch:
            return "batch"
        return "local"

    @property
    def profile(self) -> str | None:
        """Get the env profile name (from --run or --batch)."""
        return self.run or self.batch


def split_unknown_args(
    args: list[str],
    global_ctx: GlobalContext | None = None,
) -> tuple[list[str], list[str], GlobalContext]:
    """Split unknown args into dotlist overrides and passthrough args.

    Also extracts any global options that appear in unknown args
    (for when options come after subcommand).

    Dotlist overrides are tokens like `key.sub=value` (no leading `-`).
    Passthrough args are everything else (e.g., `--mock`, `--some-flag`).

    Args:
        args: List of unknown arguments from ctx.args
        global_ctx: Global context to update with extracted options

    Returns:
        Tuple of (dotlist_overrides, passthrough_args, updated_global_ctx)
    """
    if global_ctx is None:
        global_ctx = GlobalContext()

    dotlist = []
    passthrough = []

    # Known global options to extract from args
    global_opts = {
        "-c": "config",
        "--config": "config",
        "-r": "run",
        "--run": "run",
        "-b": "batch",
        "--batch": "batch",
        "-d": "dry_run",
        "--dry-run": "dry_run",
        "--stage": "stage",
    }

    i = 0
    while i < len(args):
        arg = args[i]

        # Check if it's a global option
        if arg in global_opts:
            attr = global_opts[arg]
            if attr in ("dry_run", "stage"):
                # Boolean flag
                setattr(global_ctx, attr, True)
                i += 1
            else:
                # Option with value
                if i + 1 < len(args):
                    setattr(global_ctx, attr, args[i + 1])
                    i += 2
                else:
                    i += 1
        elif "=" in arg and not arg.startswith("-"):
            # Dotlist override
            dotlist.append(arg)
            i += 1
        else:
            # Passthrough arg
            passthrough.append(arg)
            i += 1

    return dotlist, passthrough, global_ctx


def global_callback(
    ctx: typer.Context,
    config: str | None = typer.Option(
        None,
        "-c",
        "--config",
        help="Config name (looks in recipe's config/ dir) or path",
    ),
    run: str | None = typer.Option(
        None,
        "-r",
        "--run",
        help="Execute attached via nemo-run with specified env profile",
    ),
    batch: str | None = typer.Option(
        None,
        "-b",
        "--batch",
        help="Execute detached via nemo-run with specified env profile",
    ),
    dry_run: bool = typer.Option(
        False,
        "-d",
        "--dry-run",
        help="Print compiled config as rich table (no execution)",
    ),
    stage: bool = typer.Option(
        False,
        "--stage",
        help="Stage script + config to remote cluster for interactive debugging",
    ),
) -> None:
    """Global callback that captures options available to all commands.

    This callback is invoked before any subcommand and stores the global
    options in ctx.obj for access by leaf commands.
    """
    # Validate mutual exclusivity (only if both are set at this point)
    # Note: additional options may be extracted later by split_unknown_args
    if run and batch:
        typer.echo("Error: --run and --batch cannot both be set", err=True)
        raise typer.Exit(1)

    # Store in context
    ctx.ensure_object(GlobalContext)
    ctx.obj.config = config
    ctx.obj.run = run
    ctx.obj.batch = batch
    ctx.obj.dry_run = dry_run
    ctx.obj.stage = stage

    # Unknown args will be populated by the leaf command
    # since they need allow_extra_args=True on the command itself
