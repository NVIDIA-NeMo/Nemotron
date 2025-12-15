#!/usr/bin/env python3
"""Nemotron CLI - Main entry point.

Usage:
    nemotron nano3 pretrain -c test                       # local execution
    nemotron nano3 pretrain --config test --run dlw       # nemo-run attached
    nemotron nano3 pretrain -c test -r dlw train.train_iters=5000
    nemotron nano3 pretrain -c test --dry-run             # preview config
"""

from __future__ import annotations

from typing import Optional

import typer

from nemotron.kit.cli.globals import GlobalContext, global_callback

# Create root app with global callback
app = typer.Typer(
    name="nemotron",
    help="Nemotron CLI - Reproducible training recipes",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(
        None,
        "-c",
        "--config",
        help="Config name (looks in recipe's config/ dir) or path",
    ),
    run: Optional[str] = typer.Option(
        None,
        "-r",
        "--run",
        help="Execute attached via nemo-run with specified env profile",
    ),
    batch: Optional[str] = typer.Option(
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
    """Nemotron CLI - Reproducible training recipes."""
    # Delegate to global_callback
    global_callback(ctx, config, run, batch, dry_run, stage)


# Import and register recipe groups
def _register_groups() -> None:
    """Register all recipe groups with the main app."""
    from nemotron.cli.kit import kit_app
    from nemotron.cli.nano3 import nano3_app

    app.add_typer(nano3_app, name="nano3")
    app.add_typer(kit_app, name="kit")


# Register groups on import
_register_groups()


def main() -> None:
    """Entry point for the nemotron CLI."""
    app()


if __name__ == "__main__":
    main()
