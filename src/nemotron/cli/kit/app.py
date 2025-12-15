"""Kit Typer group.

Contains utility commands for cluster setup and management.
"""

from __future__ import annotations

import typer

from nemotron.cli.kit.squash import squash

# Create kit app
kit_app = typer.Typer(
    name="kit",
    help="Utility commands for cluster setup and management",
    no_args_is_help=True,
)

# Register commands
kit_app.command(name="squash")(squash)
