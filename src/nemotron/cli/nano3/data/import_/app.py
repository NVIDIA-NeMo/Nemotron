"""Data import Typer group for nano3."""

from __future__ import annotations

import typer

from nemotron.cli.nano3.data.import_.pretrain import pretrain
from nemotron.cli.nano3.data.import_.rl import rl
from nemotron.cli.nano3.data.import_.sft import sft

# Create import app
import_app = typer.Typer(
    name="import",
    help="Import data as W&B artifacts",
    no_args_is_help=True,
)

# Register commands
import_app.command(name="pretrain")(pretrain)
import_app.command(name="sft")(sft)
import_app.command(name="rl")(rl)
