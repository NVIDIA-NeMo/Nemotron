"""Nano3 Typer group.

Contains the nano3 command group with subcommands for training stages.
"""

from __future__ import annotations

import typer

from nemotron.cli.nano3.data import data_app
from nemotron.cli.nano3.model import model_app
from nemotron.cli.nano3.pretrain import pretrain
from nemotron.cli.nano3.rl import rl
from nemotron.cli.nano3.sft import sft

# Create nano3 app
nano3_app = typer.Typer(
    name="nano3",
    help="Nano3 training recipe",
    no_args_is_help=True,
)

# Register data subgroup
nano3_app.add_typer(data_app, name="data")

# Register model subgroup
nano3_app.add_typer(model_app, name="model")

# Register commands
nano3_app.command(
    name="pretrain",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)(pretrain)

nano3_app.command(
    name="sft",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)(sft)

nano3_app.command(
    name="rl",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)(rl)
