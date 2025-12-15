"""Model Typer group for nano3."""

from __future__ import annotations

import typer

from nemotron.cli.nano3.model.eval import eval_cmd
from nemotron.cli.nano3.model.import_ import import_app

# Create model app
model_app = typer.Typer(
    name="model",
    help="Model evaluation and import commands",
    no_args_is_help=True,
)

# Register import subgroup
model_app.add_typer(import_app, name="import")

# Register eval command
model_app.command(name="eval")(eval_cmd)
