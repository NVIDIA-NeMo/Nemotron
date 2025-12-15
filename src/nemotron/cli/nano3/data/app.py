"""Data command group for nano3."""

from __future__ import annotations

import typer

from nemotron.cli.nano3.data.import_ import import_app
from nemotron.cli.nano3.data.prep import prep_app

# Create data app
data_app = typer.Typer(
    name="data",
    help="Data curation and preparation commands",
    no_args_is_help=True,
)

# Register prep subgroup
data_app.add_typer(prep_app, name="prep")

# Register import subgroup
data_app.add_typer(import_app, name="import")
