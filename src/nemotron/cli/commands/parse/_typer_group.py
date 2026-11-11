"""Nemotron Parse recipe registration."""

from nemo_runspec.recipe_typer import RecipeTyper
from nemotron.cli.commands.parse.commands import data, evaluate, metadata, rlvr, sft

parse_app = RecipeTyper(name="parse", help="Parse model adaptation and evaluation; dental data preparation example",
                        no_args_is_help=True)
for name, command, panel in (("data", data, "Data"), ("sft", sft, "Training"),
                              ("rlvr", rlvr, "Training"), ("eval", evaluate, "Evaluation")):
    parse_app.add_recipe_command(command, name=name, meta=metadata(name), rich_help_panel=panel)
