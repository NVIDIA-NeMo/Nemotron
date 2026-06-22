# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RL data prep sub-stage Typer group for super3."""

from __future__ import annotations

from nemo_runspec.recipe_typer import RecipeTyper

from nemotron.cli.commands.super3.data.prep.rl.rlvr import META as RLVR_META, rlvr

rl_app = RecipeTyper(
    name="rl",
    help="RL data prep sub-stages (RLVR, SWE1, SWE2, RLHF)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

rl_app.add_recipe_command(rlvr, meta=RLVR_META, rich_help_panel="RL Sub-Stages")

# TODO: Add swe1, swe2, rlhf sub-stage data prep commands as needed.
