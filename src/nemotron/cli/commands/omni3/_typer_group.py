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

"""Omni3 Typer group."""

from __future__ import annotations

from nemotron.cli.commands.omni3.build import build
from nemotron.cli.commands.omni3.data import data_app
from nemotron.cli.commands.omni3.model import model_app
from nemotron.cli.commands.omni3.rl import rl_app
from nemotron.cli.commands.omni3.sft import META as SFT_META, sft
from nemo_runspec.recipe_typer import RecipeTyper

omni3_app = RecipeTyper(
    name="omni3",
    help="Omni3 training recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

omni3_app.command(
    "build",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    rich_help_panel="Infrastructure",
)(build)

omni3_app.add_recipe_command(sft, meta=SFT_META, rich_help_panel="Training Stages")
omni3_app.add_typer(data_app, name="data")
omni3_app.add_typer(model_app, name="model")
omni3_app.add_typer(rl_app, name="rl")
