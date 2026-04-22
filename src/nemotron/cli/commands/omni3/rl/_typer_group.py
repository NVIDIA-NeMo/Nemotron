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

"""RL Typer subgroup for omni3."""

from __future__ import annotations

from nemotron.cli.commands.omni3.rl.mpo import META as MPO_META, mpo
from nemotron.cli.commands.omni3.rl.text import META as TEXT_META, text
from nemotron.cli.commands.omni3.rl.vision import META as VISION_META, vision
from nemo_runspec.recipe_typer import RecipeTyper

rl_app = RecipeTyper(
    name="rl",
    help="Reinforcement learning sub-stages (MPO, text-only RL, vision RL)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

rl_app.add_recipe_command(mpo, meta=MPO_META, rich_help_panel="RL Sub-Stages")
rl_app.add_recipe_command(text, meta=TEXT_META, rich_help_panel="RL Sub-Stages")
rl_app.add_recipe_command(vision, meta=VISION_META, rich_help_panel="RL Sub-Stages")
