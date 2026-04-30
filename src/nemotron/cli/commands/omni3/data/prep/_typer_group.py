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

"""Data prep command group for omni3."""

from __future__ import annotations

from nemotron.cli.commands.omni3.data.prep.rl import META as RL_META, rl
from nemotron.cli.commands.omni3.data.prep.sft import META as SFT_META, sft
from nemo_runspec.recipe_typer import RecipeTyper

prep_app = RecipeTyper(
    name="prep",
    help="Prepare data for omni3 training stages",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

prep_app.add_recipe_command(sft, meta=SFT_META)
prep_app.add_recipe_command(rl, meta=RL_META)
