# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Super35 Typer group.

Nemotron 3.5 Super VL (120B-A12B): Megatron-Bridge vision-language pretraining
and SFT, then the NeMo-RL reference run (RLVR -> specialist teachers ->
two-phase MOPD).

Design: LLM-Native Recipe Architecture
- Uses RecipeTyper for standardized command registration
- Each command module has visible execution logic
"""

from __future__ import annotations

from nemo_runspec.recipe_typer import RecipeTyper
from nemotron.cli.commands.super35.data import data_app
from nemotron.cli.commands.super35.eval import META as EVAL_META
from nemotron.cli.commands.super35.eval import eval as eval_cmd
from nemotron.cli.commands.super35.model import model_app
from nemotron.cli.commands.super35.pretrain import META as PRETRAIN_META
from nemotron.cli.commands.super35.pretrain import pretrain
from nemotron.cli.commands.super35.rl import rl_app
from nemotron.cli.commands.super35.sft import META as SFT_META
from nemotron.cli.commands.super35.sft import sft

super35_app = RecipeTyper(
    name="super35",
    help="Super35 (Nemotron 3.5 Super VL) training recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

super35_app.add_typer(data_app, name="data")
super35_app.add_typer(model_app, name="model")

super35_app.add_recipe_command(pretrain, meta=PRETRAIN_META, rich_help_panel="Training Stages")
super35_app.add_recipe_command(sft, meta=SFT_META, rich_help_panel="Training Stages")
super35_app.add_typer(rl_app, name="rl")
super35_app.add_recipe_command(eval_cmd, meta=EVAL_META, rich_help_panel="Evaluation")
