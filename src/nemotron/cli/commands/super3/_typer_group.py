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

"""Super3 Typer group.

Contains the super3 command group with subcommands for training stages.

Design: LLM-Native Recipe Architecture
- Uses RecipeTyper for standardized command registration
- Each command module has visible execution logic
"""

from __future__ import annotations

from nemotron.cli.commands.super3.data import data_app
from nemotron.cli.commands.super3.eval import META as EVAL_META
from nemotron.cli.commands.super3.eval import eval as eval_cmd
from nemotron.cli.commands.super3.model import model_app
from nemotron.cli.commands.super3.pipe import META as PIPE_META, pipe
from nemotron.cli.commands.super3.pretrain import META as PRETRAIN_META, pretrain
from nemotron.cli.commands.super3.rl import META as RL_META, rl
from nemotron.cli.commands.super3.sft import META as SFT_META, sft
from nemo_runspec.recipe_typer import RecipeTyper

# Create super3 app using RecipeTyper
super3_app = RecipeTyper(
    name="super3",
    help="Super3 training recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register data subgroup
super3_app.add_typer(data_app, name="data")

# Register model subgroup
super3_app.add_typer(model_app, name="model")

# =============================================================================
# Register Training Commands
#
# Each command exports a META object with config_dir, input/output_artifacts.
# Execution logic stays visible in each command module.
# =============================================================================

super3_app.add_recipe_command(pretrain, meta=PRETRAIN_META, rich_help_panel="Training Stages")
super3_app.add_recipe_command(sft, meta=SFT_META, rich_help_panel="Training Stages")
super3_app.add_recipe_command(rl, meta=RL_META, rich_help_panel="Training Stages")
super3_app.add_recipe_command(eval_cmd, meta=EVAL_META, rich_help_panel="Evaluation")
super3_app.add_recipe_command(pipe, meta=PIPE_META, rich_help_panel="Pipeline")
