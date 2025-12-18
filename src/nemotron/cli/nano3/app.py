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

"""Nano3 Typer group.

Contains the nano3 command group with subcommands for training stages.
"""

from __future__ import annotations

import typer

from nemotron.cli.nano3.data import data_app
from nemotron.cli.nano3.help import RecipeCommand, make_recipe_command
from nemotron.cli.nano3.model import model_app
from nemotron.cli.nano3.pretrain import pretrain
from nemotron.cli.nano3.rl import rl
from nemotron.cli.nano3.sft import sft


# Create nano3 app
nano3_app = typer.Typer(
    name="nano3",
    help="Nano3 training recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register data subgroup
nano3_app.add_typer(data_app, name="data")

# Register model subgroup
nano3_app.add_typer(model_app, name="model")

# Register commands with custom command class for enhanced help
# Pretrain has data artifact override
nano3_app.command(
    name="pretrain",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    rich_help_panel="Training Stages",
    cls=make_recipe_command(
        artifact_overrides={"data": "Pretrain data artifact (bin/idx blends)"},
        config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config",
    ),
)(pretrain)

# SFT has data and model artifact overrides
nano3_app.command(
    name="sft",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    rich_help_panel="Training Stages",
    cls=make_recipe_command(
        artifact_overrides={
            "model": "Base model checkpoint artifact",
            "data": "SFT data artifact (packed .npy)",
        },
        config_dir="src/nemotron/recipes/nano3/stage1_sft/config",
    ),
)(sft)

# RL has data and model artifact overrides
nano3_app.command(
    name="rl",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    rich_help_panel="Training Stages",
    cls=make_recipe_command(
        artifact_overrides={
            "model": "SFT model checkpoint artifact",
            "data": "RL data artifact (JSONL prompts)",
        },
        config_dir="src/nemotron/recipes/nano3/stage2_rl/config",
    ),
)(rl)
