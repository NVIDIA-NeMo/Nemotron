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

"""Data prep command group for nano3."""

from __future__ import annotations

import typer

from nemotron.cli.nano3.data.prep.pretrain import pretrain
from nemotron.cli.nano3.data.prep.pretrain_xenna import pretrain_xenna
from nemotron.cli.nano3.data.prep.rl import rl
from nemotron.cli.nano3.data.prep.rl_xenna import rl_xenna
from nemotron.cli.nano3.data.prep.sft import sft
from nemotron.cli.nano3.data.prep.sft_xenna import sft_xenna
from nemotron.cli.nano3.help import make_recipe_command

# Create prep app
prep_app = typer.Typer(
    name="prep",
    help="Prepare data for training stages",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register commands with custom help and allow_extra_args for dotlist overrides
prep_app.command(
    name="pretrain",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    cls=make_recipe_command(
        config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config/data_prep",
    ),
)(pretrain)

prep_app.command(
    name="pretrain-xenna",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    cls=make_recipe_command(
        config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config/data_prep",
    ),
)(pretrain_xenna)

prep_app.command(
    name="sft",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    cls=make_recipe_command(
        config_dir="src/nemotron/recipes/nano3/stage1_sft/config/data_prep",
    ),
)(sft)

prep_app.command(
    name="sft-xenna",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    cls=make_recipe_command(
        config_dir="src/nemotron/recipes/nano3/stage1_sft/config/data_prep",
    ),
)(sft_xenna)

prep_app.command(
    name="rl",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    cls=make_recipe_command(
        config_dir="src/nemotron/recipes/nano3/stage2_rl/config/data_prep",
    ),
)(rl)

prep_app.command(
    name="rl-xenna",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    cls=make_recipe_command(
        config_dir="src/nemotron/recipes/nano3/stage2_rl/config/data_prep",
    ),
)(rl_xenna)
