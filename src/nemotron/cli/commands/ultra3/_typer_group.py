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

"""Ultra3 Typer group.

Contains the ultra3 command group with subcommands for training stages.
"""

from __future__ import annotations

from nemotron.cli.commands.ultra3.pretrain import META as PRETRAIN_META, pretrain
from nemotron.cli.commands.ultra3.sft import META as SFT_META, sft

try:
    from nemotron.cli.commands.ultra3.data import data_app

    _has_data_app = True
except ImportError:
    _has_data_app = False
from nemo_runspec.recipe_typer import RecipeTyper

# Create ultra3 app using RecipeTyper
ultra3_app = RecipeTyper(
    name="ultra3",
    help="Ultra3 training recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# =============================================================================
# Register Infrastructure + Training Commands
#
# Each training command exports a META object with config_dir,
# input/output_artifacts. Execution logic stays visible in each command module.
# =============================================================================

# Container builds moved to the shared, Slurm-explicit `nemotron kit slurm build`
# command (see src/nemotron/cli/kit/slurm/). The recipe folder keeps only its
# per-stage Dockerfile; build policy is no longer duplicated per recipe.

ultra3_app.add_recipe_command(pretrain, meta=PRETRAIN_META, rich_help_panel="Training Stages")
ultra3_app.add_recipe_command(sft, meta=SFT_META, rich_help_panel="Training Stages")

# Register data curation/preparation subgroup
if _has_data_app:
    ultra3_app.add_typer(data_app, name="data")
