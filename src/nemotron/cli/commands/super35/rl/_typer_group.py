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

"""RL Typer subgroup for super35.

Mirrors the Super 3.5 reference run:
- rlvr: text-only RLVR from SFT (stage 1)
- teacher <name>: specialist teachers branched from RLVR (stage 2)
- mopd: multi-teacher on-policy distillation, phases 1 and 2 (stage 3)
"""

from __future__ import annotations

from nemo_runspec.recipe_typer import RecipeTyper
from nemotron.cli.commands.super35.rl.mopd import META as MOPD_META
from nemotron.cli.commands.super35.rl.mopd import mopd
from nemotron.cli.commands.super35.rl.rlvr import META as RLVR_META
from nemotron.cli.commands.super35.rl.rlvr import rlvr
from nemotron.cli.commands.super35.rl.teachers import teacher_app

rl_app = RecipeTyper(
    name="rl",
    help="Reinforcement learning sub-stages (RLVR, specialist teachers, MOPD)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

rl_app.add_recipe_command(rlvr, meta=RLVR_META, rich_help_panel="RL Sub-Stages")
rl_app.add_typer(teacher_app, name="teacher")
rl_app.add_recipe_command(mopd, meta=MOPD_META, rich_help_panel="RL Sub-Stages")
