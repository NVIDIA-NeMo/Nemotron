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

"""Specialist teacher sub-stage commands (RL stage 2).

Each teacher branches from the RLVR checkpoint (RLHF teacher 1 branches from
SFT) and is later distilled into the student by MOPD. Every teacher has its
own train.py, so the table below only maps a CLI name to that script.
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta, RecipeTyper
from nemotron.cli.commands.super3.rl._base import _execute_rl

_TEACHERS_DIR = "src/nemotron/recipes/super35/stage2_rl/stage2_teachers"

# (CLI name, recipe dir, one-line description)
TEACHERS = (
    ("reasoning", "reasoning", "Science/math/competitive-coding reasoning teacher (128K generation)."),
    ("vision", "vision", "Vision tool-use teacher (crop, zoom, rotate, ... as structured tool calls)."),
    ("video", "video", "Video reasoning and SA-V object-tracking teacher."),
    ("vision-reasoning", "vision_reasoning", "Chart/document/OCR vision-reasoning teacher (2 stages: -c stage1, -c stage2)."),
    ("rlhf", "rlhf", "Alignment (RLHF/GenRM) teachers 1 and 2 (-c teacher1_stage1 ... teacher2_stage2)."),
    ("agentic", "agentic", "End-to-end Terminal-Bench + BrowseComp agentic teacher."),
)


def _make_command(cli_name: str, recipe_dir: str, description: str) -> tuple[Callable, RecipeMeta]:
    script_path = f"{_TEACHERS_DIR}/{recipe_dir}/train.py"
    spec = parse_runspec(script_path)
    meta = RecipeMeta(
        name=spec.name,
        script_path=script_path,
        config_dir=str(spec.config_dir),
        default_config=spec.config.default,
        input_artifacts={
            "model": "Starting checkpoint (RLVR, or SFT for RLHF teacher 1)",
            "data": f"{cli_name} teacher data artifact (NeMo-Gym JSONL)",
        },
        output_artifacts={"model": f"{cli_name} teacher checkpoint"},
    )

    def command(ctx: typer.Context) -> None:
        cfg = parse_recipe_config(ctx)
        _execute_rl(cfg, script_path=script_path, spec=spec)

    command.__doc__ = f"Train the {description}"
    return command, meta


teacher_app = RecipeTyper(
    name="teacher",
    help="Specialist teachers distilled into the student by MOPD",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

for _cli_name, _recipe_dir, _description in TEACHERS:
    _command, _meta = _make_command(_cli_name, _recipe_dir, _description)
    teacher_app.add_recipe_command(_command, meta=_meta, name=_cli_name, rich_help_panel="Teachers")
