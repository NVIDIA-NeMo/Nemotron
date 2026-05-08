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

"""Text-only RL sub-stage command for omni3 RL."""

from __future__ import annotations

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.cli.commands.omni3.rl._base import _execute_rl

SCRIPT_PATH = "src/nemotron/recipes/omni3/stage1_rl/stage2_text_rl/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={
        "model": "MPO-stage Omni checkpoint",
        "data": "Prepared text-only RL JSONL artifact",
    },
    output_artifacts={"model": "Text-only RL Omni checkpoint"},
)


def text(ctx: typer.Context) -> None:
    """Run the Omni text-only RL sub-stage."""
    cfg = parse_recipe_config(ctx)
    _execute_rl(cfg, script_path=SCRIPT_PATH, spec=SPEC)
