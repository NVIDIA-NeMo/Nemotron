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

"""SWE-RL Stage 2 (SWE-bench with Apptainer) sub-stage command.

Stage 2b of the RL pipeline — trains on SWE-bench tasks using
Apptainer-based sandbox containers.

Reuses the Ray execution logic from _base.py.
"""

from __future__ import annotations

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_typer import RecipeMeta
from nemo_runspec.recipe_config import parse_recipe_config
from nemotron.cli.commands.super3.rl._base import _execute_rl

SCRIPT_PATH = "src/nemotron/recipes/super3/stage2_rl/stage2_swe2/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={
        "model": "SWE-RL Stage 1 model checkpoint (from stage2_swe1)",
        "data": "SWE-bench data artifact",
    },
    output_artifacts={"model": "SWE-RL Stage 2 model checkpoint"},
)


def swe2(ctx: typer.Context) -> None:
    """Run SWE-RL Stage 2 (SWE-bench) training with Apptainer sandboxes.

    Trains on SWE-bench tasks using Apptainer-based sandbox containers
    with NeMo-RL GRPO.
    """
    cfg = parse_recipe_config(ctx)
    _execute_rl(cfg, script_path=SCRIPT_PATH, spec=SPEC)
