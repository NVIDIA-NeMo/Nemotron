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

"""MOPD sub-stage command (RL stage 3).

Multi-teacher on-policy distillation. Phase 1 (default) distills the agentic,
reasoning, RLHF-1 and vision teachers into the RLVR student; phase 2
(`-c phase2`) distills the RLHF-2, vision, vision-reasoning and video teachers
into the phase-1 student and produces the final checkpoint. `-c single_stage`
is NeMo-RL's single-stage MOPD: phase 1 without the agentic teacher.

Reuses the Ray execution logic from super3's RL backend.
"""

from __future__ import annotations

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.cli.commands.super3.rl._base import _execute_rl

SCRIPT_PATH = "src/nemotron/recipes/super35/stage2_rl/stage3_mopd/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={
        "model": "Student checkpoint (RLVR for phase 1, MOPD phase 1 for phase 2)",
        "data": "MOPD prompt blend (NeMo-Gym JSONL, routed per teacher)",
        "teacher_*": "Specialist teacher checkpoints (from rl teacher <name>)",
    },
    output_artifacts={"model": "MOPD-distilled student checkpoint"},
)


def mopd(ctx: typer.Context) -> None:
    """Run multi-teacher on-policy distillation (RL stage 3).

    Teacher checkpoints come from the config's run.teacher_* artifacts.
    """
    cfg = parse_recipe_config(ctx)
    _execute_rl(cfg, script_path=SCRIPT_PATH, spec=SPEC)
