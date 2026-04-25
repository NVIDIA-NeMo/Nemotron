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

"""BYOB (Build Your Own Benchmark) command implementation.

Runs stage4 BYOB pipeline: seed data preparation, MCQ generation,
optional translation.

Design: LLM-Native Recipe Architecture
- Execution logic in _execute.py (shared across all customize commands)
- Fork _execute.py to change how jobs are submitted
"""

from __future__ import annotations

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

from nemotron.cli.commands.customize._execute import execute_recipe

# =============================================================================
# Recipe Metadata -- uses run_prepare.py as the default entry point
# (run_generate.py and run_translate.py are additional steps)
# =============================================================================

SCRIPT_PATH = "src/nemotron/customization_recipes/nemotron/stage4_byob/run_prepare.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "Seed data for BYOB benchmark generation"},
    output_artifacts={"data": "Generated MCQ benchmark dataset"},
)

# Step-to-script mapping for BYOB pipeline stages
_STEP_SCRIPTS = {
    "prepare": "src/nemotron/customization_recipes/nemotron/stage4_byob/run_prepare.py",
    "generate": "src/nemotron/customization_recipes/nemotron/stage4_byob/run_generate.py",
    "translate": "src/nemotron/customization_recipes/nemotron/stage4_byob/run_translate.py",
}


# =============================================================================
# CLI Entry Point
# =============================================================================


def byob(ctx: typer.Context) -> None:
    """Run Build Your Own Benchmark pipeline (stage4).

    Prepares seed data, generates MCQ benchmarks, and optionally translates.
    Pass ``--step generate`` or ``--step translate`` for subsequent steps.
    Default step is ``prepare``.
    """
    # Check for --step flag in passthrough args
    step = "prepare"
    args = list(ctx.args) if ctx.args else []
    if "--step" in args:
        idx = args.index("--step")
        if idx + 1 < len(args):
            step = args[idx + 1].lower()
            args.pop(idx + 1)
            args.pop(idx)
            ctx.args = args

    script_path = _STEP_SCRIPTS.get(step, _STEP_SCRIPTS["prepare"])
    spec = parse_runspec(script_path)

    cfg = parse_recipe_config(ctx)
    execute_recipe(cfg, spec, script_path)
