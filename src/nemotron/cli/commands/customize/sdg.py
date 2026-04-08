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

"""SDG (Synthetic Data Generation) command implementation.

Runs stage1 synthetic data generation via NeMo Data Designer.

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
# Recipe Metadata (read from [tool.runspec] in script)
# =============================================================================

SCRIPT_PATH = "src/nemotron/customization_recipes/nemotron/stage1_sft/run_sdg.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "Seed data or persona definitions for SDG"},
    output_artifacts={"data": "Synthetic conversation data (JSONL)"},
)


# =============================================================================
# CLI Entry Point
# =============================================================================


def sdg(ctx: typer.Context) -> None:
    """Run synthetic data generation (stage1 SDG).

    Generates synthetic conversation data using NeMo Data Designer.
    The execution logic is in _execute.py - see execute_recipe()
    for nemo-run setup.
    """
    cfg = parse_recipe_config(ctx)
    execute_recipe(cfg, SPEC, SCRIPT_PATH)
