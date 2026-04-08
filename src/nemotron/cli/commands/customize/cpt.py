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

"""CPT (Continued Pre-Training) command implementation.

Runs stage0 CPT training: tokenize data then run distributed pre-training
using Megatron-Bridge.

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

SCRIPT_PATH = "src/nemotron/customization_recipes/nemotron/stage0_cpt/run_cpt.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "CPT data (raw text or pre-tokenized)"},
    output_artifacts={"model": "Continued pre-trained model checkpoint"},
)


# =============================================================================
# CLI Entry Point
# =============================================================================


def cpt(ctx: typer.Context) -> None:
    """Run continued pre-training (stage0).

    Tokenizes data and runs distributed pre-training using Megatron-Bridge.
    The execution logic is in _execute.py - see execute_recipe()
    for nemo-run setup.
    """
    cfg = parse_recipe_config(ctx)
    execute_recipe(cfg, SPEC, SCRIPT_PATH)
