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

"""Data-prep command implementation for customization recipes.

Supports both CPT data prep (stage1) and SFT data prep (stage2)
depending on the ``--mode`` flag. Default is CPT data prep.

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
# Recipe Metadata -- CPT data prep is the default; SFT data prep via --mode sft
# =============================================================================

CPT_SCRIPT_PATH = "src/nemotron/customization_recipes/nemotron/stage1_cpt/run_data_prep.py"
SFT_SCRIPT_PATH = "src/nemotron/customization_recipes/nemotron/stage2_sft/run_data_prep.py"

CPT_SPEC = parse_runspec(CPT_SCRIPT_PATH)
SFT_SPEC = parse_runspec(SFT_SCRIPT_PATH)

META = RecipeMeta(
    name=CPT_SPEC.name,
    script_path=CPT_SCRIPT_PATH,
    config_dir=str(CPT_SPEC.config_dir),
    default_config=CPT_SPEC.config.default,
    input_artifacts={"data": "Raw data sources (HuggingFace, JSONL, etc.)"},
    output_artifacts={"data": "Prepared JSONL / tokenized data for training"},
)


# =============================================================================
# CLI Entry Point
# =============================================================================


def data_prep(ctx: typer.Context) -> None:
    """Run data preparation for customization (stage1 CPT or stage2 SFT).

    By default runs CPT data-prep (acquire, filter, tokenize).
    Pass ``--mode sft`` to run SFT data-prep instead.
    """
    # Check for --mode flag in passthrough args
    sft_mode = False
    args = list(ctx.args) if ctx.args else []
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args) and args[idx + 1].lower() == "sft":
            sft_mode = True
            args.pop(idx + 1)
            args.pop(idx)
            ctx.args = args

    spec = SFT_SPEC if sft_mode else CPT_SPEC
    script_path = SFT_SCRIPT_PATH if sft_mode else CPT_SCRIPT_PATH

    cfg = parse_recipe_config(ctx)
    execute_recipe(cfg, spec, script_path)
