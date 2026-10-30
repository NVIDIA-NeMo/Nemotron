# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SWE-RL stage 2 data prep command (super3 stage2_swe2 sub-stage)."""

from __future__ import annotations

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

from nemotron.cli.commands.super3.data.prep.rl._base import _execute_data_prep_rl

SCRIPT_PATH = "src/nemotron/recipes/super3/stage2_rl/stage2_swe2/data_prep.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "SWE-bench prompt JSONL (swe2.jsonl)"},
    output_artifacts={"data": "Split train/val JSONL"},
)


def swe2(ctx: typer.Context) -> None:
    """Prepare data for SWE-RL stage 2 (SWE-bench): split swe2.jsonl into train/val."""
    cfg = parse_recipe_config(ctx)
    _execute_data_prep_rl(cfg, script_path=SCRIPT_PATH, spec=SPEC)
