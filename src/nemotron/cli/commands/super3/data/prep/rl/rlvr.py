# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RLVR data prep command (super3 stage1_rlvr sub-stage)."""

from __future__ import annotations

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

from nemotron.cli.commands.super3.data.prep.rl._base import _execute_data_prep_rl

SCRIPT_PATH = "src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/data_prep.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "Raw RLVR prompt JSONL (DAPO-Math-17k, Skywork-OR1-RL-Data)"},
    output_artifacts={"data": "Split train/val JSONL"},
)


def rlvr(ctx: typer.Context) -> None:
    """Prepare data for RLVR (stage1.1/1.2/1.3 — math/code/STEM prompts)."""
    cfg = parse_recipe_config(ctx)
    _execute_data_prep_rl(cfg, script_path=SCRIPT_PATH, spec=SPEC)
