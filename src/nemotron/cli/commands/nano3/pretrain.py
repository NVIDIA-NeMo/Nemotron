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

"""Pretrain command implementation.

This module defines the `pretrain` command for the nano3 recipe.
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe

# Paths relative to repository root
SCRIPT_PATH = "src/nemotron/recipes/nano3/stage0_pretrain/train.py"
CONFIG_DIR = "src/nemotron/recipes/nano3/stage0_pretrain/config"


@recipe(
    name="nano3/pretrain",
    script_path=SCRIPT_PATH,
    config_dir=CONFIG_DIR,
    default_config="default",
    packager="self_contained",
    artifacts={
        "data": {
            "default": "PretrainBlendsArtifact-default",
            "mappings": {"path": "recipe.per_split_data_args_path"},
        },
    },
)
def pretrain(ctx: typer.Context) -> None:
    """Run pretraining with Megatron-Bridge (stage0)."""
    ...
