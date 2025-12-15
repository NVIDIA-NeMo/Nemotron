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

"""SFT command implementation.

This module defines the `sft` command for the nano3 recipe.
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe

# Paths relative to repository root
SCRIPT_PATH = "src/nemotron/recipes/nano3/stage1_sft/train.py"
CONFIG_DIR = "src/nemotron/recipes/nano3/stage1_sft/config"


@recipe(
    name="nano3/sft",
    script_path=SCRIPT_PATH,
    config_dir=CONFIG_DIR,
    default_config="default",
    packager="self_contained",
)
def sft(ctx: typer.Context) -> None:
    """Run supervised fine-tuning with Megatron-Bridge (stage1).

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., train.train_iters=5000)

    Examples:
        nemotron nano3 sft -c test                       # local execution
        nemotron nano3 sft --config test --run dlw       # nemo-run attached
        nemotron nano3 sft -c test -r dlw train.train_iters=5000
        nemotron nano3 sft -c test --dry-run             # preview config
        nemotron nano3 sft -c test --batch dlw --mock    # detached + passthrough
    """
    ...
