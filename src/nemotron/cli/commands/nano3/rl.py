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

"""RL command implementation.

This module defines the `rl` command for the nano3 recipe.
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe

# Paths relative to repository root
SCRIPT_PATH = "src/nemotron/recipes/nano3/stage2_rl/train.py"
CONFIG_DIR = "src/nemotron/recipes/nano3/stage2_rl/config"


@recipe(
    name="nano3/rl",
    script_path=SCRIPT_PATH,
    config_dir=CONFIG_DIR,
    default_config="tiny",
    torchrun=False,
    ray=True,
    packager="self_contained",
    workdir="/opt/nemo-rl",
    # Files (main.py, config.yaml) are rsynced to working dir, copy to nemo-rl workdir
    pre_ray_start_commands=[
        "cp main.py /opt/nemo-rl/",
        "cp config.yaml /opt/nemo-rl/",
    ],
    run_command="uv run python {script} --config {config}",
)
def rl(ctx: typer.Context) -> None:
    """Run reinforcement learning with NeMo-RL GRPO (stage2)."""
    ...
