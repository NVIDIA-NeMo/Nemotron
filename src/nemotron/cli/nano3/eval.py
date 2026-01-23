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

"""Eval command implementation for nano3 recipe (stage3)."""

from __future__ import annotations

import typer

from nemotron.kit.cli.evaluator import evaluator

CONFIG_DIR = "src/nemotron/recipes/nano3/stage3_eval/config"


@evaluator(
    name="nano3/eval",
    config_dir=CONFIG_DIR,
    default_config="default",
)
def eval(ctx: typer.Context) -> None:
    """Run evaluation with NeMo-Evaluator (stage3).

    Evaluates the trained model using nemo-evaluator-launcher.
    By default, evaluates the RL stage output (run.model=rl:latest).

    Examples:
        # Eval on cluster (loads env.toml profile)
        nemotron nano3 eval --run MY-CLUSTER

        # Override model artifact
        nemotron nano3 eval --run MY-CLUSTER run.model=sft:v2

        # Filter specific tasks
        nemotron nano3 eval --run MY-CLUSTER -t adlr_mmlu -t hellaswag

        # Dry run (show resolved config without executing)
        nemotron nano3 eval --run MY-CLUSTER --dry-run

        # Local execution
        nemotron nano3 eval execution.type=local
    """
    ...
