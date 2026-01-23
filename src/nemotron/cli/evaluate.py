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

"""Top-level evaluate command.

Provides a generic `nemotron evaluate` command with pre-built configs for
common evaluation scenarios. Unlike recipe-specific commands (nano3/eval),
this command has no default config and requires explicit config selection.
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.evaluator import evaluator

# Config directory for generic evaluator configs
CONFIG_DIR = "src/nemotron/recipes/evaluator/config"


@evaluator(
    name="evaluate",
    config_dir=CONFIG_DIR,
    default_config="default",
    require_explicit_config=True,
)
def evaluate(ctx: typer.Context) -> None:
    """Run model evaluation with nemo-evaluator.

    Generic evaluation command with pre-built configs for common models.
    For recipe-specific evaluation with artifact resolution, use `nemotron nano3 eval`.

    Available configs:
        nemotron-3-nano-nemo-ray  NeMo Framework Ray deployment for Nemotron-3-Nano

    Examples:
        # Evaluate Nemotron-3-Nano with NeMo Ray deployment
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER

        # Override checkpoint path
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER \\
            deployment.checkpoint_path=/path/to/checkpoint

        # Filter specific tasks
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER -t adlr_mmlu

        # Dry run (preview config)
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER --dry-run

        # Use custom config file
        nemotron evaluate -c /path/to/custom.yaml --run MY-CLUSTER
    """
    ...
