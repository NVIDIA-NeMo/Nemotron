#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/rl/nemo_rl/rlhf"
# image = "nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super"
#
# [tool.runspec.run]
# launch = "ray"
# workdir = "/opt/nemo-rl"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 8
# ///

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

"""NeMo-RL RLHF step (GRPO with a learned reward / judge model).

The default config uses the generic NeMo-Gym GRPO runner for GenRM-style
comparison rewards. Configs with ``env.should_use_nemo_gym=false`` can still
fall back to the upstream generic GRPO example.
"""

from __future__ import annotations

from pathlib import Path

from nemotron.steps._runners.nemo_rl import exec_or_run_nemo_rl_grpo

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
UPSTREAM_SCRIPT = "/opt/nemo-rl/examples/run_grpo.py"


def main() -> None:
    exec_or_run_nemo_rl_grpo(
        default_config=DEFAULT_CONFIG,
        upstream_script=UPSTREAM_SCRIPT,
        description="NeMo-RL RLHF (GRPO + reward model) training step",
    )


if __name__ == "__main__":
    main()
