#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super35/rl/mopd"
# image = "/home/${oc.env:USER}/.cache/nemotron/containers/super35-rl.sqsh"
# setup = "Build the Super3.5 RL container with `nemotron kit slurm build <profile> --recipe super35 --stage rl`."
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run python {script} --config {config}"
# workdir = "/opt/nemo-rl"
#
# [tool.runspec.config]
# dir = "./config"
# default = "phase1"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 128
# gpus_per_node = 4
# ///

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Multi-teacher on-policy distillation (MOPD) for Nemotron 3.5 Super VL (RL stage 3).

MOPD is GRPO with an ``on_policy_distillation`` block: every student token
gets a reverse-KL advantage from the teacher that owns the prompt's
environment, and environments no specialist owns are routed to a frozen
copy of the student's starting checkpoint (the self-anchor).

- ``phase1`` (default): RLVR student; agentic, reasoning, RLHF teacher 1 and
  vision tool-use teachers. Selected checkpoint: step 30.
- ``phase2``: phase-1 student; RLHF teacher 2, vision tool-use,
  vision-reasoning and video teachers, proximal-teacher alpha 0.2.
  Selected (final) checkpoint: step 30.

CLI:
    nemotron super35 rl mopd [-c phase2] --run <profile>

Execution logic: src/nemotron/cli/commands/super35/rl/mopd.py
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super35.stage2_rl._nemo_rl import run

# Flag to indicate this module requires Ray execution
RAY = True

if __name__ == "__main__":
    run("mopd", default_config=str(Path(__file__).parent / "config" / "phase1.yaml"))
