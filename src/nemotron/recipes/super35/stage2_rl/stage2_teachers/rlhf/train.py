#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super35/rl/teacher/rlhf"
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
# default = "teacher1_stage1"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 32
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

"""RLHF alignment teachers for Nemotron 3.5 Super VL (RL stage 2).

Two GenRM-driven alignment teachers, each trained in two substages against a
fixed Gemma-4-31B-it baseline response per prompt (rubric-mean scoring):

- ``teacher1_stage1``: SFT -> step 120, Blend 1, reasoning-length penalty 0.12.
- ``teacher1_stage2``: full-state resume of stage 1 at step 120 -> step 500
  (penalty 0.06). Teacher 1 is distilled in MOPD phase 1.
- ``teacher2_stage1``: RLVR -> step 200, Blend 2, penalty 0.06.
- ``teacher2_stage2``: weights-only warm start from teacher 2 stage 1 step 200,
  Blend 3 + abstention reward 0.2, 120 steps. Teacher 2 is distilled in MOPD
  phase 2.

The GenRM (and, for teacher 2 stage 2, the abstention judge) are external
vLLM servers outside the Ray cluster; see README.md.

CLI:
    nemotron super35 rl teacher rlhf -c <variant> --run <profile>

Execution logic: src/nemotron/cli/commands/super35/rl/teachers.py
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super35.stage2_rl._nemo_rl import run

# Flag to indicate this module requires Ray execution
RAY = True

if __name__ == "__main__":
    run("teacher-rlhf", default_config=str(Path(__file__).parent / "config" / "teacher1_stage1.yaml"))
