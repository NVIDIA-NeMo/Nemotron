#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super35/rl/teacher/vision_reasoning"
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
# default = "stage1"
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

"""Vision-reasoning teacher for Nemotron 3.5 Super VL (RL stage 2).

Two GRPO stages on chart / document / OCR reasoning with a frozen vision tower:

- ``stage1``: RLVR step 120 -> step 114.
- ``stage2``: native policy-weights warm start from stage 1 (fresh optimizer
  and step counter), structured-answer verifier, router replay -> step 20.
  Stage 2 is distilled in MOPD phase 2.

CLI:
    nemotron super35 rl teacher vision-reasoning -c stage1 --run <profile>
    nemotron super35 rl teacher vision-reasoning -c stage2 --run <profile>

Execution logic: src/nemotron/cli/commands/super35/rl/teachers.py
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super35.stage2_rl._nemo_rl import run

# Flag to indicate this module requires Ray execution
RAY = True

if __name__ == "__main__":
    run("teacher-vision-reasoning", default_config=str(Path(__file__).parent / "config" / "stage1.yaml"))
