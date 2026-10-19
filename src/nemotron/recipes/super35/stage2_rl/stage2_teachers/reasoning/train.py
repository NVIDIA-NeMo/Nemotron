#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super35/rl/teacher/reasoning"
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
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 64
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

"""Reasoning teacher for Nemotron 3.5 Super VL (RL stage 2).

GRPO from RLVR step 120 on mixed math, STEM and competitive-coding prompts
with 128K-token generations. Math/STEM answers are graded by a DeepSeek V4
Flash judge (hosted endpoint or your own vLLM server); code runs through the
competitive-coding verifier in a NeMo-Skills sandbox. Step 80 is distilled
into the student in MOPD phase 1.

CLI:
    nemotron super35 rl teacher reasoning --run <profile>

Execution logic: src/nemotron/cli/commands/super35/rl/teachers.py
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super35.stage2_rl._nemo_rl import run

# Flag to indicate this module requires Ray execution
RAY = True

if __name__ == "__main__":
    run("teacher-reasoning", default_config=str(Path(__file__).parent / "config" / "default.yaml"))
