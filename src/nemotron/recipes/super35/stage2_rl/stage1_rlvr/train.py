#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super35/rl/rlvr"
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
# nodes = 144
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

"""Text-only RLVR for Nemotron 3.5 Super VL (RL stage 1).

Joint RLVR + RLHF over the broad NeMo-Gym environment blend, starting
from the SFT checkpoint. Step 120 of this stage is the student for MOPD
phase 1 and the starting point for every specialist teacher.

CLI:
    nemotron super35 rl rlvr --run <profile>

Execution logic: src/nemotron/cli/commands/super35/rl/rlvr.py
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super35.stage2_rl._nemo_rl import run

# Flag to indicate this module requires Ray execution
RAY = True

if __name__ == "__main__":
    run("rlvr", default_config=str(Path(__file__).parent / "config" / "default.yaml"))
