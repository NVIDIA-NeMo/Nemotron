#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super3/data/prep/rl/rlvr"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker.
# Install the nemotron package with xenna extras: uv sync --reinstall-package nemotron.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/data_prep"
# default = "rlvr1"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
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

"""Data preparation for RLVR sub-stage.

Processes local JSONL files (rlvr1/rlvr2/rlvr3) with HF placeholder resolution
for DAPO-Math-17k and Skywork-OR1-RL-Data entries, then splits into train/val.

Uses the xenna pipeline (Plan → Download → JsonlShard) for placeholder resolution,
followed by driver-side train/val splitting (last 100 rows → val).

CLI:
    nemotron super3 data prep rl rlvr -c rlvr1
    nemotron super3 data prep rl rlvr -c rlvr2
    nemotron super3 data prep rl rlvr -c rlvr3

Direct usage:
    python data_prep.py
    python data_prep.py --config rlvr2.yaml
    python data_prep.py sample=100 force=true
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super3.stage2_rl._data_prep_base import main as _main

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "rlvr1.yaml"

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


def main(cfg=None):
    return _main(
        default_config=DEFAULT_CONFIG_PATH,
        resolve_hf_placeholders=True,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
