#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super3/data/prep/rl/swe2"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker.
# Install the nemotron package: uv sync --reinstall-package nemotron.
# """
#
# [tool.runspec.run]
# launch = "python"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/data_prep"
# default = "default"
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

"""Data preparation for SWE Stage 2 (SWE-bench with Apptainer).

Splits the local swe2.jsonl file into train/val (last 100 rows → val).
No HF placeholder resolution needed — data is already complete.

CLI:
    nemotron super3 data prep rl swe2
    nemotron super3 data prep rl swe2 sample=100

Direct usage:
    python data_prep.py
    python data_prep.py sample=100 force=true
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.super3.stage2_rl._data_prep_base import main as _main

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "default.yaml"

# No Ray needed for direct split
RAY = False


def main(cfg=None):
    return _main(
        default_config=DEFAULT_CONFIG_PATH,
        resolve_hf_placeholders=False,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
