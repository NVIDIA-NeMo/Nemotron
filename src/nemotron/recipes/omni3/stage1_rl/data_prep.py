#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/data/prep/rl"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker.
# Install the nemotron package with `uv sync --reinstall-package nemotron`.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/data_prep"
# default = "mpo"
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

"""Single Omni RL data-prep entrypoint with mpo|text|vision config variants."""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.omni3.stage1_rl._data_prep_base import (
    Omni3RLDataPrepConfig,
    main as _main,
)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "data_prep" / "mpo.yaml"

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


def main(cfg: Omni3RLDataPrepConfig | None = None):
    """Entry point for omni3 RL data prep."""
    return _main(default_config=DEFAULT_CONFIG_PATH, cfg=cfg)


if __name__ == "__main__":
    main()
