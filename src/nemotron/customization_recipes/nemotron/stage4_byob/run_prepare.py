#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nemotron/byob-prepare"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron"
# setup = "NeMo Data Designer and BYOB dependencies are pre-installed."
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.config]
# dir = "./config"
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

"""BYOB seed dataset preparation (sample from source and dump for MCQ generation)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    """Entry point for BYOB seed data preparation."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
        config = apply_hydra_overrides(config, cli_overrides)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    from nemotron.customization_recipes.data_prep import prepare_byob_seed

    result = prepare_byob_seed(config)
    logger.info("BYOB seed prep complete. Output: %s", result)


if __name__ == "__main__":
    main()
