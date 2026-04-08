#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nemotron/cpt-train"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron"
# setup = "NeMo, Megatron-Bridge, and training dependencies are pre-installed."
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 2
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

"""CPT script for Nemotron: tokenize data then run distributed pre-training."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    """Entry point for CPT training."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
        config = apply_hydra_overrides(config, cli_overrides)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Data preparation: tokenize and pack into Megatron bin/idx format
    from nemotron.customization_recipes.data_prep import prepare_cpt_data

    data_result = prepare_cpt_data(config)
    logger.info("CPT data prepared: %s", data_result)

    # Training: launch Megatron-Bridge CPT via nemo_automodel
    try:
        from nemo_automodel import AutoModel

        model = AutoModel.from_config(config)
        model.train()
    except ImportError:
        logger.warning(
            "nemo_automodel not available; skipping training launch. "
            "Run training manually via torchrun with the prepared data at %s",
            data_result.get("blend_path", data_result.get("output_dir")),
        )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
