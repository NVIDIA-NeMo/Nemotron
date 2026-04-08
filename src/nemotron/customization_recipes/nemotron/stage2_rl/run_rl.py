#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nemotron/rl-train"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron"
# setup = "NeMo-RL (DPO/GRPO) and Ray dependencies are pre-installed."
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
# nodes = 4
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

"""RL training (DPO / GRPO) for Nemotron models via NeMo-RL + Ray."""

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
    """Entry point for RL training."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
        config = apply_hydra_overrides(config, cli_overrides)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Dispatch RL training based on training_type (dpo or grpo)
    from omegaconf import OmegaConf

    training_type = OmegaConf.select(config, "training_type", default="grpo")
    logger.info("RL training type: %s", training_type)

    if training_type == "grpo":
        try:
            from nemo_rl.sft.grpo import GRPOTrainer

            trainer = GRPOTrainer(config)
            trainer.fit()
        except ImportError:
            logger.error(
                "nemo-rl is required for GRPO training. "
                "Install with: pip install nemo-rl  (or use the NeMo-RL container)"
            )
            sys.exit(1)
    elif training_type == "dpo":
        try:
            from nemo_rl.sft.dpo import DPOTrainer

            trainer = DPOTrainer(config)
            trainer.fit()
        except ImportError:
            logger.error(
                "nemo-rl is required for DPO training. "
                "Install with: pip install nemo-rl  (or use the NeMo container)"
            )
            sys.exit(1)
    else:
        logger.error("Unknown training_type '%s'. Use 'dpo' or 'grpo'.", training_type)
        sys.exit(1)


if __name__ == "__main__":
    main()
