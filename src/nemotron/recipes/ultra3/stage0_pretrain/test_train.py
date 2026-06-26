#!/usr/bin/env python3
# /// script
# [tool.runspec]
# name = "ultra3/pretrain-test"
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "test"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 1
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

"""Integration-test pretrain script for Nemotron Ultra3.

Uses a *tiny* Ultra3/Nemotron-H architecture model (~7M params) that fits on a
single GPU while exercising all Ultra3-specific code paths:

- Hybrid Mamba + Attention layers  (pattern ``MEM*EME``)
- Mixture-of-Experts with latent routing  (``moe_latent_size``)
- Multi-Token Prediction  (``mtp_num_layers=2``)
- Shared expert  (``moe_shared_expert_intermediate_size``)

The full training pipeline (wandb, artifact resolution, lineage, checkpointing)
is identical to the production ``train.py`` — only the model is smaller and
precision is bf16 for portability.

Usage::

    torchrun --nproc_per_node=1 test_train.py
    torchrun --nproc_per_node=1 test_train.py --config config/test.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import (
    nemotron_3_ultra_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer
from omegaconf import DictConfig

from nemotron.kit.train_script import parse_config_and_overrides
from nemotron.recipes.ultra3.stage0_pretrain.train import run_pretrain
from nemotron.recipes.ultra3.tiny_model import make_tiny_ultra3_model

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recipe builder
# ---------------------------------------------------------------------------


def _tiny_recipe_builder(config: DictConfig) -> ConfigContainer:
    """Build a single-GPU Ultra3 pretrain config with the tiny provider."""
    # Start from the public production recipe (gets optimizer, DDP, dataset, etc.).
    # Unlike Super3, the Ultra3 recipe does not accept data kwargs; mock data is
    # fine for this smoke test because the recipe leaves cfg.dataset.blend=None.
    cfg = nemotron_3_ultra_pretrain_config()

    # Swap in the tiny model with single-node parallelism.
    cfg.model = make_tiny_ultra3_model()

    # Disable TP comm overlap (meaningless at TP=1).
    cfg.comm_overlap = None

    # Use standard bf16 precision (production uses nvfp4 which needs larger models).
    cfg.mixed_precision = "bf16_mixed"

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "test.yaml"


def main() -> None:
    """Entry point for tiny Ultra3 integration-test pretraining."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_pretrain(config_path, _tiny_recipe_builder, cli_overrides, tags=["test"])


if __name__ == "__main__":
    main()
