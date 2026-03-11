#!/usr/bin/env python3
# /// script
# [tool.runspec]
# name = "super3/sft-test"
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

"""Integration-test SFT script for Nemotron Super3.

Uses the same tiny Super3-architecture model as the pretrain integration test
(~7M params, single GPU) with full-parameter SFT (no LoRA) and packed sequences.

Exercises all Super3-specific code paths:

- Hybrid Mamba + Attention layers  (pattern ``MEM*EME``)
- Mixture-of-Experts with latent routing  (``moe_latent_size``)
- Multi-Token Prediction  (``mtp_num_layers=2``)
- Shared expert  (``moe_shared_expert_intermediate_size``)
- Packed-sequence finetuning with custom dataset builder

The full training pipeline (wandb, artifact resolution, lineage, checkpointing,
HF conversion) is identical to the production ``train.py`` — only the model is
smaller.

Usage::

    torchrun --nproc_per_node=1 test_train.py
    torchrun --nproc_per_node=1 test_train.py --config config/test.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_finetune_config,
)
from megatron.bridge.training.config import ConfigContainer
from omegaconf import DictConfig

from nemotron.kit.train_script import parse_config_and_overrides
from nemotron.recipes.super3.stage1_sft.train import run_finetune
from nemotron.recipes.super3.tiny_model import make_tiny_super3_model

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recipe builder
# ---------------------------------------------------------------------------

def _tiny_recipe_builder(config: DictConfig) -> ConfigContainer:  # noqa: ARG001
    """Build a single-GPU Super3 SFT config with the tiny provider.

    Uses full-parameter SFT (no LoRA) with packed sequences, matching the
    production Super3 SFT recipe but at tiny scale.
    """
    # Start from the public production finetune recipe
    cfg = nemotron_3_super_finetune_config(
        packed_sequence=True,
        peft=None,  # Full SFT, no LoRA
    )

    # Swap in the tiny model with single-node parallelism
    cfg.model = make_tiny_super3_model(seq_length=4096)

    # Disable TP comm overlap (meaningless at TP=1)
    cfg.comm_overlap = None

    # Use standard bf16 precision
    cfg.mixed_precision = "bf16_mixed"

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "test.yaml"


def main() -> None:
    """Entry point for tiny Super3 integration-test SFT."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_finetune(config_path, _tiny_recipe_builder, cli_overrides)


if __name__ == "__main__":
    main()
