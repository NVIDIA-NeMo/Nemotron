#!/usr/bin/env python3

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

"""Pretrain script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Dynamically loads the recipe function specified in the YAML config.

Usage:
    # With YAML config file (required)
    python /path/to/train.py --config /path/to/pretrain.yaml

    # With CLI overrides (Hydra syntax)
    python /path/to/train.py --config /path/to/pretrain.yaml train.train_iters=5000

    # As module (requires nemotron package installed)
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.train \
        --config /path/to/pretrain.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from omegaconf import OmegaConf

from nemotron.kit.recipe_loader import extract_recipe_config, import_recipe_function
from nemotron.kit.resolvers import register_resolvers_from_config
from nemotron.kit.train_script import load_omegaconf_yaml, parse_config_and_overrides
from nemotron.kit.wandb import (
    patch_wandb_checkpoint_logging,
    patch_wandb_http_handler_skip_digest_verification,
    patch_wandb_init_for_lineage,
    patch_wandb_local_file_handler_skip_digest_verification,
    patch_wandb_runid_for_seeded_random,
)

logger: logging.Logger = logging.getLogger(__name__)


# Default config path relative to this file
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

# Default recipe function
DEFAULT_RECIPE_TARGET = (
    "megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2.nemotron_next_3b_v2_pretrain_config"
)


def main() -> None:
    """Entry point for Nemotron Nano3 pretraining."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    patch_wandb_http_handler_skip_digest_verification()
    patch_wandb_local_file_handler_skip_digest_verification()

    # Fix "Invalid Client ID digest" error caused by seeded random (wandb bug)
    # See: https://github.com/wandb/wandb/pull/11039
    patch_wandb_runid_for_seeded_random()

    # Apply monkey patch for wandb checkpoint artifact logging
    patch_wandb_checkpoint_logging()

    # Resolve artifacts before wandb.init() (Megatron-Bridge initializes wandb).
    qualified_names = register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )

    # Patch wandb.init so lineage is registered immediately once MB initializes wandb.
    patch_wandb_init_for_lineage(
        artifact_qualified_names=qualified_names,
        tags=["pretrain"],
    )

    recipe_target, recipe_kwargs = extract_recipe_config(
        config,
        default_target=DEFAULT_RECIPE_TARGET,
    )

    try:
        recipe_func = import_recipe_function(recipe_target)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    cfg: ConfigContainer = recipe_func(**recipe_kwargs)

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Merge config overrides (excluding recipe field)
    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)

    if config_overrides:
        logger.debug(f"Merging config overrides: {list(config_overrides.keys())}")
        yaml_overrides_omega = OmegaConf.create(config_overrides)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("Config overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    pretrain(config=cfg, forward_step_func=forward_step)

    # # Finish wandb run to ensure all artifacts are synced
    # try:
    #     import wandb
    #     if wandb.run is not None:
    #         logger.info("[WANDB] Finishing wandb run...")
    #         wandb.finish()
    #         logger.info("[WANDB] Wandb run finished")
    # except Exception as e:
    #     logger.warning(f"[WANDB] Failed to finish wandb run: {e}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
