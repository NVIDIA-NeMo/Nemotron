#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super3/pretrain"
# image = "nvcr.io/nvidian/nemo:26.02.super.rc4"
# setup = "NeMo and all training dependencies are pre-installed in the image."
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

"""Pretrain script for Nemotron Super3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Dynamically loads the recipe function specified in the YAML config.

CLI:
    nemotron super3 pretrain              # local execution
    nemotron super3 pretrain --run dgx    # submit to cluster

Execution logic: src/nemotron/cli/commands/super3/pretrain.py

Direct usage:
    python /path/to/train.py --config /path/to/pretrain.yaml
    python /path/to/train.py --config /path/to/pretrain.yaml train.train_iters=5000
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
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
from nemo_runspec.artifacts import setup_artifact_tracking
from nemotron.kit.train_script import load_omegaconf_yaml, parse_config_and_overrides
from nemotron.kit.wandb_kit import (
    patch_checkpoint_logging_both,
    patch_manifest_checkpoint_logging,
    patch_wandb_checkpoint_logging,
    patch_wandb_init_for_lineage,
    patch_wandb_local_file_handler_skip_digest_verification,
)

logger: logging.Logger = logging.getLogger(__name__)


# Default config path relative to this file
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

DEFAULT_RECIPE_TARGET = (
    "megatron.bridge.recipes.nemotronh.nemotron_3_super.nemotron_3_super_pretrain_config"
)


RecipeBuilder = Callable[["DictConfig"], ConfigContainer]
"""Signature for a function that builds a ConfigContainer from a loaded config."""


def _default_recipe_builder(config: "DictConfig") -> ConfigContainer:
    """Build recipe from YAML ``recipe._target_`` (production path).

    The ``data`` YAML section contains recipe-function parameters (e.g.
    ``per_split_data_args_path``) that control dataset blend construction.
    These are **not** fields on ``ConfigContainer`` / ``GPTDatasetConfig``,
    so they must be forwarded as kwargs to the recipe function here — the
    later config-merge step cannot reach them.
    """
    recipe_target, recipe_kwargs = extract_recipe_config(
        config,
        default_target=DEFAULT_RECIPE_TARGET,
    )

    # Merge data section into recipe kwargs (dataset blend parameters)
    if "data" in config:
        data_kwargs = OmegaConf.to_container(config.data, resolve=True)
        if isinstance(data_kwargs, dict):
            recipe_kwargs = {**data_kwargs, **recipe_kwargs}

    try:
        recipe_func = import_recipe_function(recipe_target)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    return recipe_func(**recipe_kwargs)


def run_pretrain(
    config_path: Path,
    recipe_builder: RecipeBuilder,
    cli_overrides: list[str] | None = None,
    *,
    tags: list[str] | None = None,
) -> None:
    """Core pretrain pipeline.

    Handles wandb patches, artifact resolution, lineage, config merging,
    and calls ``pretrain()``.  The *recipe_builder* callback is the only
    extension point: ``train.py`` passes the dynamic ``_target_`` loader,
    while ``test_train.py`` can pass an inline recipe.

    Args:
        config_path: Path to the YAML config file.
        recipe_builder: Callable that receives the loaded DictConfig and
            returns a fully-constructed ConfigContainer.
        cli_overrides: Optional Hydra-style command-line overrides.
    """
    config = load_omegaconf_yaml(config_path)

    # -------------------------------------------------------------------------
    # ARTIFACT TRACKING
    #
    # setup_artifact_tracking reads config.artifacts and initializes the
    # manifest tracker + artifact resolvers.  We then apply stage-specific
    # monkey-patches based on which backends are active.
    # -------------------------------------------------------------------------
    tracking = setup_artifact_tracking(config, artifacts_key="run")

    # Wandb bug workarounds (specific to this container version)
    if tracking.wandb:
        patch_wandb_local_file_handler_skip_digest_verification()

    # Checkpoint logging patches
    if tracking.manifest and tracking.wandb:
        patch_checkpoint_logging_both()
    elif tracking.wandb:
        patch_wandb_checkpoint_logging()
    elif tracking.manifest:
        patch_manifest_checkpoint_logging()

    # Wandb lineage registration
    if tracking.wandb:
        patch_wandb_init_for_lineage(
            artifact_qualified_names=tracking.qualified_names,
            tags=["pretrain", *(tags or [])],
        )

    cfg: ConfigContainer = recipe_builder(config)

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Merge config overrides (excluding recipe and data — those are recipe-function kwargs)
    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)
    config_overrides.pop("data", None)

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

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    """Entry point for Nemotron Super3 pretraining."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_pretrain(config_path, _default_recipe_builder, cli_overrides)


if __name__ == "__main__":
    main()
