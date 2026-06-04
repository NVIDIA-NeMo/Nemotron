#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "ultra3/pretrain"
# image = "/home/${oc.env:USER}/.cache/nemotron/containers/ultra3-pretrain.sqsh"
# setup = "Build the Ultra3 pretrain container with `nemotron ultra3 build pretrain` before training."
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
# nodes = 96
# gpus_per_node = 8
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

"""Pretrain script for Nemotron Ultra3.

Uses Megatron-Bridge's ConfigContainer for full training configuration and
wraps the real Ultra recipe:
``megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_pretrain_config``.

CLI:
    nemotron ultra3 pretrain              # local execution
    nemotron ultra3 pretrain --run dgx    # submit to cluster

Execution logic: src/nemotron/cli/commands/ultra3/pretrain.py

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
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from omegaconf import DictConfig, OmegaConf

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
    "megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_pretrain_config"
)


RecipeBuilder = Callable[[DictConfig], ConfigContainer]
"""Signature for a function that builds a ConfigContainer from a loaded config."""


def _default_recipe_builder(config: DictConfig) -> ConfigContainer:
    """Build the Ultra3 Megatron-Bridge recipe from YAML ``recipe._target_``."""
    recipe_target, recipe_kwargs = extract_recipe_config(
        config,
        default_target=DEFAULT_RECIPE_TARGET,
    )

    try:
        recipe_func = import_recipe_function(recipe_target)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    return recipe_func(**recipe_kwargs)


def _build_ultra_nvfp4_precision():
    from megatron.bridge.training.mixed_precision import bf16_with_nvfp4_mixed

    mp = bf16_with_nvfp4_mixed()
    mp.first_last_layers_bf16 = True
    mp.num_layers_at_end_in_bf16 = 16
    return mp


def run_pretrain(
    config_path: Path,
    recipe_builder: RecipeBuilder,
    cli_overrides: list[str] | None = None,
    *,
    tags: list[str] | None = None,
) -> None:
    """Core pretrain pipeline."""
    config = load_omegaconf_yaml(config_path)

    # -------------------------------------------------------------------------
    # ARTIFACT TRACKING
    # -------------------------------------------------------------------------
    tracking = setup_artifact_tracking(config, artifacts_key="run")

    # Wandb bug workarounds
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

    data_kwargs = {}
    if "data" in config:
        data_config = OmegaConf.to_container(config.data, resolve=True)
        if isinstance(data_config, dict):
            data_kwargs = data_config

    blend_keys = {
        "per_split_data_args_path",
        "data_paths",
        "train_data_path",
        "valid_data_path",
        "test_data_path",
        "mock",
    }
    blend_kwargs = {key: data_kwargs[key] for key in blend_keys if key in data_kwargs}
    if blend_kwargs:
        blend, blend_per_split, split = get_blend_fields_from_data_paths(**blend_kwargs)
        cfg.dataset.blend = blend
        cfg.dataset.blend_per_split = blend_per_split
        cfg.dataset.split = split

    if "path_to_cache" in data_kwargs:
        cfg.dataset.path_to_cache = data_kwargs["path_to_cache"]

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Merge config overrides (excluding recipe/run and data — data is wired above after recipe construction)
    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)
    config_overrides.pop("run", None)
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

    if (
        isinstance(cfg.mixed_precision, str)
        and cfg.mixed_precision == "nemotron_3_ultra_bf16_with_nvfp4_mixed"
    ):
        cfg.mixed_precision = _build_ultra_nvfp4_precision()

    pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    """Entry point for Nemotron Ultra3 pretraining."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_pretrain(config_path, _default_recipe_builder, cli_overrides)


if __name__ == "__main__":
    main()
