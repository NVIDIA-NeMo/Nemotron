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

"""SFT (Supervised Fine-Tuning) script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Dynamically loads the recipe function specified in the YAML config.

Usage:
    # With YAML config file (required)
    python /path/to/train.py --config /path/to/sft.yaml

    # With CLI overrides (Hydra syntax)
    python /path/to/train.py --config /path/to/sft.yaml train.train_iters=5000

    # As module (requires nemotron package installed)
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage1_sft.train \
        --config /path/to/sft.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from omegaconf import DictConfig, OmegaConf

from nemotron.kit.recipe_loader import extract_recipe_config, import_recipe_function
from nemotron.kit.resolvers import clear_artifact_cache, register_resolvers_from_config
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
DEFAULT_RECIPE_TARGET = "megatron.bridge.recipes.nemotronh.nemotron_nano_9b_v2_finetune_config"


def _build_dataset_config(dataset_config: DictConfig, current_dataset: Any) -> FinetuningDatasetConfig:
    """Build a FinetuningDatasetConfig from YAML config.

    This creates a proper FinetuningDatasetConfig (not HFDatasetConfig) to avoid
    downloading HuggingFace datasets. The dataset config uses pre-packed .npy files.

    Args:
        dataset_config: The dataset section from YAML config (resolved)
        current_dataset: The current dataset config from the recipe (for defaults)

    Returns:
        A FinetuningDatasetConfig instance
    """
    # Build PackedSequenceSpecs if provided
    packed_specs = None
    if "packed_sequence_specs" in dataset_config:
        specs_dict = dataset_config["packed_sequence_specs"]
        # PackedSequenceSpecs.__post_init__ converts string paths to Path/MultiStoragePath
        packed_specs = PackedSequenceSpecs(
            packed_sequence_size=specs_dict.get("packed_sequence_size", -1),
            packed_train_data_path=specs_dict.get("packed_train_data_path"),
            packed_val_data_path=specs_dict.get("packed_val_data_path"),
            packed_metadata_path=specs_dict.get("packed_metadata_path"),
        )

    # Build FinetuningDatasetConfig with values from YAML, falling back to current config
    return FinetuningDatasetConfig(
        dataset_root=dataset_config.get("dataset_root", getattr(current_dataset, "dataset_root", None)),
        seq_length=dataset_config.get("seq_length", getattr(current_dataset, "seq_length", 4096)),
        packed_sequence_specs=packed_specs,
        dataloader_type=dataset_config.get("dataloader_type", getattr(current_dataset, "dataloader_type", "batch")),
    )


def main() -> None:
    """Entry point for Nemotron Nano3 supervised fine-tuning."""
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

    # Clear artifact cache to ensure fresh downloads (important for :latest resolution)
    clear_artifact_cache()

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
        tags=["sft"],
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
    # Do this BEFORE building our custom dataset config (which contains MultiStoragePath)
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Get config overrides (excluding recipe, run, and dataset)
    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)
    config_overrides.pop("run", None)
    config_overrides.pop("dataset", None)  # We handle dataset separately below

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

    # Don't let apply_overrides touch the dataset - we handle it separately
    final_overrides_as_dict.pop("dataset", None)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Handle dataset config AFTER apply_overrides - build FinetuningDatasetConfig directly
    # This avoids HFDatasetConfig which tries to download from HuggingFace
    # PackedSequenceSpecs.__post_init__ converts paths to Path/MultiStoragePath automatically
    if "dataset" in config:
        dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
        dataset_config.pop("_target_", None)
        cfg.dataset = _build_dataset_config(dataset_config, cfg.dataset)
        logger.info(f"Built dataset config: {type(cfg.dataset).__name__}")

    # Debug: print key config values
    print(f"DEBUG: checkpoint.pretrained_checkpoint = {cfg.checkpoint.pretrained_checkpoint}")
    print(f"DEBUG: dataset type = {type(cfg.dataset).__name__}")
    if hasattr(cfg.dataset, "packed_sequence_specs") and cfg.dataset.packed_sequence_specs:
        print(f"DEBUG: packed_sequence_specs.packed_train_data_path = {cfg.dataset.packed_sequence_specs.packed_train_data_path}")

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
