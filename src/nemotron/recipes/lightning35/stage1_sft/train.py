#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "lightning35/sft"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_5_lightning"
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
# nodes = 1
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

"""SFT (Supervised Fine-Tuning) script for Nemotron Lightning35.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Dynamically loads the recipe function specified in the YAML config.

CLI:
    nemotron lightning35 sft              # local execution
    nemotron lightning35 sft --run dgx    # submit to cluster

Execution logic: src/nemotron/cli/commands/lightning35/sft.py

Direct usage:
    python /path/to/train.py --config /path/to/sft.yaml
    python /path/to/train.py --config /path/to/sft.yaml train.train_iters=5000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

try:  # Megatron-Bridge >= the 2026-07 dataset refactor (builder architecture)
    from megatron.bridge.data.packing import PackedSequenceSpecs

    _LEGACY_DATASET_API = False
except ImportError:  # pre-refactor Megatron-Bridge
    from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

    _LEGACY_DATASET_API = True
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from omegaconf import DictConfig, OmegaConf

from nemotron.kit.recipe_loader import (
    derive_pad_seq_to_mult,
    extract_recipe_config,
    import_recipe_function,
)
from nemotron.kit.train_script import load_omegaconf_yaml, parse_config_and_overrides
from nemotron.kit.wandb_kit import (
    patch_checkpoint_logging_both,
    patch_manifest_checkpoint_logging,
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
DEFAULT_RECIPE_TARGET = "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning_sft_config"


def _build_dataset_config(
    dataset_config: DictConfig,
    current_dataset: Any,
    model_config: Any = None,
    dist_config: Any = None,
) -> FinetuningDatasetConfig:
    """Build a FinetuningDatasetConfig from YAML config.

    This creates a proper FinetuningDatasetConfig (not HFDatasetConfig) to avoid
    downloading HuggingFace datasets.

    Supports packed parquet specs (directory, glob, or file paths):
    - lightning35_packed_sft_dir: Single dir that auto-resolves to splits/train/ and splits/valid/
    - packed_sequence_specs.packed_train_data_path: Explicit path/glob for training data
    - packed_sequence_specs.packed_val_data_path: Explicit path/glob for validation data

    Args:
        dataset_config: The dataset section from YAML config (resolved)
        current_dataset: The current dataset config from the recipe (for defaults)

    Returns:
        A FinetuningDatasetConfig instance
    """
    # Build PackedSequenceSpecs if provided
    packed_specs = None
    has_validation_data = True  # Track if we have validation data
    if "packed_sequence_specs" in dataset_config:
        specs_dict = dict(dataset_config["packed_sequence_specs"])

        # Check for lightning35_packed_sft_dir shorthand
        # lightning35_packed_sft_dir should point to the splits directory (e.g., /path/to/output/splits)
        # which contains train/ and valid/ subdirectories
        lightning35_dir = dataset_config.get("lightning35_packed_sft_dir")
        if lightning35_dir:
            # Auto-resolve to split directories if not explicitly set
            # Only set paths if the directories actually contain parquet files
            if not specs_dict.get("packed_train_data_path"):
                train_dir = Path(f"{lightning35_dir}/train/")
                if train_dir.is_dir() and list(train_dir.glob("*.parquet")):
                    specs_dict["packed_train_data_path"] = str(train_dir)
                else:
                    raise FileNotFoundError(
                        f"No parquet files found in train split directory: {train_dir}. "
                        "Data prep may have failed or produced no training data."
                    )
            if not specs_dict.get("packed_val_data_path"):
                valid_dir = Path(f"{lightning35_dir}/valid/")
                # Validation is optional - only set if directory has files
                if valid_dir.is_dir() and list(valid_dir.glob("*.parquet")):
                    specs_dict["packed_val_data_path"] = str(valid_dir)
                else:
                    logger.info(f"No validation data found in {valid_dir}, skipping validation split")
                    has_validation_data = False
            logger.info(
                f"Resolved lightning35_packed_sft_dir: train={specs_dict.get('packed_train_data_path')}, valid={specs_dict.get('packed_val_data_path')}"
            )

        # PackedSequenceSpecs.__post_init__ converts string paths to Path/MultiStoragePath
        packed_specs = PackedSequenceSpecs(
            packed_sequence_size=specs_dict.get("packed_sequence_size", -1),
            packed_train_data_path=specs_dict.get("packed_train_data_path"),
            packed_val_data_path=specs_dict.get("packed_val_data_path"),
            packed_metadata_path=specs_dict.get("packed_metadata_path"),
            # CP/SP-aware padding options (see NVIDIA-NeMo/Megatron-Bridge#5080):
            # forward pad_cu_seqlens and derive the sample-length multiple required
            # by the parallel topology so CP>1 packed SFT works out of the box.
            pad_cu_seqlens=specs_dict.get("pad_cu_seqlens", False),
            pad_seq_to_mult=derive_pad_seq_to_mult(specs_dict.get("pad_seq_to_mult"), model_config, dist_config),
        )

    # Build FinetuningDatasetConfig with values from YAML, falling back to current config
    # Note: dataset_root can be None when using externally-prepared packed parquet data
    common_fields = dict(
        dataset_root=dataset_config.get("dataset_root", getattr(current_dataset, "dataset_root", None)),
        seq_length=dataset_config.get("seq_length", getattr(current_dataset, "seq_length", 4096)),
        dataloader_type=dataset_config.get("dataloader_type", getattr(current_dataset, "dataloader_type", "batch")),
        do_validation=has_validation_data,
        do_test=False,  # We don't use test split for lightning35 SFT
    )

    if _LEGACY_DATASET_API:
        return FinetuningDatasetConfig(packed_sequence_specs=packed_specs, **common_fields)

    # New builder API: packing moves to enable_offline_packing/offline_packing_specs,
    # and validate() demands a dataset source — anchor dataset_root on the packed
    # train split dir when data was packed externally.
    if common_fields["dataset_root"] is None and packed_specs is not None:
        common_fields["dataset_root"] = str(packed_specs.packed_train_data_path)
    return FinetuningDatasetConfig(
        enable_offline_packing=packed_specs is not None,
        offline_packing_specs=packed_specs,
        **common_fields,
    )


def main() -> None:
    """Entry point for Nemotron Lightning35 supervised fine-tuning."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # -------------------------------------------------------------------------
    # WANDB MONKEY-PATCHES
    # These patches work around bugs in wandb and Megatron-Bridge.
    # See nemotron/kit/wandb_kit.py for detailed "Why" / "Remove when" documentation.
    # -------------------------------------------------------------------------
    # Initialize artifact tracking from the config's `artifacts:` section.
    # Supports the wandb backend, the file-based manifest backend
    # ([artifacts.manifest] root=... in env.toml), or both.
    # Imported at function scope: the code packager's import inliner breaks
    # class ordering when this module is pulled in at module level.
    from nemo_runspec.artifacts import setup_artifact_tracking

    tracking = setup_artifact_tracking(config, artifacts_key="run")

    # Wandb bug workarounds (only relevant when the wandb backend is active)
    if tracking.wandb:
        patch_wandb_http_handler_skip_digest_verification()
        patch_wandb_local_file_handler_skip_digest_verification()
        patch_wandb_runid_for_seeded_random()

    # Checkpoint logging: route to the active backend(s)
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
        cfg.dataset = _build_dataset_config(
            dataset_config,
            cfg.dataset,
            model_config=cfg.model,
            dist_config=getattr(cfg, "dist", None),
        )
        logger.info(f"Built dataset config: {type(cfg.dataset).__name__}")

    # Log key config values for debugging
    logger.debug(f"checkpoint.pretrained_checkpoint = {cfg.checkpoint.pretrained_checkpoint}")
    logger.debug(f"dataset type = {type(cfg.dataset).__name__}")
    if hasattr(cfg.dataset, "packed_sequence_specs") and cfg.dataset.packed_sequence_specs:
        logger.debug(
            f"packed_sequence_specs.packed_train_data_path = {cfg.dataset.packed_sequence_specs.packed_train_data_path}"
        )

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
