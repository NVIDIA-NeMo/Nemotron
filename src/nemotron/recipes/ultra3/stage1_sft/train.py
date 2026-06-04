#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "ultra3/sft"
# image = "/home/${oc.env:USER}/.cache/nemotron/containers/ultra3-sft.sqsh"
# setup = "Build the Ultra3 SFT container with `nemotron ultra3 build sft` before training."
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
# nodes = 384
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

"""SFT script for Nemotron Ultra3.

Wraps Megatron-Bridge's OpenMathInstruct-2 packed SFT recipe:
``megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_sft_openmathinstruct2_packed_config``.

When the YAML contains a ``dataset:`` block, this script consumes externally
prepared packed-Parquet SFT data with the same ``FinetuningDatasetConfig`` +
``PackedSequenceSpecs`` path used by Super3. When ``dataset:`` is omitted, the
Megatron-Bridge recipe's OpenMathInstruct-2 auto-download/packing dataset is left
untouched as a fallback.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
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


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

DEFAULT_RECIPE_TARGET = (
    "megatron.bridge.recipes.nemotronh.nemotron_3_ultra."
    "nemotron_3_ultra_sft_openmathinstruct2_packed_config"
)


def _build_dataset_config(dataset_config: dict[str, Any], current_dataset: Any) -> FinetuningDatasetConfig:
    """Build a FinetuningDatasetConfig from Ultra3 packed-Parquet YAML config.

    This mirrors Super3's externally-packed SFT data path and avoids replacing
    the recipe dataset with an HFDatasetConfig that downloads from HuggingFace.

    Supports packed parquet specs (directory, glob, or file paths):
    - ultra3_packed_sft_dir: Single dir that auto-resolves to train/ and valid/
    - packed_sequence_specs.packed_train_data_path: Explicit path/glob for training data
    - packed_sequence_specs.packed_val_data_path: Explicit path/glob for validation data

    Args:
        dataset_config: The resolved dataset section from YAML config.
        current_dataset: The current dataset config from the recipe (for defaults).

    Returns:
        A FinetuningDatasetConfig instance.
    """
    packed_specs = None
    has_validation_data = True
    if "packed_sequence_specs" in dataset_config:
        specs_dict = dict(dataset_config["packed_sequence_specs"])

        ultra3_dir = dataset_config.get("ultra3_packed_sft_dir")
        if ultra3_dir:
            if not specs_dict.get("packed_train_data_path"):
                train_dir = Path(f"{ultra3_dir}/train/")
                if train_dir.is_dir() and list(train_dir.glob("*.parquet")):
                    specs_dict["packed_train_data_path"] = str(train_dir)
                else:
                    raise FileNotFoundError(
                        f"No parquet files found in train split directory: {train_dir}. "
                        "Data prep may have failed or produced no training data."
                    )
            if not specs_dict.get("packed_val_data_path"):
                valid_dir = Path(f"{ultra3_dir}/valid/")
                if valid_dir.is_dir() and list(valid_dir.glob("*.parquet")):
                    specs_dict["packed_val_data_path"] = str(valid_dir)
                else:
                    logger.info(f"No validation data found in {valid_dir}, skipping validation split")
                    has_validation_data = False
            logger.info(
                "Resolved ultra3_packed_sft_dir: "
                f"train={specs_dict.get('packed_train_data_path')}, "
                f"valid={specs_dict.get('packed_val_data_path')}"
            )

        packed_specs = PackedSequenceSpecs(
            packed_sequence_size=specs_dict.get("packed_sequence_size", -1),
            packed_train_data_path=specs_dict.get("packed_train_data_path"),
            packed_val_data_path=specs_dict.get("packed_val_data_path"),
            packed_metadata_path=specs_dict.get("packed_metadata_path"),
        )

    return FinetuningDatasetConfig(
        dataset_root=dataset_config.get("dataset_root", getattr(current_dataset, "dataset_root", None)),
        seq_length=dataset_config.get("seq_length", getattr(current_dataset, "seq_length", 4096)),
        packed_sequence_specs=packed_specs,
        dataloader_type=dataset_config.get("dataloader_type", getattr(current_dataset, "dataloader_type", "batch")),
        do_validation=has_validation_data,
        do_test=False,
    )


RecipeBuilder = Callable[[DictConfig], ConfigContainer]
"""Signature for a function that builds a ConfigContainer from a loaded config."""


def _default_recipe_builder(config: DictConfig) -> ConfigContainer:
    """Build the Ultra3 OpenMath SFT Megatron-Bridge recipe from YAML."""
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


def run_finetune(
    config_path: Path,
    recipe_builder: RecipeBuilder,
    cli_overrides: list[str] | None = None,
    *,
    tags: list[str] | None = None,
) -> None:
    """Core SFT pipeline for Ultra3 SFT."""
    config = load_omegaconf_yaml(config_path)

    tracking = setup_artifact_tracking(config, artifacts_key="run")

    if tracking.wandb:
        patch_wandb_local_file_handler_skip_digest_verification()

    if tracking.manifest and tracking.wandb:
        patch_checkpoint_logging_both()
    elif tracking.wandb:
        patch_wandb_checkpoint_logging()
    elif tracking.manifest:
        patch_manifest_checkpoint_logging()

    if tracking.wandb:
        patch_wandb_init_for_lineage(
            artifact_qualified_names=tracking.qualified_names,
            tags=["sft", *(tags or [])],
        )

    cfg: ConfigContainer = recipe_builder(config)

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)
    config_overrides.pop("run", None)
    config_overrides.pop("dataset", None)

    if config_overrides:
        logger.debug(f"Merging config overrides: {list(config_overrides.keys())}")
        yaml_overrides_omega = OmegaConf.create(config_overrides)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("Config overrides merged successfully.")

    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)

    final_overrides_as_dict.pop("dataset", None)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    if "dataset" in config:
        dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
        dataset_config.pop("_target_", None)
        cfg.dataset = _build_dataset_config(dataset_config, cfg.dataset)
        logger.info(f"Built dataset config: {type(cfg.dataset).__name__}")

    logger.debug(f"checkpoint.pretrained_checkpoint = {cfg.checkpoint.pretrained_checkpoint}")
    logger.debug(f"dataset type = {type(cfg.dataset).__name__}")
    if hasattr(cfg.dataset, "packed_sequence_specs") and cfg.dataset.packed_sequence_specs:
        logger.debug(
            "packed_sequence_specs.packed_train_data_path = "
            f"{cfg.dataset.packed_sequence_specs.packed_train_data_path}"
        )

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    """Entry point for Nemotron Ultra3 supervised fine-tuning."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    run_finetune(config_path, _default_recipe_builder, cli_overrides)


if __name__ == "__main__":
    main()
