#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "lightning35/pretrain"
# image = "nvcr.io/nvidia/nemo:26.08"
# # The public 26.08 tag ships at launch; until then the stage configs pin
# # nvcr.io/nvidian/nemo:26.08.rc2 and mount Megatron-Bridge main (see config/).
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

"""Pretrain script for Nemotron Lightning35.

Uses Megatron-Bridge's ConfigContainer for full training configuration.
Dynamically loads the recipe function specified in the YAML config.

CLI:
    nemotron lightning35 pretrain              # local execution
    nemotron lightning35 pretrain --run dgx    # submit to cluster

Execution logic: src/nemotron/cli/commands/lightning35/pretrain.py

Direct usage:
    python /path/to/train.py --config /path/to/pretrain.yaml
    python /path/to/train.py --config /path/to/pretrain.yaml train.train_iters=5000
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
DEFAULT_RECIPE_TARGET = "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning_pretrain_config"


def main() -> None:
    """Entry point for Nemotron Lightning35 pretraining."""
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

    # Newer Megatron-Bridge recipe functions take no data kwargs; the blend is
    # applied to cfg.dataset instead. Split the kwargs accordingly so both old
    # (kwargs-accepting) and new (zero-arg) recipe functions work.
    import inspect

    recipe_params = inspect.signature(recipe_func).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in recipe_params.values())
    blend_kwargs = {}
    for key in ("per_split_data_args_path", "data_args_path", "data_paths"):
        if key in recipe_kwargs and key not in recipe_params and not accepts_kwargs:
            blend_kwargs[key] = recipe_kwargs.pop(key)

    cfg: ConfigContainer = recipe_func(**recipe_kwargs)

    if blend_kwargs:
        from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths

        blend, blend_per_split, split = get_blend_fields_from_data_paths(**blend_kwargs)
        cfg.dataset.blend = blend
        cfg.dataset.blend_per_split = blend_per_split
        cfg.dataset.split = split
        logger.info(f"Applied data blend to dataset config: {blend_kwargs}")

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

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
