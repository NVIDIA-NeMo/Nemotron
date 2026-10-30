#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super35/sft"
# image = "nvcr.io/nvidia/nemo:26.08"
# # The stage configs use this container and mount a pinned
# # Megatron-Bridge commit that carries the Super-VL recipes (see config/).
# setup = "NeMo is pre-installed in the image; Megatron-Bridge is mounted at a pinned commit by the stage config."
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
# nodes = 8
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

"""SFT (Supervised Fine-Tuning) script for Nemotron 3.5 Super VL.

Self-contained training entry point that mirrors Megatron-Bridge's
``scripts/training/run_recipe.py`` for the Super-VL recipes: load a recipe
builder from the YAML's ``recipe:`` block, merge YAML + Hydra-style CLI
overrides, export the recipe's ``env_vars`` (HybridEP / NVTE settings), and
call ``finetune(...)`` with the Omni/Super-VL forward step
(``nemotron_omni_step``). The same script drives every config in ``config/``:
SFT stage 1 (default), SFT stage 2, LoRA (peft) and the long-context recipe.

CLI:
    nemotron super35 sft                       # SFT stage 1 (default.yaml)
    nemotron super35 sft -c stage2 --run dgx   # SFT stage 2 on a cluster

Execution logic: src/nemotron/cli/commands/super35/sft.py

Direct usage:
    torchrun --nproc-per-node=8 /path/to/train.py --config /path/to/default.yaml
    torchrun --nproc-per-node=8 /path/to/train.py --config /path/to/default.yaml \\
        train.train_iters=20 dataset.path=/path/to/energon/dataset
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from omegaconf import DictConfig, OmegaConf

from nemotron.kit.recipe_loader import derive_pad_seq_to_mult, extract_recipe_config, import_recipe_function
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

# Defaults if the YAML's ``recipe:`` block omits a field.
DEFAULT_RECIPE_TARGET = "megatron.bridge.recipes.nemotronh_multimodal.nemotron_35_super_vl_sft_config"
# Megatron-Bridge maps every ``nemotron_35_super_vl_*`` recipe to this step
# (scripts/training/recipe_metadata.py: RECIPE_FORWARD_STEP_PREFIXES).
DEFAULT_STEP_FUNC = "nemotron_omni_step"

# Step-function registry (subset of Megatron-Bridge's recipe_runner.STEP_FUNCTIONS).
# A fully-qualified ``module:attr`` spec is accepted as well.
_STEP_FUNCTIONS: dict[str, str] = {
    "nemotron_omni_step": "megatron.bridge.training.nemotron_omni_step:forward_step",
    "vlm_step": "megatron.bridge.training.vlm_step:forward_step",
    "gpt_step": "megatron.bridge.training.gpt_step:forward_step",
    "llm_step": "megatron.bridge.training.gpt_step:forward_step",
}


def _load_forward_step(name: str) -> Callable[..., Any]:
    """Resolve a forward-step function by short name or ``module:attr`` spec."""
    spec = _STEP_FUNCTIONS.get(name, name)
    module_path, _, attr = spec.partition(":")
    try:
        return getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError, ValueError) as e:
        choices = ", ".join(sorted(_STEP_FUNCTIONS))
        raise ValueError(
            f"Unknown step function: {name!r}. Choose from {{{choices}}} or pass a 'module:attr' spec. "
            f"Underlying error: {e}"
        ) from e


def _merge_yaml_and_cli_overrides(cfg: ConfigContainer, config: DictConfig, cli_overrides: list[str]) -> None:
    """Merge YAML-driven overrides + Hydra-style CLI overrides into ``cfg`` in place."""
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    config_overrides = OmegaConf.to_container(config, resolve=False)
    # ``recipe`` is consumed above; ``run`` is execution metadata, not training config.
    config_overrides.pop("recipe", None)
    config_overrides.pop("run", None)

    if config_overrides:
        logger.debug(f"Merging YAML overrides: {list(config_overrides.keys())}")
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, OmegaConf.create(config_overrides))

    if cli_overrides:
        logger.debug(f"Applying Hydra-style CLI overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final, excluded_fields)


def _sync_seq_lengths(cfg: ConfigContainer) -> None:
    """Propagate ``dataset.seq_length`` to the model (as MB's run_recipe.py does)."""
    dataset_seq_length = getattr(cfg.dataset, "seq_length", None)
    if dataset_seq_length is not None and cfg.model.seq_length != dataset_seq_length:
        logger.info("Syncing model.seq_length %s -> dataset.seq_length %s", cfg.model.seq_length, dataset_seq_length)
        cfg.model.seq_length = dataset_seq_length


def _sync_finetuning_invariants(cfg: ConfigContainer) -> None:
    """Mirror MB's run_recipe.py SFT syncs (only active for offline-packed text SFT).

    - CP > 1 needs per-token loss and no in-collective averaging.
    - Offline-packed samples must match ``dataset.seq_length`` and be padded to
      the multiple the CP/SP topology requires.
    Both are no-ops for the Energon recipes (no offline packing, CP=1).
    """
    dataset = cfg.dataset
    if not hasattr(dataset, "enable_offline_packing"):
        return

    if getattr(cfg.model, "context_parallel_size", 1) > 1:
        cfg.model.calculate_per_token_loss = True
        cfg.ddp.average_in_collective = False

    specs = getattr(dataset, "offline_packing_specs", None)
    if getattr(dataset, "enable_offline_packing", False) and specs is not None:
        if getattr(dataset, "seq_length", None) is not None:
            specs.packed_sequence_size = dataset.seq_length
        specs.pad_seq_to_mult = derive_pad_seq_to_mult(
            getattr(specs, "pad_seq_to_mult", None), cfg.model, getattr(cfg, "dist", None)
        )


def _apply_recipe_env_vars(cfg: ConfigContainer) -> None:
    """Export the recipe's ``env_vars`` (HybridEP / NVTE / CUDA settings) like MB's runner."""
    try:
        from megatron.bridge.training.config import apply_environment_variables
    except ImportError:  # older Megatron-Bridge without recipe env vars
        return
    apply_environment_variables(cfg)


def main() -> None:
    """Entry point for Nemotron 3.5 Super VL supervised fine-tuning."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # -------------------------------------------------------------------------
    # ARTIFACT TRACKING + WANDB MONKEY-PATCHES
    # See nemotron/kit/wandb_kit.py for detailed "Why" / "Remove when" notes.
    # Imported at function scope: the code packager's import inliner breaks
    # class ordering when this module is pulled in at module level.
    # -------------------------------------------------------------------------
    from nemo_runspec.artifacts import setup_artifact_tracking

    tracking = setup_artifact_tracking(config, artifacts_key="run")

    if tracking.wandb:
        patch_wandb_http_handler_skip_digest_verification()
        patch_wandb_local_file_handler_skip_digest_verification()
        patch_wandb_runid_for_seeded_random()

    if tracking.manifest and tracking.wandb:
        patch_checkpoint_logging_both()
    elif tracking.wandb:
        patch_wandb_checkpoint_logging()
    elif tracking.manifest:
        patch_manifest_checkpoint_logging()

    if tracking.wandb:
        patch_wandb_init_for_lineage(
            artifact_qualified_names=tracking.qualified_names,
            tags=["sft", "super35"],
        )

    # -------------------------------------------------------------------------
    # RECIPE / STEP-FUNC SELECTION (driven by the YAML ``recipe:`` block)
    # -------------------------------------------------------------------------
    recipe_target, recipe_kwargs = extract_recipe_config(config, default_target=DEFAULT_RECIPE_TARGET)
    step_func_name = recipe_kwargs.pop("step_func", DEFAULT_STEP_FUNC)
    try:
        recipe_func = import_recipe_function(recipe_target)
        forward_step = _load_forward_step(step_func_name)
    except (AttributeError, ImportError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    cfg: ConfigContainer = recipe_func(**recipe_kwargs)

    _merge_yaml_and_cli_overrides(cfg, config, cli_overrides)
    _sync_seq_lengths(cfg)
    _sync_finetuning_invariants(cfg)
    _apply_recipe_env_vars(cfg)

    logger.debug(f"Recipe: {recipe_target}")
    logger.debug(f"Step func: {step_func_name}")
    logger.debug(f"checkpoint.pretrained_checkpoint = {cfg.checkpoint.pretrained_checkpoint}")
    logger.debug(f"dataset type = {type(cfg.dataset).__name__}")

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
