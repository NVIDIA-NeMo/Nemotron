#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/sft"
# image = "/home/${oc.env:USER}/.cache/nemotron/containers/omni3-sft.sqsh"
# setup = "Build the Omni SFT container with `nemotron omni3 build sft` before training."
#
# [tool.runspec.run]
# launch = "torchrun"
# workdir = "/workspace/Megatron-Bridge"
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

"""SFT (Supervised Fine-Tuning) script for Nemotron Omni3.

Self-contained training entry point — does not shell out to upstream's
``scripts/training/run_recipe.py``. The logic mirrors that script: load a
recipe builder by name, optionally apply a ``--dataset`` selector, merge
YAML and Hydra-style CLI overrides, then call ``finetune(...)``. The recipe
name, step function, and dataset selector are read from the YAML's
``recipe:`` block (rather than threaded as CLI flags), matching the shape
of ``nano3/stage1_sft/train.py`` and ``super3/stage1_sft/train.py``.

CLI:
    nemotron omni3 sft              # local execution
    nemotron omni3 sft --run dlw    # submit to cluster

Execution logic: src/nemotron/cli/commands/omni3/sft.py

Direct usage (inside the omni3-sft container):
    cd /workspace/Megatron-Bridge
    torchrun --nproc-per-node=8 /path/to/train.py --config /path/to/default.yaml
    torchrun --nproc-per-node=8 /path/to/train.py --config /path/to/default.yaml \\
        train.train_iters=20 dataset.seq_length=2048
"""

from __future__ import annotations

import importlib
import inspect
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

from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config
from nemotron.kit.train_script import load_omegaconf_yaml, parse_config_and_overrides
from nemotron.kit.wandb_kit import (
    patch_wandb_checkpoint_logging,
    patch_wandb_http_handler_skip_digest_verification,
    patch_wandb_init_for_lineage,
    patch_wandb_local_file_handler_skip_digest_verification,
    patch_wandb_runid_for_seeded_random,
)

logger: logging.Logger = logging.getLogger(__name__)


# Default config path relative to this file.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

# Defaults if the YAML's ``recipe:`` block omits a field.
DEFAULT_RECIPE_NAME = "nemotron_omni_cord_v2_sft_config"
DEFAULT_STEP_FUNC = "nemotron_omni_step"


# Step-function registry. Mirrors upstream ``scripts/training/run_recipe.py``
# (megatron-bridge), extended with ``nemotron_omni_step`` for the omni branch.
# Imports are lazy so we don't pay the cost (or fail) for unused entries.
_STEP_FUNCTIONS: dict[str, str] = {
    "audio_lm_step": "megatron.bridge.training.audio_lm_step:forward_step",
    "gpt_step": "megatron.bridge.training.gpt_step:forward_step",
    "vlm_step": "megatron.bridge.training.vlm_step:forward_step",
    "llava_step": "megatron.bridge.training.llava_step:forward_step",
    "qwen3_vl_step": "megatron.bridge.models.qwen_vl.qwen3_vl_step:forward_step",
    "nemotron_omni_step": "megatron.bridge.training.nemotron_omni_step:forward_step",
}


def _import_attr(spec: str) -> Any:
    """Import ``module:attr`` (or ``module.attr``) and return the attribute."""
    if ":" in spec:
        module_path, attr = spec.split(":", 1)
    else:
        module_path, _, attr = spec.rpartition(".")
        if not module_path:
            raise ImportError(f"Cannot parse import spec: {spec!r}")
    return getattr(importlib.import_module(module_path), attr)


def _load_forward_step(name: str) -> Callable[..., Any]:
    """Resolve a forward-step function by short name or fully-qualified spec."""
    spec = _STEP_FUNCTIONS.get(name, name)
    try:
        return _import_attr(spec)
    except (ImportError, AttributeError) as e:
        choices = ", ".join(sorted(_STEP_FUNCTIONS))
        raise ValueError(
            f"Unknown step function: {name!r}. "
            f"Choose from {{{choices}}} or pass a 'module:attr' import spec. "
            f"Underlying error: {e}"
        ) from e


def _resolve_recipe_builder(recipe_section: dict[str, Any]) -> tuple[Callable[..., ConfigContainer], dict[str, Any]]:
    """Resolve the recipe builder callable and its kwargs from the ``recipe:`` block.

    Supports two shapes:
      * ``recipe._target_: module.path.func`` — fully-qualified import (matches
        the nano3/super3 convention).
      * ``recipe.name: <short_name>`` — short name resolved via
        ``getattr(megatron.bridge.recipes, name)`` (matches upstream
        ``run_recipe.py --recipe <name>``).

    Recognized control keys (``name``, ``_target_``, ``step_func``, ``dataset``)
    are stripped before returning kwargs; anything else is passed through to the
    builder if its signature accepts it (or via ``**kwargs``).
    """
    section = dict(recipe_section)
    target = section.pop("_target_", None)
    name = section.pop("name", None)
    section.pop("step_func", None)
    section.pop("dataset", None)

    if target and name:
        raise ValueError("recipe._target_ and recipe.name are mutually exclusive; set only one")

    if target:
        builder = _import_attr(target)
    else:
        recipe_name = name or DEFAULT_RECIPE_NAME
        recipes_pkg = importlib.import_module("megatron.bridge.recipes")
        if not hasattr(recipes_pkg, recipe_name):
            raise AttributeError(
                f"Recipe '{recipe_name}' not found in megatron.bridge.recipes. "
                "Make sure the recipe is exported in its family __init__.py "
                "(e.g. megatron.bridge.recipes.nemotron_omni)."
            )
        builder = getattr(recipes_pkg, recipe_name)

    # Filter kwargs to ones the builder accepts, unless it has **kwargs.
    try:
        sig = inspect.signature(builder)
        params = sig.parameters
        accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_var_keyword:
            section = {k: v for k, v in section.items() if k in params}
    except (TypeError, ValueError):
        # If we can't inspect, let the builder validate.
        pass

    return builder, section


def _apply_dataset_selector(
    cfg: ConfigContainer,
    dataset_type: str | None,
    cli_overrides: list[str],
) -> ConfigContainer:
    """Apply the ``--dataset <type>`` selector from upstream run_recipe.py.

    Mirrors the call upstream makes when its ``--dataset`` flag is set:
    pulls in ``apply_dataset_override`` from ``megatron.bridge.recipes.utils``
    and lets it stamp the right dataset block onto the config (vlm-hf,
    vlm-energon, llm-finetune, etc.).
    """
    if not dataset_type:
        return cfg
    from megatron.bridge.recipes.utils.dataset_utils import apply_dataset_override

    return apply_dataset_override(
        cfg,
        dataset_type=dataset_type,
        packed_sequence=False,
        seq_length=None,
        cli_overrides=cli_overrides,
    )


def _merge_yaml_and_cli_overrides(
    cfg: ConfigContainer,
    config: DictConfig,
    cli_overrides: list[str],
) -> None:
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
    """Keep ``model.seq_length`` and ``dataset.seq_length`` consistent.

    Matches the post-override sync that upstream ``run_recipe.py`` does so a
    single ``dataset.seq_length=...`` CLI override propagates to the model.
    """
    model = getattr(cfg, "model", None)
    dataset = getattr(cfg, "dataset", None)
    if model is None or dataset is None:
        return
    if hasattr(dataset, "seq_length") and model.seq_length != dataset.seq_length:
        logger.warning(
            "Syncing model.seq_length (%s) to dataset.seq_length (%s); "
            "override dataset.seq_length when changing Omni SFT context length.",
            model.seq_length,
            dataset.seq_length,
        )
        model.seq_length = dataset.seq_length


def main() -> None:
    """Entry point for Nemotron Omni3 supervised fine-tuning."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # -------------------------------------------------------------------------
    # WANDB MONKEY-PATCHES (best-effort; safe no-ops when wandb isn't loaded)
    # -------------------------------------------------------------------------
    patch_wandb_http_handler_skip_digest_verification()
    patch_wandb_local_file_handler_skip_digest_verification()
    patch_wandb_runid_for_seeded_random()
    patch_wandb_checkpoint_logging()

    # Clear artifact cache so ``:latest`` aliases re-resolve each run.
    clear_artifact_cache()

    # Resolve ``${art:...}`` references before Megatron-Bridge initializes wandb.
    qualified_names = register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )

    patch_wandb_init_for_lineage(
        artifact_qualified_names=qualified_names,
        tags=["sft", "omni3"],
    )

    # -------------------------------------------------------------------------
    # RECIPE / STEP-FUNC / DATASET SELECTION (driven by YAML, not CLI flags)
    # -------------------------------------------------------------------------
    recipe_section: dict[str, Any] = (
        OmegaConf.to_container(config.recipe, resolve=True) if "recipe" in config else {}
    ) or {}
    step_func_name = recipe_section.get("step_func", DEFAULT_STEP_FUNC)
    dataset_type = recipe_section.get("dataset")

    try:
        recipe_builder, recipe_kwargs = _resolve_recipe_builder(recipe_section)
    except (AttributeError, ImportError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    cfg: ConfigContainer = recipe_builder(**recipe_kwargs)

    # Apply the ``--dataset`` selector (vlm-hf, vlm-energon, llm-finetune, ...).
    cfg = _apply_dataset_selector(cfg, dataset_type, cli_overrides)

    # YAML + CLI overrides on top of whatever the recipe + dataset selector set.
    _merge_yaml_and_cli_overrides(cfg, config, cli_overrides)
    _sync_seq_lengths(cfg)

    logger.debug(f"Recipe: {recipe_builder.__module__}.{recipe_builder.__name__}")
    logger.debug(f"Step func: {step_func_name}")
    logger.debug(f"Dataset selector: {dataset_type}")
    logger.debug(f"checkpoint.pretrained_checkpoint = {getattr(cfg.checkpoint, 'pretrained_checkpoint', None)}")
    logger.debug(f"dataset type = {type(cfg.dataset).__name__}")

    forward_step = _load_forward_step(step_func_name)

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
