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

"""Shared Megatron-Bridge runner used by sft/peft/pretrain step.py wrappers.

All three step.py wrappers go through the same pipeline:

    1. Load YAML + apply CLI overrides.
    2. Build the recipe ``ConfigContainer`` from the ``recipe:`` block.
    3. Apply section overrides discovered dynamically from the recipe's
       dataclass fields (no hardcoded list).
    4. (Optional) Replace the model provider with HF weights via AutoBridge.
       Side-effects on ``cfg.checkpoint`` are explicit YAML knobs, not implicit.
    5. (Optional) Apply PEFT via ``default_peft_config`` from the YAML
       ``peft:`` block.
    6. Apply ``dataset:`` either as a normal recipe override (pretrain) or
       translate it into ``FinetuningDatasetConfig`` (SFT-style steps).
    7. Hand off to the entry point (``finetune`` / ``pretrain``).

Each step.py picks which features to enable via the runner's flags.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from nemotron.kit.recipe_loader import extract_recipe_config, import_recipe_function
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

# =============================================================================
# Section override discovery
# =============================================================================


def _discover_override_sections(cfg_container: Any) -> tuple[str, ...]:
    """Read the names of every overridable section from the ConfigContainer.

    ``ConfigContainer`` is a dataclass; its non-underscore fields are exactly
    the sections we can override (``train``, ``checkpoint``, ``model``, …).
    Discovering them dynamically means new sections in future Megatron-Bridge
    releases pick up automatically — no hardcoded tuple to maintain.
    """
    import dataclasses

    if dataclasses.is_dataclass(cfg_container):
        return tuple(f.name for f in dataclasses.fields(cfg_container) if not f.name.startswith("_"))
    # Fall back to attribute scan if the recipe returns something exotic.
    return tuple(name for name in dir(cfg_container) if not name.startswith("_"))


def _apply_section_overrides(cfg: Any, container: dict[str, Any], skip: set[str]) -> None:
    """Merge YAML override sections onto the recipe-built ConfigContainer."""
    from megatron.bridge.training.utils.omegaconf_utils import (
        apply_overrides,
        create_omegaconf_dict_config,
    )
    from omegaconf import OmegaConf

    sections = _discover_override_sections(cfg)
    overrides = {name: container[name] for name in sections if name in container and name not in skip}
    if not overrides:
        return

    omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)
    merged = OmegaConf.merge(omega_conf, OmegaConf.create(overrides))
    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded_fields)


# =============================================================================
# HF weight loading
# =============================================================================


def _maybe_load_hf_weights(cfg: Any, container: dict[str, Any]) -> None:
    """Replace ``cfg.model`` with an HF-derived provider when YAML opts in.

    Schema (all keys optional)::

        hf_model_path: <hf-id-or-path>
        load_hf_weights: true
        trust_remote_code: false
        hf_load:
          # Side effects to apply when HF weights are loaded. Defaults match
          # the prior implicit behaviour but are now explicit and overridable.
          clear_pretrained_checkpoint: true
          inherit_save_as_load: true
          mirror_model_attrs: ["tensor_model_parallel_size", "pipeline_model_parallel_size",
                               "context_parallel_size", "sequence_parallel", "seq_length",
                               "expert_model_parallel_size", "expert_tensor_parallel_size"]
    """
    hf_path = container.get("hf_model_path")
    if not hf_path:
        return

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=container.get("trust_remote_code", False))
    cfg.model = bridge.to_megatron_provider(load_weights=container.get("load_hf_weights", True))

    hf_load = dict(container.get("hf_load") or {})
    mirror = hf_load.get("mirror_model_attrs") or [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "seq_length",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    ]
    for attr in mirror:
        value = container.get("model", {}).get(attr)
        if value is not None and hasattr(cfg.model, attr):
            setattr(cfg.model, attr, value)

    if container.get("load_hf_weights", True):
        if hf_load.get("clear_pretrained_checkpoint", True):
            cfg.checkpoint.pretrained_checkpoint = None
        if hf_load.get("inherit_save_as_load", True) and cfg.checkpoint.load is None:
            cfg.checkpoint.load = cfg.checkpoint.save


# =============================================================================
# PEFT
# =============================================================================


def _maybe_apply_peft(cfg: Any, container: dict[str, Any]) -> None:
    """Build a PEFT config from the YAML ``peft:`` block when present.

    The block is forwarded to Megatron-Bridge's ``default_peft_config`` via
    its ``peft_scheme`` arg. Skip silently when ``peft.type``/``scheme`` is
    null/false/missing.
    """
    block = container.get("peft")
    if not block:
        return
    peft_kwargs = dict(block)
    scheme = peft_kwargs.pop("type", peft_kwargs.pop("scheme", None))
    if not scheme:
        return

    from megatron.bridge.recipes.utils.finetune_utils import default_peft_config

    cfg.peft = default_peft_config(peft_scheme=scheme, **peft_kwargs)


# =============================================================================
# Dataset (SFT-style)
# =============================================================================


def _maybe_build_dataset(cfg: Any, container: dict[str, Any]) -> None:
    """Translate a YAML ``dataset:`` block into a FinetuningDatasetConfig.

    Skipped when the YAML doesn't supply a ``dataset:`` block (pretrain steps
    typically rely on the recipe's own dataset config).
    """
    if "dataset" not in container:
        return

    from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
    from megatron.bridge.training.config import FinetuningDatasetConfig

    dataset_cfg = dict(container["dataset"] or {})
    dataset_cfg.pop("_target_", None)

    packed_specs = None
    has_validation = False
    if "packed_sequence_specs" in dataset_cfg:
        specs = dict(dataset_cfg["packed_sequence_specs"] or {})
        has_validation = bool(specs.get("packed_val_data_path"))
        packed_specs = PackedSequenceSpecs(
            packed_sequence_size=specs.get("packed_sequence_size", -1),
            packed_train_data_path=specs.get("packed_train_data_path"),
            packed_val_data_path=specs.get("packed_val_data_path"),
            packed_metadata_path=specs.get("packed_metadata_path"),
        )

    current = cfg.dataset
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=dataset_cfg.get("dataset_root", getattr(current, "dataset_root", None)),
        seq_length=dataset_cfg.get("seq_length", getattr(current, "seq_length", 4096)),
        packed_sequence_specs=packed_specs,
        dataloader_type=dataset_cfg.get("dataloader_type", getattr(current, "dataloader_type", "batch")),
        do_validation=has_validation,
        do_test=False,
    )


# =============================================================================
# Top-level runner
# =============================================================================


def run_megatron_bridge(
    *,
    default_recipe: str,
    default_config: Path,
    entry: Callable[..., Any],
    enable_peft: bool = True,
    enable_hf_weights: bool = True,
    dataset_mode: Literal["finetune", "override", "skip"] = "finetune",
) -> None:
    """Generic Megatron-Bridge step driver. See module docstring for the flow.

    Args:
        default_recipe: ``"module.path.fn_name"`` recipe target used when YAML
            doesn't override it.
        default_config: Path to the step's default YAML config.
        entry: ``finetune`` for SFT/PEFT, ``pretrain`` for pretrain/CPT.
        enable_peft: Apply ``peft:`` block when present.
        enable_hf_weights: Apply ``hf_model_path`` HF loading when present.
        dataset_mode: ``"finetune"`` translates ``dataset:`` into
            ``FinetuningDatasetConfig``; ``"override"`` lets normal section
            overrides merge it onto the recipe config; ``"skip"`` ignores it.
    """
    from megatron.bridge.training.gpt_step import forward_step
    from omegaconf import OmegaConf

    config_path, cli_overrides = parse_config_and_overrides(default_config=default_config)
    config = apply_hydra_overrides(load_omegaconf_yaml(config_path), cli_overrides)

    target, kwargs = extract_recipe_config(config, default_target=default_recipe)
    cfg = import_recipe_function(target)(**kwargs)

    container = OmegaConf.to_container(config, resolve=True)

    # Section overrides. SFT-style steps translate dataset separately below;
    # pretrain needs the native recipe dataset section to merge normally.
    skip_sections = {"dataset"} if dataset_mode in {"finetune", "skip"} else set()
    _apply_section_overrides(cfg, container, skip=skip_sections)

    if enable_hf_weights:
        _maybe_load_hf_weights(cfg, container)

    if enable_peft:
        _maybe_apply_peft(cfg, container)

    if dataset_mode == "finetune":
        _maybe_build_dataset(cfg, container)

    entry(config=cfg, forward_step_func=forward_step)
