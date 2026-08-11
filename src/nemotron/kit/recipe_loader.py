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

# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Helpers for loading recipe callables and extracting config-driven kwargs."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig, OmegaConf


def derive_pad_seq_to_mult(explicit: int | None, model_config: Any, dist_config: Any = None) -> int:
    """Derive the packed-sample length multiple required by the parallel topology.

    Offline-packed samples must stay divisible by the runtime's sequence-slicing
    granularity. This mirrors Megatron-Bridge's ``sync_offline_packing_alignment``:

    - context parallelism requires ``2 * CP`` (load-balanced CP splits every
      packed sample into ``2 * CP`` chunks)
    - sequence parallelism adds a ``CP * TP`` constraint when ``TP > 1``

    The result is the least common multiple of the explicit value (if any) and
    the derived constraints, so an explicitly configured ``pad_seq_to_mult`` is
    never weakened — only strengthened when the topology demands it.

    Args:
        explicit: ``pad_seq_to_mult`` from the config, or None when unset.
        model_config: Model provider/config exposing ``context_parallel_size``,
            ``tensor_model_parallel_size``, and ``sequence_parallel``. May be
            None, in which case only the explicit value is honored.
        dist_config: Optional distributed config exposing
            ``eval_context_parallel_size``; when set, the evaluation topology
            is folded into the derived multiple as well.

    Returns:
        The padding multiple to apply during offline sequence packing.
    """
    tp = getattr(model_config, "tensor_model_parallel_size", 1) or 1
    sequence_parallel = bool(getattr(model_config, "sequence_parallel", False))

    context_parallel_sizes = {getattr(model_config, "context_parallel_size", 1) or 1}
    eval_cp = getattr(dist_config, "eval_context_parallel_size", None)
    if eval_cp is not None:
        context_parallel_sizes.add(eval_cp)

    cp_mults = [2 * cp if cp > 1 else 1 for cp in context_parallel_sizes]
    sp_mults = [cp * tp if sequence_parallel and tp > 1 else 1 for cp in context_parallel_sizes]
    return math.lcm(explicit or 1, *cp_mults, *sp_mults)


def import_recipe_function(target: str) -> Callable[..., Any]:
    """Import a recipe function from a fully-qualified target string."""
    module_path, function_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, function_name)
    except AttributeError as e:
        raise AttributeError(f"Failed to import recipe '{target}': {e}") from e


def extract_recipe_config(
    config: DictConfig,
    *,
    default_target: str,
) -> tuple[str, dict[str, Any]]:
    """Extract recipe target + kwargs from a config.

    Expects:
        recipe:
          _target_: some.module.func
          <other keys>: kwargs
    """
    if "recipe" not in config:
        return default_target, {}

    recipe_dict = OmegaConf.to_container(config.recipe, resolve=True)
    if not isinstance(recipe_dict, dict):
        return default_target, {}

    target = str(recipe_dict.pop("_target_", default_target))
    kwargs = recipe_dict
    return target, kwargs
