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

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig, OmegaConf


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
