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

"""Shared AutoModel runner used by sft/peft/pretrain step.py wrappers.

The recipe class can be overridden in YAML via a top-level
``_step_recipe: "module.path:ClassName"`` key. We deliberately avoid the
``recipe._target_`` slot because AutoModel's own config loader interprets
``_target_`` values as ``file/path.py:ClassName`` (file-style) — sharing the
slot causes a parse error.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from nemotron.kit.train_script import parse_config_and_overrides

_STEP_RECIPE_KEY = "_step_recipe"


def _resolve_recipe(default_target: str, cfg) -> type:
    """Return the recipe class. YAML override via ``_step_recipe`` key, else default."""
    target = default_target
    override = None
    if hasattr(cfg, _STEP_RECIPE_KEY):
        override = getattr(cfg, _STEP_RECIPE_KEY)
    elif isinstance(cfg, dict):
        override = cfg.get(_STEP_RECIPE_KEY)
    if override:
        target = override

    module_path, _, class_name = target.partition(":")
    if not class_name:
        # Allow Python dotted form: "pkg.mod.Class".
        module_path, _, class_name = target.rpartition(".")
    if not class_name:
        raise ValueError(f"Could not resolve recipe target: {target!r}")

    return getattr(importlib.import_module(module_path), class_name)


def run_automodel(*, default_target: str, default_config: Path) -> None:
    """Load YAML config, instantiate the AutoModel recipe class, run it.

    Args:
        default_target: ``"module.path:ClassName"`` of the recipe used when the
            YAML doesn't supply ``_step_recipe``.
        default_config: Default config path (the step's ``config/default.yaml``).
    """
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    config_path, _ = parse_config_and_overrides(default_config=default_config)
    cfg = parse_args_and_load_config(str(config_path))

    recipe_cls = _resolve_recipe(default_target, cfg)
    recipe = recipe_cls(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()
