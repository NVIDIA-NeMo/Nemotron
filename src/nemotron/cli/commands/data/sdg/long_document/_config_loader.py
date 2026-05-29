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

"""Importlib-based config-class loader for long-document SDG recipes.

The recipe scripts at ``recipes/data/sdg/long-document/0X-*-sdg.py`` cannot
be imported by their dotted Python path because their parent directory uses
a dash (``long-document``) and the filenames start with a digit.  We load
each script as a private module via ``importlib.util.spec_from_file_location``
so we can pull out its ``<Stage>Config`` Pydantic class for the rich help
panel that ``RecipeMeta(config_model=...)`` powers.

The recipe scripts depend on ``data_designer`` (and other PEP 723 deps) for
their *runtime* behavior.  For the CLI to introspect them, those deps must
also be available in the parent CLI environment — install via:

    uv sync --extra data-sdg

If ``data_designer`` (or any other heavy import) is missing, this loader
raises with a clear message pointing at the extra rather than silently
degrading the help panel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


def load_recipe_module(script_path: Path, module_alias: str) -> ModuleType:
    """Load a long-document recipe script as a Python module.

    Args:
        script_path: Absolute path to the ``0X-*-sdg.py`` script.
        module_alias: Synthetic module name to register in ``sys.modules``
            (must be unique across the CLI process).

    Returns:
        The loaded module object.

    Raises:
        ImportError: If a heavy runtime dep (e.g. ``data_designer``) is not
            installed in the parent CLI environment.  The message includes
            the install hint ``uv sync --extra data-sdg``.
    """
    spec = importlib.util.spec_from_file_location(module_alias, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        # Surface a clearer error than the bare module-name failure.
        raise ImportError(
            f"Failed to import recipe module {script_path.name}: {exc.msg}.\n"
            "The Nemotron CLI introspects each long-document SDG recipe to render "
            "its rich `--help` panel; this requires the recipe's heavy deps to be "
            "installed in the CLI environment.\n"
            "Install them with: `uv sync --extra data-sdg`."
        ) from exc
    return module


def load_config_class(script_path: Path, class_name: str, module_alias: str) -> type[BaseModel]:
    """Load and return the Pydantic config class from a recipe script."""
    module = load_recipe_module(script_path, module_alias)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Recipe module {script_path.name} does not define `{class_name}`."
        )
    return cls


__all__ = ["load_recipe_module", "load_config_class"]
