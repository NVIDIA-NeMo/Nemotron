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

"""Backend registry — name → factory.

Allows registering a backend either by class or by ``"module:Class"`` string
so heavy imports (slurm pulls nemo-run + paramiko) are deferred.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import typer

from nemotron.cli.commands.step.backends.base import Backend

_BackendFactory = type[Backend] | Callable[[], Backend] | str
_REGISTRY: dict[str, _BackendFactory] = {}


def register(name: str, factory: _BackendFactory) -> None:
    """Register a backend factory under ``name``.

    ``factory`` may be a Backend class, a zero-arg callable, or a
    ``"module.path:ClassName"`` string for lazy import.
    """
    _REGISTRY[name] = factory


def get_backend(name: str) -> Backend:
    """Look up a backend by name; raise a clean Typer error if unknown."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        typer.echo(f"Unknown executor type: {name!r}. Known: {known}", err=True)
        raise typer.Exit(1)

    factory = _REGISTRY[name]
    if isinstance(factory, str):
        module_path, _, class_name = factory.partition(":")
        module = importlib.import_module(module_path)
        factory = getattr(module, class_name)
    return factory()  # type: ignore[operator]
