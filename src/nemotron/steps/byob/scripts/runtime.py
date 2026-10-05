# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin BYOB runtime dispatcher.

Benchmark-specific behavior belongs in `nemotron.steps.byob.runtime.benchmark_families`.
This module only selects the family and requested stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar, cast, overload

import yaml  # type: ignore[import-untyped]

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkRunResult
from nemotron.steps.byob.runtime.benchmark_families.registry import get_family, list_families

STAGE_CHOICES = ("prepare", "generate", "translate", "eval", "all")
StageName = Literal["prepare", "generate", "translate", "eval", "all"]


class ByobDispatchError(ValueError):
    code = "byob_stage_unsupported"
    cli_exit_code = 2


def list_family_names() -> tuple[str, ...]:
    """Return the registered benchmark families."""
    return tuple(list_families())


def load_dispatch_config(config_path: str | Path) -> dict[str, Any]:
    """Parse the BYOB YAML config; returns ``{}`` for empty/non-mapping payloads."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return cast(dict[str, Any], data) if isinstance(data, dict) else {}


_ValueT = TypeVar("_ValueT")


@overload
def resolve_dispatch_value(
    arg_value: _ValueT | None,
    yaml_dict: dict[str, Any],
    yaml_key: str,
    default: _ValueT,
) -> _ValueT: ...


@overload
def resolve_dispatch_value(
    arg_value: _ValueT | None,
    yaml_dict: dict[str, Any],
    yaml_key: str,
    default: None = None,
) -> _ValueT | None: ...


def resolve_dispatch_value(
    arg_value: _ValueT | None,
    yaml_dict: dict[str, Any],
    yaml_key: str,
    default: _ValueT | None = None,
) -> _ValueT | None:
    """Resolve CLI/YAML dispatch values without coupling to one CLI framework.

    A key present but explicitly null in the config falls back to the default, so a
    declared default is honored rather than dispatched as a missing value.
    """
    value = arg_value or yaml_dict.get(yaml_key) or default
    return cast("_ValueT | None", value)


def run_byob(
    *,
    config: str | Path,
    stage: StageName,
    family: str = "mcq",
    skip_until: str | None = None,
) -> Path | BenchmarkRunResult | None:
    """Run one BYOB stage for a benchmark family."""
    try:
        spec = get_family(family)
    except ValueError as exc:
        raise ByobDispatchError(str(exc)) from exc
    config_path = Path(config)

    if stage == "all":
        # A generation resume must validate and restore its own checkpoint before
        # any prepare hook can invalidate mutable stage caches.
        if skip_until is None or family != "bfcl":
            spec.prepare_data(config_path)
        return spec.generate(config_path, skip_until=skip_until)
    if stage == "prepare":
        return spec.prepare_data(config_path)
    if stage == "generate":
        return spec.generate(config_path, skip_until=skip_until)
    if stage == "translate":
        if spec.translate is None:
            raise ByobDispatchError(
                f"Benchmark family {family!r} does not define translation"
            )
        return spec.translate(config_path, skip_until=skip_until)
    if stage == "eval":
        if skip_until is not None:
            raise ByobDispatchError(
                "skip_until is generation-only and cannot be used by eval"
            )
        if spec.evaluate is None:
            raise ByobDispatchError(
                f"Benchmark family {family!r} does not define evaluation"
            )
        return spec.evaluate(config_path)

    raise ByobDispatchError(f"Unknown BYOB stage {stage!r}")
