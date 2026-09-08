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

"""Interfaces for BYOB benchmark-family implementations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Machine-readable result returned by a family-specific CLI stage."""

    output_path: Path
    output_format: Literal["human", "json"]
    payload: dict[str, Any]

    def render(self) -> str:
        if self.output_format == "json":
            return json.dumps(self.payload, sort_keys=True)
        # Human lines stay one-per-key, but nested values are rendered as JSON so a
        # dict or None never reaches an operator as a Python repr.
        return "\n".join(
            f"{key}: {value if isinstance(value, str) else json.dumps(value, sort_keys=True)}"
            for key, value in self.payload.items()
        )


ConfigPath = str | os.PathLike[str]


# Stage hooks are called positionally by the dispatcher, so the parameter is
# positional-only here: a family is free to name its own argument.
class PrepareHook(Protocol):
    def __call__(self, config_path: ConfigPath, /) -> Path | None: ...


class GenerateHook(Protocol):
    def __call__(
        self, config_path: ConfigPath, /, *, skip_until: str | None = None
    ) -> Path | None: ...


class EvaluateHook(Protocol):
    def __call__(self, config_path: ConfigPath, /) -> Path | BenchmarkRunResult | None: ...


@dataclass(frozen=True)
class BenchmarkFamilySpec:
    """Named hooks that let agents add new benchmark families without rewriting orchestration."""

    name: str
    description: str
    prepare_data: PrepareHook
    generate: GenerateHook
    translate: GenerateHook | None = None
    evaluate: EvaluateHook | None = None
