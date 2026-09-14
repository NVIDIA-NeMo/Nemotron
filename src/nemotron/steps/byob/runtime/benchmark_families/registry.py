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

"""Registry for BYOB benchmark families."""

from __future__ import annotations

from .base import BenchmarkFamilySpec
from .bfcl.family import SPEC as BFCL_SPEC
from .mcq.family import SPEC as MCQ_SPEC

_REGISTRY: dict[str, BenchmarkFamilySpec] = {
    BFCL_SPEC.name: BFCL_SPEC,
    MCQ_SPEC.name: MCQ_SPEC,
}


def get_family(name: str) -> BenchmarkFamilySpec:
    """Return a registered benchmark family by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown BYOB benchmark family {name!r}. Available: {sorted(_REGISTRY)}") from exc


def list_families() -> list[str]:
    """Return registered benchmark family names."""
    return sorted(_REGISTRY)
