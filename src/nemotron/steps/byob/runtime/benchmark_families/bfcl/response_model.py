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

"""Structured model responses used by BFCL surface-only stages."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_quality_contract import (
    SurfaceJudgeResult,
)

__all__ = [
    "ParaphraseResult",
    "ParaphraseVariant",
    "ReferenceProfileResult",
    "SurfaceJudgeResult",
]


class ReferenceProfileResult(BaseModel):
    style_hints: list[str] = Field(
        description="Concise style rules observed in the supplied reference conversations"
    )
    avoid: list[str] = Field(
        default_factory=list,
        description="Surface-writing patterns absent from or discouraged by the references",
    )


class ParaphraseVariant(BaseModel):
    user_turns: list[str] = Field(
        description="Ordered rewrites, one for each canonical user turn"
    )


class ParaphraseResult(BaseModel):
    variants: list[ParaphraseVariant] = Field(
        description="Ordered conversation variants; no explanations or metadata"
    )
