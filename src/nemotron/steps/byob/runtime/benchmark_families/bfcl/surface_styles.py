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

"""Structural surface style axes and their deterministic per-binding assignment.

This module holds no configuration or stage dependency so that config validation
and the paraphrase stage can share one definition of the axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Asked to rewrite the same canonical turn for many fixture bindings, a model
# converges on one preferred phrasing, so wording diversity saturates at roughly one
# surface per template no matter how many bindings exist. These axes are the request
# for diversity: each binding is assigned one, so repeated bindings are asked for
# different sentence forms rather than the same rewrite. Since Stage 11 counts masked
# surfaces, the axis count is what bounds the scale a pack can publish.
#
# Every axis is structural or register-level on purpose. An axis that asked for
# shortened numbers, abbreviated identifiers, or converted units would rewrite
# protected values, and the must_preserve guard would then reject the variant.
SURFACE_STYLE_AXES: tuple[str, ...] = (
    "direct imperative request with no opener",
    "polite request phrased as a yes/no question",
    "state the goal first, then the identifying values",
    "state the identifying values first, then the goal",
    "terse phrasing that drops optional function words",
    "one sentence that also says why the user is asking",
    "two short sentences rather than one longer sentence",
    "formal register addressed to a service desk",
    "casual register addressed to a familiar agent",
    "brief courtesy greeting, then the request",
    "the request, then a brief closing thanks",
    "phrased as confirming something the user already believes",
    "phrased as an uncertainty the user wants resolved",
    "phrased as the next step after something the user just did",
    "lead with the relevant condition, then the request",
    "indirect request that asks whether the assistant is able to help",
    "phrased as a comparison or cross-check between the values involved",
    "phrased with mild urgency about needing the answer now",
    "phrased as a follow-up to an earlier unanswered attempt",
    "phrased as delegating the task and awaiting a report back",
)


def style_plan(
    task: Mapping[str, Any],
    requested: int,
    axes: Sequence[str] = SURFACE_STYLE_AXES,
) -> list[str]:
    """Assign one style axis per requested variant, deterministically per binding.

    The offset comes from the task seed, which already mixes the global seed with the
    binding identity, so the same config yields the same plan and two bindings of one
    template normally receive different axes. Consecutive axes keep the variants of a
    single binding distinct whenever fewer variants than axes are requested.
    """
    if not axes:
        raise ValueError("surface style axes must not be empty")
    offset = int(task["seed"]) % len(axes)
    return [axes[(offset + index) % len(axes)] for index in range(max(0, requested))]
