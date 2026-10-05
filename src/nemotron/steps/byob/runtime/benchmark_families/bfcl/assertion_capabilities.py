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

"""The one reading of a pack's declared assertion capabilities.

Validation and executable projection both need to know what an assertion is
compatible with and which metric it feeds. They read it here, from the literal
assignment in the pack's ``assertions.py``, so a pack cannot be admitted by one
stage and refused by the other. Requiring a literal is what lets the reading
happen without importing pack code.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

ASSERTION_CATEGORIES: Final = (
    "state",
    "path",
    "result",
    "final_answer",
    "unclassified",
)
CAPABILITY_FIELDS: Final = ("trace", "executable", "category")
DEFAULT_ASSERTION_CAPABILITY: Final = {
    "trace": False,
    "executable": True,
    "category": "unclassified",
}
ASSERTION_CAPABILITIES_SYMBOL: Final = "ASSERTION_CAPABILITIES"


class AssertionCapabilityError(ValueError):
    """A pack's declared assertion capabilities do not follow the contract."""


def read_literal_assertion_capabilities(path: Path) -> dict[str, Any] | None:
    """Return the literal capability mapping, or ``None`` when none is declared."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise AssertionCapabilityError(
            f"{path.name} cannot be read: {type(exc).__name__}: {exc}"
        ) from exc
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and target.id == ASSERTION_CAPABILITIES_SYMBOL
            and value is not None
        ):
            try:
                declared = ast.literal_eval(value)
            except ValueError as exc:
                raise AssertionCapabilityError(
                    f"{ASSERTION_CAPABILITIES_SYMBOL} is computed rather than "
                    f"declared literally: {exc}"
                ) from exc
            if not isinstance(declared, dict):
                raise AssertionCapabilityError(
                    f"{ASSERTION_CAPABILITIES_SYMBOL} is not a mapping"
                )
            return declared
    return None


def normalized_assertion_capability(name: str, raw: Any) -> dict[str, Any]:
    """Validate one declared capability and fill the contract's defaults."""
    if not isinstance(raw, dict):
        raise AssertionCapabilityError(f"{name} does not declare a capability object")
    if unknown := sorted(set(raw) - set(CAPABILITY_FIELDS)):
        raise AssertionCapabilityError(f"{name} declares unknown field(s) {unknown}")
    capability = {**DEFAULT_ASSERTION_CAPABILITY, **raw}
    for field in ("trace", "executable"):
        if not isinstance(capability[field], bool):
            raise AssertionCapabilityError(f"{name} declares a non-boolean {field}")
    if capability["category"] not in ASSERTION_CATEGORIES:
        raise AssertionCapabilityError(
            f"{name} declares category {capability['category']!r}, "
            f"which is not one of {list(ASSERTION_CATEGORIES)}"
        )
    return capability


def assertion_capabilities(
    declared: dict[str, Any] | None,
    names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Resolve each named assertion's capability against the pack's declaration.

    Names the pack never declares fall back to the contract's defaults. Whether
    the declaration itself names something that is not an assertion is a
    whole-pack question, so validation asks it rather than this helper.
    """
    mapping = declared or {}
    return {
        name: normalized_assertion_capability(name, mapping.get(name, {}))
        for name in names
    }


__all__ = [
    "ASSERTION_CAPABILITIES_SYMBOL",
    "ASSERTION_CATEGORIES",
    "AssertionCapabilityError",
    "CAPABILITY_FIELDS",
    "DEFAULT_ASSERTION_CAPABILITY",
    "assertion_capabilities",
    "normalized_assertion_capability",
    "read_literal_assertion_capabilities",
]
