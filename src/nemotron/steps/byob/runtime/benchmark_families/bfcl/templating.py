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

"""One placeholder substitution rule shared by surface text and call arguments.

Rendering a turn and binding an argument must agree on what ``{slot}`` means, or a
conversation ends up describing a value its gold call never used. Substitution is a
single pass: a value that itself contains braces is inserted verbatim rather than
rescanned, so pack data can hold JSON or ``{`` without being re-substituted or
mistaken for an unbound placeholder.
"""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PlaceholderError(ValueError):
    """Raised when text references a name the caller did not bind."""


def placeholder_names(text: str) -> list[str]:
    """Return the names ``text`` references, in order of appearance."""
    return PLACEHOLDER.findall(text)


def sole_placeholder(text: str) -> str | None:
    """Return the name when ``text`` is exactly one placeholder, else ``None``.

    Callers that need the bound value's own type (an integer argument, say) use this
    to skip stringification.
    """
    match = PLACEHOLDER.fullmatch(text)
    return match.group(1) if match else None


def substitute(text: str, values: dict[str, Any], *, what: str = "text") -> str:
    """Replace every ``{name}`` with its bound value in one pass."""
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            missing.append(name)
            return match.group(0)
        return str(values[name])

    rendered = PLACEHOLDER.sub(replace, text)
    if missing:
        raise PlaceholderError(
            f"{what} references unbound {', '.join(sorted(set(missing)))} in {text!r}"
        )
    return rendered
