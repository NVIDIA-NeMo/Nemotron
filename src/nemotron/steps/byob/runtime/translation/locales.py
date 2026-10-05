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

"""Small, dependency-free locale normalization for BYOB translation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LOCALE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_CUSTOM = {"hinglish"}
_SCRIPT_PATTERNS = {
    "arabic": re.compile(r"[\u0600-\u06ff]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
    "greek": re.compile(r"[\u0370-\u03ff]"),
    "han": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]"),
    "hangul": re.compile(r"[\uac00-\ud7af]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "japanese": re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"),
    "thai": re.compile(r"[\u0e00-\u0e7f]"),
}
_PRIMARY_SCRIPTS = {
    "ar": "arabic",
    "el": "greek",
    "he": "hebrew",
    "hi": "devanagari",
    "ja": "japanese",
    "ko": "hangul",
    "ru": "cyrillic",
    "th": "thai",
    "zh": "han",
}
SUPPORTED_SCRIPTS = frozenset(_SCRIPT_PATTERNS)


@dataclass(frozen=True)
class LocaleTag:
    canonical: str
    primary: str
    file_slug: str


def normalize_locale(value: str) -> LocaleTag:
    """Validate a BCP-47-shaped tag and return stable comparison forms."""
    normalized = value.strip().replace("_", "-")
    if normalized.casefold() in _CUSTOM:
        primary = normalized.casefold()
        return LocaleTag(primary, primary, primary)
    if not _LOCALE.fullmatch(normalized):
        raise ValueError(f"unsupported locale tag: {value!r}")
    parts = normalized.split("-")
    primary = parts[0].lower()
    canonical_parts = [primary]
    for part in parts[1:]:
        canonical_parts.append(part.upper() if len(part) == 2 and part.isalpha() else part)
    canonical = "-".join(canonical_parts)
    return LocaleTag(
        canonical=canonical,
        primary=primary,
        file_slug="-".join(part.lower() for part in canonical_parts),
    )


def expected_script(primary_language: str) -> str | None:
    return _PRIMARY_SCRIPTS.get(primary_language.casefold())


def contains_script(text: str, script: str) -> bool:
    try:
        pattern = _SCRIPT_PATTERNS[script]
    except KeyError as exc:
        raise ValueError(f"unsupported script: {script!r}") from exc
    return pattern.search(text) is not None


__all__ = [
    "LocaleTag",
    "SUPPORTED_SCRIPTS",
    "contains_script",
    "expected_script",
    "normalize_locale",
]
