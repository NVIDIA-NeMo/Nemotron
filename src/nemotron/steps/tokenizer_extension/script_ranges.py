#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unicode ranges that identify a language's tokens in a vocabulary.

Shared by the extend step (which prunes these tokens for method=replace) and
the init_embeddings step (which uses the same set to find the base model's
existing target-language rows). One source of truth, so adding a language is a
single edit here plus a LanguageProfile in languages.py.

A token "belongs to" a language if any character of its *decoded* surface falls
in one of these ranges. The byte alphabet decodes into none of them and so is
never captured -- which is what lets a pruned script be rebuilt from bytes.
"""

from __future__ import annotations

SCRIPT_UNICODE_RANGES: dict[str, list[tuple[int, int]]] = {
    "devanagari": [(0x0900, 0x097F)],  # Hindi, Marathi, Sanskrit, Nepali, ...
    "bengali": [(0x0980, 0x09FF)],  # Bengali, Assamese
    "gurmukhi": [(0x0A00, 0x0A7F)],  # Punjabi
    "gujarati": [(0x0A80, 0x0AFF)],
    "oriya": [(0x0B00, 0x0B7F)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "arabic": [(0x0600, 0x06FF)],  # Urdu, Sindhi, Kashmiri
    # Vietnamese is Latin-script, so it has no block of its own. These are the
    # codepoints that are effectively Vietnamese-only: the precomposed
    # vowel+tone forms, the horned o/u, and the combining horn. A token is
    # residual-Vietnamese if its decoded surface contains any of them. The byte
    # alphabet does not decode into these ranges and so survives the prune,
    # which is what lets the fresh merges rebuild Vietnamese from bytes.
    "vietnamese": [(0x1EA0, 0x1EFF), (0x01A0, 0x01A1), (0x01AF, 0x01B0), (0x031B, 0x031B)],
    # Same, plus d-stroke and a-breve. Those two are shared with other
    # languages (Romanian, Sami), so pruning them reaches slightly wider.
    "vietnamese_broad": [
        (0x1EA0, 0x1EFF),
        (0x01A0, 0x01A1),
        (0x01AF, 0x01B0),
        (0x031B, 0x031B),
        (0x0110, 0x0111),
        (0x0102, 0x0103),
    ],
}


def resolve_ranges(names: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for name in names:
        key = name.strip().lower()
        if key not in SCRIPT_UNICODE_RANGES:
            raise ValueError(f"Unknown script {name!r}. Choose from: {sorted(SCRIPT_UNICODE_RANGES)}")
        ranges.extend(SCRIPT_UNICODE_RANGES[key])
    return ranges


# --------------------------------------------------------------------------- #
# Process-wide target script.
#
# The init_embeddings code needs to find the base model's existing rows for the
# target language -- they anchor the norm correction and the FOCUS candidate
# pool. That used to be hardcoded to Devanagari. It now defaults to Devanagari,
# so every existing Hindi config behaves identically, and any other language
# calls set_target_script() once at startup.
# --------------------------------------------------------------------------- #
_TARGET: list[tuple[int, int]] = list(SCRIPT_UNICODE_RANGES["devanagari"])
_TARGET_NAME = "devanagari"


def set_target_script(name: str) -> None:
    global _TARGET, _TARGET_NAME
    key = (name or "").strip().lower()
    if key not in SCRIPT_UNICODE_RANGES:
        raise ValueError(f"Unknown script {name!r}. Choose from: {sorted(SCRIPT_UNICODE_RANGES)}")
    _TARGET, _TARGET_NAME = list(SCRIPT_UNICODE_RANGES[key]), key


def target_script_name() -> str:
    return _TARGET_NAME


def is_target(text: str) -> bool:
    """True if any character of `text` belongs to the active target script."""
    if not text:
        return False
    return any(any(lo <= ord(ch) <= hi for lo, hi in _TARGET) for ch in text)
