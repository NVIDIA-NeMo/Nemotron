# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven language and script quality helpers."""

from __future__ import annotations

import re
from typing import Any


def compile_script_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a configured single-character script matcher."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid script_pattern {pattern!r}: {exc}") from exc
    if compiled.fullmatch(""):
        raise ValueError("script_pattern must not match an empty string")
    return compiled


def script_fraction(value: str, script_pattern: str) -> float:
    """Return the fraction of alphabetic characters matching a configured script."""
    pattern = compile_script_pattern(script_pattern)
    letters = sum(character.isalpha() for character in value)
    target_letters = sum(character.isalpha() and bool(pattern.fullmatch(character)) for character in value)
    return target_letters / letters if letters else 0.0


def validate_fraction(config: dict[str, Any], path: str) -> tuple[float, float]:
    """Validate and return a configured inclusive script-fraction range."""
    if "min" not in config or "max" not in config:
        raise ValueError(f"{path} requires min and max")
    minimum, maximum = float(config["min"]), float(config["max"])
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError(f"{path} must satisfy 0 <= min <= max <= 1")
    return minimum, maximum


def answer_instruction(
    *,
    question_language: str,
    answer_label: str,
    reasoning_language: str,
) -> str:
    """Build a language-independent answer contract from config values."""
    return (
        f"Answer the following multiple-choice question, which is written in {question_language}. "
        f"Explain your reasoning step by step in {reasoning_language}. "
        "The last line of your response must contain only "
        f"'{answer_label}: $LETTER' (without quotes), where $LETTER is one of A, B, C, or D."
    )
