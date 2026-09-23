# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict parsing and rendering for authored multiple-choice questions."""

from __future__ import annotations

import re


def index_to_letter(index: int) -> str:
    return chr(ord("A") + index)


def format_question(question: dict[str, object]) -> str:
    choices = question["choices"]
    if not isinstance(choices, list):
        raise TypeError("question choices must be a list")
    options = "\n".join(f"{index_to_letter(i)}) {choice}" for i, choice in enumerate(choices))
    return f"{question['question']}\n{options}"


def parse_question(response: str, *, num_options: int = 4) -> dict[str, object]:
    question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL | re.IGNORECASE)
    options_match = re.search(r"<options>(.*?)</options>", response, re.DOTALL | re.IGNORECASE)
    question = question_match.group(1).strip() if question_match else ""
    choices: list[str] = []
    if options_match:
        for index, raw_line in enumerate(options_match.group(1).strip().splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            letter = index_to_letter(index)
            match = re.match(rf"^{letter}[).]\s*(.+)$", line, re.IGNORECASE)
            if not match:
                choices = []
                break
            choices.append(match.group(1).strip())
    valid_choices = len(choices) == num_options and all(choices) and len(set(choices)) == num_options
    return {"question": question or None, "choices": choices or None, "success": bool(question and valid_choices)}
