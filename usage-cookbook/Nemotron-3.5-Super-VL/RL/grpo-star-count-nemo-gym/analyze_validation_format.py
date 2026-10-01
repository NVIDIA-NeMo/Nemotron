#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report boxed-answer coverage from NeMo RL validation JSONL logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BOXED_INTEGER = re.compile(r"\\boxed\{\d+\}")


def _unwrap_singleton(value: Any) -> Any:
    """Remove singleton batch dimensions written by the NeMo RL logger."""
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def _content_text(content: Any) -> str:
    """Convert an assistant content value to searchable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", part.get("content", "")))
            if isinstance(part, dict)
            else str(part)
            for part in content
        )
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()

    for log_path in args.logs:
        total = 0
        boxed = 0
        correct = 0
        with log_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                row = json.loads(line)
                messages = _unwrap_singleton(row["content"])
                assistant_text = "\n".join(
                    _content_text(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "assistant"
                )
                reward = _unwrap_singleton(row["rewards"])
                if isinstance(reward, list):
                    reward = reward[0]
                total += 1
                boxed += bool(BOXED_INTEGER.search(assistant_text))
                correct += float(reward) == 1.0

        if total == 0:
            raise ValueError(f"no validation rows found in {log_path}")
        print(
            f"{log_path.name}: correct={correct}/{total} ({correct / total:.2%}), "
            f"boxed={boxed}/{total} ({boxed / total:.2%})"
        )


if __name__ == "__main__":
    main()
