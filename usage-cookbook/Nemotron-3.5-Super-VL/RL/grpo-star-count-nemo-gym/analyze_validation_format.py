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


# Match the circle-count verifier exactly: no whitespace, sign, or decimal.
VERIFIER_BOXED_INTEGER = re.compile(r"\\boxed\{\d+\}")


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
                # These fields and singleton dimensions follow the validation
                # logger schema on the documented NeMo RL branch.
                messages = _unwrap_singleton(row["content"])
                assistant_messages = [
                    _content_text(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "assistant"
                ]
                assistant_text = assistant_messages[-1] if assistant_messages else ""
                reward = _unwrap_singleton(row["rewards"])
                if isinstance(reward, list):
                    reward = reward[0]
                total += 1
                boxed += bool(VERIFIER_BOXED_INTEGER.search(assistant_text))
                correct += float(reward) == 1.0

        if total == 0:
            raise ValueError(f"no validation rows found in {log_path}")
        conditional = f"{correct}/{boxed} ({correct / boxed:.2%})" if boxed else "n/a"
        print(
            f"{log_path.name}: correct={correct}/{total} ({correct / total:.2%}), "
            f"boxed={boxed}/{total} ({boxed / total:.2%}), "
            f"correct_given_boxed={conditional}"
        )


if __name__ == "__main__":
    main()
