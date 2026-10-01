#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check star-count image tokens against the recipe sequence limit."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


def _as_int(value: object) -> int:
    """Convert a processor scalar to a Python integer."""
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4_096)
    parser.add_argument("--text-reserve", type=int, default=512)
    parser.add_argument("--response-reserve", type=int, default=256)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    image_processor = processor.image_processor
    tokens_by_size: dict[tuple[int, int], int] = {}
    row_count = 0

    for data_path in args.data:
        with data_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                row = json.loads(line)
                image_url = row["responses_create_params"]["input"][1]["content"][
                    0
                ]["image_url"]
                image = Image.open(
                    io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))
                ).convert("RGB")
                if image.size not in tokens_by_size:
                    processed = image_processor(images=[image], return_tensors=None)
                    tokens_by_size[image.size] = _as_int(processed["num_tokens"][0])
                row_count += 1

    if not tokens_by_size:
        raise ValueError("no dataset rows found")

    minimum = min(tokens_by_size.values())
    maximum = max(tokens_by_size.values())
    conservative_total = maximum + args.text_reserve + args.response_reserve
    print(f"Rows checked: {row_count}")
    print(f"Distinct image sizes: {len(tokens_by_size)}")
    print(f"Image-token range: {minimum}–{maximum}")
    print(
        "Conservative sequence budget: "
        f"{maximum} image + {args.text_reserve} text + "
        f"{args.response_reserve} response = {conservative_total}"
    )
    if conservative_total > args.max_sequence_length:
        raise ValueError(
            f"sequence budget {conservative_total} exceeds "
            f"limit {args.max_sequence_length}"
        )
    print(f"Sequence budget fits within {args.max_sequence_length} tokens.")


if __name__ == "__main__":
    main()
