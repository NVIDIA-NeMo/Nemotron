#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic NeMo Gym rows for counting colored stars."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (133, 153, 0),
    "yellow": (181, 137, 0),
    "purple": (108, 113, 196),
    "orange": (203, 75, 22),
    "cyan": (42, 161, 152),
    "pink": (211, 54, 130),
}

AGENT_REF = {
    "type": "responses_api_agents",
    "name": "circle_count_simple_agent",
}

SYSTEM_PROMPT = (
    "You are a visual assistant. Count the number of stars of the specified "
    "color in the image. Output your final answer in \\boxed{} format, e.g. "
    "\\boxed{3}."
)


def _place_stars(
    count: int, canvas_size: int, radius: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Place non-overlapping stars, failing instead of silently dropping one."""
    margin = radius + 10
    positions: list[tuple[int, int]] = []
    minimum_distance = 2 * radius + 12
    for _ in range(count):
        for _ in range(2_000):
            point = (
                rng.randint(margin, canvas_size - margin),
                rng.randint(margin, canvas_size - margin),
            )
            if all(math.dist(point, other) > minimum_distance for other in positions):
                positions.append(point)
                break
        else:
            raise RuntimeError(
                f"could not place {count} stars on a {canvas_size}px canvas "
                f"with radius {radius}"
            )
    return positions


def _star_vertices(x: int, y: int, radius: int) -> list[tuple[float, float]]:
    inner_radius = radius * 0.42
    vertices = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        point_radius = radius if index % 2 == 0 else inner_radius
        vertices.append(
            (x + point_radius * math.cos(angle), y + point_radius * math.sin(angle))
        )
    return vertices


def _render(stars: list[dict[str, Any]], canvas_size: int) -> str:
    image = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for star in stars:
        draw.polygon(
            _star_vertices(star["x"], star["y"], star["radius"]),
            fill=COLORS[star["color"]],
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def make_example(
    seed: int,
    canvas_size_range: tuple[int, int],
    star_radius_range: tuple[int, int],
    num_stars_range: tuple[int, int],
    num_colors_range: tuple[int, int],
) -> dict[str, Any]:
    rng = random.Random(seed)
    canvas_size = rng.randint(*canvas_size_range)
    radius = rng.randint(*star_radius_range)
    num_stars = rng.randint(*num_stars_range)
    num_colors = min(num_stars, rng.randint(*num_colors_range))
    palette = rng.sample(list(COLORS), num_colors)

    # Include every selected color at least once so the target answer is positive.
    color_names = palette + [rng.choice(palette) for _ in range(num_stars - num_colors)]
    rng.shuffle(color_names)
    target_color = rng.choice(palette)
    positions = _place_stars(num_stars, canvas_size, radius, rng)
    stars = [
        {"x": x, "y": y, "radius": radius, "color": color}
        for (x, y), color in zip(positions, color_names, strict=True)
    ]

    # The NeMo Gym verifier calls this field `circles`, but only reads
    # each item's color. Retaining that wire-format key lets the existing
    # circle_count_simple_agent score star images without a custom service.
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": _render(stars, canvas_size),
                            "detail": "auto",
                        },
                        {
                            "type": "input_text",
                            "text": f"How many {target_color} stars are in the image?",
                        },
                    ],
                },
            ],
        },
        "circles": stars,
        "target_color": target_color,
        "agent_ref": dict(AGENT_REF),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1_024)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--canvas-size-min", type=int, default=800)
    parser.add_argument("--canvas-size-max", type=int, default=1_200)
    parser.add_argument("--radius-min", type=int, default=24)
    parser.add_argument("--radius-max", type=int, default=48)
    parser.add_argument("--num-stars-min", type=int, default=1)
    parser.add_argument("--num-stars-max", type=int, default=30)
    parser.add_argument("--num-colors-min", type=int, default=2)
    parser.add_argument("--num-colors-max", type=int, default=4)
    args = parser.parse_args()

    ranges = {
        "canvas size": (args.canvas_size_min, args.canvas_size_max),
        "star radius": (args.radius_min, args.radius_max),
        "number of stars": (args.num_stars_min, args.num_stars_max),
        "number of colors": (args.num_colors_min, args.num_colors_max),
    }
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    for name, (minimum, maximum) in ranges.items():
        if minimum <= 0 or minimum > maximum:
            raise ValueError(f"invalid {name} range: {minimum}..{maximum}")
    if args.num_colors_max > len(COLORS):
        raise ValueError(
            f"--num-colors-max cannot exceed the {len(COLORS)} available colors"
        )
    if args.canvas_size_min <= 2 * (args.radius_max + 10):
        raise ValueError("the minimum canvas is too small for the maximum radius")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for index in range(args.num_samples):
            row = make_example(
                seed=args.seed_offset + index,
                canvas_size_range=(args.canvas_size_min, args.canvas_size_max),
                star_radius_range=(args.radius_min, args.radius_max),
                num_stars_range=(args.num_stars_min, args.num_stars_max),
                num_colors_range=(args.num_colors_min, args.num_colors_max),
            )
            output.write(json.dumps(row) + "\n")

    print(f"Generated {args.num_samples} star-count rows: {args.out}")


if __name__ == "__main__":
    main()
