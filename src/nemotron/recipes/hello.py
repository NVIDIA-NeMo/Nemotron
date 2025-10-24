# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""
Tutorial Part 1: Data Preparation

Demonstrates creating a dataset artifact with validation.

Usage:
    uv run python examples/hello.py
    uv run python examples/hello.py --num-samples 500
    uv run python examples/hello.py --output-dir /tmp/custom_data
"""

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pydantic import Field

from nemotron.artifact import Artifact, print_complete


@dataclass
class Config:
    """Configuration for data preparation."""

    output_dir: Path = Path("/tmp/tutorial_data")
    num_samples: int = 1000
    seed: int = 42


class Dataset(Artifact):
    """Output: A prepared dataset."""

    num_examples: int = Field(gt=0)
    train_path: Path


def prepare_data(num_samples: int, seed: int) -> list[dict[str, str]]:
    """Generate synthetic training data.

    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of training examples
    """
    random.seed(seed)
    data = []

    for i in range(num_samples):
        data.append(
            {
                "id": i,
                "text": f"Sample text {i}: "
                + " ".join(random.choices(["hello", "world", "example"], k=10)),
                "label": random.choice(["positive", "negative", "neutral"]),
            }
        )

    return data


def save_data(data: list[dict], path: Path) -> None:
    """Save data to JSON file.

    Args:
        data: List of examples
        path: Output path
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(config: Config) -> Dataset:
    """Prepare and save training data.

    Args:
        config: Configuration with output directory and parameters

    Returns:
        Dataset artifact
    """
    # 1. Setup
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare data
    print(f"Generating {config.num_samples:,} samples...", file=sys.stderr)
    data = prepare_data(config.num_samples, seed=config.seed)

    # 3. Save data file
    train_path = config.output_dir / "train.json"
    save_data(data, train_path)
    print(f"Saved data to {train_path}", file=sys.stderr)

    # 4. Create and save artifact
    artifact = Dataset(
        path=config.output_dir,
        num_examples=len(data),
        train_path=train_path,
        metrics={"num_examples": len(data)},
    )
    artifact.save()
    print_complete({"data_prep": artifact})
    return artifact


if __name__ == "__main__":
    tyro.cli(main)
