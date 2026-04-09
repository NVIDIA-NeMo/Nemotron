# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for distributing shards into train/valid/test splits.

Provides:
- distribute_shards_to_splits: Partition shards into train/valid/test
- realize_packed_shards_into_split_dirs: Create canonical split directories with symlinks
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from nemotron.data_prep.utils.filesystem import get_filesystem

logger = logging.getLogger(__name__)


def distribute_shards_to_splits(
    data_paths: list[str],
    num_shards: int,
    *,
    valid_shards: int = 1,
    test_shards: int = 1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Distribute shard paths into train/valid/test splits.

    Collects all shards from all datasets into a pool, then randomly selects
    shards for test and valid splits. The remaining shards go to train.

    The data_paths format is: ["weight", "prefix", "weight", "prefix", ...]
    where each prefix is a shard base path WITHOUT the index suffix.
    Example: "/path/to/runs/abc/datasets/mydata/hash/shard"

    This function appends "_{shard_idx:06d}" to each prefix to create per-shard
    paths. For example, with num_shards=3:
        Input prefix: "/path/shard"
        Output paths: "/path/shard_000000", "/path/shard_000001", "/path/shard_000002"

    Note: The actual files have a .parquet extension (e.g., shard_000000.parquet).
    The output paths here are base names; realize_packed_shards_into_split_dirs()
    appends ".parquet" when creating symlinks.

    Output format compatible with Megatron-Bridge's per_split_data_args_path:
    {"train": ["weight", "path_000000", ...], "valid": [...], "test": [...]}

    Args:
        data_paths: Megatron-Bridge format path list ["weight", "prefix", ...]
            where prefix is the shard base path (see FormatResult.data_paths)
        num_shards: Total number of shards per dataset
        valid_shards: Number of shards for validation (total, not per-dataset)
        test_shards: Number of shards for test (total, not per-dataset)
        seed: Random seed for reproducible shard selection

    Returns:
        Dict with "train", "valid", "test" keys containing data_paths lists
    """
    # Parse weight/path pairs from data_paths
    # Format: ["1.0", "/path/dataset1/shard", "0.5", "/path/dataset2/shard", ...]
    pairs = []
    for i in range(0, len(data_paths), 2):
        if i + 1 < len(data_paths):
            weight = data_paths[i]
            prefix = data_paths[i + 1]
            pairs.append((weight, prefix))

    # Collect ALL shards from ALL datasets into one pool
    # Each entry is (weight, shard_path) where shard_path has the _XXXX suffix
    all_shards: list[tuple[str, str]] = []
    for weight, prefix in pairs:
        for shard_idx in range(num_shards):
            all_shards.append((weight, f"{prefix}_{shard_idx:06d}"))

    # Use seeded RNG for reproducibility
    rng = random.Random(seed)

    # Randomly select shards for test and valid
    # Ensure we don't request more shards than available
    total_shards = len(all_shards)
    actual_test_shards = min(test_shards, total_shards)
    remaining_after_test = total_shards - actual_test_shards
    actual_valid_shards = min(valid_shards, remaining_after_test)

    # Shuffle and partition
    shuffled = all_shards.copy()
    rng.shuffle(shuffled)

    test_selection = shuffled[:actual_test_shards]
    valid_selection = shuffled[actual_test_shards : actual_test_shards + actual_valid_shards]
    train_selection = shuffled[actual_test_shards + actual_valid_shards :]

    # Convert back to flat list format ["weight", "path", "weight", "path", ...]
    def flatten(shard_pairs: list[tuple[str, str]]) -> list[str]:
        result: list[str] = []
        for weight, path in shard_pairs:
            result.append(weight)
            result.append(path)
        return result

    return {
        "train": flatten(train_selection),
        "valid": flatten(valid_selection),
        "test": flatten(test_selection),
    }


def realize_packed_shards_into_split_dirs(
    *,
    output_dir: Path,
    split_to_paths: dict[str, list[str]],
) -> dict[str, Path]:
    """Create canonical split directories with symlinks to packed shard files.

    Ensures packed shard files are accessible under:
        output_dir/splits/<split>/<basename>

    This enables training to consume split dirs/globs directly without
    parsing blend.json.

    Args:
        output_dir: Base output directory for the data prep run.
        split_to_paths: Dict from distribute_shards_to_splits() with format
            {"train": ["weight", "path", ...], "valid": [...], "test": [...]}

    Returns:
        Dict mapping split name to canonical split directory Path.
        {"train": output_dir/splits/train, "valid": ..., "test": ...}

    Raises:
        FileNotFoundError: If train split has no valid shard files.
    """
    splits_base = output_dir / "splits"
    result: dict[str, Path] = {}

    # Use filesystem abstraction for checking file existence on remote filesystems
    fs, _ = get_filesystem(str(output_dir))

    for split_name, path_list in split_to_paths.items():
        split_dir = splits_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        result[split_name] = split_dir

        # path_list format: ["weight", "path", "weight", "path", ...]
        # Extract just the paths (odd indices)
        shard_paths = [path_list[i] for i in range(1, len(path_list), 2)]

        created_count = 0
        missing_paths = []

        for shard_path in shard_paths:
            # Shard path is a prefix like /path/to/shard_000000
            # Actual file is shard_000000.parquet
            parquet_path_str = f"{shard_path}.parquet"
            parquet_path = Path(parquet_path_str)

            # Use filesystem abstraction for existence check (works on Lustre, S3, etc.)
            if not fs.exists(parquet_path_str):
                missing_paths.append(parquet_path_str)
                logger.warning(f"Shard file not found: {parquet_path_str}")
                continue

            # Create symlink in split dir
            link_path = split_dir / parquet_path.name

            if link_path.exists() or link_path.is_symlink():
                # Remove existing link/file to update
                link_path.unlink()

            try:
                # Use relative symlink if possible for portability
                rel_target = os.path.relpath(parquet_path, split_dir)
                link_path.symlink_to(rel_target)
                created_count += 1
            except OSError:
                # Fall back to absolute symlink if relative fails
                link_path.symlink_to(parquet_path.resolve())
                created_count += 1

        logger.info(f"Created split dir '{split_name}' with {created_count}/{len(shard_paths)} shards: {split_dir}")

        # Fail loudly if train split has no files - this is a critical error
        if split_name == "train" and created_count == 0 and len(shard_paths) > 0:
            raise FileNotFoundError(
                f"No parquet files found for train split. Expected {len(shard_paths)} shards. "
                f"Missing files: {missing_paths[:5]}{'...' if len(missing_paths) > 5 else ''}"
            )

    return result
