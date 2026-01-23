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

"""Utilities for distributing shards into train/valid/test splits."""

from __future__ import annotations

import random


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

    The data_paths format is: ["weight", "path", "weight", "path", ...]
    where paths are shard prefixes (e.g., /path/to/shard).

    Output format compatible with Megatron-Bridge's per_split_data_args_path:
    {"train": ["weight", "path_0000", ...], "valid": [...], "test": [...]}

    Args:
        data_paths: Megatron-Bridge format path list ["weight", "path", ...]
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
