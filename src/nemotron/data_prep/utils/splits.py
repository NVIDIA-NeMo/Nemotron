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

    Note:
        Shards are pooled globally before selection, so with the common
        one-shard valid/test settings those splits usually contain a single
        source and cannot realize a weighted mixture. Blend weights are
        primarily honored for the train split (see
        realize_packed_shards_into_split_dirs); size valid/test in whole
        shards per source if a weighted mix matters there.

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


def _shard_row_count(fs, parquet_path: str) -> int:
    """Read a parquet shard's row count from its footer metadata (cheap)."""
    import pyarrow.parquet as pq

    with fs.open(parquet_path, "rb") as f:
        return pq.ParquetFile(f).metadata.num_rows


def _select_weighted_shards(
    fs,
    pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict[str, dict[str, float]]]:
    """Select a shard subset whose row mix matches the blend weights.

    Every positive-weight source is fully packed into shards upstream, so a
    uniform view of all shards realizes a mix proportional to raw source sizes
    and silently ignores fractional weights (NVIDIA-NeMo/Megatron-Bridge#5080).

    This selects, per source, shards totalling ~``t_i * T`` rows, where ``t_i``
    is the source's normalized weight and ``T = min_i(rows_i / t_i)`` is the
    largest total that keeps every source within its available rows. The
    highest weight-to-size source keeps (nearly) everything; others are
    subsampled at shard granularity. Shards are taken in the already seeded,
    shuffled order from distribute_shards_to_splits(), so selection is
    deterministic.

    Missing shard files fail loudly here: silently renormalizing a weighted
    blend over a partially missing dataset would quietly train on the wrong
    mixture.

    Args:
        fs: Filesystem abstraction for reading shard footers.
        pairs: (weight, shard_path) tuples for one split, shard_path WITHOUT
            the .parquet suffix.

    Returns:
        (selected_pairs, per_dataset_stats) where stats map dataset name to
        {"weight", "rows_total", "rows_kept", "shards_total", "shards_kept"}.

    Raises:
        FileNotFoundError: If any shard file referenced by the blend is missing.
        ValueError: If a dataset's shards carry inconsistent weights.
    """
    from collections import defaultdict

    # Group shards by source dataset (path layout: .../datasets/<name>/<hash>/shard_N)
    by_dataset: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    weights: dict[str, float] = {}
    missing: list[str] = []
    for weight, shard_path in pairs:
        parquet_str = f"{shard_path}.parquet"
        if not fs.exists(parquet_str):
            missing.append(parquet_str)
            continue
        dataset = Path(parquet_str).parent.parent.name
        numeric_weight = float(weight)
        by_dataset[dataset].append((weight, shard_path, _shard_row_count(fs, parquet_str)))
        if dataset in weights and weights[dataset] != numeric_weight:
            raise ValueError(
                f"Inconsistent blend weights for dataset '{dataset}': {weights[dataset]} vs {numeric_weight}"
            )
        weights[dataset] = numeric_weight

    if missing:
        raise FileNotFoundError(
            f"Weighted blend realization: {len(missing)} shard file(s) missing; "
            f"refusing to silently renormalize the blend. First few: {missing[:5]}"
        )

    total_weight = sum(weights.values())
    if not by_dataset or total_weight <= 0:
        return list(pairs), {}

    targets = {name: w / total_weight for name, w in weights.items()}
    rows = {name: sum(r for _, _, r in shards) for name, shards in by_dataset.items()}
    # Largest achievable total that respects every source's target share
    total_rows = min(rows[name] / targets[name] for name in by_dataset if targets[name] > 0)

    selected: list[tuple[str, str]] = []
    stats: dict[str, dict[str, float]] = {}
    for name, shards in by_dataset.items():
        target_rows = targets[name] * total_rows
        kept_rows = 0
        kept: list[tuple[str, str]] = []
        for weight, shard_path, shard_rows in shards:
            # Stop at the shard boundary closest to the target: only add when
            # it moves the kept total nearer the target (always keep >= 1).
            if kept and abs(kept_rows + shard_rows - target_rows) >= abs(kept_rows - target_rows):
                break
            kept.append((weight, shard_path))
            kept_rows += shard_rows
        selected.extend(kept)
        stats[name] = {
            "weight": weights[name],
            "rows_total": rows[name],
            "rows_kept": kept_rows,
            "shards_total": len(shards),
            "shards_kept": len(kept),
        }
    return selected, stats


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

    When the blend uses non-uniform fractional weights, each split's shard
    selection is subsampled per source so the realized row mix matches the
    normalized weights (see NVIDIA-NeMo/Megatron-Bridge#5080). Uniform-weight
    blends keep every shard, preserving the previous behavior.

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
        pairs = [(path_list[i], path_list[i + 1]) for i in range(0, len(path_list) - 1, 2)]

        # Honor fractional blend weights: with non-uniform weights, subsample
        # shards per source so the realized mix matches the normalized weights.
        # Uniformity is decided numerically ("1" and "1.0" are the same weight).
        try:
            distinct_weights = {float(weight) for weight, _ in pairs}
        except ValueError as e:
            raise ValueError(f"Split '{split_name}': non-numeric blend weight: {e}") from e
        if any(w < 0 or w != w or w == float("inf") for w in distinct_weights):
            raise ValueError(
                f"Split '{split_name}': blend weights must be finite and non-negative, got {sorted(distinct_weights)}"
            )
        if len(distinct_weights) > 1:
            pairs, weight_stats = _select_weighted_shards(fs, pairs)
            for name, s in sorted(weight_stats.items()):
                logger.info(
                    f"Split '{split_name}' blend weighting: dataset '{name}' "
                    f"(weight={s['weight']}) kept {s['shards_kept']}/{s['shards_total']} shards "
                    f"({s['rows_kept']}/{s['rows_total']} rows)"
                )
                if s["shards_kept"] < s["shards_total"]:
                    logger.warning(
                        f"Split '{split_name}': dataset '{name}' subsampled to honor "
                        f"blend weight {s['weight']} "
                        f"({s['shards_total'] - s['shards_kept']} shards excluded)"
                    )

        shard_paths = [shard_path for _, shard_path in pairs]

        created_count = 0
        missing_paths = []

        for position, shard_path in enumerate(shard_paths):
            # Shard path is a prefix like /path/to/shard_000000
            # Actual file is shard_000000.parquet
            parquet_path_str = f"{shard_path}.parquet"
            parquet_path = Path(parquet_path_str)

            # Use filesystem abstraction for existence check (works on Lustre, S3, etc.)
            if not fs.exists(parquet_path_str):
                missing_paths.append(parquet_path_str)
                logger.warning(f"Shard file not found: {parquet_path_str}")
                continue

            # ``{position:06d}`` preserves the shuffle (it drives the loader's sort order);
            # ``{dataset_name}`` keeps links unique across datasets, since every dataset
            # names its shards shard_000000.parquet... (finalize.py prefix
            # ".../<name>/<hash>/shard"). Layout: .../datasets/<name>/<plan_hash>/shard_*.parquet
            # -> parent.parent.name is the dataset name.
            dataset_name = parquet_path.parent.parent.name
            link_path = split_dir / f"{position:06d}__{dataset_name}__{parquet_path.name}"

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
