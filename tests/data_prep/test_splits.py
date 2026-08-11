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

"""Tests for shard split distribution and weighted split realization.

Regression coverage for NVIDIA-NeMo/Megatron-Bridge#5080 (§2): fractional
data_blend.json weights must affect the realized blend proportionally rather
than being flattened to a uniform mix.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemotron.data_prep.utils.splits import (
    distribute_shards_to_splits,
    realize_packed_shards_into_split_dirs,
)


def _write_dataset_shards(root, dataset_name, num_shards, rows_per_shard):
    """Create parquet shards in the pipeline's .../datasets/<name>/<hash>/ layout."""
    shard_dir = root / "datasets" / dataset_name / "confighash"
    shard_dir.mkdir(parents=True)
    prefix = shard_dir / "shard"
    for idx in range(num_shards):
        table = pa.table({"input_ids": [[1, 2, 3]] * rows_per_shard, "src": [dataset_name] * rows_per_shard})
        pq.write_table(table, f"{prefix}_{idx:06d}.parquet")
    return str(prefix)


def _realized_rows_by_dataset(split_dir):
    """Count realized rows per source dataset in a split directory."""
    counts: dict[str, int] = {}
    for link in sorted(split_dir.iterdir()):
        dataset = link.name.split("__")[1]
        rows = pq.ParquetFile(link).metadata.num_rows
        counts[dataset] = counts.get(dataset, 0) + rows
    return counts


class TestDistributeShardsToSplits:
    def test_weights_preserved_in_output(self):
        data_paths = ["0.22", "/data/datasets/a/h/shard", "0.78", "/data/datasets/b/h/shard"]
        result = distribute_shards_to_splits(data_paths, num_shards=4, valid_shards=0, test_shards=0)
        train = result["train"]
        weights = {train[i] for i in range(0, len(train), 2)}
        assert weights == {"0.22", "0.78"}
        assert len(train) == 16  # 2 datasets * 4 shards * (weight, path)

    def test_split_counts(self):
        data_paths = ["1.0", "/data/datasets/a/h/shard"]
        result = distribute_shards_to_splits(data_paths, num_shards=10, valid_shards=2, test_shards=1)
        assert len(result["train"]) // 2 == 7
        assert len(result["valid"]) // 2 == 2
        assert len(result["test"]) // 2 == 1


class TestWeightedRealization:
    def test_uniform_weights_keep_all_shards(self, tmp_path):
        prefix_a = _write_dataset_shards(tmp_path, "alpha", 4, 10)
        prefix_b = _write_dataset_shards(tmp_path, "beta", 4, 10)
        blend = distribute_shards_to_splits(
            ["1.0", prefix_a, "1.0", prefix_b], num_shards=4, valid_shards=0, test_shards=0
        )
        out = tmp_path / "out"
        dirs = realize_packed_shards_into_split_dirs(output_dir=out, split_to_paths=blend)
        counts = _realized_rows_by_dataset(dirs["train"])
        assert counts == {"alpha": 40, "beta": 40}

    def test_fractional_weights_shape_the_mix(self, tmp_path):
        """MB#5080 repro: 0.22/0.78 must not realize as a 50/50 mix."""
        prefix_a = _write_dataset_shards(tmp_path, "alpha", 10, 10)  # 100 rows
        prefix_b = _write_dataset_shards(tmp_path, "beta", 10, 10)  # 100 rows
        blend = distribute_shards_to_splits(
            ["0.22", prefix_a, "0.78", prefix_b], num_shards=10, valid_shards=0, test_shards=0
        )
        out = tmp_path / "out"
        dirs = realize_packed_shards_into_split_dirs(output_dir=out, split_to_paths=blend)
        counts = _realized_rows_by_dataset(dirs["train"])

        # beta (dominant weight) keeps everything; alpha is subsampled toward
        # 0.22/0.78 * 100 ≈ 28 rows (shard granularity: 30).
        assert counts["beta"] == 100
        assert counts["alpha"] < 50, "fractional weight was flattened to a uniform mix"
        share = counts["alpha"] / (counts["alpha"] + counts["beta"])
        assert share == pytest.approx(0.22, abs=0.05)

    def test_zero_weight_source_excluded_upstream(self, tmp_path):
        """Weight 0 sources are dropped before realization (existing behavior)."""
        prefix_a = _write_dataset_shards(tmp_path, "alpha", 2, 5)
        blend = distribute_shards_to_splits(["1.0", prefix_a], num_shards=2, valid_shards=0, test_shards=0)
        out = tmp_path / "out"
        dirs = realize_packed_shards_into_split_dirs(output_dir=out, split_to_paths=blend)
        counts = _realized_rows_by_dataset(dirs["train"])
        assert counts == {"alpha": 10}

    def test_uniform_weights_never_subsample_unequal_sources(self, tmp_path):
        """Uniform weights preserve the historical keep-everything behavior.

        Weighted subsampling only engages for non-uniform weights: the default
        all-1.0 blend must keep consuming every packed shard, even when source
        sizes differ, so existing recipes see no behavior change.
        """
        prefix_a = _write_dataset_shards(tmp_path, "alpha", 10, 30)  # 300 rows
        prefix_b = _write_dataset_shards(tmp_path, "beta", 10, 10)  # 100 rows
        blend = distribute_shards_to_splits(
            ["1.0", prefix_a, "1.0", prefix_b], num_shards=10, valid_shards=0, test_shards=0
        )
        out = tmp_path / "out"
        dirs = realize_packed_shards_into_split_dirs(output_dir=out, split_to_paths=blend)
        counts = _realized_rows_by_dataset(dirs["train"])
        assert counts == {"alpha": 300, "beta": 100}

    def test_fractional_weights_with_unequal_sources(self, tmp_path):
        """Non-uniform weights are mix shares, independent of raw source sizes."""
        prefix_a = _write_dataset_shards(tmp_path, "alpha", 10, 30)  # 300 rows
        prefix_b = _write_dataset_shards(tmp_path, "beta", 10, 10)  # 100 rows
        blend = distribute_shards_to_splits(
            ["0.25", prefix_a, "0.75", prefix_b], num_shards=10, valid_shards=0, test_shards=0
        )
        out = tmp_path / "out"
        dirs = realize_packed_shards_into_split_dirs(output_dir=out, split_to_paths=blend)
        counts = _realized_rows_by_dataset(dirs["train"])

        # beta (100 rows / 0.75 share) caps the total at ~133 rows; alpha
        # contributes ~33 rows for a 25/75 mix despite being 3x larger.
        assert counts["beta"] == 100
        share = counts["alpha"] / (counts["alpha"] + counts["beta"])
        assert share == pytest.approx(0.25, abs=0.07)
