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

"""Tests for configuration dataclasses."""

from pathlib import Path

import pytest

from nemotron.data_prep.config import (
    BinIdxOutputConfig,
    ChatSftOutputConfig,
    DatasetConfig,
    FileInfo,
    FormatResult,
    HfDownloadConfig,
    InternalOutputConfig,
    InternalTokenizerConfig,
    JsonlOutputConfig,
    ObservabilityConfig,
    OutputConfig,
    PerSplitConfig,
    PipelineConfig,
    ShardAssignment,
    ShardPlan,
    TokenizerConfig,
    VALID_OUTPUT_DTYPES,
)


# =============================================================================
# TokenizerConfig
# =============================================================================


class TestTokenizerConfig:
    def test_defaults(self) -> None:
        cfg = TokenizerConfig(model="nvidia/test")
        assert cfg.type == "huggingface"
        assert cfg.revision is None
        assert cfg.add_bos is False
        assert cfg.add_eos is True
        assert cfg.trust_remote_code is False

    def test_frozen(self) -> None:
        cfg = TokenizerConfig(model="test")
        with pytest.raises(AttributeError):
            cfg.model = "other"  # type: ignore[misc]


# =============================================================================
# BinIdxOutputConfig
# =============================================================================


class TestBinIdxOutputConfig:
    def test_defaults(self) -> None:
        cfg = BinIdxOutputConfig()
        assert cfg.format == "binidx"
        assert cfg.shard_size == "256MB"
        assert cfg.num_shards is None
        assert cfg.dtype == "int32"

    def test_shard_size_and_num_shards_exclusive(self) -> None:
        with pytest.raises(ValueError, match="either shard_size or num_shards"):
            BinIdxOutputConfig(shard_size="256MB", num_shards=10)

    def test_num_shards_only(self) -> None:
        cfg = BinIdxOutputConfig(shard_size=None, num_shards=16)
        assert cfg.num_shards == 16
        assert cfg.shard_size is None


# =============================================================================
# JsonlOutputConfig
# =============================================================================


class TestJsonlOutputConfig:
    def test_defaults(self) -> None:
        cfg = JsonlOutputConfig()
        assert cfg.format == "jsonl"
        assert cfg.compression == "none"
        assert cfg.transform is None
        assert cfg.resolve_hf_placeholders is False

    def test_shard_size_and_num_shards_exclusive(self) -> None:
        with pytest.raises(ValueError, match="either shard_size or num_shards"):
            JsonlOutputConfig(shard_size="256MB", num_shards=10)


# =============================================================================
# ChatSftOutputConfig
# =============================================================================


class TestChatSftOutputConfig:
    def test_defaults(self) -> None:
        cfg = ChatSftOutputConfig()
        assert cfg.format == "chat_sft"
        assert cfg.pack_size == 2048
        assert cfg.algorithm == "first_fit_shuffle"
        assert cfg.parquet_row_group_size == 1000
        assert cfg.parquet_compression == "zstd"

    def test_shard_size_and_num_shards_exclusive(self) -> None:
        with pytest.raises(ValueError, match="either shard_size or num_shards"):
            ChatSftOutputConfig(shard_size="256MB", num_shards=10)

    def test_pack_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="pack_size must be positive"):
            ChatSftOutputConfig(shard_size=None, pack_size=0)

    def test_pack_size_negative(self) -> None:
        with pytest.raises(ValueError, match="pack_size must be positive"):
            ChatSftOutputConfig(shard_size=None, pack_size=-1)

    def test_parquet_row_group_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="parquet_row_group_size must be positive"):
            ChatSftOutputConfig(shard_size=None, parquet_row_group_size=0)


# =============================================================================
# ObservabilityConfig
# =============================================================================


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        cfg = ObservabilityConfig()
        assert cfg.wandb_log_downloads is False
        assert cfg.wandb_log_pipeline_stats is True
        assert cfg.wandb_log_plan_table is True
        assert cfg.wandb_log_progress_table is True
        assert cfg.wandb_log_stage_table is True
        assert cfg.wandb_progress_table_interval_s == 300
        assert cfg.wandb_stage_table_interval_s == 30
        assert cfg.wandb_download_log_interval_sec == 30
        assert cfg.pipeline_logging_interval_s == 30
        assert cfg.pipeline_stats_jsonl_path is None


# =============================================================================
# HfDownloadConfig
# =============================================================================


class TestHfDownloadConfig:
    def test_defaults(self) -> None:
        cfg = HfDownloadConfig()
        assert cfg.max_concurrent == 64
        assert cfg.timeout_sec == 300
        assert cfg.max_retries == 3


# =============================================================================
# OutputConfig
# =============================================================================


class TestOutputConfig:
    def test_defaults(self) -> None:
        cfg = OutputConfig(dir=Path("/output"))
        assert isinstance(cfg.format, BinIdxOutputConfig)
        assert cfg.min_doc_chars is None
        assert cfg.max_doc_tokens is None
        assert cfg.max_rows is None


# =============================================================================
# PipelineConfig
# =============================================================================


class TestPipelineConfig:
    def test_minimal(self) -> None:
        cfg = PipelineConfig(output=OutputConfig(dir=Path("/out")))
        assert cfg.tokenizer is None
        assert cfg.sample is None
        assert cfg.sample_seed == 42
        assert cfg.force is False
        assert cfg.per_split is None
        assert cfg.max_workers is None
        assert isinstance(cfg.download, HfDownloadConfig)
        assert isinstance(cfg.observability, ObservabilityConfig)

    def test_frozen(self) -> None:
        cfg = PipelineConfig(output=OutputConfig(dir=Path("/out")))
        with pytest.raises(AttributeError):
            cfg.force = True  # type: ignore[misc]


# =============================================================================
# InternalOutputConfig
# =============================================================================


class TestInternalOutputConfig:
    def test_valid_dtypes(self) -> None:
        for dtype in VALID_OUTPUT_DTYPES:
            cfg = InternalOutputConfig(num_shards=1, dtype=dtype)
            assert cfg.dtype == dtype

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValueError, match="Invalid dtype"):
            InternalOutputConfig(num_shards=1, dtype="float32")


# =============================================================================
# FormatResult
# =============================================================================


class TestFormatResult:
    def test_construction(self) -> None:
        result = FormatResult(
            run_hash="abc123",
            run_dir="/runs/abc123",
            output_dir=Path("/output"),
            num_shards=4,
            data_paths=["1.0", "prefix_0"],
            dataset_stats={"ds1": {"tokens": 1000}},
            from_cache=False,
            total_tokens=1000,
            total_sequences=50,
        )
        assert result.run_hash == "abc123"
        assert result.total_tokens == 1000

    def test_defaults(self) -> None:
        result = FormatResult(
            run_hash="x",
            run_dir="/x",
            output_dir=Path("/x"),
            num_shards=1,
            data_paths=[],
            dataset_stats={},
            from_cache=True,
        )
        assert result.total_tokens == 0
        assert result.total_sequences == 0


# =============================================================================
# DatasetConfig
# =============================================================================


class TestDatasetConfig:
    def test_defaults(self) -> None:
        cfg = DatasetConfig(name="test", path="/data")
        assert cfg.weight == 1.0
        assert cfg.text_field == "text"
        assert cfg.include_in_blend is True
        assert cfg.split is None
        assert cfg.subset is None
        assert cfg.revision is None


# =============================================================================
# ShardPlan.from_dict
# =============================================================================


class TestShardPlan:
    def test_from_dict(self) -> None:
        data = {
            "version": "1",
            "created_at": "2025-01-01T00:00:00Z",
            "plan_hash": "abc",
            "dataset_name": "test",
            "num_shards": 2,
            "source_fingerprint": "fp",
            "config_hash": "ch",
            "determinism_constraints": {},
            "resolved_tokenizer": {"type": "huggingface"},
            "file_assignments": [
                {
                    "shard_index": 0,
                    "files": [
                        {"path": "/a.parquet", "local_path": None, "size": 100},
                    ],
                    "total_bytes": 100,
                },
                {
                    "shard_index": 1,
                    "files": [],
                    "total_bytes": 0,
                },
            ],
        }
        plan = ShardPlan.from_dict(data)
        assert plan.num_shards == 2
        assert len(plan.file_assignments) == 2
        assert plan.file_assignments[0].shard_index == 0
        assert len(plan.file_assignments[0].files) == 1
        assert plan.file_assignments[0].files[0].path == "/a.parquet"
