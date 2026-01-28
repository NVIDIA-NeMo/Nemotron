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

"""Pipeline configuration models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# Valid dtypes for indexed dataset output (must match DTYPE_CODES in indexed_dataset.py)
VALID_OUTPUT_DTYPES = {"int32", "int64", "uint16"}

# Type alias for transform functions
Transform = Callable[[dict], dict | None]


# ============================================================================
# Public Configuration Models (New API)
# ============================================================================


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer configuration.

    Attributes:
        model: HuggingFace model name/path, SentencePiece model path,
               or tiktoken encoding name
        type: Tokenizer backend (huggingface, sentencepiece, tiktoken)
        revision: Git revision/SHA for HuggingFace models. If not specified,
                  the latest revision is resolved and used. Specifying a revision
                  ensures deterministic caching across runs.
        add_bos: Prepend BOS token to each document
        add_eos: Append EOS token to each document
        trust_remote_code: Allow custom code in HF tokenizers
    """

    model: str
    type: Literal["huggingface", "sentencepiece", "tiktoken"] = "huggingface"
    revision: str | None = None
    add_bos: bool = False
    add_eos: bool = True
    trust_remote_code: bool = False


# ============================================================================
# Output Format Configurations
# ============================================================================


@dataclass(frozen=True)
class BinIdxOutputConfig:
    """Configuration for Megatron .bin/.idx indexed dataset output.

    This is the default format, producing tokenized binary files compatible
    with Megatron-Bridge and Megatron-Core.

    Attributes:
        format: Format identifier (always "binidx")
        shard_size: Target size per shard (e.g., "256MB"). Mutually exclusive with num_shards.
        num_shards: Exact number of output shards. Mutually exclusive with shard_size.
        dtype: Token dtype (int32, int64, uint16)
    """

    format: Literal["binidx"] = "binidx"
    shard_size: str | int | None = "256MB"
    num_shards: int | None = None
    dtype: Literal["int32", "int64", "uint16"] = "int32"

    def __post_init__(self) -> None:
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Specify either shard_size or num_shards, not both")


@dataclass(frozen=True)
class JsonlOutputConfig:
    """Configuration for JSONL output (no tokenization).

    Outputs structured JSONL files for SFT/RL training, applying optional
    transforms to convert records to the desired format.

    Attributes:
        format: Format identifier (always "jsonl")
        shard_size: Target size per shard (e.g., "256MB"). Mutually exclusive with num_shards.
        num_shards: Exact number of output shards. Mutually exclusive with shard_size.
        transform: Optional callable to transform records. Returns dict or None to skip.
        compression: Output compression ("none" for .jsonl, "zstd" for .jsonl.zst)
    """

    format: Literal["jsonl"] = "jsonl"
    shard_size: str | int | None = "256MB"
    num_shards: int | None = None
    transform: Transform | None = None
    compression: Literal["none", "zstd"] = "none"
    resolve_hf_placeholders: bool = False

    def __post_init__(self) -> None:
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Specify either shard_size or num_shards, not both")


@dataclass(frozen=True)
class ChatSftOutputConfig:
    """Configuration for chat-templated SFT output with loss masking.

    Applies materialize.py chat template logic to OpenAI-format messages,
    tokenizes with role-based loss masking, and outputs packed .npy files
    compatible with GPTSFTPackedDataset.

    Pipeline:
    1. Apply chat template → role-labeled chunks
    2. Tokenize chunks → input_ids
    3. Build loss_mask (0=system/user, 1=assistant)
    4. Pack sequences → .npy output

    Attributes:
        format: Format identifier (always "chat_sft")
        shard_size: Target size per shard (e.g., "256MB"). Mutually exclusive with num_shards.
        num_shards: Exact number of output shards. Mutually exclusive with shard_size.
        dtype: Token dtype (int32, int64, uint16)
        pack_size: Maximum tokens per packed sequence
        algorithm: Packing algorithm ("first_fit_decreasing", "first_fit_shuffle", "concatenative")
        chat_template: "nano3", path to .jinja file, or inline template string
        messages_field: Field name for messages in input records
        tools_field: Field name for tools in input records
        used_in_filter: Filter to only include records where used_in contains this value
        used_in_field: Field name for used_in filtering (default: "used_in")
    """

    format: Literal["chat_sft"] = "chat_sft"
    shard_size: str | int | None = "256MB"
    num_shards: int | None = None
    dtype: Literal["int32", "int64", "uint16"] = "int32"
    pack_size: int = 2048
    algorithm: Literal["first_fit_decreasing", "first_fit_shuffle", "concatenative"] = (
        "first_fit_shuffle"
    )
    chat_template: str | None = None
    messages_field: str = "messages"
    tools_field: str = "tools"
    used_in_filter: str | None = None
    used_in_field: str = "used_in"

    parquet_row_group_size: int = 1000
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"

    def __post_init__(self) -> None:
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Specify either shard_size or num_shards, not both")
        if self.pack_size <= 0:
            raise ValueError(f"pack_size must be positive, got {self.pack_size}")
        if self.parquet_row_group_size <= 0:
            raise ValueError(
                f"parquet_row_group_size must be positive, got {self.parquet_row_group_size}"
            )


# Union type for all output formats
OutputFormat = BinIdxOutputConfig | JsonlOutputConfig | ChatSftOutputConfig


@dataclass(frozen=True)
class HfDownloadConfig:
    """HuggingFace download stage settings.

    Attributes:
        max_concurrent: Maximum parallel HuggingFace file downloads
        timeout_sec: Timeout for HuggingFace downloads
        max_retries: Max retries for HuggingFace downloads
    """

    max_concurrent: int = 64
    timeout_sec: int = 300
    max_retries: int = 3


@dataclass(frozen=True)
class ObservabilityConfig:
    """Pipeline observability settings.

    Attributes:
        wandb_log_downloads: Log download progress to wandb
        wandb_log_pipeline_stats: Log pipeline stats (actors, queues, progress) to wandb
        wandb_log_plan_table: Log plan table to wandb showing datasets and processing status
        wandb_log_progress_table: Log per-dataset progress table to wandb periodically
        wandb_log_stage_table: Log stage overview table to wandb showing per-stage metrics
        wandb_progress_table_interval_s: Interval for progress table updates (default 5 min)
        wandb_stage_table_interval_s: Interval for stage table updates (default 30s)
        wandb_download_log_interval_sec: Interval for download progress logging
        pipeline_logging_interval_s: Interval for pipeline stats logging
        pipeline_stats_jsonl_path: Optional path to write pipeline stats as JSONL.
            If set, appends one JSON object per stats callback invocation.
            Should be a local filesystem path (append semantics may not work on cloud storage).
    """

    wandb_log_downloads: bool = False
    wandb_log_pipeline_stats: bool = True
    wandb_log_plan_table: bool = True
    wandb_log_progress_table: bool = True
    wandb_log_stage_table: bool = True
    wandb_progress_table_interval_s: int = 300
    wandb_stage_table_interval_s: int = 30
    wandb_download_log_interval_sec: int = 30
    pipeline_logging_interval_s: int = 30
    pipeline_stats_jsonl_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration.

    Attributes:
        dir: Output directory (local path or cloud URI)
        format: Output format configuration (BinIdxOutputConfig, JsonlOutputConfig, etc.)
        min_doc_chars: Skip documents shorter than this (for tokenized formats)
        max_doc_tokens: Truncate documents longer than this (for tokenized formats)
        max_rows: Limit rows processed per shard (useful for quick tests)
    """

    dir: Path
    format: OutputFormat = field(default_factory=BinIdxOutputConfig)
    min_doc_chars: int | None = None
    max_doc_tokens: int | None = None
    max_rows: int | None = None


@dataclass(frozen=True)
class PerSplitConfig:
    """Configuration for per-split output mode.

    Distributes shards into train/valid/test splits and outputs JSON
    with {"train": [...], "valid": [...], "test": [...]} keys.

    Attributes:
        enabled: If True, enables per-split output mode
        valid_shards: Number of shards for validation (default: 1)
        test_shards: Number of shards for test (default: 1)
    """

    enabled: bool = True
    valid_shards: int = 1
    test_shards: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration.

    All pipelines execute via cosmos-xenna. Format-specific pipeline configs
    are derived from this config using the from_pipeline_config() class methods.

    Attributes:
        output: Output settings
        tokenizer: Tokenizer settings (required for binidx/chat_sft formats, optional for jsonl)
        sample: Shard sampling spec ("10%", "5", or None for all)
        sample_seed: Random seed for sampling
        force: Force new run (ignore cached results)
        per_split: Per-split output configuration for Megatron-Bridge per_split_data_args_path
        download: HuggingFace download settings
        observability: Xenna observability settings
        max_workers: Maximum workers for shard processing. None means auto-scale.
    """

    output: OutputConfig
    tokenizer: TokenizerConfig | None = None
    sample: str | int | None = None
    sample_seed: int = 42
    force: bool = False
    per_split: PerSplitConfig | None = None
    download: HfDownloadConfig = field(default_factory=HfDownloadConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    max_workers: int | None = None


# ============================================================================
# Format Result (returned by per-format run() methods)
# ============================================================================


@dataclass
class FormatResult:
    """Result from running a format-specific pipeline.

    This is the internal result type returned by *PipelineConfig.run().
    The public API uses PipelineResult which is built from FormatResult(s).

    Attributes:
        run_hash: Deterministic hash identifying this run configuration
        run_dir: Path to the runs/{run_hash} directory
        output_dir: User-facing output root directory
        num_shards: Number of shards produced
        data_paths: Megatron-Bridge format ["weight", "prefix", ...] where each prefix
            is a shard base path WITHOUT the shard index suffix. For example:
            [".../runs/abc123/datasets/mydata/hash/shard", ...]
            The actual shard files are named: shard_000000.parquet, shard_000001.parquet, etc.
            Consumers like distribute_shards_to_splits() append _{shard_idx:06d} to each prefix.
        dataset_stats: Per-dataset statistics {name: {tokens, sequences, ...}}
        from_cache: True if all results were served from cache
        total_tokens: Total tokens across all datasets (0 for non-tokenized formats)
        total_sequences: Total sequences/records across all datasets
    """

    run_hash: str
    run_dir: str
    output_dir: Path
    num_shards: int
    data_paths: list[str]
    dataset_stats: dict[str, dict]
    from_cache: bool
    total_tokens: int = 0
    total_sequences: int = 0


# NOTE: PretrainPipelineConfig, JsonlPipelineConfig, and ChatSftPipelineConfig have been
# removed as part of xenna-native migration. Use recipes/pretrain.py and recipes/sft.py instead.


# ============================================================================
# Internal Configuration Classes (Used by pipeline internals)
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source (internal use)."""

    name: str  # Unique identifier
    path: str  # hf://..., s3://..., or local path/glob
    weight: float = 1.0  # Blend weight
    text_field: str = "text"
    include_in_blend: bool = True

    # HuggingFace-specific
    split: str | None = None  # Required for hf://
    subset: str | None = None  # HF dataset config
    revision: str | None = None  # Git revision (resolved to SHA)


@dataclass
class InternalTokenizerConfig:
    """Configuration for the tokenizer (internal use)."""

    type: Literal["huggingface", "sentencepiece", "tiktoken"]
    model: str  # Model name or path
    revision: str | None = None  # Model revision (resolved to SHA)
    add_eos: bool = True
    add_bos: bool = False
    trust_remote_code: bool = False


@dataclass
class InternalOutputConfig:
    """Configuration for output generation (internal use)."""

    num_shards: int  # Required - explicit shard count
    dtype: str = "int32"
    min_doc_chars: int | None = None
    max_doc_tokens: int | None = None
    max_rows: int | None = None  # Limit rows processed per shard (useful for quick tests)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.dtype not in VALID_OUTPUT_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}'. Must be one of: {sorted(VALID_OUTPUT_DTYPES)}"
            )
        # Validate dtype is actually a valid numpy dtype
        try:
            np.dtype(self.dtype)
        except TypeError as e:
            raise ValueError(f"Invalid numpy dtype '{self.dtype}': {e}")


@dataclass
class FileInfo:
    """Metadata for an input file."""

    path: str
    local_path: str | None  # Resolved local path (for HF cache) - None for HF files
    size: int
    etag: str | None = None
    # Additional fingerprint fields
    mtime: float | None = None  # For local files
    version_id: str | None = None  # For S3/GCS versioned objects
    # HuggingFace-specific fields for deferred download
    hf_repo_id: str | None = None  # e.g., "allenai/c4"
    hf_filename: str | None = None  # e.g., "en/c4-train.00000-of-01024.json.gz"
    hf_revision: str | None = None  # Resolved SHA for determinism


@dataclass
class ShardAssignment:
    """Files assigned to a shard."""

    shard_index: int
    files: list[FileInfo] = field(default_factory=list)
    total_bytes: int = 0


@dataclass
class ShardPlan:
    """Deterministic shard assignment, frozen at first run."""

    version: str
    created_at: str
    plan_hash: str
    dataset_name: str
    num_shards: int
    source_fingerprint: str
    config_hash: str
    determinism_constraints: dict
    resolved_tokenizer: dict
    file_assignments: list[ShardAssignment]

    @classmethod
    def from_dict(cls, data: dict) -> "ShardPlan":
        """Create ShardPlan from dictionary."""
        file_assignments = []
        for fa in data["file_assignments"]:
            files = [FileInfo(**f) for f in fa["files"]]
            file_assignments.append(
                ShardAssignment(
                    shard_index=fa["shard_index"],
                    files=files,
                    total_bytes=fa["total_bytes"],
                )
            )
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            plan_hash=data["plan_hash"],
            dataset_name=data["dataset_name"],
            num_shards=data["num_shards"],
            source_fingerprint=data["source_fingerprint"],
            config_hash=data["config_hash"],
            determinism_constraints=data["determinism_constraints"],
            resolved_tokenizer=data["resolved_tokenizer"],
            file_assignments=file_assignments,
        )


class SourceChangedError(Exception):
    """Raised when source data has changed since plan creation."""

    pass


