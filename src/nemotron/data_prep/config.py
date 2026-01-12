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
        add_bos: Prepend BOS token to each document
        add_eos: Append EOS token to each document
        trust_remote_code: Allow custom code in HF tokenizers
    """

    model: str
    type: Literal["huggingface", "sentencepiece", "tiktoken"] = "huggingface"
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
        shard_size: Target size per shard (e.g., "256MB"). When set, shard count
            is computed per-dataset based on individual dataset sizes. This is
            recommended for blends with datasets of varying sizes, as it prevents
            empty shard files for small datasets. Mutually exclusive with num_shards.
        num_shards: Exact number of output shards applied to ALL datasets.
            Use shard_size instead when datasets have very different sizes to
            avoid empty shards. Mutually exclusive with shard_size.
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

    def __post_init__(self) -> None:
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Specify either shard_size or num_shards, not both")


@dataclass(frozen=True)
class PackedOutputConfig:
    """Configuration for packed sequence output.

    Tokenizes and packs sequences into efficient batches compatible with
    GPTSFTPackedDataset. Output is .npy files with packed sequences.

    Attributes:
        format: Format identifier (always "packed")
        shard_size: Target size per shard (e.g., "256MB"). Mutually exclusive with num_shards.
        num_shards: Exact number of output shards. Mutually exclusive with shard_size.
        dtype: Token dtype (int32, int64, uint16)
        pack_size: Maximum tokens per packed sequence
        algorithm: Packing algorithm ("first_fit_decreasing", "first_fit_shuffle", "concatenative")
    """

    format: Literal["packed"] = "packed"
    shard_size: str | int | None = "256MB"
    num_shards: int | None = None
    dtype: Literal["int32", "int64", "uint16"] = "int32"
    pack_size: int = 2048
    algorithm: Literal["first_fit_decreasing", "first_fit_shuffle", "concatenative"] = (
        "first_fit_shuffle"
    )

    def __post_init__(self) -> None:
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Specify either shard_size or num_shards, not both")
        if self.pack_size <= 0:
            raise ValueError(f"pack_size must be positive, got {self.pack_size}")


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

    def __post_init__(self) -> None:
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Specify either shard_size or num_shards, not both")
        if self.pack_size <= 0:
            raise ValueError(f"pack_size must be positive, got {self.pack_size}")


# Union type for all output formats
OutputFormat = BinIdxOutputConfig | JsonlOutputConfig | PackedOutputConfig | ChatSftOutputConfig


@dataclass(frozen=True)
class RayDataConfig:
    """Configuration for Ray Data shard-task execution.

    These settings map directly to Ray Data's ActorPoolStrategy and
    map_batches parameters, providing explicit control over resource usage.

    When enabled, uses Ray Data's streaming executor for shard processing
    instead of manual actor pool management. Benefits include:
    - Automatic actor lifecycle management (no leaked actors)
    - Integrated backpressure with Ray's resource manager
    - Explicit CPU accounting per actor

    Attributes:
        enabled: Enable Ray Data execution (vs legacy manual actors)
        min_actors: Minimum actors to keep alive (warm pool)
        max_actors: Maximum actors. None means use all available CPUs.
        cpus_per_actor: CPUs allocated per actor (explicit accounting)
        max_tasks_in_flight_per_actor: Pipelining depth to reduce scheduling
            bubbles and keep actors fed. Note: does not by itself parallelize
            a single actor; true I/O latency hiding requires either more actors
            (with fractional num_cpus) or async internal concurrency.
    """

    enabled: bool = False
    min_actors: int = 2
    max_actors: int | None = None  # None = use all available CPUs
    cpus_per_actor: float = 1.0
    max_tasks_in_flight_per_actor: int = 2


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration.

    Attributes:
        dir: Output directory (local path or cloud URI)
        format: Output format configuration (BinIdxOutputConfig, JsonlOutputConfig, etc.)
        min_doc_chars: Skip documents shorter than this (for tokenized formats)
        max_doc_tokens: Truncate documents longer than this (for tokenized formats)
        max_rows: Limit rows processed per shard (useful for quick tests)

    Deprecated attributes (for backward compatibility):
        num_shards: Use format.num_shards instead
        dtype: Use format.dtype instead
    """

    dir: Path
    format: OutputFormat = field(default_factory=BinIdxOutputConfig)
    min_doc_chars: int | None = None
    max_doc_tokens: int | None = None
    max_rows: int | None = None
    # Deprecated - for backward compatibility
    num_shards: int | None = None
    dtype: Literal["int32", "int64", "uint16"] | None = None

    def __post_init__(self) -> None:
        # Handle backward compatibility: if old-style num_shards/dtype provided,
        # we need to handle them. But since this is frozen, we can't modify.
        # The pipeline should check for these and warn.
        pass


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

    Attributes:
        output: Output settings
        tokenizer: Tokenizer settings (required for binidx/packed formats, optional for jsonl)
        sample: Shard sampling spec ("10%", "5", or None for all)
        sample_seed: Random seed for sampling
        force: Force new run (ignore cached results)
        split: Split ratio for single-blend mode (e.g., "99990,8,2"). Deprecated.
        per_split: Per-split output configuration for Megatron-Bridge per_split_data_args_path
        ray_data: Ray Data execution configuration. When enabled and ray_data.enabled=True,
            uses Ray Data's ActorPoolStrategy for shard processing instead of manual actors.
    """

    output: OutputConfig
    tokenizer: TokenizerConfig | None = None
    sample: str | int | None = None
    sample_seed: int = 42
    force: bool = False
    split: str | None = None  # Deprecated - use per_split instead
    per_split: PerSplitConfig | None = None
    ray_data: RayDataConfig | None = None


# ============================================================================
# Internal Configuration Classes (Used by pipeline internals)
# ============================================================================


@dataclass
class InternalDatasetConfig:
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


# ============================================================================
# Aliases for backward compatibility with existing internal code
# ============================================================================

# These aliases allow existing internal modules (planning.py, discovery.py, etc.)
# to continue using the old names without modification
DatasetConfig = InternalDatasetConfig
# Note: TokenizerConfig is now the public frozen dataclass
# Internal code should use InternalTokenizerConfig
# OutputConfig is now the public frozen dataclass
# Internal code should use InternalOutputConfig
