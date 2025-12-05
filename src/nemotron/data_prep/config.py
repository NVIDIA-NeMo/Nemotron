"""Configuration dataclasses for the tokenization pipeline."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Valid dtypes for indexed dataset output (must match DTYPE_CODES in indexed_dataset.py)
VALID_OUTPUT_DTYPES = {"int32", "int64", "uint16"}


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""

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
class TokenizerConfig:
    """Configuration for the tokenizer."""

    type: Literal["huggingface", "sentencepiece"]
    model: str  # Model name or path
    revision: str | None = None  # Model revision (resolved to SHA)
    add_eos: bool = True
    add_bos: bool = False
    trust_remote_code: bool = False


@dataclass
class OutputConfig:
    """Configuration for output generation."""

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
class PipelineConfig:
    """Complete pipeline configuration."""

    datasets: list[DatasetConfig]
    tokenizer: TokenizerConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create config from dictionary (e.g., loaded from JSON)."""
        datasets = [DatasetConfig(**d) for d in data["datasets"]]
        tokenizer = TokenizerConfig(**data["tokenizer"])
        output = OutputConfig(**data["output"])
        return cls(datasets=datasets, tokenizer=tokenizer, output=output)


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
