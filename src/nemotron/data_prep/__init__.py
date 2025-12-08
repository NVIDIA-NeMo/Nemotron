"""Data preparation for Megatron training.

Tokenizes raw text data from HuggingFace, S3, or local sources into
.bin/.idx format compatible with Megatron-Bridge and Megatron-Core.

Quick Start:
    from nemotron.data_prep import DataPrepConfig, run_data_prep
    from pathlib import Path

    # Create config
    config = DataPrepConfig(
        blend_path=Path("data_blend.json"),
        output_dir=Path("./output"),
        tokenizer_model="meta-llama/Llama-3.2-1B",
    )

    # Run data preparation
    artifact = run_data_prep(config)

    # Use output with Megatron-Bridge
    print(f"Blend file: {artifact.blend_path}")

Output Format:
    The generated blend.json is directly compatible with Megatron-Bridge's
    get_blend_fields_from_data_paths() function.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from pydantic import Field

from nemotron.data_prep.blend import DataBlend, Dataset
from nemotron.data_prep.config import PipelineConfig, TokenizerConfig, OutputConfig
from nemotron.data_prep.pipeline import tokenize, PipelineResult, SplitResult
from nemotron.kit import Artifact


class DataPrepArtifact(Artifact):
    """Output artifact for data preparation step.

    The blend_path points to a Megatron-Bridge compatible blend.json that can
    be passed directly to training recipes.

    Example:
        >>> from nemotron.data_prep import DataPrepArtifact
        >>> artifact = DataPrepArtifact(
        ...     path=Path("./output"),
        ...     blend_path=Path("./output/blend.json"),
        ...     total_tokens=1_000_000,
        ...     total_sequences=10_000,
        ...     elapsed_sec=120.5,
        ... )
    """

    blend_path: Annotated[Path, Field(description="Path to blend.json for Megatron-Bridge")]
    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[float, Field(ge=0, description="Processing time in seconds")]
    split_ratio: Annotated[
        str | None, Field(default=None, description="Split ratio if single-blend mode (e.g., '99990,8,2')")
    ]
    is_per_split: Annotated[
        bool, Field(default=False, description="True if train/valid/test were tokenized separately")
    ]


@dataclass
class DataPrepConfig:
    """Configuration for data preparation.

    Generic configuration that can be customized per-recipe.

    Example:
        >>> from nemotron.data_prep import DataPrepConfig, run_data_prep
        >>> config = DataPrepConfig(
        ...     blend_path=Path("data_blend.json"),
        ...     output_dir=Path("./output"),
        ...     tokenizer_model="meta-llama/Llama-3.2-1B",
        ... )
        >>> artifact = run_data_prep(config)
    """

    # Data source
    blend_path: Path = field(default_factory=lambda: Path("data_blend.json"))
    """Path to data blend JSON file"""

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    """Output directory for tokenized data"""

    num_shards: int = 128
    """Number of output shards for parallel loading"""

    split: str | None = "99990,8,2"
    """Train:valid:test ratio (e.g., '99990,8,2') or None to disable"""

    # Tokenizer
    tokenizer_model: str = "meta-llama/Llama-3.2-1B"
    """HuggingFace tokenizer model name"""

    add_bos: bool = False
    """Prepend BOS token to documents"""

    add_eos: bool = True
    """Append EOS token to documents"""

    # Processing
    text_field: str = "text"
    """Default text field name in datasets"""

    min_doc_chars: int | None = None
    """Skip documents shorter than this"""

    max_doc_tokens: int | None = None
    """Truncate documents longer than this"""

    # Execution
    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""


def run_data_prep(config: DataPrepConfig) -> DataPrepArtifact:
    """Execute data preparation pipeline.

    Loads the data blend, tokenizes all datasets, and produces a
    Megatron-Bridge compatible blend.json.

    Args:
        config: Data preparation configuration

    Returns:
        DataPrepArtifact with blend.json path and metrics

    Example:
        >>> from nemotron.data_prep import DataPrepConfig, run_data_prep
        >>> config = DataPrepConfig(
        ...     blend_path=Path("data_blend.json"),
        ...     output_dir=Path("./output"),
        ... )
        >>> artifact = run_data_prep(config)
        >>> print(f"Blend file: {artifact.blend_path}")
    """
    # Load data blend specification
    blend = DataBlend.load(config.blend_path)

    # Apply default text_field to datasets that use default
    for split_datasets in blend.splits.values():
        for dataset in split_datasets:
            if dataset.text_field == "text" and config.text_field != "text":
                # Use object.__setattr__ since Dataset is a Pydantic model
                object.__setattr__(dataset, "text_field", config.text_field)

    # Auto-detect num_actors from CPU count
    num_actors = config.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    # Build pipeline config
    # When sampling, use 1 shard to get exactly `sample` rows per dataset
    num_shards = config.num_shards
    if config.sample is not None:
        num_shards = 1

    pipeline_config = PipelineConfig(
        tokenizer=TokenizerConfig(
            model=config.tokenizer_model,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
        ),
        output=OutputConfig(
            dir=config.output_dir,
            num_shards=num_shards,
            min_doc_chars=config.min_doc_chars,
            max_doc_tokens=config.max_doc_tokens,
            max_rows=config.sample,
        ),
        num_actors=num_actors,
        force=config.force,
        split=config.split,
    )

    # Run tokenization pipeline
    result = tokenize(blend, pipeline_config)

    # Build output artifact
    artifact = DataPrepArtifact(
        path=result.output_dir,
        blend_path=result.blend_path,
        total_tokens=result.total_tokens,
        total_sequences=result.total_sequences,
        elapsed_sec=result.elapsed_sec,
        split_ratio=result.split_ratio,
        is_per_split=result.is_per_split,
        metrics={
            "total_tokens": float(result.total_tokens),
            "total_sequences": float(result.total_sequences),
            "elapsed_sec": result.elapsed_sec,
        },
    )
    artifact.save()

    return artifact


__all__ = [
    # Input specification
    "DataBlend",
    "Dataset",
    # High-level API
    "DataPrepConfig",
    "run_data_prep",
    "DataPrepArtifact",
    # Low-level configuration
    "PipelineConfig",
    "TokenizerConfig",
    "OutputConfig",
    # Execution
    "tokenize",
    # Results
    "PipelineResult",
    "SplitResult",
]
