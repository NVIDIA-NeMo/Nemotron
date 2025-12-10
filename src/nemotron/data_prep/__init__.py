"""Data preparation for Megatron training.

Processes raw text data from HuggingFace, S3, or local sources into
various training formats compatible with Megatron-Bridge and Megatron-Core.

Supported output formats:
- **binidx**: Tokenized .bin/.idx files (default, for pretraining)
- **jsonl**: JSONL files with optional transforms (for SFT/RL, no tokenization)
- **packed**: Packed sequences in .npy format (for efficient SFT training)

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
    print(f"Blend path: {artifact.path}")

Low-Level API (last_mile_process):
    from nemotron.data_prep import last_mile_process, DataBlend, PipelineConfig
    from nemotron.data_prep.config import OutputConfig, JsonlOutputConfig
    from nemotron.data_prep.formats.transforms import sft

    blend = DataBlend.load("data_blend.json")

    # JSONL output for SFT
    config = PipelineConfig(
        output=OutputConfig(
            dir=Path("./sft_data"),
            format=JsonlOutputConfig(transform=sft(input="instruction", output="response")),
        ),
    )
    result = last_mile_process(blend, config)

Output Format:
    The generated blend.json is directly compatible with Megatron-Bridge's
    get_blend_fields_from_data_paths() function.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep.blend import DataBlend, Dataset
from nemotron.data_prep.config import (
    PipelineConfig,
    TokenizerConfig,
    OutputConfig,
    BinIdxOutputConfig,
    JsonlOutputConfig,
    PackedOutputConfig,
    ChatSftOutputConfig,
    Transform,
)
from nemotron.data_prep.pipeline import (
    last_mile_process,
    tokenize,
    PipelineResult,
    SplitResult,
)
from nemotron.data_prep.formats.transforms import (
    sft,
    openai_chat,
    sharegpt,
    passthrough,
    select,
    rename,
    SftRecord,
    OpenAIChatRecord,
    ShareGPTRecord,
)
from nemotron.kit.artifact import DataBlendsArtifact
from nemotron.kit.trackers import InputDatasetInfo, tokenizer_to_uri
from nemotron.kit.wandb import finish_wandb
from nemotron.data_prep.discovery import get_dataset_metadata


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

    artifact_name: str | None = None
    """Semantic artifact name (e.g., 'nano3/pretrain/data')"""


def run_data_prep(config: DataPrepConfig) -> DataBlendsArtifact:
    """Execute data preparation pipeline.

    Loads the data blend, tokenizes all datasets, and produces a
    Megatron-Bridge compatible blend.json.

    Args:
        config: Data preparation configuration

    Returns:
        DataBlendsArtifact with blend.json path and metrics

    Example:
        >>> from nemotron.data_prep import DataPrepConfig, run_data_prep
        >>> config = DataPrepConfig(
        ...     blend_path=Path("data_blend.json"),
        ...     output_dir=Path("./output"),
        ... )
        >>> artifact = run_data_prep(config)
        >>> print(f"Blend path: {artifact.path}")
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

    # Build output config with format that has num_shards
    # When num_shards is specified, clear shard_size to avoid conflict
    output_format = BinIdxOutputConfig(
        num_shards=num_shards,
        shard_size=None if num_shards is not None else "256MB",
    )

    pipeline_config = PipelineConfig(
        output=OutputConfig(
            dir=config.output_dir,
            format=output_format,
            min_doc_chars=config.min_doc_chars,
            max_doc_tokens=config.max_doc_tokens,
            max_rows=config.sample,
        ),
        tokenizer=TokenizerConfig(
            model=config.tokenizer_model,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
        ),
        num_actors=num_actors,
        force=config.force,
        split=config.split,
    )

    # Run processing pipeline
    result = last_mile_process(blend, pipeline_config)

    # Collect source datasets with metadata for lineage tracking
    source_datasets: list[InputDatasetInfo] = []
    seen_keys: set[str] = set()
    for split_datasets in blend.splits.values():
        for dataset in split_datasets:
            # Use path+subset as key since same path can have different subsets
            key = f"{dataset.path}|{dataset.subset or ''}"
            if key not in seen_keys:
                seen_keys.add(key)
                # Build dataset config for metadata fetching
                from nemotron.data_prep.config import DatasetConfig

                ds_config = DatasetConfig(
                    name=dataset.name,
                    path=dataset.path,
                    split=dataset.split,
                    subset=dataset.subset,
                    text_field=dataset.text_field,
                )
                hf_metadata = get_dataset_metadata(ds_config)
                source_datasets.append(
                    InputDatasetInfo(
                        uri=dataset.path,
                        name=dataset.name,
                        weight=dataset.weight,
                        split=dataset.split,
                        subset=dataset.subset,
                        text_field=dataset.text_field,
                        num_rows=hf_metadata.num_rows,
                        size_bytes=hf_metadata.size_bytes,
                    )
                )

    # Create tokenizer URI for lineage tracking
    tok_uri = tokenizer_to_uri(config.tokenizer_model)

    # Build output artifact - path points to blend.json
    artifact = DataBlendsArtifact(
        path=result.blend_path,
        total_tokens=result.total_tokens,
        total_sequences=result.total_sequences,
        elapsed_sec=result.elapsed_sec,
        source_datasets=source_datasets,
        tokenizer_uri=tok_uri,
        name=config.artifact_name,  # Semantic name for W&B artifact naming
    )
    artifact.save()

    # Mark wandb run as successful (before Ray shutdown to avoid socket noise)
    finish_wandb(exit_code=0)

    # Gracefully shutdown Ray - suppress stderr during cleanup
    try:
        import ray
        if ray.is_initialized():
            import sys
            import io

            # Temporarily suppress stderr during Ray shutdown (socket cleanup noise)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                ray.shutdown()
            finally:
                sys.stderr = old_stderr
    except Exception:
        pass

    return artifact


__all__ = [
    # Input specification
    "DataBlend",
    "Dataset",
    # High-level API
    "DataPrepConfig",
    "run_data_prep",
    "DataBlendsArtifact",
    # Low-level configuration
    "PipelineConfig",
    "TokenizerConfig",
    "OutputConfig",
    # Output format configs
    "BinIdxOutputConfig",
    "JsonlOutputConfig",
    "PackedOutputConfig",
    "ChatSftOutputConfig",
    "Transform",
    # Transform factories
    "sft",
    "openai_chat",
    "sharegpt",
    "passthrough",
    "select",
    "rename",
    # Transform type definitions
    "SftRecord",
    "OpenAIChatRecord",
    "ShareGPTRecord",
    # Execution
    "last_mile_process",
    "tokenize",  # Deprecated alias
    # Results
    "PipelineResult",
    "SplitResult",
]
