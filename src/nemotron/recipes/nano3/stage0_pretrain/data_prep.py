"""Data preparation for Nemotron Nano3 pretraining.

Tokenizes raw text data into Megatron-Bridge format for pretraining.

CLI Usage:
    # With defaults
    uv run python -m nemotron.recipes.nano3.stage0_pretrain.data_prep

    # Fast iteration (10% sample)
    uv run python -m nemotron.recipes.nano3.stage0_pretrain.data_prep --sample 10%

    # Custom settings
    uv run python -m nemotron.recipes.nano3.stage0_pretrain.data_prep \\
        --output-dir /data/tokenized \\
        --num-shards 256 \\
        --tokenizer-model nvidia/Llama-3.1-Nemotron-Nano-8B-Instruct

Programmatic Usage:
    from nemotron.recipes.nano3.stage0_pretrain.data_prep import (
        nano3_data_prep_config,
        run_data_prep,
    )

    # Get config with Nano3 defaults
    config = nano3_data_prep_config()

    # Or customize
    config = nano3_data_prep_config(sample="10%", num_shards=64)

    # Execute
    artifact = run_data_prep(config)
    print(f"Blend file: {artifact.blend_path}")
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

from pydantic import Field

from nemotron.artifact import Artifact, print_complete
from nemotron.data_prep import (
    DataBlend,
    PipelineConfig,
    TokenizerConfig,
    OutputConfig,
    tokenize,
)


# Default blend file for Nano3 pretraining
DATA_BLEND_PATH = Path(__file__).parent / "data_blend.json"


# ============================================================================
# Output Artifact
# ============================================================================


class DataPrepArtifact(Artifact):
    """Output artifact for data preparation step.

    The blend_path points to a Megatron-Bridge compatible blend.json that can
    be passed directly to training recipes.
    """

    blend_path: Path = Field(description="Path to blend.json for Megatron-Bridge")
    total_tokens: int = Field(ge=0, description="Total tokens processed")
    total_sequences: int = Field(ge=0, description="Total documents processed")
    elapsed_sec: float = Field(ge=0, description="Processing time in seconds")
    split_ratio: str | None = Field(
        default=None, description="Split ratio if single-blend mode (e.g., '99990,8,2')"
    )
    is_per_split: bool = Field(
        default=False, description="True if train/valid/test were tokenized separately"
    )


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DataPrepConfig:
    """Configuration for Nano3 data preparation.

    All parameters have sensible defaults for Nano3 pretraining.
    Use nano3_data_prep_config() for recommended settings.
    """

    # Data source
    blend_path: Path = field(default_factory=lambda: DATA_BLEND_PATH)

    # Output
    output_dir: Path = Path("./output/nano3/stage0_pretrain")
    num_shards: int = 128
    split: str | None = "99990,8,2"  # train:valid:test ratio

    # Tokenizer (Nano3 uses Llama-3.2 base)
    tokenizer_model: str = "meta-llama/Llama-3.2-1B"
    add_bos: bool = False
    add_eos: bool = True

    # Processing
    text_field: str = "text"
    min_doc_chars: int | None = None
    max_doc_tokens: int | None = None

    # Execution
    sample: int | None = None  # Limit rows per dataset (for quick tests)
    num_actors: int | None = None  # Auto-detect from CPU count
    force: bool = False


# ============================================================================
# Config Helper
# ============================================================================


def nano3_data_prep_config(
    blend_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    num_shards: int = 128,
    split: str | None = "99990,8,2",
    tokenizer_model: str = "meta-llama/Llama-3.2-1B",
    add_bos: bool = False,
    add_eos: bool = True,
    text_field: str = "text",
    min_doc_chars: int | None = None,
    max_doc_tokens: int | None = None,
    sample: int | None = None,
    num_actors: int | None = None,
    force: bool = False,
) -> DataPrepConfig:
    """Return data preparation config with Nano3 defaults.

    Recommended defaults:
    - Llama-3.2-1B tokenizer (Nano3 base model)
    - 128 shards (suitable for multi-node training)
    - 99990:8:2 split ratio (~99.99% train)
    - EOS tokens appended (standard for pretraining)

    Args:
        blend_path: Path to data blend JSON (default: built-in Nano3 blend)
        output_dir: Output directory for tokenized data
        num_shards: Number of output shards for parallel loading
        split: Train:valid:test ratio (e.g., "99990,8,2") or None to disable
        tokenizer_model: HuggingFace tokenizer model name
        add_bos: Prepend BOS token to documents
        add_eos: Append EOS token to documents
        text_field: Default text field name in datasets
        min_doc_chars: Skip documents shorter than this
        max_doc_tokens: Truncate documents longer than this
        sample: Limit rows per dataset (for quick tests, e.g., 1000)
        num_actors: Ray actors for parallel processing (None = auto)
        force: Force new run, ignoring cache

    Returns:
        DataPrepConfig ready for run_data_prep()

    Example:
        # Default config
        config = nano3_data_prep_config()

        # Fast iteration
        config = nano3_data_prep_config(sample="10%")

        # Custom tokenizer and shards
        config = nano3_data_prep_config(
            tokenizer_model="nvidia/Llama-3.1-Nemotron-Nano-8B-Instruct",
            num_shards=256,
        )
    """
    return DataPrepConfig(
        blend_path=Path(blend_path) if blend_path else DATA_BLEND_PATH,
        output_dir=Path(output_dir) if output_dir else Path("./output/nano3/stage0_pretrain"),
        num_shards=num_shards,
        split=split,
        tokenizer_model=tokenizer_model,
        add_bos=add_bos,
        add_eos=add_eos,
        text_field=text_field,
        min_doc_chars=min_doc_chars,
        max_doc_tokens=max_doc_tokens,
        sample=sample,
        num_actors=num_actors,
        force=force,
    )


# ============================================================================
# Execution
# ============================================================================


def run_data_prep(config: DataPrepConfig) -> DataPrepArtifact:
    """Execute data preparation pipeline.

    Loads the data blend, tokenizes all datasets, and produces a
    Megatron-Bridge compatible blend.json.

    Args:
        config: Data preparation configuration

    Returns:
        DataPrepArtifact with blend.json path and metrics
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


def main(config: DataPrepConfig) -> DataPrepArtifact:
    """CLI entry point for data preparation.

    Runs the pipeline and prints completion summary.
    """
    artifact = run_data_prep(config)
    print_complete({"data_prep": artifact})
    return artifact


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
