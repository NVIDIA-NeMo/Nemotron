"""Data preparation for Nano3 SFT stage.

Applies chat templates to OpenAI-format messages, tokenizes with role-based
loss masking, and outputs packed .npy files compatible with GPTSFTPackedDataset.

Pipeline:
1. Apply nano3 chat template → role-labeled chunks
2. Tokenize chunks → input_ids
3. Build loss_mask (0=system/user, 1=assistant)
4. Pack sequences → .npy output

Usage:
    python -m nemotron.recipes.nano3.stage1_sft.data_prep [options]
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep import (
    DataBlend,
    PipelineConfig,
    OutputConfig,
    TokenizerConfig,
    ChatSftOutputConfig,
    last_mile_process,
)
from nemotron.data_prep.config import DatasetConfig
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.kit import DataBlendsArtifact, cli, print_step_complete
from nemotron.kit.trackers import InputDatasetInfo, tokenizer_to_uri
from nemotron.kit.wandb import add_wandb_tags, finish_wandb

STAGE_PATH = Path(__file__).parent


@dataclass
class SFTDataPrepConfig:
    """SFT data preparation config using chat template.

    Applies chat templates to OpenAI-format messages, tokenizes with role-based
    loss masking, and outputs packed .npy files.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: Path("./output/nano3/stage1_sft"))
    """Output directory for packed .npy data"""

    # Tokenizer
    tokenizer_model: str = "nvidia/Nemotron-4-340B-Instruct"
    """HuggingFace tokenizer model name"""

    # Packing
    pack_size: int = 4096
    """Maximum tokens per packed sequence"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    # Chat template
    chat_template: str = "nano3"
    """Chat template: 'nano3', path to .jinja file, or inline template"""

    messages_field: str = "messages"
    """Field name for OpenAI-format messages in input records"""

    tools_field: str = "tools"
    """Field name for tools definition in input records"""

    # Processing limits
    max_doc_tokens: int | None = None
    """Truncate sequences longer than this"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    def __post_init__(self) -> None:
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def main(cfg: SFTDataPrepConfig) -> DataBlendsArtifact:
    """Run SFT data preparation with chat template."""
    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "sft"])

    # Load data blend
    blend = DataBlend.load(cfg.blend_path)

    # Auto-detect num_actors from CPU count
    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    # Build pipeline config with ChatSftOutputConfig
    format_config = ChatSftOutputConfig(
        shard_size=cfg.shard_size,
        pack_size=cfg.pack_size,
        chat_template=cfg.chat_template,
        messages_field=cfg.messages_field,
        tools_field=cfg.tools_field,
    )

    pipeline_config = PipelineConfig(
        output=OutputConfig(
            dir=cfg.output_dir,
            format=format_config,
            max_doc_tokens=cfg.max_doc_tokens,
            max_rows=cfg.sample,
        ),
        tokenizer=TokenizerConfig(model=cfg.tokenizer_model),
        num_actors=num_actors,
        force=cfg.force,
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
    tok_uri = tokenizer_to_uri(cfg.tokenizer_model)

    # Build output artifact
    artifact = DataBlendsArtifact(
        path=result.blend_path,
        total_tokens=result.total_tokens,
        total_sequences=result.total_sequences,
        elapsed_sec=result.elapsed_sec,
        source_datasets=source_datasets,
        tokenizer_uri=tok_uri,
    )
    artifact.name = f"nano3/sft/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    # Mark wandb run as successful
    finish_wandb(exit_code=0)

    print_step_complete(data_prep=artifact)
    return artifact


if __name__ == "__main__":
    cli(main)
