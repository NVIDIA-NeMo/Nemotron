"""Data preparation for Nano3 SFT stage.

Converts datasets to JSONL format with input/output fields.

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
    last_mile_process,
)
from nemotron.data_prep.config import JsonlOutputConfig
from nemotron.data_prep.formats.transforms import select
from nemotron.kit import DataBlendsArtifact, cli, print_step_complete

STAGE_PATH = Path(__file__).parent


@dataclass
class SFTDataPrepConfig:
    """SFT data preparation config.

    Converts to JSONL with input/output fields for supervised fine-tuning.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: Path("./output/nano3/stage1_sft"))
    """Output directory for JSONL data"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    text_field: str = "text"
    """Field name for text in source data"""

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
    """Run SFT data preparation."""
    # Load data blend
    blend = DataBlend.load(cfg.blend_path)

    # Auto-detect num_actors from CPU count
    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    # Build pipeline config with JSONL output format
    # Use select transform to extract just the text field
    format_config = JsonlOutputConfig(
        shard_size=cfg.shard_size,
        transform=select(cfg.text_field),
    )

    pipeline_config = PipelineConfig(
        output=OutputConfig(
            dir=cfg.output_dir,
            format=format_config,
            max_rows=cfg.sample,
        ),
        tokenizer=None,  # JSONL doesn't need tokenizer
        num_actors=num_actors,
        force=cfg.force,
    )

    # Run processing pipeline
    result = last_mile_process(blend, pipeline_config)

    # Build output artifact
    artifact = DataBlendsArtifact(
        path=result.blend_path,
        total_tokens=result.total_tokens,
        total_sequences=result.total_sequences,
        elapsed_sec=result.elapsed_sec,
    )
    artifact.name = f"nano3/sft/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    print_step_complete(data_prep=artifact)
    return artifact


if __name__ == "__main__":
    cli(main)
