"""Data preparation for Nano3 RL stage.

Converts datasets to JSONL format with OpenAI chat messages.

Usage:
    python -m nemotron.recipes.nano3.stage2_rl.data_prep [options]

Examples:
    # Basic processing (flat output)
    python -m nemotron.recipes.nano3.stage2_rl.data_prep

    # With train/val/test split
    python -m nemotron.recipes.nano3.stage2_rl.data_prep --split_output=train_val_test

    # Quick test with sample
    python -m nemotron.recipes.nano3.stage2_rl.data_prep --sample=100
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nemotron.data_prep import (
    DataBlend,
    PipelineConfig,
    OutputConfig,
    last_mile_process,
)
from nemotron.data_prep.config import JsonlOutputConfig
from nemotron.data_prep.formats.transforms import openai_chat
from nemotron.kit import DataBlendsArtifact, cli, print_step_complete

STAGE_PATH = Path(__file__).parent


@dataclass
class RLDataPrepConfig:
    """RL data preparation config.

    Converts to JSONL with OpenAI chat format for RLHF training.
    RL doesn't support blends, so output is a flat directory of JSONL files.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: Path("./output/nano3/stage2_rl"))
    """Output directory for JSONL data"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    messages_field: str = "messages"
    """Field name for messages in source data"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    split_output: Literal["none", "train_val_test"] = "none"
    """Split output into train/val/test directories (default: single flat output)"""

    train_ratio: float = 0.98
    """Ratio of data for training when split_output='train_val_test'"""

    val_ratio: float = 0.01
    """Ratio of data for validation when split_output='train_val_test'"""

    def __post_init__(self) -> None:
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"
        # Validate split ratios
        if self.split_output == "train_val_test":
            test_ratio = 1.0 - self.train_ratio - self.val_ratio
            if test_ratio < -0.001:  # Allow small floating point errors
                raise ValueError(
                    f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                    f"must not exceed 1.0"
                )


def _run_single_blend(
    blend: DataBlend, cfg: RLDataPrepConfig, num_actors: int
) -> DataBlendsArtifact:
    """Process blend without train/val/test splitting."""
    # Build pipeline config with JSONL output format
    format_config = JsonlOutputConfig(
        shard_size=cfg.shard_size,
        transform=openai_chat(messages=cfg.messages_field),
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
    return artifact


def _run_split_blend(
    blend: DataBlend, cfg: RLDataPrepConfig, num_actors: int
) -> DataBlendsArtifact:
    """Process blend with train/val/test splitting.

    Creates separate output directories for train, val, and test splits.
    The split is done by assigning output shards to different splits based on ratios.
    """
    import json
    import time

    start_time = time.time()
    total_sequences = 0
    split_paths = {}

    # Calculate split ratios
    train_ratio = cfg.train_ratio
    val_ratio = cfg.val_ratio
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    splits_config = [
        ("train", train_ratio),
        ("val", val_ratio),
        ("test", test_ratio),
    ]
    # Filter out zero-ratio splits
    splits_config = [(name, ratio) for name, ratio in splits_config if ratio > 0]

    # Process each split
    for split_name, ratio in splits_config:
        split_output_dir = cfg.output_dir / split_name

        # Build pipeline config for this split
        format_config = JsonlOutputConfig(
            shard_size=cfg.shard_size,
            transform=openai_chat(messages=cfg.messages_field),
        )

        # Calculate max_rows for this split based on ratio
        split_max_rows = None
        if cfg.sample is not None:
            split_max_rows = int(cfg.sample * ratio)

        pipeline_config = PipelineConfig(
            output=OutputConfig(
                dir=split_output_dir,
                format=format_config,
                max_rows=split_max_rows,
            ),
            tokenizer=None,
            num_actors=num_actors,
            force=cfg.force,
            sample=str(ratio),  # Use ratio as sampling spec
            sample_seed=42 + hash(split_name) % 1000,  # Different seed per split
        )

        # Run processing for this split
        result = last_mile_process(blend, pipeline_config)
        total_sequences += result.total_sequences
        split_paths[split_name] = result.blend_path

    # Create a combined manifest
    manifest = {
        "train": str(split_paths.get("train", "")),
        "val": str(split_paths.get("val", "")),
        "test": str(split_paths.get("test", "")),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
    }

    manifest_path = cfg.output_dir / "manifest.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time

    # Return artifact pointing to manifest
    artifact = DataBlendsArtifact(
        path=manifest_path,
        total_tokens=0,
        total_sequences=total_sequences,
        elapsed_sec=elapsed,
    )
    return artifact


def main(cfg: RLDataPrepConfig) -> DataBlendsArtifact:
    """Run RL data preparation."""
    # Load data blend
    blend = DataBlend.load(cfg.blend_path)

    # Auto-detect num_actors from CPU count
    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    # Run appropriate processing
    if cfg.split_output == "train_val_test":
        artifact = _run_split_blend(blend, cfg, num_actors)
    else:
        artifact = _run_single_blend(blend, cfg, num_actors)

    artifact.name = f"nano3/rl/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    print_step_complete(data_prep=artifact)
    return artifact


if __name__ == "__main__":
    cli(main, ray=True)
