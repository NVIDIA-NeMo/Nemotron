#!/usr/bin/env python3
"""Data preparation for Nano3 RL stage.

Converts datasets to JSONL format with OpenAI chat messages.

Usage:
    # With default config
    python data_prep.py

    # With custom config file
    python data_prep.py --config /path/to/config.yaml

    # With CLI overrides (Hydra-style)
    python data_prep.py sample=100 force=true

    # Flat output without splitting
    python data_prep.py split_output=none

    # Via nemotron CLI with nemo-run
    nemotron nano3 data prep rl --run prep --sample 10000
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nemotron.data_prep import (
    DataBlend,
    PipelineConfig,
    OutputConfig,
    last_mile_process,
)
from nemotron.data_prep.config import DatasetConfig, JsonlOutputConfig
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.data_prep.formats.transforms import nemotron_rl, passthrough
from nemotron.kit import SplitJsonlDataArtifact, print_step_complete
from nemotron.kit.trackers import InputDatasetInfo
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)
from nemotron.kit.wandb import add_wandb_tags, finish_wandb

STAGE_PATH = Path(__file__).parent

# Default config path relative to this file
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class RLDataPrepConfig:
    """RL data preparation config.

    Converts to JSONL with OpenAI chat format for RLHF training.
    By default outputs train/val/test splits as separate JSONL files.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage2_rl")
    """Output directory for JSONL data"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    # Note: messages_field removed - nemotron_rl transform extracts from
    # responses_create_params.input directly

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    split_output: Literal["none", "train_val_test"] = "train_val_test"
    """Split output into train/val/test directories (default: train_val_test)"""

    train_ratio: float = 0.98
    """Ratio of data for training when split_output='train_val_test'"""

    val_ratio: float = 0.01
    """Ratio of data for validation when split_output='train_val_test'"""

    mode: Literal["process", "copy"] = "copy"
    """Mode: 'copy' uses passthrough transform and HF dataset's existing splits,
    'process' applies nemotron_rl transform and custom splits."""

    def __post_init__(self) -> None:
        # Ensure paths are Path objects
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Add sample suffix to output_dir if sampling
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


def _discover_hf_splits(dataset_path: str) -> list[str]:
    """Discover available splits from a HuggingFace dataset.

    Args:
        dataset_path: HuggingFace dataset path (e.g., "nvidia/Nemotron-3-Nano-RL-Training-Blend")

    Returns:
        List of split names (e.g., ["train", "validation", "test"])
    """
    from datasets import get_dataset_split_names

    # Handle hf:// prefix if present
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]

    return get_dataset_split_names(dataset_path)


def _run_copy_mode(
    blend: DataBlend,
    cfg: RLDataPrepConfig,
    num_actors: int,
    source_datasets: list[InputDatasetInfo],
) -> SplitJsonlDataArtifact:
    """Copy HF dataset using its existing splits with passthrough transform.

    Downloads the HF dataset and outputs JSONL files for each split found
    in the dataset (train, validation, test, etc.) without applying any
    field transforms.
    """
    import json
    import time

    start_time = time.time()
    total_sequences = 0
    split_paths: dict[str, Path] = {}

    # Get the dataset from blend (copy mode expects single dataset in blend)
    if len(blend.datasets) != 1:
        raise ValueError(
            f"Copy mode expects exactly one dataset in blend, got {len(blend.datasets)}. "
            "Use 'process' mode for multi-dataset blends."
        )

    dataset = blend.datasets[0]

    # Discover available splits from HF
    available_splits = _discover_hf_splits(dataset.path)

    # Normalize split names for output directories
    # HF uses "validation" but we output as "val" for consistency
    split_name_mapping = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    # Process each split
    for hf_split in available_splits:
        output_split_name = split_name_mapping.get(hf_split, hf_split)
        split_output_dir = cfg.output_dir / output_split_name

        # Create a single-dataset blend for this split
        split_blend = DataBlend(
            datasets=[
                DataBlend.Dataset(
                    name=dataset.name,
                    path=dataset.path,
                    split=hf_split,  # Use the HF split name
                    subset=dataset.subset,
                    weight=1.0,
                    text_field=dataset.text_field,
                )
            ]
        )

        # Build pipeline config with passthrough transform (no field extraction)
        format_config = JsonlOutputConfig(
            shard_size=cfg.shard_size,
            transform=passthrough(),
        )

        pipeline_config = PipelineConfig(
            output=OutputConfig(
                dir=split_output_dir,
                format=format_config,
                max_rows=cfg.sample,
            ),
            tokenizer=None,
            num_actors=num_actors,
            force=cfg.force,
        )

        # Run processing for this split
        result = last_mile_process(split_blend, pipeline_config)
        total_sequences += result.total_sequences
        split_paths[output_split_name] = result.blend_path

    # Create a combined manifest
    manifest = {
        "train": str(split_paths.get("train", "")),
        "val": str(split_paths.get("val", "")),
        "test": str(split_paths.get("test", "")),
        "mode": "copy",
        "source_splits": available_splits,
    }

    manifest_path = cfg.output_dir / "manifest.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time

    # Build artifact
    artifact = SplitJsonlDataArtifact(
        path=manifest_path,
        total_sequences=total_sequences,
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
    )

    # Add split paths to metadata for artifact resolution
    for split_name, split_path in split_paths.items():
        artifact.metadata[split_name] = str(split_path)

    return artifact


def _run_single_blend(
    blend: DataBlend,
    cfg: RLDataPrepConfig,
    num_actors: int,
    source_datasets: list[InputDatasetInfo],
) -> SplitJsonlDataArtifact:
    """Process blend without train/val/test splitting."""
    # Build pipeline config with JSONL output format
    format_config = JsonlOutputConfig(
        shard_size=cfg.shard_size,
        transform=nemotron_rl(),
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

    # Build output artifact with source datasets for lineage tracking
    # Using SplitJsonlDataArtifact since JSONL doesn't tokenize
    artifact = SplitJsonlDataArtifact(
        path=result.blend_path,
        total_sequences=result.total_sequences,
        elapsed_sec=result.elapsed_sec,
        source_datasets=source_datasets,
    )
    return artifact


def _run_split_blend(
    blend: DataBlend,
    cfg: RLDataPrepConfig,
    num_actors: int,
    source_datasets: list[InputDatasetInfo],
) -> SplitJsonlDataArtifact:
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
            transform=nemotron_rl(),
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

    # Return artifact pointing to manifest with source datasets for lineage
    # Using SplitJsonlDataArtifact since JSONL doesn't tokenize
    artifact = SplitJsonlDataArtifact(
        path=manifest_path,
        total_sequences=total_sequences,
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
    )

    # Add train/val paths directly to metadata for artifact resolution
    # These are used by the rl training command via artifact mappings
    if "train" in split_paths:
        artifact.metadata["train"] = str(split_paths["train"])
    if "val" in split_paths:
        artifact.metadata["val"] = str(split_paths["val"])
    if "test" in split_paths:
        artifact.metadata["test"] = str(split_paths["test"])

    return artifact


def run_data_prep_main(cfg: RLDataPrepConfig) -> SplitJsonlDataArtifact:
    """Run RL data preparation.

    Args:
        cfg: RL data prep configuration.

    Returns:
        SplitJsonlDataArtifact with paths to JSONL data.
    """
    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "rl"])

    # Load data blend
    blend = DataBlend.load(cfg.blend_path)

    # Auto-detect num_actors from CPU count
    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    # Collect source datasets with metadata for lineage tracking
    source_datasets: list[InputDatasetInfo] = []
    seen_keys: set[str] = set()
    for dataset in blend.datasets:
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

    # Run appropriate processing based on mode
    if cfg.mode == "copy":
        artifact = _run_copy_mode(blend, cfg, num_actors, source_datasets)
    elif cfg.split_output == "train_val_test":
        artifact = _run_split_blend(blend, cfg, num_actors, source_datasets)
    else:
        artifact = _run_single_blend(blend, cfg, num_actors, source_datasets)

    artifact.name = f"nano3/rl/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    # Mark wandb run as successful
    finish_wandb(exit_code=0)

    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: RLDataPrepConfig | None = None) -> SplitJsonlDataArtifact:
    """Entry point for RL data preparation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        SplitJsonlDataArtifact with paths to JSONL data.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        # Load YAML config
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Apply CLI overrides (Hydra-style: key=value)
        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        # Convert to dataclass
        cfg = omegaconf_to_dataclass(config, RLDataPrepConfig)

    # Initialize wandb from environment variables (set by nemo-run)
    init_wandb_from_env()

    # Run data prep
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
