#!/usr/bin/env python3

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

"""Data preparation for Nano3 RL stage (merge mode).

Merges datasets from a raw data blend into a single JSONL output, then splits
into train/val/test based on configurable ratios.

This is useful when you have multiple source datasets that need to be combined
and then split into training splits, rather than using pre-existing HF splits.

For placeholder resolution (DAPO, Skywork datasets), use data_prep.py instead.

Usage:
    # With default config
    python data_prep_merge.py

    # With custom config file
    python data_prep_merge.py --config /path/to/config.yaml

    # With CLI overrides (Hydra-style)
    python data_prep_merge.py sample=100 force=true train_ratio=0.9

    # Via nemotron CLI with nemo-run
    nemotron nano3 data prep rl --run data_prep_merge --sample 10000
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep import (
    DataBlend,
    OutputConfig,
    PipelineConfig,
    last_mile_process,
)
from nemotron.data_prep.config import DatasetConfig, JsonlOutputConfig
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.data_prep.formats.transforms import passthrough
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
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "merge.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class RLMergeDataPrepConfig:
    """RL data preparation config (merge mode).

    Merges datasets from a raw data blend into JSONL, then splits into
    train/val/test based on configurable ratios.
    """

    blend_path: Path = field(
        default_factory=lambda: STAGE_PATH / "config" / "data_prep" / "data_blend_raw.json"
    )
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage2_rl")
    """Output directory for JSONL data"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.98
    """Fraction of data for training split"""

    valid_ratio: float = 0.01
    """Fraction of data for validation split"""

    test_ratio: float = 0.01
    """Fraction of data for test split"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    seed: int = 42
    """Random seed for shuffling before split"""

    def __post_init__(self) -> None:
        # Ensure paths are Path objects
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate split ratios sum to 1.0
        total_ratio = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} "
                f"(train={self.train_ratio}, valid={self.valid_ratio}, test={self.test_ratio})"
            )

        # Add sample suffix to output_dir if sampling
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def _split_jsonl_files(
    shards_dir: Path,
    output_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, dict]:
    """Load all JSONL shards, shuffle, split by ratio, and save to split directories.

    Args:
        shards_dir: Directory containing shard_*.jsonl files from pipeline.
        output_dir: Directory to write train/, val/, test/ subdirectories.
        train_ratio: Fraction of data for training.
        valid_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for test.
        seed: Random seed for shuffling before split.

    Returns:
        Dict with split statistics and paths.
    """
    import glob
    import random

    # Find all shard files
    shard_pattern = str(shards_dir / "**" / "*.jsonl")
    shard_files = sorted(glob.glob(shard_pattern, recursive=True))

    if not shard_files:
        raise ValueError(f"No JSONL shard files found matching {shard_pattern}")

    print(f"Loading {len(shard_files)} shard files from {shards_dir}")

    # Load all records from shards
    all_records: list[str] = []
    for shard_file in shard_files:
        with open(shard_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(line)

    total_records = len(all_records)
    print(f"Total records loaded: {total_records}")

    # Shuffle before splitting (for reproducibility)
    random.seed(seed)
    random.shuffle(all_records)

    # Calculate split boundaries
    train_end = int(total_records * train_ratio)
    valid_end = train_end + int(total_records * valid_ratio)

    # Split records
    train_records = all_records[:train_end]
    valid_records = all_records[train_end:valid_end]
    test_records = all_records[valid_end:]

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write split files
    def write_split(records: list[str], split_dir: Path, split_name: str) -> Path:
        output_path = split_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for record in records:
                f.write(record + "\n")
        print(f"Saved {len(records)} records to {output_path}")
        return output_path

    train_path = write_split(train_records, train_dir, "train")
    val_path = write_split(valid_records, val_dir, "val")
    test_path = write_split(test_records, test_dir, "test")

    return {
        "train": {"sequences": len(train_records), "path": train_path},
        "val": {"sequences": len(valid_records), "path": val_path},
        "test": {"sequences": len(test_records), "path": test_path},
        "total_sequences": total_records,
    }


def _run_merge(
    blend: DataBlend,
    cfg: RLMergeDataPrepConfig,
    num_actors: int,
    source_datasets: list[InputDatasetInfo],
) -> SplitJsonlDataArtifact:
    """Merge datasets and split into train/val/test.

    Processes all datasets in the blend, merges them together, shuffles,
    and splits into train/val/test based on configured ratios.
    """
    start_time = time.time()

    # Use a temporary shards directory for pipeline output
    shards_dir = cfg.output_dir / "_shards"

    # Build pipeline config with passthrough transform
    format_config = JsonlOutputConfig(
        shard_size=cfg.shard_size,
        transform=passthrough(),
    )

    pipeline_config = PipelineConfig(
        output=OutputConfig(
            dir=shards_dir,
            format=format_config,
            max_rows=cfg.sample,
        ),
        tokenizer=None,
        num_actors=num_actors,
        force=cfg.force,
    )

    # Run processing pipeline to generate merged shards
    print("Running pipeline to merge datasets...")
    result = last_mile_process(blend, pipeline_config)

    # Split the merged data by ratio
    print("Splitting merged data by ratio...")
    split_stats = _split_jsonl_files(
        shards_dir=shards_dir,
        output_dir=cfg.output_dir,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    # Clean up intermediate shards directory
    if shards_dir.exists():
        print(f"Cleaning up intermediate shards directory: {shards_dir}")
        shutil.rmtree(shards_dir)

    # Create a combined manifest
    manifest = {
        "train": str(split_stats["train"]["path"]),
        "val": str(split_stats["val"]["path"]),
        "test": str(split_stats["test"]["path"]),
        "mode": "merge",
        "train_ratio": cfg.train_ratio,
        "valid_ratio": cfg.valid_ratio,
        "test_ratio": cfg.test_ratio,
        "seed": cfg.seed,
        "total_sequences": split_stats["total_sequences"],
    }

    manifest_path = cfg.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time

    # Build artifact
    artifact = SplitJsonlDataArtifact(
        path=manifest_path,
        total_sequences=split_stats["total_sequences"],
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
    )

    # Add split paths to metadata for artifact resolution
    artifact.metadata["train"] = str(split_stats["train"]["path"])
    artifact.metadata["val"] = str(split_stats["val"]["path"])
    artifact.metadata["test"] = str(split_stats["test"]["path"])

    return artifact


def run_data_prep_main(cfg: RLMergeDataPrepConfig) -> SplitJsonlDataArtifact:
    """Run RL data preparation (merge mode).

    Args:
        cfg: RL merge data prep configuration.

    Returns:
        SplitJsonlDataArtifact with paths to JSONL data.
    """
    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "rl", "merge"])

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

    # Run merge processing
    artifact = _run_merge(blend, cfg, num_actors, source_datasets)

    artifact.name = f"nano3/rl/data-merged{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    # Mark wandb run as successful
    finish_wandb(exit_code=0)

    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: RLMergeDataPrepConfig | None = None) -> SplitJsonlDataArtifact:
    """Entry point for RL data preparation (merge mode).

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
        cfg = omegaconf_to_dataclass(config, RLMergeDataPrepConfig)

    # Initialize wandb from environment variables (set by nemo-run)
    init_wandb_from_env()

    # Run data prep
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
