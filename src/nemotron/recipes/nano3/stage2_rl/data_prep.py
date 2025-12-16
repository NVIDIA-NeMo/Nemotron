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

"""Data preparation for Nano3 RL stage.

Processes the nvidia/Nemotron-3-Nano-RL-Training-Blend dataset and resolves
placeholder entries that reference external HuggingFace datasets (DAPO, Skywork).

Placeholder records have an `_hf_placeholder` field containing row indices and
question templates. This script:
1. Detects placeholder records by the presence of `_hf_placeholder` field
2. Fetches the actual data from the external HF dataset
3. Applies template restoration (DAPO prefix/suffix, Skywork {question} replacement)
4. Outputs resolved JSONL with train/val/test splits

For simple copy/passthrough (no placeholder resolution), use data_prep_copy.py instead.

Usage:
    # With default config
    python data_prep.py

    # With custom config file
    python data_prep.py --config /path/to/config.yaml

    # With CLI overrides (Hydra-style)
    python data_prep.py sample=100 force=true

    # Via nemotron CLI with nemo-run
    nemotron nano3 data prep rl --sample 10000
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep import (
    DataBlend,
    Dataset,
    OutputConfig,
    PipelineConfig,
    last_mile_process,
)
from nemotron.data_prep.config import DatasetConfig, JsonlOutputConfig
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.data_prep.formats.transforms import resolve_hf_placeholders
from nemotron.data_prep.hf_placeholder import HFPlaceholderResolver
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
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class RLDataPrepConfig:
    """RL data preparation config with HuggingFace placeholder resolution.

    Processes nvidia/Nemotron-3-Nano-RL-Training-Blend and resolves placeholder
    entries by fetching from external datasets (DAPO, Skywork).

    Outputs JSONL with resolved records containing:
    - question: Full question text with template applied
    - expected_answer: Answer from source dataset
    - responses_create_params: OpenAI-format messages for RL training

    For simple copy/passthrough, use data_prep_copy.py instead.
    """

    blend_path: Path = field(
        default_factory=lambda: STAGE_PATH / "config" / "data_prep" / "data_blend_raw.json"
    )
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage2_rl_resolved")
    """Output directory for resolved JSONL data"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    def __post_init__(self) -> None:
        # Ensure paths are Path objects
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Add sample suffix to output_dir if sampling
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def _run_resolve(
    blend: DataBlend,
    cfg: RLDataPrepConfig,
    num_actors: int,
    source_datasets: list[InputDatasetInfo],
    resolver: HFPlaceholderResolver,
) -> SplitJsonlDataArtifact:
    """Process blend with HuggingFace placeholder resolution.

    Downloads the HF dataset and outputs JSONL files for each split found
    in the dataset (train, validation, test), resolving any placeholder records.
    """
    from datasets import get_dataset_split_names

    start_time = time.time()
    total_sequences = 0
    split_paths: dict[str, Path] = {}

    # Get the dataset from blend (expects single dataset in blend)
    if len(blend.datasets) != 1:
        raise ValueError(
            f"Resolve mode expects exactly one dataset in blend, got {len(blend.datasets)}."
        )

    dataset = blend.datasets[0]

    # Handle hf:// prefix if present
    dataset_path = dataset.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]

    # Discover available splits from HF
    available_splits = get_dataset_split_names(dataset_path)

    # Normalize split names for output directories
    # HF uses "validation" but we output as "val" for consistency
    split_name_mapping = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    # Create the resolve transform with pre-loaded resolver
    transform = resolve_hf_placeholders(resolver=resolver)

    # Process each split
    for hf_split in available_splits:
        output_split_name = split_name_mapping.get(hf_split, hf_split)
        split_output_dir = cfg.output_dir / output_split_name

        # Create a single-dataset blend for this split
        split_blend = DataBlend(
            datasets=[
                Dataset(
                    name=dataset.name,
                    path=dataset.path,
                    split=hf_split,  # Use the HF split name
                    subset=dataset.subset,
                    weight=1.0,
                    text_field=dataset.text_field,
                )
            ]
        )

        # Build pipeline config with resolve transform
        format_config = JsonlOutputConfig(
            shard_size=cfg.shard_size,
            transform=transform,
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

    # Resolve output_dir to absolute path for W&B artifact storage
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve split paths to absolute paths
    resolved_split_paths = {k: str(Path(v).resolve()) for k, v in split_paths.items() if v}

    # Create a combined manifest with absolute paths
    manifest = {
        "train": resolved_split_paths.get("train", ""),
        "val": resolved_split_paths.get("val", ""),
        "test": resolved_split_paths.get("test", ""),
        "mode": "resolve",
        "source_splits": available_splits,
    }

    manifest_path = output_dir / "manifest.json"
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

    # Add split paths to metadata for artifact resolution (using resolved absolute paths)
    for split_name, split_path in resolved_split_paths.items():
        artifact.metadata[split_name] = split_path

    return artifact


def run_data_prep_main(cfg: RLDataPrepConfig) -> SplitJsonlDataArtifact:
    """Run RL data preparation with placeholder resolution.

    Args:
        cfg: Resolve data prep configuration.

    Returns:
        SplitJsonlDataArtifact with paths to resolved JSONL data.
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

    # Pre-load the HF placeholder resolver (loads DAPO and Skywork datasets)
    print("Loading external HuggingFace datasets for placeholder resolution...")
    resolver = HFPlaceholderResolver.create()

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

    # Add external placeholder datasets (DAPO, Skywork) for lineage tracking
    for ext_ds_info in resolver.get_loaded_datasets_info():
        source_datasets.append(
            InputDatasetInfo(
                uri=ext_ds_info["uri"],
                name=ext_ds_info["name"],
                split=ext_ds_info["split"],
                num_rows=ext_ds_info["num_rows"],
            )
        )

    # Run resolve processing
    artifact = _run_resolve(blend, cfg, num_actors, source_datasets, resolver)

    artifact.name = f"nano3/rl/data-resolved{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    # Mark wandb run as successful
    finish_wandb(exit_code=0)

    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: RLDataPrepConfig | None = None) -> SplitJsonlDataArtifact:
    """Entry point for RL data preparation with placeholder resolution.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        SplitJsonlDataArtifact with paths to resolved JSONL data.
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
