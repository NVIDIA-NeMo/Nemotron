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

"""Data preparation for Nano3 SFT stage.

Applies chat templates to OpenAI-format messages, tokenizes with role-based
loss masking, and outputs packed .npy files compatible with GPTSFTPackedDataset.

Output structure (Megatron-Bridge compatible):
    output_dir/
        training_{pack_size}.npy    # All training data concatenated
        validation_{pack_size}.npy  # All validation data concatenated
        test_{pack_size}.npy        # All test data concatenated
        {pack_size}_metadata.jsonl  # Megatron-Bridge compatible packing metadata
        metadata.json               # Nemotron metadata with split info

Compatible with Megatron-Bridge's FinetuningDatasetConfig with PackedSequenceSpecs.

Pipeline:
1. Apply nano3 chat template → role-labeled chunks
2. Tokenize chunks → input_ids
3. Build loss_mask (0=system/user, 1=assistant)
4. Pack sequences → sharded .npy files
5. Concatenate shards and split by ratio → single .npy per split

Usage:
    # With default config
    python data_prep.py

    # With custom config file
    python data_prep.py --config /path/to/config.yaml

    # With CLI overrides (Hydra-style)
    python data_prep.py sample=100 force=true

    # Via nemotron CLI with nemo-run
    nemotron nano3 data prep sft --run prep --sample 10000
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nemotron.data_prep import (
    ChatSftOutputConfig,
    DataBlend,
    OutputConfig,
    PipelineConfig,
    TokenizerConfig,
    last_mile_process,
)
from nemotron.data_prep.config import DatasetConfig
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.kit import SFTDataArtifact, print_step_complete
from nemotron.kit.trackers import InputDatasetInfo, tokenizer_to_uri
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)
from nemotron.kit.wandb import add_wandb_tags, finish_wandb

logger = logging.getLogger(__name__)

STAGE_PATH = Path(__file__).parent

# Default config path relative to this file
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class SFTDataPrepConfig:
    """SFT data preparation config using chat template.

    Applies chat templates to OpenAI-format messages, tokenizes with role-based
    loss masking, and outputs single packed .npy files per split (training.npy,
    validation.npy, test.npy) compatible with Megatron-Bridge's FinetuningDatasetConfig.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "stage1_sft")
    """Output directory for packed .npy data"""

    # Tokenizer
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    """HuggingFace tokenizer model name"""

    # Packing
    pack_size: int = 4096
    """Maximum tokens per packed sequence"""

    shard_size: str = "256MB"
    """Target size per shard (e.g., '256MB', '1GB')"""

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.98
    """Fraction of data for training split"""

    valid_ratio: float = 0.01
    """Fraction of data for validation split"""

    test_ratio: float = 0.01
    """Fraction of data for test split"""

    # Chat template
    chat_template: str = "nano3"
    """Chat template: 'nano3', path to .jinja file, or inline template"""

    messages_field: str = "messages"
    """Field name for OpenAI-format messages in input records"""

    tools_field: str = "tools"
    """Field name for tools definition in input records"""

    used_in_filter: str | None = None
    """Filter to only include records where used_in contains this value (e.g., 'nano_v3')"""

    used_in_field: str = "used_in"
    """Field name for used_in filtering"""

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


def _concatenate_and_split_npy(
    shards_dir: Path,
    output_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    pack_size: int,
    seed: int = 42,
    data_paths: list[str] | None = None,
) -> dict:
    """Load all shards, concatenate, split by ratio, and save single files per split.

    Args:
        shards_dir: Directory containing shard_*.npy files from pipeline.
        output_dir: Directory to write training.npy, validation.npy, test.npy.
        train_ratio: Fraction of data for training.
        valid_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for test.
        pack_size: Pack size (for metadata).
        seed: Random seed for shuffling before split.
        data_paths: Optional list of weight/path pairs from PipelineResult.
            Format: ["weight", "path_prefix", ...] where path_prefix is the
            shard path prefix (without the _XXXXXX.npy suffix).

    Returns:
        Dict with split statistics: {
            "train": {"sequences": N, "path": "..."},
            "valid": {"sequences": N, "path": "..."},
            "test": {"sequences": N, "path": "..."},
            "total_sequences": N,
        }
    """
    # Find all shard files
    shard_files: list[str] = []

    if data_paths:
        # Extract shard prefixes from data_paths (format: ["weight", "path", ...])
        # and find all matching shard files
        for i in range(1, len(data_paths), 2):
            prefix = data_paths[i]
            # Find all shard files matching this prefix
            pattern = f"{prefix}_*.npy"
            matching = sorted(glob.glob(pattern))
            shard_files.extend(matching)

    if not shard_files:
        # Fallback: try recursive search under shards_dir
        shard_pattern = str(shards_dir / "**" / "shard_*.npy")
        shard_files = sorted(glob.glob(shard_pattern, recursive=True))

    if not shard_files:
        raise ValueError(f"No shard files found matching {shard_pattern}")

    logger.info(f"Loading {len(shard_files)} shard files from {shards_dir}")

    # Load and concatenate all shards
    all_sequences = []
    for shard_file in shard_files:
        shard_data = np.load(shard_file, allow_pickle=True)
        all_sequences.extend(shard_data)
        logger.debug(f"Loaded {len(shard_data)} sequences from {shard_file}")

    total_sequences = len(all_sequences)
    logger.info(f"Total sequences loaded: {total_sequences}")

    # Shuffle before splitting (for reproducibility)
    rng = np.random.default_rng(seed)
    indices = np.arange(total_sequences)
    rng.shuffle(indices)

    # Calculate split boundaries
    train_end = int(total_sequences * train_ratio)
    valid_end = train_end + int(total_sequences * valid_ratio)

    # Split indices
    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    # Extract sequences for each split
    train_sequences = [all_sequences[i] for i in train_indices]
    valid_sequences = [all_sequences[i] for i in valid_indices]
    test_sequences = [all_sequences[i] for i in test_indices]

    # Ensure output directory exists and resolve to absolute path
    # This is critical for W&B artifacts - paths must be absolute for remote access
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split as a single .npy file with pack_size in filename
    # Format: {split}_{pack_size}.npy (Megatron-Bridge compatible)
    train_path = output_dir / f"training_{pack_size}.npy"
    valid_path = output_dir / f"validation_{pack_size}.npy"
    test_path = output_dir / f"test_{pack_size}.npy"

    np.save(train_path, train_sequences, allow_pickle=True)
    logger.info(f"Saved {len(train_sequences)} training sequences to {train_path}")

    np.save(valid_path, valid_sequences, allow_pickle=True)
    logger.info(f"Saved {len(valid_sequences)} validation sequences to {valid_path}")

    np.save(test_path, test_sequences, allow_pickle=True)
    logger.info(f"Saved {len(test_sequences)} test sequences to {test_path}")

    # Compute packing statistics for Megatron-Bridge compatible metadata
    def compute_packing_stats(sequences: list) -> dict:
        """Compute packing statistics for a list of sequences."""
        if not sequences:
            return {
                "max_samples_per_bin": 0,
                "dataset_max_seqlen": 0,
                "packing_factor": 0.0,
                "packing_efficiency": 0.0,
                "pack_size": pack_size,
                "min_packed_seqlen": 0,
            }
        # Count samples per bin and sequence lengths
        samples_per_bin = []
        packed_seqlens = []
        max_seqlen = 0
        for seq in sequences:
            num_samples = len(seq.get("seq_start_id", [1]))  # Default to 1 if no boundaries
            samples_per_bin.append(num_samples)
            seq_len = len(seq.get("input_ids", []))
            packed_seqlens.append(seq_len)
            # Track max sequence length across all sub-sequences
            for i, start in enumerate(seq.get("seq_start_id", [0])):
                if i + 1 < len(seq.get("seq_start_id", [])):
                    end = seq["seq_start_id"][i + 1]
                else:
                    end = len(seq.get("input_ids", []))
                max_seqlen = max(max_seqlen, end - start)

        total_tokens = sum(packed_seqlens)

        packing_factor = round(sum(samples_per_bin) / len(sequences), 2) if sequences else 0.0
        packing_efficiency = (
            round(total_tokens / (len(sequences) * pack_size) * 100, 2) if sequences else 0.0
        )
        return {
            "max_samples_per_bin": max(samples_per_bin) if samples_per_bin else 0,
            "dataset_max_seqlen": max_seqlen,
            "packing_factor": packing_factor,
            "packing_efficiency": packing_efficiency,
            "pack_size": pack_size,
            "min_packed_seqlen": min(packed_seqlens) if packed_seqlens else 0,
        }

    # Compute stats for each split
    train_stats = compute_packing_stats(train_sequences)
    valid_stats = compute_packing_stats(valid_sequences)
    test_stats = compute_packing_stats(test_sequences)

    # Write Megatron-Bridge compatible metadata.jsonl
    # Format: JSON list with one entry per dataset (train, valid, test)
    mb_metadata_path = output_dir / f"{pack_size}_metadata.jsonl"
    mb_metadata = [train_stats, valid_stats, test_stats]
    with open(mb_metadata_path, "w") as f:
        json.dump(mb_metadata, f, indent=2)
    logger.info(f"Saved Megatron-Bridge compatible metadata to {mb_metadata_path}")

    # Write nemotron metadata.json (backward compatible)
    metadata = {
        "pack_size": pack_size,
        "seed": seed,
        "splits": {
            "train": {"sequences": len(train_sequences), "path": str(train_path), **train_stats},
            "valid": {"sequences": len(valid_sequences), "path": str(valid_path), **valid_stats},
            "test": {"sequences": len(test_sequences), "path": str(test_path), **test_stats},
        },
        "total_sequences": total_sequences,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return {
        "train": {"sequences": len(train_sequences), "path": str(train_path)},
        "valid": {"sequences": len(valid_sequences), "path": str(valid_path)},
        "test": {"sequences": len(test_sequences), "path": str(test_path)},
        "total_sequences": total_sequences,
        "training_path": str(train_path),
        "validation_path": str(valid_path),
        "test_path": str(test_path),
        "metadata_path": str(mb_metadata_path),
    }


def run_data_prep_main(cfg: SFTDataPrepConfig) -> SFTDataArtifact:
    """Run SFT data preparation with chat template.

    Processes data through pipeline to generate shards, then concatenates
    and splits into single .npy files per split (training.npy, validation.npy, test.npy).

    Args:
        cfg: SFT data prep configuration.

    Returns:
        SFTDataArtifact with paths to packed data.
    """
    import shutil
    import time

    start_time = time.time()

    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "sft"])

    # Load data blend
    blend = DataBlend.load(cfg.blend_path)

    # Auto-detect num_actors from CPU count
    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    # Use a temporary shards directory for pipeline output
    shards_dir = cfg.output_dir / "_shards"

    # Build pipeline config with ChatSftOutputConfig (no per-split config)
    format_config = ChatSftOutputConfig(
        shard_size=cfg.shard_size,
        pack_size=cfg.pack_size,
        chat_template=cfg.chat_template,
        messages_field=cfg.messages_field,
        tools_field=cfg.tools_field,
        used_in_filter=cfg.used_in_filter,
        used_in_field=cfg.used_in_field,
    )

    pipeline_config = PipelineConfig(
        output=OutputConfig(
            dir=shards_dir,
            format=format_config,
            max_doc_tokens=cfg.max_doc_tokens,
            max_rows=cfg.sample,
        ),
        tokenizer=TokenizerConfig(model=cfg.tokenizer_model),
        num_actors=num_actors,
        force=cfg.force,
        # No per_split config - we handle splitting ourselves
    )

    # Run processing pipeline to generate shards
    logger.info("Running pipeline to generate shards...")
    result = last_mile_process(blend, pipeline_config)

    # Concatenate shards and split by ratio
    # Extract data_paths from pipeline result to locate actual shard files
    data_paths = result.splits["all"].data_paths if "all" in result.splits else None
    logger.info("Concatenating shards and splitting by ratio...")
    split_stats = _concatenate_and_split_npy(
        shards_dir=shards_dir,
        output_dir=cfg.output_dir,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
        test_ratio=cfg.test_ratio,
        pack_size=cfg.pack_size,
        data_paths=data_paths,
    )

    # Clean up intermediate shards directory
    if shards_dir.exists():
        logger.info(f"Cleaning up intermediate shards directory: {shards_dir}")
        shutil.rmtree(shards_dir)

    elapsed_sec = time.time() - start_time

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

    # Build output artifact - path points to output_dir (contains training_{pack_size}.npy, etc.)
    # Use resolved absolute path for W&B artifact storage
    artifact = SFTDataArtifact(
        path=cfg.output_dir.resolve(),
        total_tokens=result.total_tokens,
        total_sequences=split_stats["total_sequences"],
        elapsed_sec=elapsed_sec,
        pack_size=cfg.pack_size,
        source_datasets=source_datasets,
        tokenizer_uri=tok_uri,
        training_path=split_stats["training_path"],
        validation_path=split_stats["validation_path"],
        test_path=split_stats["test_path"],
        metadata_path=split_stats["metadata_path"],
    )
    artifact.name = f"nano3/sft/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    # Mark wandb run as successful
    finish_wandb(exit_code=0)

    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: SFTDataPrepConfig | None = None) -> SFTDataArtifact:
    """Entry point for SFT data preparation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        SFTDataArtifact with paths to packed data.
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
        cfg = omegaconf_to_dataclass(config, SFTDataPrepConfig)

    # Initialize wandb from environment variables (set by nemo-run)
    init_wandb_from_env()

    # Run data prep
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
