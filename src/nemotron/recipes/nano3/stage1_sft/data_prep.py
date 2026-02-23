#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nano3/data/prep/sft"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker.
# Install the nemotron package with xenna extras: uv sync --reinstall-package nemotron.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/data_prep"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

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
loss masking, packs sequences, and outputs packed Parquet shards using the
3-stage pipeline: SftPlanStage → DownloadStage → PackedSftParquetStage.

Output structure (Megatron-Bridge compatible):
    output_dir/
        blend.json                  # Per-split blend {"train": [...], "valid": [...], "test": [...]}
        splits/                     # Canonical split directories for training
            train/                  # Symlinks to training shards
                shard_000000.parquet -> ../runs/.../shard_000000.parquet
                ...
            valid/                  # Symlinks to validation shards
                ...
            test/                   # Symlinks to test shards
                ...
        runs/{run_hash}/            # Run directory
            datasets/{name}/{hash}/ # Per-dataset outputs
                shard_000000.parquet
                ...

Training can use either:
- Split directories: packed_train_data_path=/path/to/output_dir/splits/train/
- Globs: packed_train_data_path=/path/to/output_dir/splits/train/*.parquet
- blend.json for provenance and exact shard paths

Compatible with Megatron-Bridge's FinetuningDatasetConfig with packed Parquet shards.

Pipeline:
1. Apply nano3 chat template → role-labeled chunks
2. Tokenize chunks → input_ids + loss_mask
3. Pack sequences → Parquet shards with seq_start_id
4. Distribute shards to train/valid/test splits

CLI:
    nemotron nano3 data prep sft                       # local execution
    nemotron nano3 data prep sft --run ray --sample 10000  # submit to cluster

Execution logic: src/nemotron/cli/commands/nano3/data/prep/sft.py

Direct usage:
    python data_prep.py
    python data_prep.py --config /path/to/config.yaml
    python data_prep.py sample=100 force=true
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nemotron.data_prep.blend import DataBlend
from nemotron.data_prep.config import ObservabilityConfig, TokenizerConfig
from nemotron.data_prep.utils.splits import distribute_shards_to_splits, realize_packed_shards_into_split_dirs
from nemotron.data_prep.recipes import run_sft_pipeline
from nemotron.data_prep.stages import (
    DownloadStageConfig,
    PackedSftParquetStageConfig,
    SftPlanStageConfig,
)
from nemotron.kit import SFTDataArtifact, print_step_complete
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
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class SFTDataPrepConfig:
    """SFT data preparation config using chat template.

    Applies chat templates to OpenAI-format messages, tokenizes with role-based
    loss masking, and outputs packed Parquet shards with per-split blend.json
    compatible with Megatron-Bridge's FinetuningDatasetConfig.

    Structure:
        - Data: blend_path, output_dir, num_shards
        - Tokenizer: nested TokenizerConfig
        - Packing: pack_size, algorithm, seed, parquet_*
        - Split ratios: train_ratio, valid_ratio, test_ratio
        - Chat: chat_template, messages_field, tools_field, used_in_*
        - Processing: max_doc_tokens
        - Stage configs: plan, download, tokenization
        - Pipeline config: observability
        - Run control: sample, sample_seed, force, config_name
    """

    # Data paths
    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_prep/data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "stage1_sft")
    """Output directory for packed Parquet data"""

    # Sharding
    num_shards: int = 128
    """Number of output shards for parallel loading"""

    # Tokenizer config (nested)
    tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    ))
    """Tokenizer configuration"""

    # Packing
    pack_size: int = 4096
    """Maximum tokens per packed sequence"""

    algorithm: str = "first_fit_shuffle"
    """Packing algorithm: 'first_fit_decreasing', 'first_fit_shuffle', 'concatenative'"""

    seed: int | None = None
    """Packing seed (defaults to sample_seed when None)"""

    # Parquet output options
    parquet_row_group_size: int = 1000
    """Parquet row group size (bins per group)"""

    parquet_compression: str = "zstd"
    """Parquet compression codec: 'zstd', 'snappy', 'gzip', 'none'"""

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

    # Run control
    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    sample_seed: int = 42
    """Random seed for sampling"""

    force: bool = False
    """Force new run, ignoring cache"""

    execution_mode: str = "auto"
    """Execution mode: 'auto' (default), 'streaming', or 'batch'.
    'auto' uses STREAMING if cluster CPUs suffice, BATCH otherwise."""

    config_name: str = "default"
    """Config name used for artifact naming"""

    # Stage configs (nested)
    plan: SftPlanStageConfig = field(default_factory=SftPlanStageConfig)
    """Configuration for planning stage"""

    download: DownloadStageConfig = field(default_factory=DownloadStageConfig)
    """Configuration for download stage"""

    tokenization: PackedSftParquetStageConfig = field(default_factory=PackedSftParquetStageConfig)
    """Configuration for tokenization stage"""

    # Pipeline-level config
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    """Pipeline observability settings"""

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


def run_data_prep_main(cfg: SFTDataPrepConfig) -> SFTDataArtifact:
    """Run SFT data preparation pipeline.

    Args:
        cfg: SFT data prep configuration.

    Returns:
        SFTDataArtifact with paths to packed Parquet data.
    """
    start_time = time.time()

    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "sft"])

    # Log config to W&B
    try:
        import wandb

        if wandb.run is not None:
            config_dict = asdict(cfg)
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
            wandb.config.update(config_dict)
    except ImportError:
        pass

    # Load blend and validate
    blend = DataBlend.load(cfg.blend_path)
    if blend.datasets is None:
        raise ValueError(
            f"Nano3 SFT expects single-blend format (datasets list), "
            f"but got per-split blend from {cfg.blend_path}"
        )

    # Sampling mode: use single shard for exact sample count
    num_shards_effective = 1 if cfg.sample is not None else cfg.num_shards

    # Determine packing seed (defaults to sample_seed)
    packing_seed = cfg.seed if cfg.seed is not None else cfg.sample_seed

    # Run SFT pipeline
    logger.info("Running SFT pipeline...")
    format_result = run_sft_pipeline(
        blend=blend,
        output_dir=cfg.output_dir,
        tokenizer=cfg.tokenizer,
        num_shards=num_shards_effective,
        messages_field_default=cfg.messages_field,
        tools_field_default=cfg.tools_field,
        chat_template=cfg.chat_template,
        used_in_filter=cfg.used_in_filter,
        used_in_field=cfg.used_in_field,
        pack_size=cfg.pack_size,
        algorithm=cfg.algorithm,
        seed=packing_seed,
        parquet_row_group_size=cfg.parquet_row_group_size,
        parquet_compression=cfg.parquet_compression,
        max_doc_tokens=cfg.max_doc_tokens,
        max_rows=cfg.sample,
        sample_seed=cfg.sample_seed,
        force=cfg.force,
        execution_mode=cfg.execution_mode,
        # Stage configs
        plan_stage=cfg.plan,
        download_stage=cfg.download,
        tokenization_stage=cfg.tokenization,
        # Pipeline config
        observability=cfg.observability,
    )

    # Convert ratios to shard counts for per-split distribution
    # Must guarantee at least 1 train shard, so valid+test <= total-1
    total_shards = format_result.num_shards

    def _ratio_to_shards(ratio: float, total: int) -> int:
        """Convert ratio to shard count, respecting ratio=0 as 0 shards."""
        if ratio <= 0.0:
            return 0
        return max(1, int(round(total * ratio)))

    if total_shards <= 2:
        # With very few shards, put everything in train (no valid/test)
        test_shards = 0
        valid_shards = 0
        logger.warning(
            f"Only {total_shards} shard(s) available; skipping valid/test splits. "
            "Increase num_shards or disable sampling for proper split distribution."
        )
    else:
        test_shards = _ratio_to_shards(cfg.test_ratio, total_shards)
        valid_shards = _ratio_to_shards(cfg.valid_ratio, total_shards)

        # Ensure train gets at least 1 shard
        max_non_train = total_shards - 1
        if test_shards + valid_shards > max_non_train:
            # Scale down proportionally, maintaining at least 1 for train
            scale = max_non_train / (test_shards + valid_shards)
            test_shards = max(0, int(test_shards * scale))
            valid_shards = max(0, int(valid_shards * scale))
            # If still over, prefer valid over test
            if test_shards + valid_shards > max_non_train:
                test_shards = max(0, max_non_train - valid_shards)

    # Validate data_paths before split distribution
    if not format_result.data_paths:
        raise ValueError(
            f"Pipeline produced no data_paths. This usually means no shards were completed. "
            f"Check logs for pipeline errors. run_dir={format_result.run_dir}"
        )

    logger.info(
        f"Distributing {format_result.num_shards} shards across splits "
        f"(train: {format_result.num_shards - valid_shards - test_shards}, "
        f"valid: {valid_shards}, test: {test_shards})"
    )

    # Generate per-split blend.json
    blend_data = distribute_shards_to_splits(
        data_paths=format_result.data_paths,
        num_shards=format_result.num_shards,
        valid_shards=valid_shards,
        test_shards=test_shards,
        seed=packing_seed,
    )

    # Validate train split has shards
    train_shard_count = len(blend_data.get("train", [])) // 2  # path_list is [weight, path, ...]
    if train_shard_count == 0:
        raise ValueError(
            f"Train split has no shards after distribution. "
            f"data_paths={format_result.data_paths}, num_shards={format_result.num_shards}"
        )

    logger.info(
        f"Split distribution: train={train_shard_count}, "
        f"valid={len(blend_data.get('valid', [])) // 2}, "
        f"test={len(blend_data.get('test', [])) // 2}"
    )

    # Ensure output directory exists
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    blend_json_path = cfg.output_dir / "blend.json"
    with open(blend_json_path, "w") as f:
        json.dump(blend_data, f, indent=2)

    logger.info(f"Wrote per-split blend.json to {blend_json_path}")

    # Create canonical split directories with symlinks
    split_dirs = realize_packed_shards_into_split_dirs(
        output_dir=cfg.output_dir,
        split_to_paths=blend_data,
    )
    logger.info(f"Created split directories: {list(split_dirs.keys())}")

    elapsed_sec = time.time() - start_time

    # Build artifact using classmethod
    sample_suffix = f"?sample={cfg.sample}" if cfg.sample else ""
    artifact_name = f"nano3/sft/data{sample_suffix}"

    artifact = SFTDataArtifact.from_result(
        format_result=format_result,
        blend=blend,
        tokenizer_model=cfg.tokenizer.model,
        blend_json_path=blend_json_path,
        pack_size=cfg.pack_size,
        messages_field_default=cfg.messages_field,
        elapsed_sec=elapsed_sec,
        name=artifact_name,
    )
    artifact.save()

    # Finish W&B and print completion
    finish_wandb(exit_code=0)
    print_step_complete(data_prep=artifact)

    return artifact


def main(cfg: SFTDataPrepConfig | None = None) -> SFTDataArtifact:
    """Entry point for SFT data preparation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        SFTDataArtifact with paths to packed Parquet data.
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
