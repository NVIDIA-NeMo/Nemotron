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

"""Data preparation for Nano3 pretraining stage.

Tokenizes raw text data into Megatron bin/idx format using the
3-stage pipeline: PlanStage → DownloadStage → BinIdxTokenizationStage.

Outputs blend.json with {"train": [...], "valid": [...], "test": [...]} format
compatible with Megatron-Bridge's per_split_data_args_path parameter.

Usage:
    # With default config
    uv run python data_prep.py

    # With custom config file
    uv run python data_prep.py --config /path/to/config.yaml

    # With CLI overrides (Hydra-style)
    uv run python data_prep.py sample=100 force=true

    # Via nemotron CLI with nemo-run
    nemotron nano3 data prep pretrain --run prep --sample 10000
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nemotron.data_prep.blend import DataBlend
from nemotron.data_prep.config import ObservabilityConfig, TokenizerConfig
from nemotron.data_prep.utils.splits import distribute_shards_to_splits
from nemotron.data_prep.recipes import run_pretrain_pipeline
from nemotron.data_prep.stages import (
    BinIdxTokenizationStageConfig,
    DownloadStageConfig,
    PlanStageConfig,
)
from nemotron.kit import PretrainBlendsArtifact, print_step_complete
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
class PreTrainDataPrepConfig:
    """Pretrain data preparation config.

    Tokenizes text into Megatron bin/idx format for pretraining.
    Outputs {"train": [...], "valid": [...], "test": [...]} JSON format.

    Structure:
        - Data: blend_path, output_dir, num_shards, valid_shards, test_shards
        - Tokenizer: nested TokenizerConfig
        - Document processing: text_field, min_doc_chars, max_doc_tokens
        - Stage configs: plan, download, tokenization
        - Pipeline config: observability
        - Run control: sample, force, config_name
    """

    # Data paths
    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_prep/data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage0_pretrain")
    """Output directory for tokenized data"""

    # Output sharding
    num_shards: int = 128
    """Number of output shards for parallel loading"""

    valid_shards: int = 1
    """Number of shards for validation split"""

    test_shards: int = 1
    """Number of shards for test split"""

    # Tokenizer config (nested)
    tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        add_bos=False,
        add_eos=True,
    ))
    """Tokenizer configuration"""

    # Document processing
    text_field: str = "text"
    """Default text field name in datasets"""

    min_doc_chars: int | None = None
    """Skip documents shorter than this"""

    max_doc_tokens: int | None = None
    """Truncate documents longer than this"""

    # Run control
    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    force: bool = False
    """Force new run, ignoring cache"""

    execution_mode: str = "auto"
    """Execution mode: 'auto' (default), 'streaming', or 'batch'.
    'auto' uses STREAMING if cluster CPUs suffice, BATCH otherwise."""

    config_name: str = "default"
    """Config name used for artifact naming"""

    # Stage configs (nested)
    plan: PlanStageConfig = field(default_factory=PlanStageConfig)
    """Configuration for planning stage"""

    download: DownloadStageConfig = field(default_factory=DownloadStageConfig)
    """Configuration for download stage"""

    tokenization: BinIdxTokenizationStageConfig = field(default_factory=BinIdxTokenizationStageConfig)
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

        # Add sample suffix to output_dir if sampling
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def run_data_prep_main(cfg: PreTrainDataPrepConfig) -> PretrainBlendsArtifact:
    """Run pretrain data preparation pipeline.

    Args:
        cfg: Data prep configuration.

    Returns:
        PretrainBlendsArtifact with paths to tokenized data.
    """
    start_time = time.time()

    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "pretrain", cfg.config_name])

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
            f"Nano3 pretrain expects single-blend format (datasets list), "
            f"but got per-split blend from {cfg.blend_path}"
        )

    # Override text_field if needed (preserves legacy behavior)
    if cfg.text_field != "text":
        for d in blend.datasets:
            if d.text_field == "text" or d.text_field is None:
                object.__setattr__(d, "text_field", cfg.text_field)

    # Sampling mode: use single shard for exact sample count
    num_shards_effective = 1 if cfg.sample is not None else cfg.num_shards

    # Run pipeline with structured configs
    format_result = run_pretrain_pipeline(
        blend=blend,
        output_dir=cfg.output_dir,
        tokenizer=cfg.tokenizer,
        num_shards=num_shards_effective,
        text_field_default=cfg.text_field,
        min_doc_chars=cfg.min_doc_chars,
        max_doc_tokens=cfg.max_doc_tokens,
        max_rows=cfg.sample,  # sample acts as max_rows
        force=cfg.force,
        execution_mode=cfg.execution_mode,
        # Stage configs
        plan_stage=cfg.plan,
        download_stage=cfg.download,
        tokenization_stage=cfg.tokenization,
        # Pipeline config
        observability=cfg.observability,
    )

    # Generate per-split blend.json
    blend_data = distribute_shards_to_splits(
        data_paths=format_result.data_paths,
        num_shards=format_result.num_shards,
        valid_shards=cfg.valid_shards,
        test_shards=cfg.test_shards,
    )

    # Ensure output directory exists
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    blend_json_path = cfg.output_dir / "blend.json"
    with open(blend_json_path, "w") as f:
        json.dump(blend_data, f, indent=2)

    # Build artifact using classmethod
    elapsed_sec = time.time() - start_time
    sample_suffix = f"?sample={cfg.sample}" if cfg.sample else ""
    artifact_name = f"nano3/{cfg.config_name}/data{sample_suffix}"

    artifact = PretrainBlendsArtifact.from_result(
        format_result=format_result,
        blend=blend,
        tokenizer_model=cfg.tokenizer.model,
        blend_json_path=blend_json_path,
        text_field_default=cfg.text_field,
        elapsed_sec=elapsed_sec,
        name=artifact_name,
    )
    artifact.save()

    # Finish W&B and print completion
    finish_wandb(exit_code=0)
    print_step_complete(data_prep=artifact)

    return artifact


def main(cfg: PreTrainDataPrepConfig | None = None) -> PretrainBlendsArtifact:
    """Entry point for pretrain data preparation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        PretrainBlendsArtifact with paths to tokenized data.
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
        cfg = omegaconf_to_dataclass(config, PreTrainDataPrepConfig)

    # Initialize wandb from environment variables (set by nemo-run)
    init_wandb_from_env()

    # Run data prep
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
