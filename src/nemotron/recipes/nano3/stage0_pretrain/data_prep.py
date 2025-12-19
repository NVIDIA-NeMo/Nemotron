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

Tokenizes raw text data into Megatron bin/idx format.

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

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep import DataPrepConfig, PerSplitConfig, run_data_prep
from nemotron.kit import PretrainBlendsArtifact, print_step_complete
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)
from nemotron.kit.wandb import add_wandb_tags

STAGE_PATH = Path(__file__).parent

# Default config path relative to this file
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class PreTrainDataPrepConfig:
    """Pretrain data preparation config.

    Tokenizes text into Megatron bin/idx format for pretraining.
    Outputs {"train": [...], "valid": [...], "test": [...]} JSON format.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage0_pretrain")
    """Output directory for tokenized data"""

    num_shards: int = 128
    """Number of output shards for parallel loading"""

    valid_shards: int = 1
    """Number of shards for validation split"""

    test_shards: int = 1
    """Number of shards for test split"""

    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    """HuggingFace tokenizer model name"""

    add_bos: bool = False
    """Prepend BOS token to documents"""

    add_eos: bool = True
    """Append EOS token to documents"""

    text_field: str = "text"
    """Default text field name in datasets"""

    min_doc_chars: int | None = None
    """Skip documents shorter than this"""

    max_doc_tokens: int | None = None
    """Truncate documents longer than this"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    config_name: str = "default"
    """Config name used for artifact naming (e.g., 'default', 'tiny', 'test')"""

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
    """Run pretrain data preparation.

    Args:
        cfg: Data prep configuration.

    Returns:
        PretrainBlendsArtifact with paths to tokenized data.
    """
    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "pretrain", cfg.config_name])

    # Log config to W&B
    try:
        import wandb
        from dataclasses import asdict

        if wandb.run is not None:
            # Convert config to dict, handling Path objects
            config_dict = asdict(cfg)
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
            wandb.config.update(config_dict)
    except ImportError:
        pass

    # Build artifact name using config_name.
    # Example: "nano3/default/data" or "nano3/tiny/data?sample=100".
    sample_suffix = f"?sample={cfg.sample}" if cfg.sample else ""
    artifact_name = f"nano3/{cfg.config_name}/data{sample_suffix}"

    data_prep_config = DataPrepConfig(
        blend_path=cfg.blend_path,
        output_dir=cfg.output_dir,
        num_shards=cfg.num_shards,
        per_split=PerSplitConfig(
            enabled=True,
            valid_shards=cfg.valid_shards,
            test_shards=cfg.test_shards,
        ),
        tokenizer_model=cfg.tokenizer_model,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        text_field=cfg.text_field,
        min_doc_chars=cfg.min_doc_chars,
        max_doc_tokens=cfg.max_doc_tokens,
        sample=cfg.sample,
        num_actors=cfg.num_actors,
        force=cfg.force,
        artifact_name=artifact_name,
    )
    artifact = run_data_prep(data_prep_config)
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
