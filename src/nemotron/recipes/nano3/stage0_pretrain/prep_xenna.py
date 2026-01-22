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

"""Data preparation for Nano3 pretraining using Xenna execution."""

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
    """Pretrain data preparation config."""

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_blend_raw.json")
    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/nano3/stage0_pretrain")
    num_shards: int = 128
    valid_shards: int = 1
    test_shards: int = 1
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    add_bos: bool = False
    add_eos: bool = True
    text_field: str = "text"
    min_doc_chars: int | None = None
    max_doc_tokens: int | None = None
    sample: int | None = None
    num_actors: int | None = None
    ray_data_max_actors: int | None = None
    xenna_max_shard_workers: int | None = None
    force: bool = False
    config_name: str = "default"

    def __post_init__(self) -> None:
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def run_data_prep_main(cfg: PreTrainDataPrepConfig) -> PretrainBlendsArtifact:
    """Run pretrain data preparation (Xenna execution)."""
    add_wandb_tags(["data-prep", "pretrain", cfg.config_name, "xenna"])

    try:
        import wandb
        from dataclasses import asdict

        if wandb.run is not None:
            config_dict = asdict(cfg)
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
            wandb.config.update(config_dict)
    except ImportError:
        pass

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
        force=cfg.force,
        artifact_name=artifact_name,
        console_mode=getattr(cfg, "console_mode", "simple"),
        simple_log_interval_sec=getattr(cfg, "simple_log_interval_sec", 30),
        ray_data_max_actors=cfg.ray_data_max_actors,
        xenna_max_shard_workers=cfg.xenna_max_shard_workers,
        execution_engine="xenna",
    )
    artifact = run_data_prep(data_prep_config)
    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: PreTrainDataPrepConfig | None = None) -> PretrainBlendsArtifact:
    """Entry point for Xenna pretrain data preparation."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        cfg = omegaconf_to_dataclass(config, PreTrainDataPrepConfig)

    init_wandb_from_env()
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
