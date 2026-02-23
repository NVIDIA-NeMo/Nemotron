#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nano3/data/prep/rl"
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

"""Data preparation for Nano3 RL stage.

Processes the nvidia/Nemotron-3-Nano-RL-Training-Blend dataset and resolves
placeholder entries that reference external HuggingFace datasets (DAPO, Skywork).

Placeholder records have an `_hf_placeholder` field containing row indices and
question templates. This script:
1. Detects placeholder records by the presence of `_hf_placeholder` field
2. Fetches the actual data from the external HF dataset
3. Applies template restoration (DAPO prefix/suffix, Skywork {question} replacement)
4. Outputs resolved JSONL with train/val/test splits

Uses the cosmos-xenna multi-stage pipeline pattern:
    JsonlPlanStage → DownloadStage → JsonlShardStage

For simple copy/passthrough (no placeholder resolution), use data_prep_copy.py instead.

CLI:
    nemotron nano3 data prep rl                       # local execution
    nemotron nano3 data prep rl --run ray --sample 10000  # submit to cluster

Execution logic: src/nemotron/cli/commands/nano3/data/prep/rl.py

Direct usage:
    python data_prep.py
    python data_prep.py --config /path/to/config.yaml
    python data_prep.py sample=100 force=true
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep.blend import DataBlend
from nemotron.data_prep.config import DatasetConfig
from nemotron.data_prep.utils.discovery import get_dataset_metadata
from nemotron.data_prep.utils.hf_placeholder import HFPlaceholderResolver
from nemotron.data_prep.recipes.rl import run_rl_resolve_pipeline
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

logger = logging.getLogger(__name__)

STAGE_PATH = Path(__file__).parent

# Default config path relative to this file
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True  # Uses cosmos-xenna pipeline (requires Ray runtime)


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

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    force: bool = False
    """Force new run, ignoring cache"""

    execution_mode: str = "auto"
    """Execution mode: 'auto' (default), 'streaming', or 'batch'.
    'auto' uses STREAMING if cluster CPUs suffice, BATCH otherwise."""

    def __post_init__(self) -> None:
        # Ensure paths are Path objects
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Add sample suffix to output_dir if sampling
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def run_data_prep_main(cfg: RLDataPrepConfig) -> SplitJsonlDataArtifact:
    """Run RL data preparation with placeholder resolution.

    Uses the cosmos-xenna multi-stage pipeline:
        JsonlPlanStage → DownloadStage → JsonlShardStage

    Args:
        cfg: Resolve data prep configuration.

    Returns:
        SplitJsonlDataArtifact with paths to resolved JSONL data.
    """
    start_time = time.time()

    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "rl"])

    # Load data blend
    blend = DataBlend.load(cfg.blend_path)

    # Collect source datasets with metadata for lineage tracking
    source_datasets: list[InputDatasetInfo] = []
    seen_keys: set[str] = set()
    for dataset in blend.datasets:
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

    # Run the xenna-native RL resolve pipeline
    result = run_rl_resolve_pipeline(
        blend=blend,
        output_dir=cfg.output_dir,
        sample=cfg.sample,
        force=cfg.force,
        resolve_hf_placeholders=True,
        execution_mode=cfg.execution_mode,
    )

    # Add external placeholder datasets (DAPO, Skywork) for lineage tracking
    # The resolver is loaded on pipeline workers, but we need metadata on the driver
    # for W&B artifact lineage. This is a separate load (unavoidable since the
    # resolver's PyArrow tables are not picklable).
    print("Loading external HuggingFace dataset metadata for lineage tracking...")
    resolver = HFPlaceholderResolver.create()
    for ext_ds_info in resolver.get_loaded_datasets_info():
        source_datasets.append(
            InputDatasetInfo(
                uri=ext_ds_info["uri"],
                name=ext_ds_info["name"],
                split=ext_ds_info["split"],
                num_rows=ext_ds_info["num_rows"],
            )
        )

    elapsed = time.time() - start_time

    # Build artifact with split paths
    artifact = SplitJsonlDataArtifact(
        path=Path(result.manifest_path),
        total_sequences=result.total_records,
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
        train=result.split_paths.get("train"),
        val=result.split_paths.get("val"),
        test=result.split_paths.get("test"),
    )

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
