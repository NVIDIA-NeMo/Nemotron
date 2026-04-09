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

"""Shared base for per-sub-stage RL data preparation scripts.

Provides the common config dataclass, orchestration logic, and entry point
used by all RL sub-stage data_prep.py scripts (RLVR, SWE1, SWE2, RLHF).

Each sub-stage data_prep.py is a thin wrapper that calls ``main()``
with its own default config path and ``resolve_hf_placeholders`` flag.

Design: mirrors the ``_base.py`` pattern used by RL training commands.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep.recipes.rl_local import (
    LocalSplitResult,
    run_resolve_and_split,
    split_local_jsonl,
)
from nemotron.kit import SplitJsonlDataArtifact, print_step_complete
from nemotron.kit import wandb_kit
from nemotron.kit.trackers import InputDatasetInfo
from nemo_runspec.artifacts import ArtifactTrackingResult, log_artifact, setup_artifact_tracking
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

logger = logging.getLogger(__name__)

# Use NEMO_RUN_DIR for output when running via nemo-run (avoids writing to code dir)
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


@dataclass
class SubStageDataPrepConfig:
    """Configuration for per-sub-stage RL data preparation.

    Each sub-stage (RLVR, SWE1, SWE2, RLHF) uses this config with
    different defaults for input_path and output_dir.

    Attributes:
        input_path: Path to the source JSONL file (e.g., /lustre/.../rlvr1.jsonl).
        output_dir: Output directory for processed data.
        val_holdout: Number of rows to hold out for validation (from end of file).
        sample: Limit total rows per dataset (for quick tests).
        force: Force new run, ignoring cache.
        resolve_hf_placeholders: Whether to resolve _hf_placeholder entries.
            True for RLVR stages (DAPO/Skywork placeholders), False for SWE/RLHF.
        execution_mode: Pipeline execution mode for placeholder resolution.
    """

    input_path: Path = field(default_factory=lambda: Path("data.jsonl"))
    """Path to source JSONL file"""

    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/super3/stage2_rl")
    """Output directory for processed data"""

    val_holdout: int = 100
    """Number of rows to hold out for validation (from end of file)"""

    sample: int | None = None
    """Limit total rows (for quick tests)"""

    force: bool = False
    """Force new run, ignoring cache"""

    resolve_hf_placeholders: bool = False
    """Whether to resolve _hf_placeholder entries (True for RLVR)"""

    execution_mode: str = "auto"
    """Pipeline execution mode: 'auto', 'streaming', or 'batch'"""

    def __post_init__(self) -> None:
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Add sample suffix to output_dir if sampling
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def run_substage_data_prep(
    cfg: SubStageDataPrepConfig,
    tracking: ArtifactTrackingResult | None = None,
) -> SplitJsonlDataArtifact:
    """Run data preparation for an RL sub-stage.

    Dispatches to either the direct split path (SWE/RLHF) or the
    pipeline path with placeholder resolution (RLVR).

    Args:
        cfg: Sub-stage data prep configuration.
        tracking: Optional artifact tracking config.

    Returns:
        SplitJsonlDataArtifact with paths to train/val JSONL files.
    """
    start_time = time.time()
    stage_name = cfg.input_path.stem if cfg.input_path else "unknown"

    # Add stage-specific tags to wandb
    wandb_kit.add_run_tags(["data-prep", "rl", stage_name])
    wandb_kit.log_wandb_config(cfg)

    # Dispatch based on whether placeholder resolution is needed
    if cfg.resolve_hf_placeholders:
        logger.info(
            f"Running pipeline path (placeholder resolution) for {cfg.input_path}"
        )
        result = run_resolve_and_split(
            input_path=cfg.input_path,
            output_dir=cfg.output_dir,
            val_holdout=cfg.val_holdout,
            sample=cfg.sample,
            force=cfg.force,
            execution_mode=cfg.execution_mode,
        )
    else:
        logger.info(f"Running direct split for {cfg.input_path}")
        result = split_local_jsonl(
            input_path=cfg.input_path,
            output_dir=cfg.output_dir,
            val_holdout=cfg.val_holdout,
            sample=cfg.sample,
            force=cfg.force,
        )

    elapsed = time.time() - start_time

    # Build source dataset info for lineage
    source_datasets = [
        InputDatasetInfo(
            uri=str(cfg.input_path),
            name=stage_name,
            num_rows=result.train_rows + result.val_rows,
        )
    ]

    # Build artifact
    artifact = SplitJsonlDataArtifact(
        path=Path(result.manifest_path),
        total_sequences=result.train_rows + result.val_rows,
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
        train=result.train_path,
        val=result.val_path,
        test=None,
    )

    sample_suffix = f"?sample={cfg.sample}" if cfg.sample else ""
    artifact.name = f"super3/rl/{stage_name}/data{sample_suffix}"

    # Log to all active backends
    if tracking is not None:
        log_artifact(artifact, tracking)
    else:
        artifact.save()

    # Finish W&B and print completion
    wandb_kit.finish_run(exit_code=0)
    print_step_complete(data_prep=artifact)

    return artifact


def main(
    default_config: Path,
    resolve_hf_placeholders: bool,
    cfg: SubStageDataPrepConfig | None = None,
) -> SplitJsonlDataArtifact:
    """Generic entry point for sub-stage data preparation.

    Each sub-stage data_prep.py calls this with its defaults.

    Args:
        default_config: Path to the default YAML config file.
        resolve_hf_placeholders: Whether to resolve HF placeholders.
        cfg: Pre-built config (from CLI framework), or None for direct script usage.

    Returns:
        SplitJsonlDataArtifact with paths to processed data.
    """
    if cfg is None:
        # Called directly as script — parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=default_config
        )

        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        # Setup artifact tracking BEFORE dataclass conversion
        tracking = setup_artifact_tracking(config)

        cfg = omegaconf_to_dataclass(config, SubStageDataPrepConfig)
    else:
        tracking = None

    # Override resolve_hf_placeholders from the script-level default
    # (config can also set this, but script default takes precedence)
    cfg.resolve_hf_placeholders = resolve_hf_placeholders

    # Initialize wandb
    if tracking is None or tracking.wandb:
        init_wandb_from_env()

    return run_substage_data_prep(cfg, tracking=tracking)
