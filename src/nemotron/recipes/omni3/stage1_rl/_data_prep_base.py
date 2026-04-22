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

"""Shared base for Omni RL data preparation.

Dispatches across the three Omni RL data-prep variants:
- MPO / MMPR-public
- text-only RL / Nemotron-3-Nano-RL-Training-Blend
- vision RL / MMPR-Tiny
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nemotron.kit import Artifact, SplitJsonlDataArtifact, print_step_complete
from nemo_runspec.artifacts import ArtifactTrackingResult, log_artifact, setup_artifact_tracking
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class Omni3RLDataArtifact(Artifact):
    """Artifact for single-file or directory-based Omni RL data outputs."""

    stage: str
    dataset_name: str
    source_uri: str | None = None

    def _get_output_dir(self) -> Path:
        return self.path.parent if self.path.is_file() or self.path.suffix else self.path

    def get_wandb_files(self) -> list[tuple[str, str]]:
        files: list[tuple[str, str]] = []
        if self.path.is_file() and self.path.exists():
            files.append((str(self.path), self.path.name))
        metadata_path = self._get_output_dir() / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        target = self.path.parent if self.path.is_file() else self.path
        return [(f"file://{target.resolve()}", "output")]

    def get_input_uris(self) -> list[str]:
        return [self.source_uri] if self.source_uri else []


@dataclass
class Omni3RLDataPrepConfig:
    """Configuration for Omni RL data preparation."""

    stage: str = "mpo"
    dataset_name: str = "mmpr_public"
    source_uri: str | None = None
    input_dir: Path = Path("data")
    input_file: Path | None = None
    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "output/omni3/stage1_rl/data_prep")
    meta_name: str = "meta_public.json"
    train_file_name: str = "train.jsonl"
    val_file_name: str = "val.jsonl"
    validation_strategy: str = "same_as_train"
    builder_command: str | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _command_context(cfg: Omni3RLDataPrepConfig) -> dict[str, str]:
    input_file = cfg.input_file or (cfg.input_dir / "train.jsonl")
    train_path = cfg.output_dir / cfg.train_file_name
    val_path = cfg.output_dir / cfg.val_file_name
    return {
        "input_dir": str(cfg.input_dir.expanduser()),
        "input_file": str(input_file.expanduser()),
        "output_dir": str(cfg.output_dir.expanduser()),
        "meta_name": cfg.meta_name,
        "train_path": str(train_path.expanduser()),
        "val_path": str(val_path.expanduser()),
    }


def _run_builder_command(
    cfg: Omni3RLDataPrepConfig,
    *,
    required_paths: list[Path],
) -> None:
    required_paths = [path.expanduser() for path in required_paths]
    if not cfg.force and all(path.exists() for path in required_paths):
        return
    if not cfg.builder_command:
        missing = ", ".join(str(path) for path in required_paths if not path.exists())
        raise FileNotFoundError(
            f"Missing prepared outputs for stage={cfg.stage}: {missing}. "
            "Set builder_command or provide the prepared inputs."
        )

    cfg.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    command = cfg.builder_command.format(**_command_context(cfg))
    subprocess.check_call(["bash", "-lc", command])

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Data-prep builder command completed but required outputs are missing: {missing}"
        )


def _prepare_mpo(cfg: Omni3RLDataPrepConfig) -> Omni3RLDataArtifact:
    output_dir = cfg.output_dir.expanduser()
    meta_path = output_dir / cfg.meta_name
    _run_builder_command(cfg, required_paths=[meta_path])
    return Omni3RLDataArtifact(
        path=meta_path,
        stage=cfg.stage,
        dataset_name=cfg.dataset_name,
        source_uri=cfg.source_uri,
        name="omni3/rl/mpo/data",
    )


def _prepare_text(cfg: Omni3RLDataPrepConfig) -> SplitJsonlDataArtifact:
    output_dir = cfg.output_dir.expanduser()
    train_path = output_dir / cfg.train_file_name
    _run_builder_command(cfg, required_paths=[train_path])

    val_path = output_dir / cfg.val_file_name
    if cfg.validation_strategy == "copy":
        if cfg.force or not val_path.exists():
            shutil.copy2(train_path, val_path)
        resolved_val_path = val_path
    elif cfg.validation_strategy == "same_as_train":
        resolved_val_path = train_path
    else:
        raise ValueError(
            "validation_strategy must be one of: same_as_train, copy"
        )

    total_sequences = _count_lines(train_path)
    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "stage": cfg.stage,
            "dataset_name": cfg.dataset_name,
            "train": str(train_path),
            "val": str(resolved_val_path),
            "total_sequences": total_sequences,
        },
    )

    artifact = SplitJsonlDataArtifact(
        path=manifest_path,
        total_sequences=total_sequences,
        elapsed_sec=0.0,
        source_datasets=[
            cfg.source_uri
            or f"file://{(cfg.input_file or (cfg.input_dir / 'train.jsonl')).expanduser()}"
        ],
        train=str(train_path),
        val=str(resolved_val_path),
        test=None,
    )
    artifact.name = "omni3/rl/text/data"
    return artifact


def _prepare_vision(cfg: Omni3RLDataPrepConfig) -> Omni3RLDataArtifact:
    output_dir = cfg.output_dir.expanduser()
    ready_marker = output_dir / ".mmpr_ready"
    parquet_path = output_dir / "mmpr_tiny.parquet"
    _run_builder_command(cfg, required_paths=[ready_marker if cfg.force or ready_marker.exists() else parquet_path])

    if not ready_marker.exists() and not parquet_path.exists():
        raise FileNotFoundError(
            f"Vision RL data prep did not produce {ready_marker} or {parquet_path}"
        )

    return Omni3RLDataArtifact(
        path=output_dir,
        stage=cfg.stage,
        dataset_name=cfg.dataset_name,
        source_uri=cfg.source_uri,
        name="omni3/rl/vision/data",
    )


def run_data_prep(
    cfg: Omni3RLDataPrepConfig,
    tracking: ArtifactTrackingResult | None = None,
):
    """Run the selected Omni RL data-prep variant."""
    start_time = time.time()
    stage = cfg.stage.lower()
    dataset_name = cfg.dataset_name.lower()

    if stage == "mpo" or dataset_name in {"mmpr", "mmpr_public"}:
        artifact = _prepare_mpo(cfg)
    elif stage == "text" or dataset_name in {
        "nemotron_3_nano_rl_training_blend",
        "text_only_rl_stage1",
    }:
        artifact = _prepare_text(cfg)
    elif stage == "vision" or dataset_name == "mmpr_tiny":
        artifact = _prepare_vision(cfg)
    else:
        raise ValueError(
            f"Unsupported omni3 RL data-prep stage={cfg.stage!r} dataset_name={cfg.dataset_name!r}"
        )

    elapsed_sec = time.time() - start_time
    if isinstance(artifact, SplitJsonlDataArtifact):
        artifact.elapsed_sec = elapsed_sec
    else:
        artifact.metadata["elapsed_sec"] = elapsed_sec

    if tracking is not None:
        log_artifact(artifact, tracking)
    else:
        artifact.save()

    print_step_complete(data=artifact)
    return artifact


def main(
    default_config: Path,
    cfg: Omni3RLDataPrepConfig | None = None,
):
    """Generic entry point for Omni RL data-prep wrappers."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=default_config
        )
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        tracking = setup_artifact_tracking(config)
        cfg = omegaconf_to_dataclass(config, Omni3RLDataPrepConfig)
    else:
        tracking = None

    return run_data_prep(cfg, tracking=tracking)
