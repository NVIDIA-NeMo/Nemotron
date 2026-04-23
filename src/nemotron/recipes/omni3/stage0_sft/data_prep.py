#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/data/prep/sft"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker.
# Install the nemotron package with `uv sync --reinstall-package nemotron`.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run python {script} --config {config}"
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

"""Stage data for Omni3 SFT — open HF dataset pull or Energon dataset validation.

Two flavors, selected via the config:

    hf_dataset_id set → HF-Hub flow (open data, no shard building required).
                        The training container pulls from the Hub on-demand;
                        this step just records the dataset ID in the manifest
                        so artifact lineage flows through W&B.

    dataset_path set  → Energon flow (e.g. Valor32k-AVQA). Validates that the
                        prepared shard directory exists and emits a manifest.
                        If the path doesn't exist and a builder_command is
                        provided, runs that first to assemble the shards.

The default config uses the HF flow (CORD-v2) so `nemotron omni3 data prep sft`
works without internal dataset access. Use `-c valor32k` for the Energon path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "data_prep" / "default.yaml"


@dataclass
class Omni3EnergonDataPrepConfig:
    """Configuration for staging an SFT dataset.

    Exactly one of ``hf_dataset_id`` or ``dataset_path`` must be set. HF
    datasets are pulled by the training container from the Hub on demand
    (no local shard building); Energon datasets must already exist as a
    prepared directory at ``dataset_path``.
    """

    dataset_name: str = "cord_v2"
    hf_dataset_id: str | None = None
    dataset_path: Path | None = None
    metadata_dir: Path = Path("output/omni3/stage0_sft/data_prep")
    link_path: Path | None = None
    builder_command: str | None = None
    modality_filter: str | None = None
    sample: int | None = None
    force: bool = False


def _materialize_link(source: Path, target: Path, force: bool) -> None:
    """Create or refresh a convenience symlink."""
    if target.exists() or target.is_symlink():
        if not force:
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def _run_hf_prep(cfg: Omni3EnergonDataPrepConfig) -> Path:
    """HF-Hub flow: emit a manifest referencing the HF dataset ID.

    The training container pulls CORD-v2 (or whatever HF dataset is
    configured) on-demand via Megatron-Bridge's ``vlm-hf`` loader. There
    is no shard building to do here — we just record the dataset ID and
    sample cap for downstream artifact lineage.
    """
    metadata_dir = cfg.metadata_dir.expanduser()
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": cfg.dataset_name,
        "source": "huggingface",
        "hf_dataset_id": cfg.hf_dataset_id,
        "modality_filter": cfg.modality_filter,
        "sample": cfg.sample,
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return manifest_path


def _run_energon_prep(cfg: Omni3EnergonDataPrepConfig) -> Path:
    """Energon flow: validate a prepared dataset and write a manifest."""
    assert cfg.dataset_path is not None  # Enforced by run_data_prep pre-check
    dataset_path = cfg.dataset_path.expanduser()
    metadata_dir = cfg.metadata_dir.expanduser()

    if not dataset_path.exists():
        if not cfg.builder_command:
            raise FileNotFoundError(
                f"Energon dataset not found at {dataset_path}. "
                "Set dataset_path to a prepared Energon directory, set hf_dataset_id "
                "to use the open HF-Hub flow, or provide builder_command to assemble shards."
            )
        command = cfg.builder_command.format(dataset_path=dataset_path, metadata_dir=metadata_dir)
        subprocess.check_call(["bash", "-lc", command])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path still does not exist after builder_command: {dataset_path}")

    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": cfg.dataset_name,
        "source": "energon",
        "dataset_path": str(dataset_path.resolve()),
        "modality_filter": cfg.modality_filter,
        "sample": cfg.sample,
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if cfg.link_path:
        _materialize_link(dataset_path.resolve(), cfg.link_path.expanduser(), cfg.force)

    print(json.dumps(manifest, indent=2))
    return dataset_path


def run_data_prep(cfg: Omni3EnergonDataPrepConfig) -> Path:
    """Dispatch to HF or Energon staging based on which field is set."""
    if cfg.hf_dataset_id and cfg.dataset_path:
        raise ValueError(
            "Set exactly one of hf_dataset_id (HF-Hub flow) or dataset_path (Energon flow), not both."
        )
    if cfg.hf_dataset_id:
        return _run_hf_prep(cfg)
    if cfg.dataset_path:
        return _run_energon_prep(cfg)
    raise ValueError(
        "Either hf_dataset_id or dataset_path must be set. See data_prep/default.yaml "
        "for the open HF default or data_prep/valor32k.yaml for the Energon path."
    )


def main(cfg: Omni3EnergonDataPrepConfig | None = None) -> Path:
    """Entry point for omni3 SFT data prep."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        cfg = omegaconf_to_dataclass(config, Omni3EnergonDataPrepConfig)

    return run_data_prep(cfg)


if __name__ == "__main__":
    main()
