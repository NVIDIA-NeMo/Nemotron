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

"""Data preparation for Nano3 RL stage using Xenna execution."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import ray

from nemotron.data_prep import (
    DataBlend,
    Dataset,
    OutputConfig,
    PipelineConfig,
    last_mile_process,
)
from nemotron.data_prep.config import DatasetConfig, JsonlOutputConfig
from nemotron.data_prep.discovery import get_dataset_metadata
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
from nemotron.recipes.nano3.stage2_rl.data_prep import RLDataPrepConfig

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "default.yaml"

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


def _run_resolve(
    blend: DataBlend,
    cfg: RLDataPrepConfig,
    num_actors: int,
    source_datasets: list[InputDatasetInfo],
) -> SplitJsonlDataArtifact:
    from datasets import get_dataset_split_names

    start_time = time.time()
    total_sequences = 0
    split_paths: dict[str, Path] = {}

    if len(blend.datasets) != 1:
        raise ValueError(
            f"Resolve mode expects exactly one dataset in blend, got {len(blend.datasets)}."
        )

    dataset = blend.datasets[0]

    dataset_path = dataset.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]

    available_splits = get_dataset_split_names(dataset_path)

    split_name_mapping = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for hf_split in available_splits:
        output_split_name = split_name_mapping.get(hf_split, hf_split)
        split_output_dir = cfg.output_dir / output_split_name

        split_blend = DataBlend(
            datasets=[
                Dataset(
                    name=dataset.name,
                    path=dataset.path,
                    split=hf_split,
                    subset=dataset.subset,
                    weight=1.0,
                    text_field=dataset.text_field,
                )
            ]
        )

        format_config = JsonlOutputConfig(
            shard_size=cfg.shard_size,
            transform=None,
            resolve_hf_placeholders=True,
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
            execution_engine="xenna",
        )

        result = last_mile_process(split_blend, pipeline_config)
        total_sequences += result.total_sequences

        shard_prefix = result.splits["all"].data_paths[1]
        shard_files = sorted(Path(shard_prefix).parent.glob("shard_*.jsonl"))
        if shard_files:
            jsonl_path = shard_files[0]
        else:
            raise FileNotFoundError(f"No JSONL shard files found at {shard_prefix}")

        split_paths[output_split_name] = str(jsonl_path.resolve())

    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_split_paths = {k: str(Path(v).resolve()) for k, v in split_paths.items() if v}

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

    artifact = SplitJsonlDataArtifact(
        path=manifest_path,
        total_sequences=total_sequences,
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
        train=resolved_split_paths.get("train"),
        val=resolved_split_paths.get("val"),
        test=resolved_split_paths.get("test"),
    )

    return artifact


def _init_ray_with_hf_env() -> None:
    """Initialize Ray with runtime_env for HF_HOME and HF_TOKEN propagation."""
    if not ray.is_initialized():
        runtime_env = {
            "excludes": [
                "output/",
                "outputs/",
                "wandb/",
                "data/",
                "checkpoints/",
                "*.bin",
                "*.idx",
                "*.npy",
                "__pycache__/",
                ".git/",
                ".venv/",
                "*.egg-info/",
            ],
            "env_vars": {},
        }
        if os.environ.get("HF_HOME"):
            runtime_env["env_vars"]["HF_HOME"] = os.environ["HF_HOME"]
        if os.environ.get("HF_TOKEN"):
            runtime_env["env_vars"]["HF_TOKEN"] = os.environ["HF_TOKEN"]
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)


def run_data_prep_main(cfg: RLDataPrepConfig) -> SplitJsonlDataArtifact:
    """Run RL data preparation with placeholder resolution (Xenna)."""
    add_wandb_tags(["data-prep", "rl", "xenna"])

    # Initialize Ray with HF environment propagation
    _init_ray_with_hf_env()

    blend = DataBlend.load(cfg.blend_path)

    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

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

    try:
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
    except Exception:
        pass

    artifact = _run_resolve(blend, cfg, num_actors, source_datasets)

    artifact.name = f"nano3/rl/data-resolved{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    artifact.save()

    finish_wandb(exit_code=0)
    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: RLDataPrepConfig | None = None) -> SplitJsonlDataArtifact:
    """Entry point for Xenna RL data preparation."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        cfg = omegaconf_to_dataclass(config, RLDataPrepConfig)

    init_wandb_from_env()
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
