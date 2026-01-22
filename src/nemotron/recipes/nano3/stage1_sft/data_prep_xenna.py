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

"""Data preparation for Nano3 SFT stage using Xenna execution."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import ray

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
from nemotron.recipes.nano3.stage1_sft.data_prep import (
    SFTDataPrepConfig,
    _concatenate_and_split_npy,
)

logger = logging.getLogger(__name__)

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep.yaml"
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


def run_data_prep_main(cfg: SFTDataPrepConfig) -> SFTDataArtifact:
    """Run SFT data preparation with Xenna execution."""
    start_time = time.time()
    add_wandb_tags(["data-prep", "sft", "xenna"])

    blend = DataBlend.load(cfg.blend_path)

    num_actors = cfg.num_actors
    if num_actors is None:
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, cpu_count * 3 // 4))

    shards_dir = cfg.output_dir / "_shards"

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
        execution_engine="xenna",
    )

    # Initialize Ray with runtime_env for HF_HOME and HF_TOKEN propagation
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

    logger.info("Running pipeline to generate shards (Xenna)...")
    result = last_mile_process(blend, pipeline_config)

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

    if shards_dir.exists():
        import shutil

        logger.info(f"Cleaning up intermediate shards directory: {shards_dir}")
        shutil.rmtree(shards_dir)

    elapsed_sec = time.time() - start_time

    source_datasets: list[InputDatasetInfo] = []
    seen_keys: set[str] = set()
    for split_datasets in blend.splits.values():
        for dataset in split_datasets:
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

    tok_uri = tokenizer_to_uri(cfg.tokenizer_model)

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

    finish_wandb(exit_code=0)
    print_step_complete(data_prep=artifact)
    return artifact


def main(cfg: SFTDataPrepConfig | None = None) -> SFTDataArtifact:
    """Entry point for Xenna SFT data preparation."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        cfg = omegaconf_to_dataclass(config, SFTDataPrepConfig)

    init_wandb_from_env()
    return run_data_prep_main(cfg)


if __name__ == "__main__":
    main()
