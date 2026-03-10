#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "data/curate/nemotron-cc/exact-dedup"
# image = "nvcr.io/nvidia/nemo:25.02"
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra curator python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/exact_dedup"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Exact deduplication for the Nemotron-CC pipeline.

This script performs exact deduplication in two phases that can be run
together or independently:

  1. Identification (identify=true) -- hash every document and find exact
     duplicates (GPU-accelerated). Writes duplicate IDs and an ID
     generator mapping to cache_dir.

  2. Removal (remove=true) -- read the duplicate IDs from cache_dir and
     remove them from the original dataset, writing deduplicated output
     to output_dir.

CLI:
    nemotron data curate nemotron-cc exact-dedup                # local execution
    nemotron data curate nemotron-cc exact-dedup --run dlw      # submit to cluster

Direct usage:
    uv run python step_2a-exact_dedup.py
    uv run python step_2a-exact_dedup.py --config /path/to/config.yaml
    uv run python step_2a-exact_dedup.py identify=false remove=true

See README.md in this directory for detailed usage instructions.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from nemo_curator.tasks import EmptyTask

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "exact_dedup" / "default.yaml"

EXACT_DEDUP_IDS_SUBDIR = "ExactDuplicateIds"
ID_GENERATOR_FILENAME = "exact_id_generator.json"

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


def _parse_memory_value(value: str | int | None) -> int | Literal["auto"] | None:
    """Parse a memory value that can be an int, 'auto', or None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value_str = str(value).lower()
    if value_str == "none":
        return None
    if value_str == "auto":
        return "auto"
    return int(value_str)


@dataclass
class ExactDedupConfig:
    """Config for exact deduplication."""

    # Operation flags
    identify: bool = True
    remove: bool = True

    # Paths
    input_dir: str = "./data/cleaned_extracted"
    cache_dir: str = "./data/exact_dedup_cache"
    output_dir: str = "./data/exact_deduplicated"

    # Input/output format
    input_filetype: str = "jsonl"
    text_field: str = "text"
    output_filetype: str = "jsonl"

    # Identification settings
    input_blocksize: str = "256MiB"
    identification_batchsize: int = 12
    total_nparts: int | None = None
    rmm_pool_size: str | int | None = "auto"
    spill_memory_limit: str | int | None = "auto"

    # Cloud storage
    storage_options: str | None = None

    # Ray cluster
    num_gpus: int | None = None
    num_cpus: int | None = None


def run_identification(cfg: ExactDedupConfig) -> None:
    """Run exact duplicate identification using ExactDeduplicationWorkflow."""
    storage_options = json.loads(cfg.storage_options) if cfg.storage_options else None

    logger.info("Starting exact duplicate identification")
    logger.info(f"  Input: {cfg.input_dir}")
    logger.info(f"  Cache dir: {cfg.cache_dir}")
    start_time = time.perf_counter()

    workflow = ExactDeduplicationWorkflow(
        input_path=cfg.input_dir,
        output_path=cfg.cache_dir,
        input_filetype=cfg.input_filetype,
        text_field=cfg.text_field,
        input_blocksize=cfg.input_blocksize,
        identification_batchsize=cfg.identification_batchsize,
        assign_id=True,
        total_nparts=cfg.total_nparts,
        rmm_pool_size=_parse_memory_value(cfg.rmm_pool_size),
        spill_memory_limit=_parse_memory_value(cfg.spill_memory_limit),
        read_kwargs={"storage_options": storage_options} if storage_options else None,
    )
    workflow_result = workflow.run(initial_tasks=None)
    elapsed = time.perf_counter() - start_time

    num_duplicates = workflow_result.metadata.get("num_duplicates", 0)
    identification_time = workflow_result.metadata.get("identification_time", 0.0)
    input_filegroups_time = workflow_result.metadata.get("input_filegroups_time", 0.0)

    logger.info(f"Identification completed in {elapsed:.1f}s")
    logger.info(f"  Time taken to group files by blocksize: {input_filegroups_time:.1f}s")
    logger.info(f"  Identification time: {identification_time:.1f}s")
    logger.info(f"  Exact duplicates found: {num_duplicates}")


def run_removal(cfg: ExactDedupConfig) -> None:
    """Remove identified exact duplicates using TextDuplicatesRemovalWorkflow."""
    storage_options = json.loads(cfg.storage_options) if cfg.storage_options else None
    cache_base = cfg.cache_dir.rstrip("/")
    output_base = cfg.output_dir.rstrip("/")
    ids_to_remove_path = f"{cache_base}/{EXACT_DEDUP_IDS_SUBDIR}"
    id_generator_path = f"{cache_base}/{ID_GENERATOR_FILENAME}"
    deduplicated_output_path = f"{output_base}/exact_deduplicated"

    output_kwargs = {}
    if cfg.output_filetype == "jsonl":
        output_kwargs["force_ascii"] = False
    if storage_options:
        output_kwargs["storage_options"] = storage_options

    logger.info("Starting duplicate removal")
    logger.info(f"  Input: {cfg.input_dir}")
    logger.info(f"  Cache dir (IDs): {ids_to_remove_path}")
    logger.info(f"  Output: {deduplicated_output_path}")
    start_time = time.perf_counter()

    file_partitioning_stage = FilePartitioningStage(
        file_paths=cfg.input_dir,
        blocksize=cfg.input_blocksize,
        file_extensions=None,
        storage_options=storage_options,
    )
    logger.info("Running file partitioning pipeline...")
    file_partitioning_stage.setup()
    initial_tasks = file_partitioning_stage.process(EmptyTask)
    logger.info(f"File partitioning pipeline completed with {len(initial_tasks)} initial tasks")

    workflow = TextDuplicatesRemovalWorkflow(
        input_path=cfg.input_dir,
        ids_to_remove_path=ids_to_remove_path,
        output_path=deduplicated_output_path,
        input_filetype=cfg.input_filetype,
        input_blocksize=cfg.input_blocksize,
        duplicate_id_field=CURATOR_DEDUP_ID_STR,
        id_generator_path=id_generator_path,
        output_filetype=cfg.output_filetype,
        output_fields=["url", "warc_id", "source_id", "language", "text", "file_name"],
        input_kwargs={"storage_options": storage_options} if storage_options else None,
        output_kwargs=output_kwargs or None,
    )
    workflow_result = workflow.run(executor=RayDataExecutor(), initial_tasks=initial_tasks)
    elapsed = time.perf_counter() - start_time

    num_removed = workflow_result.metadata.get("num_duplicates_removed", 0)

    logger.info(f"Removal completed in {elapsed:.1f}s")
    logger.info(f"  Duplicates removed: {num_removed}")


def run_exact_dedup(cfg: ExactDedupConfig) -> None:
    """Run exact deduplication pipeline."""
    if not cfg.identify and not cfg.remove:
        raise ValueError("No operation specified. Set identify=true and/or remove=true.")

    ray_client = RayClient(num_gpus=cfg.num_gpus, num_cpus=cfg.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC exact deduplication")

    if cfg.identify:
        run_identification(cfg)

    if cfg.remove:
        run_removal(cfg)

    ray_client.stop()


def main(cfg: ExactDedupConfig | None = None) -> None:
    """Entry point for exact deduplication.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.
    """
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        cfg = omegaconf_to_dataclass(config, ExactDedupConfig)

    run_exact_dedup(cfg)


if __name__ == "__main__":
    main()
