#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "data/curate/nemotron-cc/fuzzy-dedup"
# image = "nvcr.io/nvidia/nemo:25.02"
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra curator python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/fuzzy_dedup"
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

"""Fuzzy deduplication for the Nemotron-CC pipeline.

This script performs fuzzy deduplication in two phases that can be run
together or independently:

  1. Identification (identify=true) -- compute MinHash signatures, perform
     Locality Sensitive Hashing, and find fuzzy duplicates via connected
     components (GPU-accelerated).

  2. Removal (remove=true) -- read the duplicate IDs and remove them from
     the original dataset, writing deduplicated output.

CLI:
    nemotron data curate nemotron-cc fuzzy-dedup                # local execution
    nemotron data curate nemotron-cc fuzzy-dedup --run dlw      # submit to cluster

Direct usage:
    uv run python step_2b-fuzzy_dedup.py
    uv run python step_2b-fuzzy_dedup.py --config /path/to/config.yaml
    uv run python step_2b-fuzzy_dedup.py identify=false remove=true

See README.md in this directory for detailed usage instructions.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
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
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "fuzzy_dedup" / "default.yaml"

FUZZY_DEDUP_IDS_SUBDIR = "FuzzyDuplicateIds"
ID_GENERATOR_FILENAME = "fuzzy_id_generator.json"

# Nemotron-CC defaults
DEFAULT_CHAR_NGRAMS = 24
DEFAULT_NUM_BANDS = 20
DEFAULT_MINHASHES_PER_BAND = 13

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class FuzzyDedupConfig:
    """Config for fuzzy deduplication."""

    # Operation flags
    identify: bool = True
    remove: bool = True

    # Paths
    input_dir: str = "./data/exact_deduplicated/exact_deduplicated"
    cache_dir: str = "./data/fuzzy_dedup_cache"
    output_dir: str = "./data/fuzzy_deduplicated"

    # Input/output format
    input_filetype: str = "jsonl"
    text_field: str = "text"
    output_filetype: str = "jsonl"

    # Fuzzy dedup settings
    input_blocksize: str = "256MiB"
    bands_per_iteration: int = 5
    total_nparts: int | None = None

    # Cloud storage
    storage_options: str | None = None

    # Ray cluster
    num_gpus: int | None = None
    num_cpus: int | None = None


def run_identification(cfg: FuzzyDedupConfig) -> None:
    """Run fuzzy duplicate identification using FuzzyDeduplicationWorkflow."""
    storage_options = json.loads(cfg.storage_options) if cfg.storage_options else None
    storage_kwargs = {"storage_options": storage_options} if storage_options else None

    logger.info("Starting fuzzy duplicate identification")
    logger.info(f"  Input: {cfg.input_dir}")
    logger.info(f"  Cache dir: {cfg.cache_dir}")
    logger.info(f"  Output dir: {cfg.output_dir}")
    logger.info(f"  Config: char_ngrams={DEFAULT_CHAR_NGRAMS}, num_bands={DEFAULT_NUM_BANDS}, "
                f"minhashes_per_band={DEFAULT_MINHASHES_PER_BAND}, bands_per_iteration={cfg.bands_per_iteration}")
    start_time = time.perf_counter()

    workflow = FuzzyDeduplicationWorkflow(
        input_path=cfg.input_dir,
        cache_path=cfg.cache_dir,
        output_path=cfg.output_dir,
        input_filetype=cfg.input_filetype,
        input_blocksize=cfg.input_blocksize,
        text_field=cfg.text_field,
        read_kwargs=storage_kwargs,
        cache_kwargs=storage_kwargs,
        write_kwargs=storage_kwargs,
        char_ngrams=DEFAULT_CHAR_NGRAMS,
        num_bands=DEFAULT_NUM_BANDS,
        minhashes_per_band=DEFAULT_MINHASHES_PER_BAND,
        bands_per_iteration=cfg.bands_per_iteration,
        lsh_num_output_partitions=cfg.total_nparts,
    )
    workflow_result = workflow.run()
    elapsed = time.perf_counter() - start_time

    num_duplicates = workflow_result.metadata.get("num_duplicates", 0)
    minhash_time = workflow_result.metadata.get("minhash_time", 0.0)
    lsh_time = workflow_result.metadata.get("lsh_time", 0.0)
    cc_time = workflow_result.metadata.get("connected_components_pipeline_time", 0.0)

    logger.info(f"Identification completed in {elapsed:.1f}s")
    logger.info(f"  MinHash time: {minhash_time:.1f}s")
    logger.info(f"  LSH time: {lsh_time:.1f}s")
    logger.info(f"  Connected components time: {cc_time:.1f}s")
    logger.info(f"  Fuzzy duplicates found: {num_duplicates}")


def run_removal(cfg: FuzzyDedupConfig) -> None:
    """Remove identified fuzzy duplicates using TextDuplicatesRemovalWorkflow."""
    storage_options = json.loads(cfg.storage_options) if cfg.storage_options else None
    storage_kwargs = {"storage_options": storage_options} if storage_options else None
    output_base = cfg.output_dir.rstrip("/")
    ids_to_remove_path = f"{output_base}/{FUZZY_DEDUP_IDS_SUBDIR}"
    id_generator_path = f"{output_base}/{ID_GENERATOR_FILENAME}"
    deduplicated_output_path = f"{output_base}/fuzzy_deduplicated"

    output_kwargs = {}
    if cfg.output_filetype == "jsonl":
        output_kwargs["force_ascii"] = False
    if storage_options:
        output_kwargs["storage_options"] = storage_options

    logger.info("Starting duplicate removal")
    logger.info(f"  Input: {cfg.input_dir}")
    logger.info(f"  IDs to remove: {ids_to_remove_path}")
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
        input_kwargs=storage_kwargs,
        output_kwargs=output_kwargs or None,
    )
    workflow_result = workflow.run(executor=RayDataExecutor(), initial_tasks=initial_tasks)
    elapsed = time.perf_counter() - start_time

    num_removed = workflow_result.metadata.get("num_duplicates_removed", 0)

    logger.info(f"Removal completed in {elapsed:.1f}s")
    logger.info(f"  Duplicates removed: {num_removed}")


def run_fuzzy_dedup(cfg: FuzzyDedupConfig) -> None:
    """Run fuzzy deduplication pipeline."""
    if not cfg.identify and not cfg.remove:
        raise ValueError("No operation specified. Set identify=true and/or remove=true.")

    ray_client = RayClient(num_gpus=cfg.num_gpus, num_cpus=cfg.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC fuzzy deduplication")

    if cfg.identify:
        run_identification(cfg)

    if cfg.remove:
        run_removal(cfg)

    ray_client.stop()


def main(cfg: FuzzyDedupConfig | None = None) -> None:
    """Entry point for fuzzy deduplication.

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

        cfg = omegaconf_to_dataclass(config, FuzzyDedupConfig)

    run_fuzzy_dedup(cfg)


if __name__ == "__main__":
    main()
