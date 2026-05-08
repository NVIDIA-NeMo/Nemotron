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

  1. Identification (--identify) — hash every document and find exact
     duplicates (GPU-accelerated). Writes duplicate IDs and an ID
     generator mapping to --cache-dir.

  2. Removal (--remove) — read the duplicate IDs from --cache-dir and
     remove them from the original dataset, writing deduplicated output
     to --output-dir.

See README.md in this directory for detailed usage instructions.
"""

import argparse
import json
import time
from typing import Literal

from loguru import logger

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import EmptyTask
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow


EXACT_DEDUP_IDS_SUBDIR = "ExactDuplicateIds"
ID_GENERATOR_FILENAME = "exact_id_generator.json"


def _parse_memory_arg(value: str) -> int | Literal["auto"] | None:
    """Parse a memory argument that can be an int, 'auto', or None."""
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        return "auto"
    return int(value)


def run_identification(args: argparse.Namespace) -> None:
    """Run exact duplicate identification using ExactDeduplicationWorkflow.

    Writes ExactDuplicateIds/ and exact_id_generator.json into --cache-dir.
    """
    storage_options = json.loads(args.storage_options) if args.storage_options else None

    logger.info("Starting exact duplicate identification")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Cache dir: {args.cache_dir}")
    start_time = time.perf_counter()

    workflow = ExactDeduplicationWorkflow(
        input_path=args.input_dir,
        output_path=args.cache_dir,
        input_filetype=args.input_filetype,
        text_field=args.text_field,
        input_blocksize=args.input_blocksize,
        identification_batchsize=args.identification_batchsize,
        assign_id=True,
        total_nparts=args.total_nparts,
        rmm_pool_size=args.rmm_pool_size,
        spill_memory_limit=args.spill_memory_limit,
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


def run_removal(args: argparse.Namespace) -> None:
    """Remove identified exact duplicates using TextDuplicatesRemovalWorkflow.

    Reads duplicate IDs and ID generator from --cache-dir, writes
    deduplicated output to --output-dir.
    """
    storage_options = json.loads(args.storage_options) if args.storage_options else None
    cache_base = args.cache_dir.rstrip("/")
    output_base = args.output_dir.rstrip("/")
    ids_to_remove_path = f"{cache_base}/{EXACT_DEDUP_IDS_SUBDIR}"
    id_generator_path = f"{cache_base}/{ID_GENERATOR_FILENAME}"
    deduplicated_output_path = f"{output_base}/exact_deduplicated"

    output_kwargs = {}
    if args.output_filetype == "jsonl":
        output_kwargs["force_ascii"] = False
    if storage_options:
        output_kwargs["storage_options"] = storage_options

    logger.info("Starting duplicate removal")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Cache dir (IDs): {ids_to_remove_path}")
    logger.info(f"  Output: {deduplicated_output_path}")
    start_time = time.perf_counter()

    file_partitioning_stage = FilePartitioningStage(
        file_paths=args.input_dir,
        blocksize=args.input_blocksize,
        file_extensions=None,
        storage_options=storage_options,
    )
    logger.info("Running file partitioning pipeline...")
    file_partitioning_stage.setup()
    initial_tasks = file_partitioning_stage.process(EmptyTask)
    logger.info(f"File partitioning pipeline completed with {len(initial_tasks)} initial tasks")

    workflow = TextDuplicatesRemovalWorkflow(
        input_path=args.input_dir,
        ids_to_remove_path=ids_to_remove_path,
        output_path=deduplicated_output_path,
        input_filetype=args.input_filetype,
        input_blocksize=args.input_blocksize,
        duplicate_id_field=CURATOR_DEDUP_ID_STR,
        id_generator_path=id_generator_path,
        output_filetype=args.output_filetype,
        output_fields=["url", "warc_id", "source_id", "language", "text", "file_name"],
        input_kwargs={"storage_options": storage_options} if storage_options else None,
        output_kwargs=output_kwargs or None,
    )
    workflow_result = workflow.run(executor=RayDataExecutor(), initial_tasks=initial_tasks)
    elapsed = time.perf_counter() - start_time

    num_removed = workflow_result.metadata.get("num_duplicates_removed", 0)

    logger.info(f"Removal completed in {elapsed:.1f}s")
    logger.info(f"  Duplicates removed: {num_removed}")


def main(args: argparse.Namespace) -> None:
    # If neither flag is specified, default to running identification
    if not args.identify and not args.remove:
        raise ValueError("No operation specified. Use --identify and/or --remove flags.")

    ray_client = RayClient(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC exact deduplication")

    if args.identify:
        run_identification(args)

    if args.remove:
        run_removal(args)

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact deduplication for Nemotron-CC: identification and removal of duplicate documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Operation flags
    parser.add_argument(
        "--identify",
        action="store_true",
        help="Run the identification phase to find exact duplicates.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Run the removal phase to remove identified duplicates.",
    )

    # Paths
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the input dataset (output of step 1).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory for intermediate identification artifacts (ExactDuplicateIds/, ID generator).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/exact_deduplicated",
        help="Directory to write deduplicated output. Required when --remove is set.",
    )

    # Input format
    parser.add_argument(
        "--input-filetype",
        type=str,
        default="jsonl",
        choices=["parquet", "jsonl"],
        help="Format of the input files.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the field containing the document text.",
    )

    # Output format
    parser.add_argument(
        "--output-filetype",
        type=str,
        default="jsonl",
        choices=["parquet", "jsonl"],
        help="Format of the deduplicated output files.",
    )

    # Identification settings
    parser.add_argument(
        "--input-blocksize",
        type=str,
        default="256MiB",
        help="Target partition size for input data (e.g., '256MiB', '512MiB', '2GiB').",
    )
    parser.add_argument(
        "--identification-batchsize",
        type=int,
        default=12,
        help="Number of partitions to process per identification batch.",
    )
    parser.add_argument(
        "--total-nparts",
        type=int,
        default=None,
        help="Total number of output partitions for identification. Auto-determined if not set.",
    )
    parser.add_argument(
        "--rmm-pool-size",
        type=_parse_memory_arg,
        default="auto",
        help="Size of the RMM GPU memory pool in bytes, 'auto' for 90%% of free GPU memory, or 'none'.",
    )
    parser.add_argument(
        "--spill-memory-limit",
        type=_parse_memory_arg,
        default="auto",
        help="Device memory limit in bytes for spilling to host, 'auto' for 80%% of RMM pool, or 'none'.",
    )

    # Cloud storage
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON string of fsspec storage options for cloud I/O (e.g., \'{"endpoint_url": "...", "key": "...", "secret": "..."}\').',
    )

    # Ray cluster — these only apply when starting a new local Ray cluster.
    # When connecting to an existing cluster (e.g., via RAY_ADDRESS), they are ignored.
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for a local Ray cluster (default: all available). Ignored when connecting to an external cluster.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for a local Ray cluster (default: all available). Ignored when connecting to an external cluster.",
    )

    return parser


if __name__ == "__main__":
    args = attach_args().parse_args()
    if args.remove and args.output_dir is None:
        attach_args().error("--output-dir is required when --remove is set")
    main(args)
