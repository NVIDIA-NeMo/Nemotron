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

  1. Identification (--identify) — compute MinHash signatures, perform
     Locality Sensitive Hashing, and find fuzzy duplicates via connected
     components (GPU-accelerated). Writes duplicate IDs and an ID
     generator mapping to --output-dir, with intermediate artifacts
     stored in --cache-dir.

  2. Removal (--remove) — read the duplicate IDs from --output-dir and
     remove them from the original dataset, writing deduplicated output
     to --output-dir/fuzzy_deduplicated.

See README.md in this directory for detailed usage instructions.
"""

import argparse
import json
import time

from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from nemo_curator.tasks import EmptyTask


FUZZY_DEDUP_IDS_SUBDIR = "FuzzyDuplicateIds"
ID_GENERATOR_FILENAME = "fuzzy_id_generator.json"

# Nemotron-CC defaults
DEFAULT_CHAR_NGRAMS = 24
DEFAULT_NUM_BANDS = 20
DEFAULT_MINHASHES_PER_BAND = 13


def run_identification(args: argparse.Namespace) -> None:
    """Run fuzzy duplicate identification using FuzzyDeduplicationWorkflow.

    Writes FuzzyDuplicateIds/ and fuzzy_id_generator.json into --output-dir,
    with intermediate artifacts (minhashes, LSH buckets, etc.) in --cache-dir.
    """
    storage_options = json.loads(args.storage_options) if args.storage_options else None
    storage_kwargs = {"storage_options": storage_options} if storage_options else None

    logger.info("Starting fuzzy duplicate identification")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Cache dir: {args.cache_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Config: char_ngrams={DEFAULT_CHAR_NGRAMS}, num_bands={DEFAULT_NUM_BANDS}, "
                f"minhashes_per_band={DEFAULT_MINHASHES_PER_BAND}, bands_per_iteration={args.bands_per_iteration}")
    start_time = time.perf_counter()

    workflow = FuzzyDeduplicationWorkflow(
        input_path=args.input_dir,
        cache_path=args.cache_dir,
        output_path=args.output_dir,
        input_filetype=args.input_filetype,
        input_blocksize=args.input_blocksize,
        text_field=args.text_field,
        read_kwargs=storage_kwargs,
        cache_kwargs=storage_kwargs,
        write_kwargs=storage_kwargs,
        char_ngrams=DEFAULT_CHAR_NGRAMS,
        num_bands=DEFAULT_NUM_BANDS,
        minhashes_per_band=DEFAULT_MINHASHES_PER_BAND,
        bands_per_iteration=args.bands_per_iteration,
        lsh_num_output_partitions=args.total_nparts,
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


def run_removal(args: argparse.Namespace) -> None:
    """Remove identified fuzzy duplicates using TextDuplicatesRemovalWorkflow.

    Reads duplicate IDs and ID generator from --output-dir, writes
    deduplicated output to --output-dir/deduplicated.
    """
    storage_options = json.loads(args.storage_options) if args.storage_options else None
    storage_kwargs = {"storage_options": storage_options} if storage_options else None
    output_base = args.output_dir.rstrip("/")
    ids_to_remove_path = f"{output_base}/{FUZZY_DEDUP_IDS_SUBDIR}"
    id_generator_path = f"{output_base}/{ID_GENERATOR_FILENAME}"
    deduplicated_output_path = f"{output_base}/fuzzy_deduplicated"

    output_kwargs = {}
    if args.output_filetype == "jsonl":
        output_kwargs["force_ascii"] = False
    if storage_options:
        output_kwargs["storage_options"] = storage_options

    logger.info("Starting duplicate removal")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  IDs to remove: {ids_to_remove_path}")
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
        input_kwargs=storage_kwargs,
        output_kwargs=output_kwargs or None,
    )
    workflow_result = workflow.run(executor=RayDataExecutor(), initial_tasks=initial_tasks)
    elapsed = time.perf_counter() - start_time

    num_removed = workflow_result.metadata.get("num_duplicates_removed", 0)

    logger.info(f"Removal completed in {elapsed:.1f}s")
    logger.info(f"  Duplicates removed: {num_removed}")


def main(args: argparse.Namespace) -> None:
    if not args.identify and not args.remove:
        raise ValueError("No operation specified. Use --identify and/or --remove flags.")

    ray_client = RayClient(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC fuzzy deduplication")

    if args.identify:
        run_identification(args)

    if args.remove:
        run_removal(args)

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuzzy deduplication for Nemotron-CC: identification and removal of near-duplicate documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Operation flags
    parser.add_argument(
        "--identify",
        action="store_true",
        help="Run the identification phase to find fuzzy duplicates.",
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
        help="Directory for intermediate artifacts (minhashes, LSH buckets, edges, connected components).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for duplicate IDs, ID generator, and deduplicated output.",
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

    # Fuzzy dedup settings
    parser.add_argument(
        "--input-blocksize",
        type=str,
        default="256MiB",
        help="Target partition size for input data (e.g., '256MiB', '512MiB', '2GiB').",
    )
    parser.add_argument(
        "--bands-per-iteration",
        type=int,
        default=5,
        help="Number of LSH bands to shuffle concurrently. Higher values are faster but use more memory.",
    )

    # Partitioning
    parser.add_argument(
        "--total-nparts",
        type=int,
        default=None,
        help="Total number of output partitions for the LSH shuffle. Auto-determined if not set.",
    )

    # Cloud storage
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON string of fsspec storage options for cloud I/O (e.g., \'{"endpoint_url": "...", "key": "...", "secret": "..."}\').',
    )

    # Ray cluster
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
    main(args)
