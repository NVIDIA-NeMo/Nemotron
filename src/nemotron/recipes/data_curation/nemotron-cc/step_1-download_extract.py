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

"""Download, extract, and preprocess Common Crawl data for the Nemotron-CC pipeline.

See README.md in this directory for detailed usage instructions.
"""

import argparse
import ast
import json
import os
import pickle
import time
import urllib.request
from pathlib import Path

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.fasttext import FastTextLangId
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.stages.text.io.writer import ParquetWriter

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_FILENAME = "lid.176.bin"


class LanguageFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract language codes from FastTextLangId scores, optionally filtering to specific languages.

    FastTextLangId produces scores in the format "[0.95, 'EN']" (stringified list).
    This stage parses that field and replaces it with just the language code.
    If target_languages is provided, only documents matching those languages are kept.
    """

    def __init__(self, target_languages: list[str] | None = None, language_field: str = "language") -> None:
        self.target_languages = {lang.upper() for lang in target_languages} if target_languages else None
        self.language_field = language_field
        self.name = "language_filter"

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        df = task.to_pandas()
        # Parse "[0.95, 'EN']" -> 'EN'
        df[self.language_field] = df[self.language_field].apply(lambda v: ast.literal_eval(v)[1])
        if self.target_languages:
            df = df[df[self.language_field].isin(self.target_languages)]
            if len(df) == 0:
                return None
        task.data = df
        return task


def download_fasttext_model(model_dir: str) -> str:
    """Download the FastText language identification model if not already present.

    Args:
        model_dir: Directory that should contain the FastText model file.

    Returns:
        The full path to the model file.
    """
    model_path = os.path.join(model_dir, FASTTEXT_MODEL_FILENAME)

    if os.path.exists(model_path):
        logger.info(f"FastText model already exists at {model_path}")
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Downloading FastText language ID model to {model_path}")
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, model_path)  # noqa: S310
    logger.info("Download complete")
    return model_path


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the download-extract-preprocess pipeline."""
    output_dir = args.output_dir
    cache_dir = str(Path(args.cache_dir).resolve())
    download_dir = os.path.join(cache_dir, "cc_downloads")
    model_dir = os.path.join(cache_dir, "model")

    # Ensure FastText model is available locally (downloads if missing)
    fasttext_model_path = download_fasttext_model(model_dir)

    storage_options = json.loads(args.storage_options) if args.storage_options else {}

    stages = [
        # 1. Download and extract Common Crawl data using JusText.
        #    The JusText extractor was chosen for the Nemotron-CC pipeline.
        CommonCrawlDownloadExtractStage(
            start_snapshot=args.start_snapshot,
            end_snapshot=args.end_snapshot,
            download_dir=download_dir,
            crawl_type="main",
            html_extraction="justext",
            url_limit=args.url_limit,
            record_limit=args.record_limit,
        ),
        # 2. Language identification using FastText lid.176.bin (threshold 0.3 per paper).
        ScoreFilter(
            FastTextLangId(
                model_path=fasttext_model_path,
                min_langid_score=0.3,
            ),
            score_field="language",
        ),
        # 3. Extract language code, optionally filter to requested languages.
        LanguageFilter(
            target_languages=args.languages,
            language_field="language",
        ),
        # 4. Fix unicode issues on all documents.
        Modify(UnicodeReformatter()),
        # 5. Write output
        ParquetWriter(output_dir, write_kwargs={"storage_options": storage_options}),
    ]

    return Pipeline(
        name="nemotron-cc-download-extract",
        description="Download, extract, and preprocess Common Crawl data with language ID and unicode fixing.",
        stages=stages,
    )


def main(args: argparse.Namespace) -> None:
    storage_options = json.loads(args.storage_options) if args.storage_options else {}
    fs, fs_path = url_to_fs(args.output_dir, **storage_options)
    fs.mkdirs(fs_path, exist_ok=True)
    cache_dir = str(Path(args.cache_dir).resolve())
    os.makedirs(cache_dir, exist_ok=True)

    ray_client = RayClient(num_cpus=args.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC download and preprocessing pipeline")
    logger.info(f"  Snapshots: {args.start_snapshot} to {args.end_snapshot}")
    logger.info(f"  Languages: {args.languages or 'all'}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    if args.url_limit is not None:
        logger.info(f"  URL limit: {args.url_limit}")
    if args.record_limit is not None:
        logger.info(f"  Record limit: {args.record_limit}")

    pipeline = create_pipeline(args)
    logger.info(f"\n{pipeline.describe()}")

    executor = RayDataExecutor()

    start_time = time.perf_counter()
    results = pipeline.run(executor=executor)
    elapsed = time.perf_counter() - start_time

    total_documents = sum(task.num_items for task in results) if results else 0
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"Total output files: {total_documents}")

    # Dump result tasks (with _stage_perf timing stats) for later analysis
    results_file = os.path.join(cache_dir, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Task results saved to {results_file}")

    # Aggregate and save per-stage metrics (mean/std/sum for each metric)
    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    metrics_file = os.path.join(cache_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Aggregated metrics saved to {metrics_file}")

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, extract, and preprocess Common Crawl data for Nemotron-CC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Snapshot range
    parser.add_argument(
        "--start-snapshot",
        type=str,
        required=True,
        help="Start Common Crawl snapshot (e.g., '2024-46').",
    )
    parser.add_argument(
        "--end-snapshot",
        type=str,
        required=True,
        help="End Common Crawl snapshot (e.g., '2024-51').",
    )

    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/cleaned_extracted",
        help="Directory to write the preprocessed extracted content.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/cache",
        help="Cache directory for intermediate files. Layout: cache_dir/cc_downloads (WARC files), cache_dir/model (FastText model), plus results.pkl and metrics.json.",
    )

    # Common Crawl options
    parser.add_argument(
        "--url-limit",
        type=int,
        default=None,
        help="Limit number of WARC files to download per snapshot (useful for testing).",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=None,
        help="Limit number of records to extract per WARC file (useful for testing).",
    )

    # Language filtering
    parser.add_argument(
        "--languages",
        nargs="+",
        type=str,
        default=None,
        help="Language codes to keep (e.g., EN DE FR). If omitted, all languages are written.",
    )
    # Cloud storage
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON string of fsspec storage options for cloud output paths (e.g., \'{"key": "...", "secret": "..."}\').',
    )

    # Ray cluster
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for the Ray cluster (default: all available).",
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
