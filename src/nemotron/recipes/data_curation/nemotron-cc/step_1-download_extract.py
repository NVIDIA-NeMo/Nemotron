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
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.filters import FastTextLangId
from nemo_curator.stages.text.modifiers import UnicodeReformatter
from nemo_curator.stages.text.modules import Modify, ScoreFilter
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.stages.resources import Resources

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


class LanguagePartitionWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Write documents to language-partitioned subdirectories.

    Groups by (language, file_name) to preserve WARC provenance in output filenames.
    Supports both local and cloud paths (s3://, gs://, etc.) via fsspec.

    Output structure:
        output_dir/EN/crawl-data-CC-MAIN-2024-51-...-00002.warc.jsonl
        output_dir/DE/crawl-data-CC-MAIN-2024-51-...-00002.warc.jsonl
        ...
    """

    def __init__(self, output_dir: str, storage_options: dict | None = None) -> None:
        self.output_dir = output_dir
        self.storage_options = storage_options or {}
        self.fs, self._fs_path = url_to_fs(self.output_dir, **self.storage_options)
        self._is_remote = is_remote_url(self.output_dir)
        self.name = "language_partition_writer"

    def process(self, task: DocumentBatch) -> FileGroupTask:
        df = task.to_pandas()
        files = []
        for (language, file_name), group in df.groupby(["language", "file_name"]):
            lang_dir = self.fs.sep.join([self._fs_path, language])
            self.fs.mkdirs(lang_dir, exist_ok=True)
            stem, _ = os.path.splitext(file_name)
            file_path = self.fs.sep.join([lang_dir, f"{stem}.jsonl"])

            file_path_with_protocol = self.fs.unstrip_protocol(file_path) if self._is_remote else file_path
            group.to_json(file_path_with_protocol, lines=True, orient="records", storage_options=self.storage_options, force_ascii=False)
            files.append(file_path_with_protocol)

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=files,
            _metadata={**task._metadata, "format": "jsonl"},
            _stage_perf=task._stage_perf,
        )


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
    download_dir = str(Path(args.download_dir).resolve())

    fasttext_model_path = download_fasttext_model(args.lang_id_model_path)

    storage_options = json.loads(args.storage_options) if args.storage_options else {}

    stages = [
        # JusText is the Nemotron-CC choice for html extraction
        CommonCrawlDownloadExtractStage(
            start_snapshot=args.start_snapshot,
            end_snapshot=args.end_snapshot,
            download_dir=download_dir,
            crawl_type="main",
            html_extraction="justext",
            url_limit=args.url_limit,
            record_limit=args.record_limit,
        ).with_({"iterate_extract_commoncrawlwarciterator_commoncrawlhtmlextractor":{"resources":Resources(cpus=2.0)}}),
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
        # 5. Write output partitioned by language (e.g., output_dir/EN/, output_dir/DE/).
        LanguagePartitionWriter(output_dir, storage_options=storage_options),
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
    os.makedirs(args.download_dir, exist_ok=True)

    ray_client = RayClient(num_cpus=args.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC download and preprocessing pipeline")
    logger.info(f"  Snapshots: {args.start_snapshot} to {args.end_snapshot}")
    logger.info(f"  Languages: {args.languages or 'all'}")
    logger.info(f"  Download dir: {args.download_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    if args.url_limit is not None:
        logger.info(f"  URL limit: {args.url_limit}")
    if args.record_limit is not None:
        logger.info(f"  Record limit: {args.record_limit}")

    pipeline = create_pipeline(args)
    logger.info(f"\n{pipeline.describe()}")

    executor = RayDataExecutor() if args.executor == "ray-data" else XennaExecutor()
    logger.info(f"Using executor: {args.executor}")

    start_time = time.perf_counter()
    results = pipeline.run(executor=executor)
    elapsed = time.perf_counter() - start_time

    total_documents = sum(task.num_items for task in results) if results else 0
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"Total output files: {total_documents}")

    # Dump result tasks (with _stage_perf timing stats) for later analysis
    results_file = fs.sep.join([fs_path, "results.pkl"])
    with fs.open(results_file, "wb") as f:
        pickle.dump(results, f)
    results_file_display = fs.unstrip_protocol(results_file) if is_remote_url(args.output_dir) else results_file
    logger.info(f"Task results saved to {results_file_display}")

    # Aggregate and save per-stage metrics (mean/std/sum for each metric)
    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    metrics_file = fs.sep.join([fs_path, "metrics.json"])
    with fs.open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    metrics_file_display = fs.unstrip_protocol(metrics_file) if is_remote_url(args.output_dir) else metrics_file
    logger.info(f"Aggregated metrics saved to {metrics_file_display}")

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
        default="./data/preprocessed",
        help="Directory to write the preprocessed output, partitioned by language.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./data/cc_downloads",
        help="Directory for intermediate Common Crawl WARC downloads.",
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
    parser.add_argument(
        "--lang-id-model-path",
        type=str,
        required=True,
        help="Directory for the FastText lid.176.bin model. Downloads automatically if not present in the directory.",
    )

    # Cloud storage
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON string of fsspec storage options for cloud output paths (e.g., \'{"key": "...", "secret": "..."}\').',
    )

    # Executor
    parser.add_argument(
        "--executor",
        type=str,
        choices=["xenna", "ray-data"],
        default="ray-data",
        help="Pipeline executor backend.",
    )

    # Local Ray cluster
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for the Ray cluster (default: all available).",
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
