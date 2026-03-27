#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "data/curate/nemotron-cc/download-extract"
# image = "anyscale/ray:2.49.2-py312"
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra curator python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/download_extract"
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

"""Download, extract, and preprocess Common Crawl data for the Nemotron-CC pipeline.

CLI:
    nemotron data curate nemotron-cc download-extract           # local execution
    nemotron data curate nemotron-cc download-extract --run dlw  # submit to cluster

Direct usage:
    uv run python step_1-download_extract.py
    uv run python step_1-download_extract.py --config /path/to/config.yaml
    uv run python step_1-download_extract.py start_snapshot=2025-01

See README.md in this directory for detailed usage instructions.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import sys
import time
import urllib.request
from dataclasses import dataclass, field
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
from nemo_curator.stages.text.io.writer import JsonlWriter

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "download_extract" / "default.yaml"

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_FILENAME = "lid.176.bin"

# Module-level flag for Ray execution (used by nemotron CLI)
RAY = True


@dataclass
class DownloadExtractConfig:
    """Config for Common Crawl download, extraction, and preprocessing."""

    start_snapshot: str = "2024-46"
    end_snapshot: str = "2024-51"
    output_dir: str = "./data/cleaned_extracted"
    cache_dir: str = "./data/cache"
    url_limit: int | None = None
    record_limit: int | None = None
    languages: list[str] | None = None
    storage_options: str | None = None
    num_cpus: int | None = None


class LanguageFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract language codes from FastTextLangId scores, optionally filtering to specific languages."""

    def __init__(
        self, target_languages: list[str] | None = None, language_field: str = "language"
    ) -> None:
        self.target_languages = (
            {lang.upper() for lang in target_languages} if target_languages else None
        )
        self.language_field = language_field
        self.name = "language_filter"

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        df = task.to_pandas()
        df[self.language_field] = df[self.language_field].apply(lambda v: ast.literal_eval(v)[1])
        if self.target_languages:
            df = df[df[self.language_field].isin(self.target_languages)]
            if len(df) == 0:
                return None
        task.data = df
        return task


def download_fasttext_model(model_dir: str) -> str:
    """Download the FastText language identification model if not already present."""
    model_path = os.path.join(model_dir, FASTTEXT_MODEL_FILENAME)

    if os.path.exists(model_path):
        logger.info(f"FastText model already exists at {model_path}")
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Downloading FastText language ID model to {model_path}")
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, model_path)  # noqa: S310
    logger.info("Download complete")
    return model_path


def create_pipeline(cfg: DownloadExtractConfig) -> Pipeline:
    """Build the download-extract-preprocess pipeline."""
    output_dir = cfg.output_dir
    cache_dir = str(Path(cfg.cache_dir).resolve())
    download_dir = os.path.join(cache_dir, "cc_downloads")
    model_dir = os.path.join(cache_dir, "model")

    fasttext_model_path = download_fasttext_model(model_dir)

    storage_options = json.loads(cfg.storage_options) if cfg.storage_options else {}

    stages = [
        CommonCrawlDownloadExtractStage(
            start_snapshot=cfg.start_snapshot,
            end_snapshot=cfg.end_snapshot,
            download_dir=download_dir,
            crawl_type="main",
            html_extraction="justext",
            url_limit=cfg.url_limit,
            record_limit=cfg.record_limit,
        ),
        ScoreFilter(
            FastTextLangId(
                model_path=fasttext_model_path,
                min_langid_score=0.3,
            ),
            score_field="language",
        ),
        LanguageFilter(
            target_languages=cfg.languages,
            language_field="language",
        ),
        Modify(UnicodeReformatter()),
        JsonlWriter(
            output_dir, write_kwargs={"storage_options": storage_options, "force_ascii": False}
        ),
    ]

    return Pipeline(
        name="nemotron-cc-download-extract",
        description="Download, extract, and preprocess Common Crawl data with language ID and unicode fixing.",
        stages=stages,
    )


def run_download_extract(cfg: DownloadExtractConfig) -> None:
    """Run the download-extract pipeline."""
    storage_options = json.loads(cfg.storage_options) if cfg.storage_options else {}
    fs, fs_path = url_to_fs(cfg.output_dir, **storage_options)
    fs.mkdirs(fs_path, exist_ok=True)
    cache_dir = str(Path(cfg.cache_dir).resolve())
    os.makedirs(cache_dir, exist_ok=True)

    ray_client = RayClient(num_cpus=cfg.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC download and preprocessing pipeline")
    logger.info(f"  Snapshots: {cfg.start_snapshot} to {cfg.end_snapshot}")
    logger.info(f"  Languages: {cfg.languages or 'all'}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Output dir: {cfg.output_dir}")
    if cfg.url_limit is not None:
        logger.info(f"  URL limit: {cfg.url_limit}")
    if cfg.record_limit is not None:
        logger.info(f"  Record limit: {cfg.record_limit}")

    pipeline = create_pipeline(cfg)
    logger.info(f"\n{pipeline.describe()}")

    executor = RayDataExecutor()

    start_time = time.perf_counter()
    results = pipeline.run(executor=executor)
    elapsed = time.perf_counter() - start_time

    total_documents = sum(task.num_items for task in results) if results else 0
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"Total output files: {total_documents}")

    results_file = os.path.join(cache_dir, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Task results saved to {results_file}")

    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    metrics_file = os.path.join(cache_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Aggregated metrics saved to {metrics_file}")

    ray_client.stop()


def main(cfg: DownloadExtractConfig | None = None) -> None:
    """Entry point for download-extract.

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

        cfg = omegaconf_to_dataclass(config, DownloadExtractConfig)

    run_download_extract(cfg)


if __name__ == "__main__":
    main()
