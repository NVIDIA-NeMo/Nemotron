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

"""Quality classification and bucketing for the Nemotron-CC pipeline.

This script performs model-based quality labeling in two phases that can
be run together or independently:

  1. Classification (--classify) — filter to English, then run an ensemble
     of three quality classifiers (FineWebNemotronEduClassifier,
     FineWebMixtralEduClassifier, and fasttext-oh-eli5) on the
     deduplicated data.  Writes classification results (with float scores)
     to parquet.

  2. Ensemble & Bucketing (--ensemble) — compute token-weighted percentile
     thresholds from the classification scores, map float scores to integer
     bins (0–19), take the per-document max across classifiers, and write
     the bucketed results partitioned by ensemble-max-int.

See README.md in this directory for detailed usage instructions.
"""

import argparse
import ctypes
import json
import os
import time

import numpy as np
import pandas as pd
from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.text.classifiers.fineweb_edu import (
    FineWebMixtralEduClassifier,
    FineWebNemotronEduClassifier,
)
from nemo_curator.stages.text.filters.fasttext import FastTextQualityFilter
from nemo_curator.stages.text.filters.score_filter import Filter, Score
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.utils import TaskPerfUtils

# ---------------------------------------------------------------------------
# Classifier score column names
# ---------------------------------------------------------------------------
CLASSIFIER_SCORES = {
    "nemotron": {
        "float_score": "fineweb-nemotron-edu-score",
        "int_score": "fineweb-nemotron-edu-score-int",
        "label": "fineweb-nemotron-edu-score-label",
    },
    "mixtral": {
        "float_score": "fineweb-mixtral-edu-score",
        "int_score": "fineweb-mixtral-edu-score-int",
        "label": "fineweb-mixtral-edu-score-label",
    },
    "fasttext": {
        "float_score": "fasttext-quality-score",
    },
}

FASTTEXT_HQ_MODEL_REPO = "mlfoundations/fasttext-oh-eli5"
FASTTEXT_HQ_MODEL_FILENAME = "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"


# ---------------------------------------------------------------------------
# Threshold / binning helpers
# ---------------------------------------------------------------------------

def weighted_percentile(data: np.ndarray, percentiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted percentiles using the inverted-CDF method."""
    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]

    cum_weights = np.cumsum(weights_sorted)
    total_weight = cum_weights[-1]
    normalized = cum_weights / total_weight

    results = []
    for p in percentiles:
        idx = np.searchsorted(normalized, p / 100.0, side="left")
        results.append(data_sorted[idx])
    return np.array(results)


def compute_thresholds(score_ar: np.ndarray, token_ar: np.ndarray) -> dict[int, float]:
    """Return {percentile: threshold} for 5th, 10th, …, 95th percentiles."""
    percentiles = np.arange(5, 100, 5)
    thresholds = weighted_percentile(score_ar, percentiles, weights=token_ar)
    return {int(p): float(t) for p, t in zip(percentiles, thresholds)}


def compute_thresholds_for_score_columns(
    df: pd.DataFrame,
    text_col_name: str,
    score_col_names: list[str],
) -> dict[str, dict[int, float]]:
    """Compute percentile-based thresholds for each score column.

    ``text_col_name`` should reference a column of pre-computed integer
    byte-lengths (e.g. ``"token_length"``).
    """
    token_ar = df[text_col_name].to_numpy()
    threshold_dict = {}
    for score_col in score_col_names:
        threshold_dict[score_col] = compute_thresholds(
            df[score_col].to_numpy(), token_ar,
        )
    return threshold_dict


def map_scores(
    df: pd.DataFrame,
    score_col_name: str,
    score_int_name: str,
    bins: np.ndarray,
) -> pd.DataFrame:
    """Map float scores to integer bins via np.digitize."""
    pred_orig = np.array(df[score_col_name])
    df[score_int_name] = np.digitize(pred_orig, bins)
    return df


def map_score_columns(
    df: pd.DataFrame,
    score_col_names: list[str],
    threshold_dict: dict[str, dict[int, float]],
) -> pd.DataFrame:
    """Apply score→int mapping for every classifier."""
    for score_col_name in score_col_names:
        score_int_name = score_col_name + "-int"
        thresholds = threshold_dict.get(score_col_name)
        if thresholds is None:
            msg = f"No thresholds found for score column '{score_col_name}'"
            raise ValueError(msg)
        sorted_keys = sorted(thresholds.keys(), key=int)
        bins = np.array([thresholds[k] for k in sorted_keys])
        df = map_scores(df, score_col_name, score_int_name, bins)
    return df


def save_thresholds(threshold_dict: dict, file_name: str) -> None:
    with open(file_name, "w") as fout:
        json.dump(threshold_dict, fout, indent=4)
    logger.info(f"Thresholds saved to {file_name}")


def _save_metrics(metrics: dict, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {file_path}")


# ---------------------------------------------------------------------------
# Phase 1 — Classify
# ---------------------------------------------------------------------------

def run_classification(args: argparse.Namespace) -> None:
    """Filter to English, run 3 classifiers, write parquet."""
    from huggingface_hub import hf_hub_download

    classification_results_dir = os.path.join(args.output_dir, "classification_results")
    os.makedirs(classification_results_dir, exist_ok=True)

    # Download the fasttext quality model
    fasttext_model_path = hf_hub_download(
        repo_id=FASTTEXT_HQ_MODEL_REPO,
        filename=FASTTEXT_HQ_MODEL_FILENAME,
    )
    logger.info(f"FastText quality model path: {fasttext_model_path}")

    logger.info("Starting quality classification pipeline")
    logger.info(f"  Input: {args.input_dir}")
    logger.info(f"  Output: {classification_results_dir}")
    start_time = time.perf_counter()

    # --- Build the pipeline ---
    pipeline = Pipeline(name="quality-classification")

    # 1. Read deduplicated JSONL
    pipeline.add_stage(JsonlReader(args.input_dir))

    # 2. Filter to English inline
    pipeline.add_stage(
        Filter(
            filter_fn=lambda lang: lang == "EN",
            filter_field="language",
        )
    )

    # 3a. Score with FastText quality (CPU, uses Score to add column)
    pipeline.add_stage(
        Score(
            score_fn=FastTextQualityFilter(model_path=fasttext_model_path, label="__label__hq"),
            score_field=CLASSIFIER_SCORES["fasttext"]["float_score"],
            text_field="text",
        )
    )

    # 3b. Nemotron-4 edu classifier (GPU)
    pipeline.add_stage(
        FineWebNemotronEduClassifier(
            float_score_field=CLASSIFIER_SCORES["nemotron"]["float_score"],
            int_score_field=CLASSIFIER_SCORES["nemotron"]["int_score"],
            label_field=CLASSIFIER_SCORES["nemotron"]["label"],
        )
    )

    # 3c. Mixtral edu classifier (GPU)
    pipeline.add_stage(
        FineWebMixtralEduClassifier(
            float_score_field=CLASSIFIER_SCORES["mixtral"]["float_score"],
            int_score_field=CLASSIFIER_SCORES["mixtral"]["int_score"],
            label_field=CLASSIFIER_SCORES["mixtral"]["label"],
        )
    )

    # 4. Drop label and original int_score columns — the ensemble step
    #    recomputes 0–19 int bins from the float scores.
    cols_to_drop = [
        v[k]
        for v in CLASSIFIER_SCORES.values()
        for k in ("int_score", "label")
        if k in v
    ]

    @processing_stage(name="drop-unused-classifier-columns")
    def drop_unused_cols(batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        batch.data = df.drop(columns=cols_to_drop, errors="ignore")
        return batch

    pipeline.add_stage(drop_unused_cols)

    # 5. Write classification results to parquet
    pipeline.add_stage(ParquetWriter(classification_results_dir, mode="overwrite"))

    results = pipeline.run()
    elapsed = time.perf_counter() - start_time

    logger.info(f"Classification completed in {elapsed:.1f}s")
    logger.info(f"  Results written to: {classification_results_dir}")

    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    metrics["total_elapsed_s"] = round(elapsed, 2)
    _save_metrics(metrics, os.path.join(args.output_dir, "classification_metrics.json"))


# ---------------------------------------------------------------------------
# Phase 2 — Ensemble & Bucket
# ---------------------------------------------------------------------------

def _sample_threshold_data(
    classification_results_dir: str,
    score_col_names: list[str],
    sample_frac: float,
) -> pd.DataFrame:
    """Read score columns + token byte-lengths from every classification
    result file, sampling a fraction of rows from each.

    Uses Ray to read files in parallel.  Computes byte-lengths of the
    ``text`` column inside each Ray task so that only the scores and a
    small integer ``token_length`` column are transferred to the driver,
    keeping driver memory proportional to the number of score columns
    rather than the size of the raw text.
    """
    import glob

    import ray

    parquet_files = sorted(glob.glob(os.path.join(classification_results_dir, "*.parquet")))
    if not parquet_files:
        msg = f"No parquet files found in {classification_results_dir}"
        raise FileNotFoundError(msg)

    logger.info(f"  Reading {len(parquet_files)} files with sample_frac={sample_frac}")

    columns_to_read = score_col_names + ["text"]

    @ray.remote
    def _read_and_sample(path: str, columns: list[str], frac: float) -> pd.DataFrame:
        df = pd.read_parquet(path, columns=columns)
        if frac < 1.0:
            df = df.sample(frac=frac)
        # Compute byte-length on the worker and drop text before returning
        df["token_length"] = df["text"].str.encode("utf-8").apply(len)
        df = df.drop(columns=["text"])
        return df

    futures = [_read_and_sample.remote(f, columns_to_read, sample_frac) for f in parquet_files]
    dfs = ray.get(futures)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    logger.info(f"  Sampled {len(df)} documents for threshold computation")
    return df


def run_ensemble(args: argparse.Namespace) -> None:
    """Compute thresholds, map to int bins, ensemble, write bucketed parquet."""
    classification_results_dir = os.path.join(args.output_dir, "classification_results")
    thresholds_path = os.path.join(args.output_dir, "classifier_thresholds.json")
    bucketed_results_dir = os.path.join(args.output_dir, "bucketed_results")
    os.makedirs(bucketed_results_dir, exist_ok=True)

    logger.info("Starting ensemble & bucketing pipeline")
    logger.info(f"  Classification results: {classification_results_dir}")
    logger.info(f"  Output: {bucketed_results_dir}")
    start_time = time.perf_counter()

    # Float score column names for all three classifiers
    score_col_names = [v["float_score"] for v in CLASSIFIER_SCORES.values()]

    # Integer score column names (will be created by map_score_columns)
    int_column_names = [col + "-int" for col in score_col_names]

    # --- Step 1: Compute thresholds from a sample of classification results ---
    logger.info("Computing token-weighted percentile thresholds...")
    t_sample_start = time.perf_counter()
    df_sample = _sample_threshold_data(
        classification_results_dir, score_col_names, args.threshold_sample_frac
    )
    num_sampled_docs = len(df_sample)
    t_sample_elapsed = time.perf_counter() - t_sample_start
    logger.info(f"  Sampling completed in {t_sample_elapsed:.1f}s ({num_sampled_docs} docs)")

    t_thresh_start = time.perf_counter()
    threshold_dict = compute_thresholds_for_score_columns(
        df_sample, text_col_name="token_length", score_col_names=score_col_names
    )
    t_thresh_elapsed = time.perf_counter() - t_thresh_start
    logger.info(f"  Threshold computation completed in {t_thresh_elapsed:.1f}s")

    save_thresholds(threshold_dict, thresholds_path)
    del df_sample
    # glibc malloc keeps freed pages mapped; trim them back to the OS so the
    # subsequent pipeline doesn't start from an inflated RSS baseline.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    # --- Step 2: Map scores to ints, ensemble, write bucketed ---
    logger.info("Running ensemble & bucketing pipeline...")
    t_pipeline_start = time.perf_counter()

    pipeline = Pipeline(name="ensemble-quality-scores")
    pipeline.add_stage(ParquetReader(classification_results_dir))

    @processing_stage(name="ensemble-score")
    def ensemble_score(batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        df = map_score_columns(df, score_col_names, threshold_dict)
        df["ensemble-max-int"] = df[int_column_names].max(axis=1)
        batch.data = df
        return batch

    pipeline.add_stage(ensemble_score)

    # Write partitioned by ensemble bucket
    pipeline.add_stage(
        _PartitionedParquetWriter(
            bucketed_results_dir,
            write_kwargs={"partition_cols": ["ensemble-max-int"]},
            mode="overwrite",
        )
    )

    results = pipeline.run(executor=RayDataExecutor())
    t_pipeline_elapsed = time.perf_counter() - t_pipeline_start
    elapsed = time.perf_counter() - start_time

    logger.info(f"  Pipeline completed in {t_pipeline_elapsed:.1f}s")
    logger.info(f"Ensemble & bucketing completed in {elapsed:.1f}s")
    logger.info(f"  Bucketed results written to: {bucketed_results_dir}")

    # Print bucket distribution
    buckets = sorted(os.listdir(bucketed_results_dir))
    logger.info(f"  Buckets created: {buckets}")

    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    metrics["threshold_sampling_elapsed_s"] = round(t_sample_elapsed, 2)
    metrics["threshold_sampling_num_docs"] = num_sampled_docs
    metrics["threshold_sampling_frac"] = args.threshold_sample_frac
    metrics["threshold_computation_elapsed_s"] = round(t_thresh_elapsed, 2)
    metrics["pipeline_elapsed_s"] = round(t_pipeline_elapsed, 2)
    metrics["total_elapsed_s"] = round(elapsed, 2)
    _save_metrics(metrics, os.path.join(args.output_dir, "bucketing_metrics.json"))


class _PartitionedParquetWriter(ParquetWriter):
    """ParquetWriter that partitions output by a column (e.g. ensemble-max-int)."""

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        df = task.to_pandas().reset_index(drop=True)
        df.to_parquet(
            os.path.dirname(file_path),
            **{"index": None, **self.write_kwargs},
        )


# ---------------------------------------------------------------------------
# Main & CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if not args.classify and not args.ensemble:
        raise ValueError("No operation specified. Use --classify and/or --ensemble flags.")

    ray_client = RayClient(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC quality classification")

    if args.classify:
        run_classification(args)

    if args.ensemble:
        run_ensemble(args)

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quality classification and bucketing for Nemotron-CC: "
            "filter to English, run an ensemble of quality classifiers, "
            "and write bucketed output (0-19)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Operation flags
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run the classification phase: filter to English and score with 3 quality classifiers.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run the ensemble phase: compute thresholds, map to int bins, and write bucketed output.",
    )

    # Paths
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing deduplicated JSONL input (e.g. output of step 2b fuzzy dedup).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/quality_labeling",
        help="Base output directory. Sub-directories will be created for classification_results/, "
        "classifier_thresholds.json, and bucketed_results/.",
    )

    # Threshold sampling
    parser.add_argument(
        "--threshold-sample-frac",
        type=float,
        default=0.01,
        help="Fraction of rows to sample per file when computing percentile thresholds. "
        "Use < 1.0 (e.g. 0.01) to reduce memory at large scale.",
    )

    # Ray cluster
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for a local Ray cluster (default: all available).",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for a local Ray cluster (default: all available).",
    )

    return parser


if __name__ == "__main__":
    args = attach_args().parse_args()
    main(args)
