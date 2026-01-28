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

"""Evaluation script for embedding models.

Evaluates embedding models on retrieval metrics using BEIR framework.
Compares base model vs fine-tuned model on nDCG, Recall, and Precision.

Supports evaluation of:
- Local HuggingFace models
- NIM API endpoints (OpenAI-compatible embeddings API)

Usage:
    # With default config
    nemotron embed eval -c default

    # With custom config
    nemotron embed eval -c /path/to/config.yaml

    # With CLI overrides
    nemotron embed eval -c default finetuned_model_path=/path/to/model

    # Evaluate NIM endpoint
    nemotron embed eval -c default eval_nim=true nim_url=http://localhost:8001
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from nemotron.kit.config_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class NIMEmbeddingModel:
    """Embedding model that uses NIM API for inference.

    Compatible with BEIR's dense retrieval framework.
    Handles the NIM-specific `input_type` parameter for queries vs passages.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8001",
        model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        batch_size: int = 32,
        timeout: int = 60,
    ):
        """Initialize NIM embedding model.

        Args:
            api_url: Base URL for NIM API.
            model: Model name for API requests.
            batch_size: Batch size for API requests.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url.rstrip("/")
        self.embeddings_url = f"{self.api_url}/v1/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if NIM API is reachable."""
        import urllib.request
        import urllib.error

        try:
            health_url = f"{self.api_url}/v1/health/ready"
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status != 200:
                    print(f"Warning: NIM health check returned status {response.status}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Warning: Could not reach NIM at {self.api_url}: {e}")

    def _encode_batch(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """Encode a batch of texts using NIM API.

        Args:
            texts: List of texts to encode.
            input_type: Either 'query' or 'passage'.

        Returns:
            List of embedding vectors.
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "input": texts,
            "model": self.model,
            "input_type": input_type,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        req = urllib.request.Request(
            self.embeddings_url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                # Sort by index to ensure correct order
                embeddings_data = sorted(result["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings_data]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"NIM API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"NIM API connection error: {e}") from e

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Encode queries using NIM API.

        Args:
            queries: List of query texts.
            batch_size: Batch size (uses default if None).
            **kwargs: Additional arguments (ignored for API compatibility).

        Returns:
            List of query embedding vectors.
        """
        import numpy as np

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            embeddings = self._encode_batch(batch, input_type="query")
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, dict[str, str]],
        batch_size: int | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Encode corpus documents using NIM API.

        Args:
            corpus: Corpus as list of dicts with 'title' and 'text' keys,
                   or dict mapping doc_id to document dict.
            batch_size: Batch size (uses default if None).
            **kwargs: Additional arguments (ignored for API compatibility).

        Returns:
            List of document embedding vectors.
        """
        import numpy as np

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        # Handle both list and dict corpus formats
        if isinstance(corpus, dict):
            corpus_list = list(corpus.values())
        else:
            corpus_list = corpus

        # Combine title and text for each document
        texts = []
        for doc in corpus_list:
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                texts.append(f"{title} {text}")
            else:
                texts.append(text)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._encode_batch(batch, input_type="passage")
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


class EvalConfig(RecipeSettings):
    """Evaluation configuration for embedding models."""

    model_config = ConfigDict(extra="forbid")

    # Model paths
    base_model: str = Field(default="nvidia/llama-nemotron-embed-1b-v2", description="Base embedding model for comparison.")
    finetuned_model_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated", description="Path to fine-tuned model checkpoint.")

    # Evaluation data
    eval_data_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage1_data_prep/eval_beir", description="Path to BEIR-formatted evaluation data.")

    # Output settings
    output_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage3_eval", description="Directory for saving evaluation results.")

    # Evaluation settings
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 100], description="K values for Recall@k and Precision@k metrics.")
    batch_size: int = Field(default=128, gt=0, description="Batch size for encoding.")
    max_length: int = Field(default=512, gt=0, description="Maximum sequence length.")
    corpus_chunk_size: int = Field(default=50000, gt=0, description="Chunk size for corpus encoding.")

    # Model settings
    pooling: Literal["mean", "cls", "max"] = Field(default="mean", description="Pooling strategy (BEIR naming: mean=avg, cls=cls, max=last).")
    normalize: bool = Field(default=True, description="Whether to L2 normalize embeddings.")
    query_prefix: str = Field(default="query:", description="Prefix for query inputs.")
    passage_prefix: str = Field(default="passage:", description="Prefix for passage inputs.")

    # Evaluation mode
    eval_base: bool = Field(default=True, description="Whether to evaluate the base model.")
    eval_finetuned: bool = Field(default=True, description="Whether to evaluate the fine-tuned model.")

    # NIM API evaluation settings
    eval_nim: bool = Field(default=False, description="Whether to evaluate a NIM API endpoint.")
    nim_url: str = Field(default="http://localhost:8000", description="NIM API base URL.")
    nim_model: str = Field(default="nvidia/llama-3.2-nv-embedqa-1b-v2", description="Model name for NIM API requests.")
    nim_batch_size: int = Field(default=32, gt=0, description="Batch size for NIM API requests.")
    nim_timeout: int = Field(default=60, gt=0, description="Timeout in seconds for NIM API requests.")


def evaluate_model(
    model_path: str | Path,
    dataset_path: Path,
    max_length: int = 512,
    batch_size: int = 128,
    corpus_chunk_size: int = 50000,
    k_values: list[int] | None = None,
    pooling: str = "mean",
    normalize: bool = True,
    query_prefix: str = "query:",
    passage_prefix: str = "passage:",
) -> tuple[dict, dict]:
    """Evaluate an embedding model on a BEIR dataset.

    Args:
        model_path: Path to the model.
        dataset_path: Path to BEIR-formatted evaluation data.
        max_length: Maximum sequence length.
        batch_size: Batch size for encoding.
        corpus_chunk_size: Chunk size for corpus encoding.
        k_values: K values for metrics.
        pooling: Pooling strategy.
        normalize: Whether to normalize embeddings.
        query_prefix: Prefix for queries.
        passage_prefix: Prefix for passages.

    Returns:
        Tuple of (metrics dict, results dict).
    """
    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval import models
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.dense.exact_search import (
            DenseRetrievalExactSearch as DRES,
        )
    except ImportError:
        print("Error: BEIR is required for evaluation. Install with: pip install beir")
        sys.exit(1)

    if k_values is None:
        k_values = [1, 5, 10, 100]

    dense_model = models.HuggingFace(
        model_path=str(model_path),
        max_length=max_length,
        append_eos_token=False,
        pooling=pooling,
        normalize=normalize,
        prompts={"query": query_prefix, "passage": passage_prefix},
        torch_dtype="bfloat16",
    )

    dres_model = DRES(
        dense_model,
        corpus_chunk_size=corpus_chunk_size,
        batch_size=batch_size,
    )

    retriever = EvaluateRetrieval(
        dres_model,
        score_function="dot",
        k_values=k_values,
    )

    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split="test")
    results = retriever.retrieve(corpus, queries)
    metrics = retriever.evaluate(qrels, results, retriever.k_values)

    return metrics, results


def evaluate_nim(
    nim_url: str,
    nim_model: str,
    dataset_path: Path,
    batch_size: int = 32,
    timeout: int = 60,
    k_values: list[int] | None = None,
) -> tuple[dict, dict]:
    """Evaluate a NIM API endpoint on a BEIR dataset.

    Args:
        nim_url: Base URL for NIM API.
        nim_model: Model name for API requests.
        dataset_path: Path to BEIR-formatted evaluation data.
        batch_size: Batch size for API requests.
        timeout: Request timeout in seconds.
        k_values: K values for metrics.

    Returns:
        Tuple of (metrics dict, results dict).
    """
    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.dense.exact_search import (
            DenseRetrievalExactSearch as DRES,
        )
    except ImportError:
        print("Error: BEIR is required for evaluation. Install with: pip install beir")
        sys.exit(1)

    if k_values is None:
        k_values = [1, 5, 10, 100]

    # Create NIM embedding model
    nim_model_instance = NIMEmbeddingModel(
        api_url=nim_url,
        model=nim_model,
        batch_size=batch_size,
        timeout=timeout,
    )

    # Wrap in DRES for BEIR compatibility
    dres_model = DRES(
        nim_model_instance,
        corpus_chunk_size=50000,
        batch_size=batch_size,
    )

    retriever = EvaluateRetrieval(
        dres_model,
        score_function="dot",
        k_values=k_values,
    )

    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split="test")
    results = retriever.retrieve(corpus, queries)
    metrics = retriever.evaluate(qrels, results, retriever.k_values)

    return metrics, results


def _print_summary_metrics(metrics: tuple, k_values: list[int]) -> None:
    """Print NDCG and Recall at the highest available k value."""
    k = max(k_values)
    for name, idx in [("NDCG", 0), ("Recall", 2)]:
        key = f"{name}@{k}"
        val = metrics[idx].get(key)
        if val is not None:
            print(f"   {key}:{' ' * (10 - len(key))}{val:.5f}")
        else:
            print(f"   {key}:{' ' * (10 - len(key))}N/A")


def run_eval(cfg: EvalConfig) -> dict:
    """Run embedding model evaluation.

    Args:
        cfg: Evaluation configuration.

    Returns:
        Dictionary with evaluation results.
    """
    # Trust remote code for HuggingFace models (e.g. nvidia/llama-nemotron-embed)
    # to avoid interactive prompts during evaluation.
    os.environ.setdefault("HF_HUB_TRUST_REMOTE_CODE", "1")
    print(f"📊 Embedding Model Evaluation")
    print(f"=" * 60)
    print(f"Eval data:       {cfg.eval_data_path}")
    print(f"Base model:      {cfg.base_model}")
    print(f"Finetuned model: {cfg.finetuned_model_path}")
    if cfg.eval_nim:
        print(f"NIM endpoint:    {cfg.nim_url}")
        print(f"NIM model:       {cfg.nim_model}")
    print(f"K values:        {cfg.k_values}")
    print(f"=" * 60)
    print()

    # Validate inputs
    if not cfg.eval_data_path.exists():
        print(f"Error: Eval data path not found: {cfg.eval_data_path}", file=sys.stderr)
        print("       Please run stage1_data_prep first or provide eval data.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Evaluate base model
    if cfg.eval_base:
        print(f"📈 Evaluating base model: {cfg.base_model}")
        base_metrics, _ = evaluate_model(
            model_path=cfg.base_model,
            dataset_path=cfg.eval_data_path,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            corpus_chunk_size=cfg.corpus_chunk_size,
            k_values=cfg.k_values,
            pooling=cfg.pooling,
            normalize=cfg.normalize,
            query_prefix=cfg.query_prefix,
            passage_prefix=cfg.passage_prefix,
        )
        results["base"] = base_metrics
        _print_summary_metrics(base_metrics, cfg.k_values)
        print()

    # Evaluate fine-tuned model
    if cfg.eval_finetuned:
        if not cfg.finetuned_model_path.exists():
            print(f"Warning: Fine-tuned model not found at {cfg.finetuned_model_path}")
            print("         Skipping fine-tuned model evaluation.")
        else:
            print(f"📈 Evaluating fine-tuned model: {cfg.finetuned_model_path}")
            ft_metrics, _ = evaluate_model(
                model_path=cfg.finetuned_model_path,
                dataset_path=cfg.eval_data_path,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                corpus_chunk_size=cfg.corpus_chunk_size,
                k_values=cfg.k_values,
                pooling=cfg.pooling,
                normalize=cfg.normalize,
                query_prefix=cfg.query_prefix,
                passage_prefix=cfg.passage_prefix,
            )
            results["finetuned"] = ft_metrics
            _print_summary_metrics(ft_metrics, cfg.k_values)
            print()

    # Evaluate NIM endpoint
    if cfg.eval_nim:
        print(f"📈 Evaluating NIM endpoint: {cfg.nim_url}")
        try:
            nim_metrics, _ = evaluate_nim(
                nim_url=cfg.nim_url,
                nim_model=cfg.nim_model,
                dataset_path=cfg.eval_data_path,
                batch_size=cfg.nim_batch_size,
                timeout=cfg.nim_timeout,
                k_values=cfg.k_values,
            )
            results["nim"] = nim_metrics
            _print_summary_metrics(nim_metrics, cfg.k_values)
            print()
        except Exception as e:
            print(f"   Error evaluating NIM: {e}")
            print()

    # Print comparison
    if "base" in results and "finetuned" in results:
        print(f"📊 Comparison (Base -> Fine-tuned)")
        print(f"=" * 60)

        metric_names = ["NDCG", "Recall"]
        metric_indices = [0, 2]

        for name, idx in zip(metric_names, metric_indices):
            print(f"  {name}:")
            for k in results["base"][idx]:
                base_val = results["base"][idx][k]
                ft_val = results["finetuned"][idx][k]
                diff = ft_val - base_val
                sign = "+" if diff > 0 else ""
                print(f"    {k}: {base_val:.5f} → {ft_val:.5f} ({sign}{diff:.5f})")
        print()

    # Print NIM vs Fine-tuned comparison (accuracy check for export)
    if "finetuned" in results and "nim" in results:
        print(f"📊 Comparison (Fine-tuned -> NIM)")
        print(f"=" * 60)
        print(f"   This verifies the exported model matches the checkpoint accuracy.")
        print()

        metric_names = ["NDCG", "Recall"]
        metric_indices = [0, 2]

        for name, idx in zip(metric_names, metric_indices):
            print(f"  {name}:")
            for k in results["finetuned"][idx]:
                ft_val = results["finetuned"][idx][k]
                nim_val = results["nim"][idx][k]
                diff = nim_val - ft_val
                sign = "+" if diff > 0 else ""
                # ONNX/TensorRT conversion introduces small numerical differences.
                # Low-k metrics (e.g. @1) are noisier since a single rank swap
                # changes the score; higher-k metrics should be more stable.
                at_k = int(k.split("@")[1]) if "@" in k else 1
                threshold = 0.03 if at_k < 5 else 0.01
                status = "✓" if abs(diff) < threshold else "⚠️"
                print(f"    {k}: {ft_val:.5f} → {nim_val:.5f} ({sign}{diff:.5f}) {status}")
        print()

    # Save results
    results_file = cfg.output_dir / "eval_results.json"

    # Convert metrics tuples to dicts for JSON serialization
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            "NDCG": metrics[0],
            "MAP": metrics[1],
            "Recall": metrics[2],
            "Precision": metrics[3],
        }

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"✅ Evaluation complete!")
    print(f"   Results saved to: {results_file}")

    return results


def main(cfg: EvalConfig | None = None) -> dict:
    """Entry point for evaluation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Dictionary with evaluation results.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, EvalConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_eval(cfg)


if __name__ == "__main__":
    main()
