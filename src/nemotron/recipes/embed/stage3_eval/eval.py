#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "embed/eval"
# image = "nvcr.io/nvidia/pytorch:25.12-py3"
# setup = "PyTorch pre-installed. Stage dependencies resolved via UV at runtime."
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 1
# ///

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

import gc
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

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
        invalid_embedding_retries: int = 3,
        expected_dimension: int | None = None,
    ):
        """Initialize NIM embedding model.

        Args:
            api_url: Base URL for NIM API.
            model: Model name for API requests.
            batch_size: Batch size for API requests.
            timeout: Request timeout in seconds.
            invalid_embedding_retries: Retry limit for non-numeric NIM vectors.
            expected_dimension: Required embedding dimension, if known.
        """
        self.api_url = api_url.rstrip("/")
        self.embeddings_url = f"{self.api_url}/v1/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.invalid_embedding_retries = invalid_embedding_retries
        self.expected_dimension = expected_dimension
        self.embedding_dimension = expected_dimension
        self.invalid_embedding_retry_requests = 0
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if NIM API is reachable."""
        import urllib.error
        import urllib.request

        try:
            health_url = f"{self.api_url}/v1/health/ready"
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status != 200:
                    print(f"Warning: NIM health check returned status {response.status}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Warning: Could not reach NIM at {self.api_url}: {e}")

    @staticmethod
    def _embedding_is_valid(embedding: object) -> bool:
        """Return whether an API embedding is a finite numeric vector."""
        return (
            isinstance(embedding, list)
            and bool(embedding)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in embedding
            )
        )

    def _request_batch(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """Send one embeddings request and return validated vectors in input order."""
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "input": texts,
                "model": self.model,
                "input_type": input_type,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.embeddings_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8") if error.fp else ""
            raise RuntimeError(f"NIM API error {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"NIM API connection error: {error}") from error

        served_model = result.get("model")
        if served_model is not None and served_model != self.model:
            raise RuntimeError(f"NIM returned model {served_model!r}; expected {self.model!r}")

        embeddings_data = result.get("data")
        if not isinstance(embeddings_data, list):
            raise RuntimeError("NIM response is missing a data list")
        if not all(isinstance(item, dict) for item in embeddings_data):
            raise RuntimeError("NIM response data entries must be objects")

        indices = [item.get("index") for item in embeddings_data]
        expected_indices = list(range(len(texts)))
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
            raise RuntimeError(f"NIM returned non-integer indices: {indices!r}")
        if sorted(indices) != expected_indices:
            raise RuntimeError(f"NIM returned indices {indices!r}; expected {expected_indices!r}")

        return [item.get("embedding") for item in sorted(embeddings_data, key=lambda item: item["index"])]

    def _validate_embedding_dimensions(self, embeddings: list[list[float]]) -> None:
        """Require one stable embedding dimension across the full evaluation."""
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise RuntimeError(f"NIM returned inconsistent embedding dimensions: {sorted(dimensions)}")
        dimension = dimensions.pop()
        if self.embedding_dimension is None:
            self.embedding_dimension = dimension
        elif dimension != self.embedding_dimension:
            raise RuntimeError(f"NIM returned embedding dimension {dimension}; expected {self.embedding_dimension}")

    def _encode_batch(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """Encode a batch, retrying transient invalid vectors independently."""
        embeddings = self._request_batch(texts, input_type)
        if len(embeddings) != len(texts):
            raise RuntimeError(f"NIM returned {len(embeddings)} embeddings for {len(texts)} inputs")

        for retry_number in range(self.invalid_embedding_retries + 1):
            invalid_indices = [
                index for index, embedding in enumerate(embeddings) if not self._embedding_is_valid(embedding)
            ]
            if not invalid_indices:
                self._validate_embedding_dimensions(embeddings)
                return embeddings
            if retry_number == self.invalid_embedding_retries:
                break

            attempt = retry_number + 1
            print(
                f"Warning: NIM returned {len(invalid_indices)} invalid embedding(s); "
                f"retrying affected inputs ({attempt}/{self.invalid_embedding_retries})."
            )
            for index in invalid_indices:
                self.invalid_embedding_retry_requests += 1
                retry_embeddings = self._request_batch([texts[index]], input_type)
                if len(retry_embeddings) != 1:
                    raise RuntimeError(f"NIM returned {len(retry_embeddings)} retry embeddings for 1 input")
                embeddings[index] = retry_embeddings[0]

        invalid_indices = [
            index for index, embedding in enumerate(embeddings) if not self._embedding_is_valid(embedding)
        ]
        raise RuntimeError(
            f"NIM returned invalid embeddings after {self.invalid_embedding_retries} retries "
            f"at batch indices {invalid_indices}"
        )

    def diagnostics(self) -> dict[str, int | str | None]:
        """Return response-validation diagnostics for result provenance."""
        return {
            "requested_model": self.model,
            "embedding_dimension": self.embedding_dimension,
            "invalid_embedding_retry_requests": self.invalid_embedding_retry_requests,
        }

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

        return np.asarray(all_embeddings, dtype=np.float32)

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

        return np.asarray(all_embeddings, dtype=np.float32)


class EvalConfig(RecipeSettings):
    """Evaluation configuration for embedding models."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: Path = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/embed/nemotron-3-1b",
        description="Root directory for this model profile's pipeline artifacts.",
    )

    # Model paths
    base_model: str = Field(
        default="nvidia/Nemotron-3-Embed-1B-BF16", description="Base embedding model for comparison."
    )
    finetuned_model_path: Path = Field(
        default_factory=lambda data: data["artifact_root"] / "stage2_finetune/checkpoints/LATEST/model/consolidated",
        description="Path to fine-tuned model checkpoint.",
    )

    # Evaluation data
    eval_data_path: Path = Field(
        default_factory=lambda data: data["artifact_root"] / "stage1_data_prep/eval_beir",
        description="Path to BEIR-formatted evaluation data.",
    )

    # Output settings
    output_dir: Path = Field(
        default_factory=lambda data: data["artifact_root"] / "stage3_eval",
        description="Directory for saving evaluation results.",
    )

    # Evaluation settings
    k_values: list[int] = Field(
        default_factory=lambda: [1, 5, 10, 100], description="K values for Recall@k and Precision@k metrics."
    )
    batch_size: int = Field(default=4, gt=0, description="Batch size for encoding.")
    max_length: int = Field(default=512, gt=0, description="Maximum sequence length.")
    corpus_chunk_size: int = Field(default=50000, gt=0, description="Chunk size for corpus encoding.")

    # Model settings
    pooling: Literal["mean", "cls", "max"] = Field(
        default="mean", description="Pooling strategy (BEIR naming: mean=avg, cls=cls, max=last)."
    )
    normalize: bool = Field(default=True, description="Whether to L2 normalize embeddings.")
    query_prefix: str = Field(default="query: ", description="Prefix for query inputs.")
    passage_prefix: str = Field(default="passage: ", description="Prefix for passage inputs.")

    # Evaluation mode
    eval_base: bool = Field(default=True, description="Whether to evaluate the base model.")
    eval_finetuned: bool = Field(default=True, description="Whether to evaluate the fine-tuned model.")

    # NIM API evaluation settings
    eval_nim: bool = Field(default=False, description="Whether to evaluate a NIM API endpoint.")
    nim_url: str = Field(default="http://localhost:8000", description="NIM API base URL.")
    nim_model: str = Field(default="nvidia/nemotron-3-embed-1b", description="Model name for NIM API requests.")
    nim_batch_size: int = Field(default=32, gt=0, description="Batch size for NIM API requests.")
    nim_timeout: int = Field(default=60, gt=0, description="Timeout in seconds for NIM API requests.")
    nim_invalid_embedding_retries: int = Field(
        default=32,
        ge=0,
        description="Retry limit for null or non-finite NIM embedding vectors.",
    )
    nim_embedding_dimension: int | None = Field(
        default=2048,
        gt=0,
        description="Expected NIM embedding dimension. None learns it from the first response.",
    )
    nim_metric_tolerance: float = Field(
        default=0.01,
        ge=0,
        description="Informational absolute metric-drift tolerance for k >= 5.",
    )
    nim_metric_low_k_tolerance: float = Field(
        default=0.03,
        ge=0,
        description="Informational absolute metric-drift tolerance for k < 5.",
    )
    fail_on_nim_metric_drift: bool = Field(
        default=False,
        description="Fail when NIM/checkpoint metric drift exceeds configured tolerances.",
    )


@contextmanager
def _allow_beir_tokenizer_remote_code(beir_huggingface_module):
    """Make BEIR trust custom tokenizer code while constructing its wrapper.

    BEIR 2.2 passes ``trust_remote_code=True`` to ``AutoModel`` but not to
    ``AutoTokenizer``. Custom embedding checkpoints therefore prompt on a
    non-interactive Slurm job and fail. Replace only the module-local tokenizer
    reference while constructing the wrapper, then restore it immediately.
    """
    tokenizer_class = beir_huggingface_module.AutoTokenizer

    class _TrustedAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs["trust_remote_code"] = True
            return tokenizer_class.from_pretrained(*args, **kwargs)

    beir_huggingface_module.AutoTokenizer = _TrustedAutoTokenizer
    try:
        yield
    finally:
        beir_huggingface_module.AutoTokenizer = tokenizer_class


def evaluate_model(
    model_path: str | Path,
    dataset_path: Path,
    max_length: int = 512,
    batch_size: int = 4,
    corpus_chunk_size: int = 50000,
    k_values: list[int] | None = None,
    pooling: str = "mean",
    normalize: bool = True,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
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
        from beir.retrieval.models import huggingface as beir_huggingface
        from beir.retrieval.search.dense.exact_search import (
            DenseRetrievalExactSearch as DRES,  # noqa: N817
        )
    except ImportError:
        print("Error: BEIR is required for evaluation. Install with: pip install beir")
        sys.exit(1)

    if k_values is None:
        k_values = [1, 5, 10, 100]

    with _allow_beir_tokenizer_remote_code(beir_huggingface):
        dense_model = models.HuggingFace(
            model_path=str(model_path),
            max_length=max_length,
            append_eos_token=False,
            pooling=pooling,
            normalize=normalize,
            prompts={"query": query_prefix, "passage": passage_prefix},
            dtype="bfloat16",
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
    invalid_embedding_retries: int = 3,
    expected_dimension: int | None = None,
    k_values: list[int] | None = None,
) -> tuple[dict, dict, dict[str, int | str | None]]:
    """Evaluate a NIM API endpoint on a BEIR dataset.

    Args:
        nim_url: Base URL for NIM API.
        nim_model: Model name for API requests.
        dataset_path: Path to BEIR-formatted evaluation data.
        batch_size: Batch size for API requests.
        timeout: Request timeout in seconds.
        invalid_embedding_retries: Retry limit for invalid NIM vectors.
        expected_dimension: Required embedding dimension, if known.
        k_values: K values for metrics.

    Returns:
        Tuple of (metrics dict, results dict, response diagnostics).
    """
    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.dense.exact_search import (
            DenseRetrievalExactSearch as DRES,  # noqa: N817
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
        invalid_embedding_retries=invalid_embedding_retries,
        expected_dimension=expected_dimension,
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

    return metrics, results, nim_model_instance.diagnostics()


def _release_cuda_memory() -> None:
    """Release model references and cached CUDA allocations between eval modes."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
    print("📊 Embedding Model Evaluation")
    print("=" * 60)
    print(f"Eval data:       {cfg.eval_data_path}")
    print(f"Base model:      {cfg.base_model}")
    print(f"Finetuned model: {cfg.finetuned_model_path}")
    if cfg.eval_nim:
        print(f"NIM endpoint:    {cfg.nim_url}")
        print(f"NIM model:       {cfg.nim_model}")
    print(f"K values:        {cfg.k_values}")
    print("=" * 60)
    print()

    # Validate inputs
    if not cfg.eval_data_path.exists():
        print(f"Error: Eval data path not found: {cfg.eval_data_path}", file=sys.stderr)
        print("       Please run stage1_data_prep first or provide eval data.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    metadata = {
        "eval_data_path": str(cfg.eval_data_path.resolve()),
        "k_values": cfg.k_values,
        "base_model": cfg.base_model if cfg.eval_base else None,
        "finetuned_model_path": (str(cfg.finetuned_model_path.resolve()) if cfg.eval_finetuned else None),
        "nim_url": cfg.nim_url if cfg.eval_nim else None,
        "nim_model": cfg.nim_model if cfg.eval_nim else None,
    }
    nim_diagnostics: dict[str, int | str | None] | None = None
    nim_metric_comparison: dict | None = None
    drift_failure = False

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
        _release_cuda_memory()

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
            _release_cuda_memory()

    # Evaluate NIM endpoint
    if cfg.eval_nim:
        print(f"📈 Evaluating NIM endpoint: {cfg.nim_url}")
        try:
            nim_metrics, _, nim_diagnostics = evaluate_nim(
                nim_url=cfg.nim_url,
                nim_model=cfg.nim_model,
                dataset_path=cfg.eval_data_path,
                batch_size=cfg.nim_batch_size,
                timeout=cfg.nim_timeout,
                invalid_embedding_retries=cfg.nim_invalid_embedding_retries,
                expected_dimension=cfg.nim_embedding_dimension,
                k_values=cfg.k_values,
            )
            results["nim"] = nim_metrics
            _print_summary_metrics(nim_metrics, cfg.k_values)
            print()
        except Exception as error:
            print(f"   Error evaluating NIM: {error}")
            raise

    # Print comparison
    if "base" in results and "finetuned" in results:
        print("📊 Comparison (Base -> Fine-tuned)")
        print("=" * 60)

        metric_names = ["NDCG", "Recall"]
        metric_indices = [0, 2]

        for name, idx in zip(metric_names, metric_indices):
            print(f"  {name}:")
            for k in results["base"][idx]:
                base_val = results["base"][idx][k]
                ft_val = results["finetuned"][idx][k]
                diff = ft_val - base_val
                sign = "+" if diff > 0 else ""
                pct = (diff / base_val * 100) if base_val != 0 else float("inf")
                print(f"    {k}: {base_val:.5f} → {ft_val:.5f} ({sign}{diff:.5f}, {sign}{pct:.1f}%)")
        print()

    # Compare aggregate retrieval behavior. This is not a model-identity proof:
    # local Hugging Face and NIM preprocessing/runtime paths may differ.
    if "finetuned" in results and "nim" in results:
        print("📊 Behavioral metric comparison (Fine-tuned -> NIM)")
        print("=" * 60)
        print("   Informational unless fail_on_nim_metric_drift=true.")
        print("   Artifact mount/fingerprint validation establishes deployment identity.")
        print()

        metric_names = ["NDCG", "Recall"]
        metric_indices = [0, 2]
        deltas = {}
        within_tolerance = True

        for name, idx in zip(metric_names, metric_indices):
            print(f"  {name}:")
            deltas[name] = {}
            for key in results["finetuned"][idx]:
                ft_val = results["finetuned"][idx][key]
                nim_val = results["nim"][idx][key]
                diff = nim_val - ft_val
                at_k = int(key.split("@")[1]) if "@" in key else 1
                threshold = cfg.nim_metric_low_k_tolerance if at_k < 5 else cfg.nim_metric_tolerance
                metric_within_tolerance = abs(diff) <= threshold
                within_tolerance = within_tolerance and metric_within_tolerance
                deltas[name][key] = {
                    "checkpoint": ft_val,
                    "nim": nim_val,
                    "delta": diff,
                    "tolerance": threshold,
                    "within_tolerance": metric_within_tolerance,
                }
                label = "within tolerance" if metric_within_tolerance else "drift"
                print(f"    {key}: {ft_val:.5f} → {nim_val:.5f} ({diff:+.5f}) [{label}]")
        print()

        nim_metric_comparison = {
            "kind": "aggregate_behavioral_metric_drift",
            "model_identity_proof": False,
            "within_tolerance": within_tolerance,
            "fail_on_drift": cfg.fail_on_nim_metric_drift,
            "deltas": deltas,
        }
        drift_failure = cfg.fail_on_nim_metric_drift and not within_tolerance

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

    metadata["nim_diagnostics"] = nim_diagnostics
    metadata["nim_metric_comparison"] = nim_metric_comparison
    serializable_results["_metadata"] = metadata

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    if drift_failure:
        raise RuntimeError(
            f"NIM behavioral metric drift exceeds the configured tolerance; details saved to {results_file}"
        )

    print("✅ Evaluation complete!")
    print(f"   Results saved to: {results_file}")

    # Save artifact (registers with artifact registry if kit.init() was called)
    try:
        from nemotron.kit.artifacts.base import Artifact

        artifact = Artifact(path=cfg.output_dir)
        artifact.save(name="embed/eval")
    except Exception:
        pass  # Artifact save is best-effort — don't break the pipeline

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
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        try:
            cfg = load_config(config_path, cli_overrides, EvalConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_eval(cfg)


if __name__ == "__main__":
    main()
