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

import base64
import gc
import json
import math
import mimetypes
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class NIMEmbeddingModel:
    """Embedding model that uses a NIM or vLLM API for inference.

    Compatible with BEIR's dense retrieval framework.
    Both backends receive their native ``input_type`` parameter. vLLM uses its
    Cohere-compatible ``/v2/embed`` endpoint so the checkpoint applies prompts.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8001",
        model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        batch_size: int = 32,
        timeout: int = 60,
        invalid_embedding_retries: int = 3,
        expected_dimension: int | None = None,
        api_backend: Literal["nim", "vllm"] = "nim",
        dataset_path: Path | None = None,
        api_truncate: Literal["NONE", "START", "END"] | None = None,
    ):
        """Initialize NIM embedding model.

        Args:
            api_url: Base URL for NIM API.
            model: Model name for API requests.
            batch_size: Batch size for API requests.
            timeout: Request timeout in seconds.
            invalid_embedding_retries: Retry limit for non-numeric NIM vectors.
            expected_dimension: Required embedding dimension, if known.
            api_backend: Request and health-check conventions used by the server.
            api_truncate: Optional Cohere truncation policy. Requires the vLLM backend.
        """
        if api_truncate not in (None, "NONE", "START", "END"):
            raise ValueError("api_truncate must be None, NONE, START, or END")
        if api_truncate is not None and api_backend != "vllm":
            raise ValueError("api_truncate requires api_backend=vllm")
        self.service_name = "vLLM" if api_backend == "vllm" else "NIM"
        self.api_url = api_url.rstrip("/")
        endpoint = "/v2/embed" if api_backend == "vllm" else "/v1/embeddings"
        self.embeddings_url = f"{self.api_url}{endpoint}"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.invalid_embedding_retries = invalid_embedding_retries
        self.expected_dimension = expected_dimension
        self.api_backend = api_backend
        self.api_truncate = api_truncate
        self.embedding_dimension = expected_dimension
        self.invalid_embedding_retry_requests = 0
        self.dataset_path = dataset_path
        self._check_connection()

    def _check_connection(self) -> None:
        """Check whether the selected embedding API is reachable."""
        import urllib.error
        import urllib.request

        service_name = "vLLM" if self.api_backend == "vllm" else "NIM"
        try:
            health_path = "/health" if self.api_backend == "vllm" else "/v1/health/ready"
            health_url = f"{self.api_url}{health_path}"
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status != 200:
                    print(f"Warning: {service_name} health check returned status {response.status}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Warning: Could not reach {service_name} at {self.api_url}: {e}")

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
        inputs: list[str | dict],
        input_type: str,
    ) -> list[list[float]]:
        """Send one embeddings request and return validated vectors in input order."""
        import urllib.error
        import urllib.request

        if self.api_backend == "vllm":
            payload_data = {
                "model": self.model,
                "input_type": "document" if input_type == "passage" else input_type,
                "embedding_types": ["float"],
            }
            if self.api_truncate is not None:
                payload_data["truncate"] = self.api_truncate
            if any(not isinstance(item, str) for item in inputs):
                payload_data["inputs"] = [
                    {"content": [{"type": "text", "text": item}]} if isinstance(item, str) else item for item in inputs
                ]
            else:
                payload_data["texts"] = inputs
        else:
            if any(not isinstance(item, str) for item in inputs):
                raise ValueError("Multimodal inputs require api_backend=vllm")
            payload_data = {"input": inputs, "model": self.model, "input_type": input_type}
        payload = json.dumps(payload_data).encode("utf-8")

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
            service_name = "vLLM" if self.api_backend == "vllm" else "NIM"
            raise RuntimeError(f"{service_name} API error {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            service_name = "vLLM" if self.api_backend == "vllm" else "NIM"
            raise RuntimeError(f"{service_name} API connection error: {error}") from error

        if self.api_backend == "vllm":
            embeddings = result.get("embeddings")
            if not isinstance(embeddings, dict) or not isinstance(embeddings.get("float"), list):
                raise RuntimeError("vLLM response is missing embeddings.float")
            vectors = embeddings["float"]
            if len(vectors) != len(inputs):
                raise RuntimeError(f"vLLM returned {len(vectors)} embeddings; expected {len(inputs)}")
            return vectors

        served_model = result.get("model")
        if served_model is not None and served_model != self.model:
            raise RuntimeError(f"NIM returned model {served_model!r}; expected {self.model!r}")

        embeddings_data = result.get("data")
        if not isinstance(embeddings_data, list):
            raise RuntimeError("NIM response is missing a data list")
        if not all(isinstance(item, dict) for item in embeddings_data):
            raise RuntimeError("NIM response data entries must be objects")

        indices = [item.get("index") for item in embeddings_data]
        expected_indices = list(range(len(inputs)))
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
            raise RuntimeError(f"NIM returned non-integer indices: {indices!r}")
        if sorted(indices) != expected_indices:
            raise RuntimeError(f"NIM returned indices {indices!r}; expected {expected_indices!r}")

        return [item.get("embedding") for item in sorted(embeddings_data, key=lambda item: item["index"])]

    def _validate_embedding_dimensions(self, embeddings: list[list[float]]) -> None:
        """Require one stable embedding dimension across the full evaluation."""
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise RuntimeError(f"{self.service_name} returned inconsistent embedding dimensions: {sorted(dimensions)}")
        dimension = dimensions.pop()
        if self.embedding_dimension is None:
            self.embedding_dimension = dimension
        elif dimension != self.embedding_dimension:
            raise RuntimeError(
                f"{self.service_name} returned embedding dimension {dimension}; expected {self.embedding_dimension}"
            )

    def _encode_batch(
        self,
        inputs: list[str | dict],
        input_type: str,
    ) -> list[list[float]]:
        """Encode a batch, retrying transient invalid vectors independently."""
        embeddings = self._request_batch(inputs, input_type)
        if len(embeddings) != len(inputs):
            raise RuntimeError(f"{self.service_name} returned {len(embeddings)} embeddings for {len(inputs)} inputs")

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
                f"Warning: {self.service_name} returned {len(invalid_indices)} invalid embedding(s); "
                f"retrying affected inputs ({attempt}/{self.invalid_embedding_retries})."
            )
            for index in invalid_indices:
                self.invalid_embedding_retry_requests += 1
                retry_embeddings = self._request_batch([inputs[index]], input_type)
                if len(retry_embeddings) != 1:
                    raise RuntimeError(
                        f"{self.service_name} returned {len(retry_embeddings)} retry embeddings for 1 input"
                    )
                embeddings[index] = retry_embeddings[0]

        invalid_indices = [
            index for index, embedding in enumerate(embeddings) if not self._embedding_is_valid(embedding)
        ]
        raise RuntimeError(
            f"{self.service_name} returned invalid embeddings after {self.invalid_embedding_retries} retries "
            f"at batch indices {invalid_indices}"
        )

    def diagnostics(self) -> dict[str, int | str | None]:
        """Return response-validation diagnostics for result provenance."""
        diagnostics = {
            "api_backend": self.api_backend,
            "requested_model": self.model,
            "embedding_dimension": self.embedding_dimension,
            "invalid_embedding_retry_requests": self.invalid_embedding_retry_requests,
        }
        if self.api_truncate is not None:
            diagnostics["api_truncate"] = self.api_truncate
        return diagnostics

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

        # Materialize image bytes only for the current request, not the BEIR chunk.
        for i in range(0, len(corpus_list), batch_size):
            inputs = [self._corpus_input(doc) for doc in corpus_list[i : i + batch_size]]
            embeddings = self._encode_batch(inputs, input_type="passage")
            all_embeddings.extend(embeddings)
            del inputs

        return np.asarray(all_embeddings, dtype=np.float32)

    def _corpus_input(self, document: dict[str, str]) -> str | dict:
        """Prepare one document without retaining image bytes between batches."""
        title = document.get("title", "")
        text = document.get("text", "")
        document_text = f"{title} {text}".strip()
        image_path = document.get("image_path")
        if not image_path:
            return document_text
        if self.api_backend != "vllm":
            raise ValueError("Corpus contains image_path but the selected backend is not vLLM")
        if self.dataset_path is None:
            raise ValueError("dataset_path is required to resolve multimodal corpus images")
        image = self._resolve_image_path(str(image_path))
        mime_type = mimetypes.guess_type(image.name)[0] or "image/png"
        image_data = base64.b64encode(image.read_bytes()).decode("ascii")
        content: list[dict] = [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}]
        if document_text:
            content.append({"type": "text", "text": document_text})
        return {"content": content}

    def _resolve_image_path(self, relative_path: str) -> Path:
        """Resolve an image path relative to a BEIR view or one of its parents."""
        assert self.dataset_path is not None
        for root in (self.dataset_path, *self.dataset_path.parents):
            candidate = root / relative_path
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not resolve corpus image {relative_path!r} from {self.dataset_path}")


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
    image_root: Path | None = Field(
        default=None,
        description="Artifact root for portable multimodal image paths; defaults to eval_data_path.",
    )
    sdg_input_path: Path | None = Field(
        default=None,
        description="Stage 0 generation-result manifest used to resolve the exact portable evaluation bundle.",
    )
    retrieval_view: Literal["text", "image", "image_and_text"] | None = Field(
        default=None,
        description="Portable evaluation view selected from sdg_input_path.",
    )
    retrieval_split_protocol: Literal["grouped_query_disjoint"] | None = Field(
        default=None,
        description="Expected portable bundle split protocol; omitted means use the manifest declaration.",
    )
    ignore_identical_ids: bool = Field(
        default=True,
        description="Preserve BEIR's legacy query/corpus identical-ID exclusion when true.",
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
    query_prefix: str | None = Field(default="query: ", description="Prefix for query inputs.")
    passage_prefix: str | None = Field(default="passage: ", description="Prefix for passage inputs.")
    model_family: Literal["text", "mistral3_vl"] = Field(default="text", description="Local evaluation model family.")
    query_max_length: int | None = Field(default=None, gt=0, description="Multimodal query sequence limit.")
    passage_max_length: int | None = Field(default=None, gt=0, description="Multimodal document sequence limit.")
    image_longest_edge: int | None = Field(default=None, gt=0, description="Multimodal image resize limit.")
    use_text_in_document: bool = Field(default=True, description="Retain text alongside document images.")
    local_backend: Literal["huggingface", "automodel"] = Field(
        default="huggingface",
        description="Local checkpoint runtime. AutoModel preserves saved retrieval attention semantics.",
    )
    tokenizer_force_default: bool = Field(
        default=False,
        description="Use AutoModel's Hugging Face tokenizer wrapper instead of model-type registry dispatch.",
    )

    # Evaluation mode
    eval_base: bool = Field(default=True, description="Whether to evaluate the base model.")
    eval_finetuned: bool = Field(default=True, description="Whether to evaluate the fine-tuned model.")

    # NIM API evaluation settings
    eval_nim: bool = Field(default=False, description="Whether to evaluate a NIM API endpoint.")
    nim_url: str = Field(default="http://localhost:8000", description="NIM API base URL.")
    nim_model: str = Field(default="nvidia/nemotron-3-embed-1b", description="Model name for NIM API requests.")
    nim_batch_size: int = Field(default=32, gt=0, description="Batch size for NIM API requests.")
    nim_timeout: int = Field(default=60, gt=0, description="Timeout in seconds for NIM API requests.")
    embedding_api_backend: Literal["nim", "vllm"] = Field(
        default="nim",
        description="Request and health-check conventions used by the deployed embedding service.",
    )
    embedding_api_truncate: Literal["NONE", "START", "END"] | None = Field(
        default=None,
        description="Optional Cohere truncation policy for vLLM; None preserves the server default.",
    )
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

    @model_validator(mode="after")
    def _validate_portable_evaluation_source(self):
        if (self.sdg_input_path is None) != (self.retrieval_view is None):
            raise ValueError("sdg_input_path and retrieval_view must be set together for portable evaluation")
        if self.retrieval_split_protocol is not None and self.sdg_input_path is None:
            raise ValueError("retrieval_split_protocol requires sdg_input_path and retrieval_view")
        return self

    @model_validator(mode="after")
    def _validate_api_truncation(self):
        if self.embedding_api_truncate is not None and self.embedding_api_backend != "vllm":
            raise ValueError("embedding_api_truncate requires embedding_api_backend=vllm")
        return self

    @model_validator(mode="after")
    def _validate_multimodal_local_backend(self):
        if self.model_family != "mistral3_vl":
            return self
        if (self.eval_base or self.eval_finetuned) and self.local_backend != "automodel":
            raise ValueError("model_family=mistral3_vl local evaluation requires local_backend=automodel")
        if any(value is None for value in (self.query_max_length, self.passage_max_length)):
            raise ValueError("model_family=mistral3_vl requires query and passage limits")
        return self

    @model_validator(mode="after")
    def _validate_metric_drift_gate(self):
        if self.fail_on_nim_metric_drift and not self.eval_finetuned:
            raise ValueError("fail_on_nim_metric_drift=true requires eval_finetuned=true")
        if self.fail_on_nim_metric_drift and not self.eval_nim:
            raise ValueError("fail_on_nim_metric_drift=true requires eval_nim=true")
        return self


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
    query_prefix: str | None = "query: ",
    passage_prefix: str | None = "passage: ",
    local_backend: Literal["huggingface", "automodel"] = "huggingface",
    tokenizer_force_default: bool = False,
    model_family: Literal["text", "mistral3_vl"] = "text",
    query_max_length: int | None = None,
    passage_max_length: int | None = None,
    image_longest_edge: int | None = None,
    use_text_in_document: bool = True,
    image_root: Path | None = None,
    ignore_identical_ids: bool = True,
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
        local_backend: Runtime for loading the local retrieval checkpoint.
        tokenizer_force_default: Bypass AutoModel tokenizer registry dispatch for local native evaluation.
        model_family: Explicit local model family.
        query_max_length: Native multimodal query sequence limit.
        passage_max_length: Native multimodal document sequence limit.
        image_longest_edge: Native multimodal image resize limit.
        use_text_in_document: Retain document text alongside images.
        image_root: Artifact root containing portable corpus image paths.
        ignore_identical_ids: Apply BEIR's legacy query/corpus identical-ID exclusion.

    Returns:
        Tuple of (metrics dict, results dict).
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

    if local_backend == "automodel":
        from nemotron.recipes.embed.stage3_eval.automodel_backend import AutoModelBEIREncoder

        if pooling != "mean" or not normalize:
            raise ValueError("local_backend=automodel requires pooling=mean and normalize=true")
        multimodal_config = None
        if model_family == "mistral3_vl":
            if query_max_length is None or passage_max_length is None:
                raise ValueError("mistral3_vl evaluation requires query and passage limits")
            from nemo_automodel._transformers.mining import CheckpointMiningEncoderConfig

            processor_overrides = {
                "q_max_length": query_max_length,
                "p_max_length": passage_max_length,
                "query_prefix": None if query_prefix is None else query_prefix.removesuffix(" "),
                "passage_prefix": None if passage_prefix is None else passage_prefix.removesuffix(" "),
                "image_longest_edge": image_longest_edge,
                "use_text_in_document": use_text_in_document,
                "use_images": True,
            }
            multimodal_config = CheckpointMiningEncoderConfig(
                **{name: value for name, value in processor_overrides.items() if value is not None}
            )
        dense_model = AutoModelBEIREncoder(
            model_path=model_path,
            max_length=max_length,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            tokenizer_force_default=tokenizer_force_default,
            multimodal_config=multimodal_config,
        )
    else:
        if model_family == "mistral3_vl":
            raise ValueError("model_family=mistral3_vl requires local_backend=automodel")
        try:
            from beir.retrieval import models
            from beir.retrieval.models import huggingface as beir_huggingface
        except ImportError:
            print("Error: BEIR Hugging Face model dependencies are required for local_backend=huggingface")
            sys.exit(1)
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
    if local_backend == "automodel":
        declared_images = _restore_corpus_image_paths(
            corpus,
            dataset_path,
            image_root=image_root,
            resolve=model_family == "mistral3_vl",
            strict=True,
        )
        if model_family == "text" and declared_images:
            raise ValueError(
                "Native text evaluation cannot discard corpus image_path fields; select model_family=mistral3_vl"
            )
    return _retrieve_and_evaluate(retriever, corpus, queries, qrels, ignore_identical_ids)


def evaluate_nim(
    nim_url: str,
    nim_model: str,
    dataset_path: Path,
    batch_size: int = 32,
    timeout: int = 60,
    invalid_embedding_retries: int = 3,
    expected_dimension: int | None = None,
    api_backend: Literal["nim", "vllm"] = "nim",
    k_values: list[int] | None = None,
    corpus_chunk_size: int = 50000,
    api_truncate: Literal["NONE", "START", "END"] | None = None,
    ignore_identical_ids: bool = True,
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
        api_backend: Request and health-check conventions used by the server.
        k_values: K values for metrics.
        corpus_chunk_size: Maximum number of documents in one retrieval search chunk.
        api_truncate: Optional Cohere truncation policy. Requires the vLLM backend.
        ignore_identical_ids: Apply BEIR's legacy query/corpus identical-ID exclusion.

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
        api_backend=api_backend,
        dataset_path=dataset_path,
        api_truncate=api_truncate,
    )

    # Wrap in DRES for BEIR compatibility
    dres_model = DRES(
        nim_model_instance,
        corpus_chunk_size=corpus_chunk_size,
        batch_size=batch_size,
    )

    retriever = EvaluateRetrieval(
        dres_model,
        score_function="dot",
        k_values=k_values,
    )

    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split="test")
    _restore_corpus_image_paths(corpus, dataset_path)
    metrics, results = _retrieve_and_evaluate(retriever, corpus, queries, qrels, ignore_identical_ids)

    return metrics, results, nim_model_instance.diagnostics()


def _retrieve_and_evaluate(
    retriever,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    ignore_identical_ids: bool,
) -> tuple[dict, dict]:
    """Retrieve and score while optionally preserving legitimate ID collisions."""
    # BEIR's loader represents an omitted optional title as None. Its dense
    # search sorts by title + text before calling our encoder, so normalize
    # absence at this shared boundary for both local and API evaluation.
    # Copy records to preserve the loaded inputs and all multimodal metadata.
    corpus = {
        document_id: {**document, "title": "" if document.get("title") is None else document["title"]}
        for document_id, document in corpus.items()
    }
    if ignore_identical_ids:
        results = retriever.retrieve(corpus, queries)
        return retriever.evaluate(qrels, results, retriever.k_values), results
    retrieval_queries, aliases = _alias_query_ids(queries, corpus)
    aliased_results = retriever.retrieve(corpus, retrieval_queries)
    results = {aliases[alias]: ranking for alias, ranking in aliased_results.items()}
    metrics = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    return metrics, results


def _alias_query_ids(
    queries: dict[str, str], corpus: dict[str, dict[str, str]]
) -> tuple[dict[str, str], dict[str, str]]:
    """Give every query a temporary ID that cannot collide with a corpus ID."""
    used_ids = set(queries) | set(corpus)
    aliased_queries = {}
    aliases = {}
    for index, (query_id, query) in enumerate(queries.items()):
        alias = f"__nemotron_query_{index}__"
        while alias in used_ids:
            alias = f"_{alias}"
        used_ids.add(alias)
        aliased_queries[alias] = query
        aliases[alias] = query_id
    return aliased_queries, aliases


def _resolve_portable_image_path(image_root: Path, image_path: str) -> Path:
    """Resolve an image strictly within its configured artifact root."""
    relative = Path(image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Corpus image_path must be portable and relative: {image_path!r}")
    root = image_root.resolve(strict=True)
    candidate = root / relative
    if not candidate.exists():
        raise FileNotFoundError(f"Could not resolve corpus image {image_path!r} within {root}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Corpus image_path escapes its configured image root: {image_path!r}") from error
    if not resolved.is_file():
        raise ValueError(f"Corpus image_path is not a regular file: {image_path!r}")
    return resolved


def _restore_corpus_image_paths(
    corpus: dict,
    dataset_path: Path,
    *,
    image_root: Path | None = None,
    resolve: bool = False,
    strict: bool = False,
) -> int:
    """Restore image fields dropped by BEIR and return the declaration count."""
    declared_images = 0
    corpus_path = dataset_path / "corpus.jsonl"
    with corpus_path.open() as file:
        for line in file:
            raw_document = json.loads(line)
            if "image_path" not in raw_document:
                continue
            declared_images += 1
            document_id = raw_document.get("_id")
            image_path = raw_document["image_path"]
            if not isinstance(image_path, str) or not image_path.strip():
                if strict:
                    raise ValueError(f"Corpus document {document_id!r} declares an empty or malformed image_path")
                continue
            if document_id not in corpus:
                if strict:
                    raise ValueError(f"Corpus image document {document_id!r} was not loaded by BEIR")
                continue
            corpus[document_id]["image_path"] = (
                str(_resolve_portable_image_path(image_root or dataset_path, image_path)) if resolve else image_path
            )
    return declared_images


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
    if cfg.sdg_input_path is not None:
        from nemotron.recipes.embed.sdg_manifest import resolve_portable_evaluation_input

        evaluation_dir, image_root = resolve_portable_evaluation_input(
            cfg.sdg_input_path, cfg.retrieval_view, cfg.retrieval_split_protocol
        )
        cfg = cfg.model_copy(update={"eval_data_path": evaluation_dir, "image_root": image_root})

    print("📊 Embedding Model Evaluation")
    print("=" * 60)
    print(f"Eval data:       {cfg.eval_data_path}")
    print(f"Base model:      {cfg.base_model}")
    print(f"Finetuned model: {cfg.finetuned_model_path}")
    if cfg.eval_nim:
        print(f"API backend:     {cfg.embedding_api_backend}")
        print(f"API endpoint:    {cfg.nim_url}")
        print(f"API model:       {cfg.nim_model}")
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
        "retrieval_split_protocol_requested": cfg.retrieval_split_protocol,
        "sdg_input_path": str(cfg.sdg_input_path.resolve()) if cfg.sdg_input_path else None,
        "eval_data_path": str(cfg.eval_data_path.resolve()),
        "image_root": str((cfg.image_root or cfg.eval_data_path).resolve()),
        "ignore_identical_ids": cfg.ignore_identical_ids,
        "k_values": cfg.k_values,
        "base_model": cfg.base_model if cfg.eval_base else None,
        "finetuned_model_path": (str(cfg.finetuned_model_path.resolve()) if cfg.eval_finetuned else None),
        "local_backend": cfg.local_backend,
        "tokenizer_force_default": cfg.tokenizer_force_default,
        "model_family": cfg.model_family,
        "query_max_length": cfg.query_max_length,
        "passage_max_length": cfg.passage_max_length,
        "image_longest_edge": cfg.image_longest_edge,
        "use_text_in_document": cfg.use_text_in_document,
        "nim_url": cfg.nim_url if cfg.eval_nim else None,
        "nim_model": cfg.nim_model if cfg.eval_nim else None,
        "embedding_api_backend": cfg.embedding_api_backend if cfg.eval_nim else None,
    }
    api_result_key = cfg.embedding_api_backend
    api_display_name = "vLLM" if cfg.embedding_api_backend == "vllm" else "NIM"
    api_diagnostics: dict[str, int | str | None] | None = None
    api_rankings: dict | None = None
    base_rankings: dict | None = None
    finetuned_rankings: dict | None = None
    api_metric_comparison: dict | None = None
    drift_failure = False

    # Evaluate base model
    if cfg.eval_base:
        print(f"📈 Evaluating base model: {cfg.base_model}")
        base_metrics, base_rankings = evaluate_model(
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
            local_backend=cfg.local_backend,
            tokenizer_force_default=cfg.tokenizer_force_default,
            model_family=cfg.model_family,
            query_max_length=cfg.query_max_length,
            passage_max_length=cfg.passage_max_length,
            image_longest_edge=cfg.image_longest_edge,
            use_text_in_document=cfg.use_text_in_document,
            image_root=cfg.image_root,
            ignore_identical_ids=cfg.ignore_identical_ids,
        )
        results["base"] = base_metrics
        _print_summary_metrics(base_metrics, cfg.k_values)
        print()
        _release_cuda_memory()

    # Evaluate fine-tuned model
    if cfg.eval_finetuned:
        if not cfg.finetuned_model_path.exists():
            requirement = (
                "required for metric-drift gate" if cfg.fail_on_nim_metric_drift else "requested for evaluation"
            )
            raise FileNotFoundError(
                f"Fine-tuned model {requirement} was not found at {cfg.finetuned_model_path}; "
                "set eval_finetuned=false to skip it explicitly"
            )
        else:
            print(f"📈 Evaluating fine-tuned model: {cfg.finetuned_model_path}")
            ft_metrics, finetuned_rankings = evaluate_model(
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
                local_backend=cfg.local_backend,
                tokenizer_force_default=cfg.tokenizer_force_default,
                model_family=cfg.model_family,
                query_max_length=cfg.query_max_length,
                passage_max_length=cfg.passage_max_length,
                image_longest_edge=cfg.image_longest_edge,
                use_text_in_document=cfg.use_text_in_document,
                image_root=cfg.image_root,
                ignore_identical_ids=cfg.ignore_identical_ids,
            )
            results["finetuned"] = ft_metrics
            _print_summary_metrics(ft_metrics, cfg.k_values)
            print()
            _release_cuda_memory()

    # Evaluate NIM endpoint
    if cfg.eval_nim:
        print(f"📈 Evaluating {cfg.embedding_api_backend} endpoint: {cfg.nim_url}")
        try:
            api_metrics, api_rankings, api_diagnostics = evaluate_nim(
                nim_url=cfg.nim_url,
                nim_model=cfg.nim_model,
                dataset_path=cfg.eval_data_path,
                batch_size=cfg.nim_batch_size,
                timeout=cfg.nim_timeout,
                invalid_embedding_retries=cfg.nim_invalid_embedding_retries,
                expected_dimension=cfg.nim_embedding_dimension,
                api_backend=cfg.embedding_api_backend,
                k_values=cfg.k_values,
                corpus_chunk_size=cfg.corpus_chunk_size,
                api_truncate=cfg.embedding_api_truncate,
                ignore_identical_ids=cfg.ignore_identical_ids,
            )
            results[api_result_key] = api_metrics
            _print_summary_metrics(api_metrics, cfg.k_values)
            print()
        except Exception as error:
            print(f"   Error evaluating embedding API: {error}")
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
    if "finetuned" in results and api_result_key in results:
        print(f"📊 Behavioral metric comparison (Fine-tuned -> {api_display_name})")
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
                api_val = results[api_result_key][idx][key]
                diff = api_val - ft_val
                at_k = int(key.split("@")[1]) if "@" in key else 1
                threshold = cfg.nim_metric_low_k_tolerance if at_k < 5 else cfg.nim_metric_tolerance
                metric_within_tolerance = abs(diff) <= threshold
                within_tolerance = within_tolerance and metric_within_tolerance
                deltas[name][key] = {
                    "checkpoint": ft_val,
                    api_result_key: api_val,
                    "delta": diff,
                    "tolerance": threshold,
                    "within_tolerance": metric_within_tolerance,
                }
                label = "within tolerance" if metric_within_tolerance else "drift"
                print(f"    {key}: {ft_val:.5f} → {api_val:.5f} ({diff:+.5f}) [{label}]")
        print()

        api_metric_comparison = {
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

    metadata[f"{api_result_key}_diagnostics"] = api_diagnostics
    metadata[f"{api_result_key}_metric_comparison"] = api_metric_comparison
    serializable_results["_metadata"] = metadata

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    for rankings, filename in (
        (base_rankings, "base_retrieval_results.json"),
        (finetuned_rankings, "finetuned_retrieval_results.json"),
    ):
        if rankings is not None:
            with (cfg.output_dir / filename).open("w") as file:
                json.dump(rankings, file)
    if api_rankings is not None:
        rankings_file = cfg.output_dir / "retrieval_results.json"
        with rankings_file.open("w") as file:
            json.dump(api_rankings, file)

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
