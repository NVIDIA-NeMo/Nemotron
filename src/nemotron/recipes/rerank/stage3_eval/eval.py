#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "rerank/eval"
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

"""Evaluation script for cross-encoder reranking models.

Evaluates reranking models by first running dense retrieval to get candidates,
then re-ranking with the cross-encoder and measuring nDCG@k improvement.

Supports evaluation of:
- Local HuggingFace models (base and fine-tuned)
- NIM API endpoints

Usage:
    # With default config
    nemotron rerank eval -c default

    # With CLI overrides
    nemotron rerank eval -c default finetuned_model_path=/path/to/model

    # Evaluate NIM endpoint
    nemotron rerank eval -c default eval_nim=true nim_url=http://localhost:8000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

# Must be set before transformers is imported (BEIR imports it at module level)
os.environ.setdefault("HF_HUB_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("TRUST_REMOTE_CODE", "True")

from pydantic import ConfigDict, Field

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class EvalConfig(RecipeSettings):
    """Evaluation configuration for cross-encoder reranking models."""

    model_config = ConfigDict(extra="forbid")

    # Model paths
    base_model: str = Field(default="nvidia/llama-nemotron-rerank-1b-v2", description="Base reranking model for comparison.")
    finetuned_model_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage2_finetune/checkpoints/LATEST/model/consolidated", description="Path to fine-tuned model checkpoint.")

    # First-stage retrieval model (for generating candidates to re-rank)
    retrieval_model: str = Field(default="nvidia/llama-nemotron-embed-1b-v2", description="Dense retrieval model for first-stage candidate generation.")

    # Evaluation data
    eval_data_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage1_prep/eval_beir", description="Path to BEIR-formatted evaluation data.")

    # Output settings
    output_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage3_eval", description="Directory for saving evaluation results.")

    # Evaluation settings
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 100], description="K values for nDCG@k metrics.")
    top_k: int = Field(default=100, gt=0, description="Number of first-stage candidates to re-rank.")
    batch_size: int = Field(default=128, gt=0, description="Batch size for reranker scoring.")
    retrieval_batch_size: int = Field(default=32, gt=0, description="Batch size for first-stage retrieval encoding. Lower than batch_size because the embedding model processes longer sequences.")
    max_length: int = Field(default=512, gt=0, description="Maximum sequence length.")
    corpus_chunk_size: int = Field(default=50000, gt=0, description="Chunk size for corpus encoding.")

    # Retrieval model settings
    retrieval_pooling: Literal["avg", "cls", "last"] = Field(default="avg", description="Pooling strategy for retrieval model.")
    retrieval_normalize: bool = Field(default=True, description="Whether to L2 normalize retrieval embeddings.")
    query_prefix: str = Field(default="query:", description="Prefix for query inputs (retrieval model).")
    passage_prefix: str = Field(default="passage:", description="Prefix for passage inputs (retrieval model).")

    # Evaluation mode
    eval_base: bool = Field(default=True, description="Whether to evaluate the base reranker.")
    eval_finetuned: bool = Field(default=True, description="Whether to evaluate the fine-tuned reranker.")

    # Reranker prompt template (must match training config)
    prompt_template: str = Field(default="question:{query} \n \n passage:{passage}", description="Template for formatting query-passage pairs. Must match the template used during training.")

    # NIM API evaluation settings
    eval_nim: bool = Field(default=False, description="Whether to evaluate a NIM API endpoint.")
    nim_url: str = Field(default="http://localhost:8000", description="NIM API base URL.")
    nim_model: str = Field(default="nvidia/llama-nemotron-rerank-1b-v2", description="Model name for NIM API requests.")
    nim_batch_size: int = Field(default=32, gt=0, description="Batch size for NIM API requests.")
    nim_timeout: int = Field(default=60, gt=0, description="Timeout in seconds for NIM API requests.")


class _SentenceTransformerRetriever:
    """BEIR-compatible retriever using SentenceTransformer.

    Wraps SentenceTransformer to provide encode_queries() and encode_corpus()
    methods expected by BEIR's DenseRetrievalExactSearch. Unlike BEIR's
    built-in models.HuggingFace, SentenceTransformer correctly passes
    trust_remote_code=True when loading models with custom code.
    """

    def __init__(
        self,
        model_path: str,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        self._pool = None
        if torch.cuda.device_count() > 1:
            print(f"   Starting multi-GPU pool ({torch.cuda.device_count()} GPUs) for retrieval model")
            self._pool = self.model.start_multi_process_pool()

    def encode_queries(self, queries: list[str], batch_size: int = 128, **kwargs):
        prompts = [f"{self.query_prefix} {q}" for q in queries]
        if self._pool:
            return self.model.encode_multi_process(prompts, self._pool, batch_size=batch_size)
        return self.model.encode(prompts, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int = 128, **kwargs):
        texts = []
        for doc in corpus:
            title = doc.get("title", "")
            text = doc.get("text", "")
            texts.append(f"{self.passage_prefix} {title} {text}".strip())
        if self._pool:
            return self.model.encode_multi_process(texts, self._pool, batch_size=batch_size)
        return self.model.encode(texts, batch_size=batch_size, **kwargs)


def _get_first_stage_results(
    retrieval_model: str,
    dataset_path: Path,
    batch_size: int = 32,
    corpus_chunk_size: int = 50000,
    k_values: list[int] | None = None,
    query_prefix: str = "query:",
    passage_prefix: str = "passage:",
) -> tuple[dict, dict, dict, dict]:
    """Run first-stage dense retrieval to get candidates for re-ranking.

    Returns:
        Tuple of (corpus, queries, qrels, first_stage_results).
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

    dense_model = _SentenceTransformerRetriever(
        model_path=str(retrieval_model),
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
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

    return corpus, queries, qrels, results


def evaluate_reranker(
    model_path: str | Path,
    corpus: dict,
    queries: dict,
    qrels: dict,
    first_stage_results: dict,
    top_k: int = 100,
    batch_size: int = 128,
    max_length: int = 512,
    k_values: list[int] | None = None,
    prompt_template: str | None = None,
) -> tuple[dict, dict]:
    """Evaluate a cross-encoder reranker on first-stage retrieval results.

    Args:
        model_path: Path to the cross-encoder model.
        corpus: BEIR corpus dict.
        queries: BEIR queries dict.
        qrels: BEIR relevance judgments.
        first_stage_results: First-stage retrieval results to re-rank.
        top_k: Number of candidates to re-rank per query.
        batch_size: Batch size for cross-encoder scoring.
        max_length: Maximum sequence length.
        k_values: K values for metrics.
        prompt_template: Template for formatting query-passage pairs (e.g.
            "question:{query} \\n \\n passage:{passage}"). When set, inputs
            are formatted with this template to match training. When None,
            falls back to BEIR's default pair formatting.

    Returns:
        Tuple of (metrics dict, reranked results dict).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError:
        print("Error: BEIR is required for evaluation.")
        print("  Install with: pip install beir")
        sys.exit(1)

    if k_values is None:
        k_values = [1, 5, 10, 100]

    # Load model and tokenizer directly to control input formatting
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"   Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Build inputs
    formatted_inputs = []
    pair_info = []
    for qid in queries:
        if qid not in first_stage_results:
            continue
        sorted_candidates = sorted(
            first_stage_results[qid].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        for doc_id, _ in sorted_candidates:
            pair_info.append((qid, doc_id))
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            passage = f"{title} {text}".strip() if title else text
            if prompt_template:
                formatted_inputs.append(
                    prompt_template.format(query=queries[qid], passage=passage)
                )
            else:
                formatted_inputs.append((queries[qid], passage))

    # Score in batches
    all_scores = []
    for batch_start in range(0, len(formatted_inputs), batch_size):
        batch = formatted_inputs[batch_start:batch_start + batch_size]
        if prompt_template:
            features = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
        else:
            texts_a = [pair[0] for pair in batch]
            texts_b = [pair[1] for pair in batch]
            features = tokenizer(
                texts_a, texts_b, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
        features = {k: v.to(device) for k, v in features.items()}
        with torch.no_grad():
            logits = model(**features).logits.squeeze(-1)
        if logits.dim() == 0:
            all_scores.append(logits.item())
        else:
            all_scores.extend(logits.cpu().tolist())

    reranked_results: dict[str, dict[str, float]] = {}
    for (qid, doc_id), score in zip(pair_info, all_scores):
        if qid not in reranked_results:
            reranked_results[qid] = {}
        reranked_results[qid][doc_id] = float(score)

    # Evaluate re-ranked results
    evaluator = EvaluateRetrieval(k_values=k_values)
    metrics = evaluator.evaluate(qrels, reranked_results, k_values)

    return metrics, reranked_results


def evaluate_nim_reranker(
    nim_url: str,
    nim_model: str,
    corpus: dict,
    queries: dict,
    qrels: dict,
    first_stage_results: dict,
    top_k: int = 100,
    batch_size: int = 32,
    timeout: int = 60,
    k_values: list[int] | None = None,
) -> tuple[dict, dict]:
    """Evaluate a NIM reranker endpoint on first-stage retrieval results.

    Args:
        nim_url: Base URL for NIM API.
        nim_model: Model name for API requests.
        corpus: BEIR corpus dict.
        queries: BEIR queries dict.
        qrels: BEIR relevance judgments.
        first_stage_results: First-stage retrieval results to re-rank.
        top_k: Number of candidates to re-rank per query.
        batch_size: Batch size for API requests.
        timeout: Request timeout in seconds.
        k_values: K values for metrics.

    Returns:
        Tuple of (metrics dict, reranked results dict).
    """
    import urllib.request
    import urllib.error

    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError:
        print("Error: BEIR is required for evaluation.")
        sys.exit(1)

    if k_values is None:
        k_values = [1, 5, 10, 100]

    ranking_url = f"{nim_url.rstrip('/')}/v1/ranking"
    reranked_results: dict[str, dict[str, float]] = {}

    query_ids = list(queries.keys())
    for i, qid in enumerate(query_ids):
        query_text = queries[qid]

        # Get top-k candidates from first-stage results
        candidate_ids = sorted(
            first_stage_results.get(qid, {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        if not candidate_ids:
            reranked_results[qid] = {}
            continue

        # Build passages list
        doc_ids = [did for did, _ in candidate_ids]
        passages = []
        for did in doc_ids:
            doc = corpus[did]
            title = doc.get("title", "")
            text = doc.get("text", "")
            passages.append({"text": f"{title} {text}".strip() if title else text})

        # Score in batches
        all_scores = []
        for batch_start in range(0, len(passages), batch_size):
            batch_passages = passages[batch_start:batch_start + batch_size]

            payload = json.dumps({
                "model": nim_model,
                "query": {"text": query_text},
                "passages": batch_passages,
            }).encode("utf-8")

            req = urllib.request.Request(
                ranking_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    rankings = sorted(result["rankings"], key=lambda x: x["index"])
                    all_scores.extend([r["logit"] for r in rankings])
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                raise RuntimeError(f"NIM API error {e.code}: {error_body}") from e

        reranked_results[qid] = {did: score for did, score in zip(doc_ids, all_scores)}

        if (i + 1) % 50 == 0:
            print(f"   Re-ranked {i + 1}/{len(query_ids)} queries...")

    # Evaluate re-ranked results
    evaluator = EvaluateRetrieval(k_values=k_values)
    metrics = evaluator.evaluate(qrels, reranked_results, k_values)

    return metrics, reranked_results


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
    """Run cross-encoder reranking model evaluation.

    Args:
        cfg: Evaluation configuration.

    Returns:
        Dictionary with evaluation results.
    """
    print(f"Reranking Model Evaluation")
    print(f"=" * 60)
    print(f"Eval data:         {cfg.eval_data_path}")
    print(f"Retrieval model:   {cfg.retrieval_model}")
    print(f"Base reranker:     {cfg.base_model}")
    print(f"Finetuned reranker:{cfg.finetuned_model_path}")
    print(f"Top-k to re-rank:  {cfg.top_k}")
    print(f"K values:          {cfg.k_values}")
    print(f"=" * 60)
    print()

    # Validate inputs
    if not cfg.eval_data_path.exists():
        print(f"Error: Eval data path not found: {cfg.eval_data_path}", file=sys.stderr)
        print("       Please run 'nemotron embed prep' first or provide eval data.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Step 1: Run first-stage dense retrieval
    print(f"Running first-stage retrieval with: {cfg.retrieval_model}")
    corpus, queries, qrels, first_stage_results = _get_first_stage_results(
        retrieval_model=cfg.retrieval_model,
        dataset_path=cfg.eval_data_path,
        batch_size=cfg.retrieval_batch_size,
        corpus_chunk_size=cfg.corpus_chunk_size,
        k_values=cfg.k_values,
        query_prefix=cfg.query_prefix,
        passage_prefix=cfg.passage_prefix,
    )
    print(f"   Retrieved candidates for {len(queries)} queries")
    print()

    # Step 2: Evaluate base reranker
    if cfg.eval_base:
        print(f"Evaluating base reranker: {cfg.base_model}")
        base_metrics, _ = evaluate_reranker(
            model_path=cfg.base_model,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            first_stage_results=first_stage_results,
            top_k=cfg.top_k,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            k_values=cfg.k_values,
            prompt_template=cfg.prompt_template,
        )
        results["base"] = base_metrics
        _print_summary_metrics(base_metrics, cfg.k_values)
        print()

    # Step 3: Evaluate fine-tuned reranker
    if cfg.eval_finetuned:
        if not cfg.finetuned_model_path.exists():
            print(f"Warning: Fine-tuned model not found at {cfg.finetuned_model_path}")
            print("         Skipping fine-tuned model evaluation.")
        else:
            print(f"Evaluating fine-tuned reranker: {cfg.finetuned_model_path}")
            ft_metrics, _ = evaluate_reranker(
                model_path=cfg.finetuned_model_path,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                first_stage_results=first_stage_results,
                top_k=cfg.top_k,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
                k_values=cfg.k_values,
                prompt_template=cfg.prompt_template,
            )
            results["finetuned"] = ft_metrics
            _print_summary_metrics(ft_metrics, cfg.k_values)
            print()

    # Step 4: Evaluate NIM reranker endpoint
    if cfg.eval_nim:
        print(f"Evaluating NIM reranker endpoint: {cfg.nim_url}")

        import urllib.request
        import urllib.error

        nim_healthy = False
        try:
            health_url = f"{cfg.nim_url.rstrip('/')}/v1/health/ready"
            with urllib.request.urlopen(health_url, timeout=10) as response:
                nim_healthy = response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        if not nim_healthy:
            print(f"   Error: NIM endpoint is not reachable at {cfg.nim_url}", file=sys.stderr)
            print(f"   Ensure the NIM service is running and healthy before evaluating.", file=sys.stderr)
            print()
        else:
            try:
                nim_metrics, _ = evaluate_nim_reranker(
                    nim_url=cfg.nim_url,
                    nim_model=cfg.nim_model,
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    first_stage_results=first_stage_results,
                    top_k=cfg.top_k,
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
        print(f"Comparison (Base -> Fine-tuned)")
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
                pct = (diff / base_val * 100) if base_val != 0 else float("inf")
                print(f"    {k}: {base_val:.5f} -> {ft_val:.5f} ({sign}{diff:.5f}, {sign}{pct:.1f}%)")
        print()

    # Print NIM vs Fine-tuned comparison (accuracy check for export)
    if "finetuned" in results and "nim" in results:
        print(f"Comparison (Fine-tuned -> NIM)")
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
                at_k = int(k.split("@")[1]) if "@" in k else 1
                threshold = 0.03 if at_k < 5 else 0.01
                status = "ok" if abs(diff) < threshold else "MISMATCH"
                pct = (diff / ft_val * 100) if ft_val != 0 else float("inf")
                print(f"    {k}: {ft_val:.5f} -> {nim_val:.5f} ({sign}{diff:.5f}, {sign}{pct:.1f}%) {status}")
        print()

    # Save results
    results_file = cfg.output_dir / "eval_results.json"

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

    print(f"Evaluation complete!")
    print(f"   Results saved to: {results_file}")

    # Save artifact
    try:
        from nemotron.kit.artifacts.base import Artifact

        artifact = Artifact(path=cfg.output_dir)
        artifact.save(name="rerank/eval")
    except Exception:
        pass

    return results


def main(cfg: EvalConfig | None = None) -> dict:
    """Entry point for evaluation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Dictionary with evaluation results.
    """
    if cfg is None:
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
