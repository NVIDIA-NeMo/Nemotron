# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-native and served scoring for Nemotron 3.5 rerank evaluation."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _document_text(document: dict, use_text_in_document: bool) -> str:
    if document.get("image_path") and not use_text_in_document:
        return ""
    return " ".join(str(document.get(key) or "") for key in ("title", "text")).strip()


class AutoModelReranker:
    """Use the training processor and saved learned scoring head without rescaling."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_length: int = 4096,
        image_longest_edge: int | None = None,
        use_text_in_document: bool = True,
        revision: str | None = None,
    ) -> None:
        import torch
        from nemo_automodel import NeMoAutoModelCrossEncoder
        from nemo_automodel.components.models.ministral_bidirectional.processor import Mistral3BiEncoderProcessor

        options: dict[str, Any] = {
            "attn_implementation": "sdpa",
            "use_liger_kernel": False,
            "use_sdpa_patching": True,
            "torch_dtype": torch.bfloat16,
            "key_mapping": {r"^language_model\.score\.weight$": "score.weight"},
        }
        if revision is not None and not Path(model_path).exists():
            options["revision"] = revision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeMoAutoModelCrossEncoder.from_pretrained(str(model_path), **options).to(self.device).eval()
        if self.model.is_causal or self.model.config.pooling != "avg":
            raise ValueError("Nemotron 3.5 rerank evaluation requires saved bidirectional mean pooling")
        processor_options = {"rerank_max_length": max_length, "use_prompt_template": True, "pad_to_multiple_of": 8}
        if image_longest_edge is not None:
            processor_options["image_longest_edge"] = image_longest_edge
        resolved_revision = getattr(self.model.config, "_commit_hash", None) or options.get("revision")
        if resolved_revision is not None and not Path(model_path).exists():
            processor_options["revision"] = resolved_revision
        self.processor = Mistral3BiEncoderProcessor.from_pretrained(str(model_path), **processor_options)
        self.use_text_in_document = use_text_in_document

    def score_pairs(self, pairs: list[tuple[str, dict]], *, batch_size: int = 1) -> list[float]:
        import torch
        from tqdm import tqdm

        scores = []
        for start in tqdm(
            range(0, len(pairs), batch_size), desc="Reranking (AutoModel)", unit="batch", mininterval=1.0
        ):
            features = [
                {
                    "question": query,
                    "doc_text": _document_text(doc, self.use_text_in_document),
                    "doc_image": doc.get("image_path") or "",
                }
                for query, doc in pairs[start : start + batch_size]
            ]
            inputs = self.processor.process_queries_documents_crossencoder(features, return_tensors="pt", padding=True)
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            if inputs.get("pixel_values") is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.model.dtype)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            if logits.shape != (len(features), 1):
                raise ValueError(f"Expected one rerank logit per pair, received shape {tuple(logits.shape)}")
            values = logits.detach().float().cpu().flatten().tolist()
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Reranker returned a non-finite logit")
            scores.extend(values)
        return scores


class VLLMReranker:
    """Read raw classifier logits from vLLM's multimodal /classify endpoint."""

    def __init__(self, url: str, model: str, *, timeout: int = 120, use_text_in_document: bool = True) -> None:
        self.url = f"{url.rstrip('/')}/classify"
        self.model = model
        self.timeout = timeout
        self.use_text_in_document = use_text_in_document

    def score_pairs(self, pairs: list[tuple[str, dict]], *, batch_size: int = 1) -> list[float]:
        from tqdm import tqdm

        scores = []
        # /classify accepts one structured conversation per request. Preserve pair order.
        for query, document in tqdm(pairs, desc="Reranking (vLLM)", unit="pair", mininterval=1.0):
            content = []
            text = _document_text(document, self.use_text_in_document)
            if text:
                content.append({"type": "text", "text": text})
            if document.get("image_path"):
                image = Path(document["image_path"])
                mime = mimetypes.guess_type(image.name)[0] or "image/png"
                encoded = base64.b64encode(image.read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            if not content:
                raise ValueError("Rerank documents must contain text or an image")
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "query", "content": [{"type": "text", "text": query}]},
                    {"role": "document", "content": content},
                ],
                "add_special_tokens": False,
                "use_activation": False,
            }
            request = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"vLLM classification failed ({error.code}): {error.read().decode()}") from error
            data = result.get("data", [])
            values = data[0].get("probs", []) if len(data) == 1 else []
            score = values[0] if len(values) == 1 else None
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("vLLM must return exactly one finite raw classification score per pair")
            scores.append(float(score))
        return scores


def evaluate_multimodal_reranker(
    scorer: AutoModelReranker | VLLMReranker,
    *,
    corpus: dict,
    queries: dict,
    qrels: dict,
    first_stage_results: dict,
    top_k: int,
    batch_size: int,
    k_values: list[int],
) -> tuple[tuple, dict]:
    """Score the same candidate set locally or through the deployed classifier."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    pair_ids, pairs = [], []
    for query_id, query in queries.items():
        candidates = sorted(first_stage_results.get(query_id, {}).items(), key=lambda item: item[1], reverse=True)
        for document_id, _ in candidates[:top_k]:
            pair_ids.append((query_id, document_id))
            pairs.append((query, corpus[document_id]))
    scores = scorer.score_pairs(pairs, batch_size=batch_size)
    if len(scores) != len(pair_ids):
        raise ValueError("Reranker must return one score per query-document pair")
    results = {query_id: {} for query_id in queries}
    for (query_id, document_id), score in zip(pair_ids, scores, strict=True):
        results[query_id][document_id] = score
    metrics = EvaluateRetrieval(k_values=k_values).evaluate(qrels, results, k_values, ignore_identical_ids=False)
    return metrics, results
