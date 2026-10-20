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

"""Optional endpoint-based hard-negative mining for embedding recipes."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Literal


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def mine_vllm_hard_negatives(
    train_data_path: Path,
    output_path: Path,
    *,
    api_url: str,
    model: str,
    batch_size: int,
    hard_negatives_to_mine: int,
    hard_neg_margin: float,
    expected_dimension: int | None = None,
    use_images: bool = True,
    use_text_in_document: bool = True,
    api_truncate: Literal["NONE", "START", "END"] | None = None,
) -> Path:
    """Mine scored VL negatives from a DataDesigner Parquet corpus via vLLM."""
    import numpy as np
    import pyarrow.parquet as pq

    from nemotron.recipes.embed.stage3_eval.eval import NIMEmbeddingModel

    payload = _load_json(train_data_path)
    corpus = payload.get("corpus")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("path"), str) or not corpus["path"]:
        raise ValueError("Training data corpus.path must be a nonempty string")
    # Prepared training files can reference a sibling corpus or an absolute path.
    # Portable bundle containment is checked when resolving the Stage 0 manifest.
    corpus_path = (train_data_path.parent / corpus["path"]).resolve()
    parquet_files = sorted(corpus_path.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet corpus shards found under {corpus_path}")

    available = set(pq.read_schema(parquet_files[0]).names)
    id_columns = available.intersection({"docid", "id", "image_filename"})
    if len(id_columns) != 1:
        raise ValueError("Corpus must have one supported document ID column")
    id_column = id_columns.pop()
    has_images = use_images and "image" in available
    if "text" not in available and not has_images:
        raise ValueError("Corpus has no usable text or enabled images")
    columns = [id_column, *(["text"] if "text" in available else []), *(["image"] if has_images else [])]
    table = pq.read_table(parquet_files, columns=columns)
    doc_ids = table[id_column].to_pylist()
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Corpus contains duplicate document IDs")
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("Training data must contain a non-empty data list")

    client = NIMEmbeddingModel(
        api_url=api_url,
        model=model,
        batch_size=batch_size,
        timeout=600,
        expected_dimension=expected_dimension,
        api_backend="vllm",
        api_truncate=api_truncate,
    )
    query_embeddings = client.encode_queries([str(record.get("question", "")) for record in records])

    document_inputs: list[str | dict[str, Any]] = []
    texts = table["text"].to_pylist() if "text" in available else [""] * len(doc_ids)
    images = table["image"].to_pylist() if has_images else [None] * len(doc_ids)
    for text, image in zip(texts, images, strict=True):
        document_text = str(text or "").strip()
        if image:
            raw_image = image.get("bytes") if isinstance(image, dict) else image
            encoded = base64.b64encode(bytes(raw_image)).decode("ascii")
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            ]
            if document_text and use_text_in_document:
                content.append({"type": "text", "text": document_text})
            document_inputs.append({"content": content})
        else:
            document_inputs.append(document_text)

    document_batches = []
    for start in range(0, len(document_inputs), batch_size):
        batch = document_inputs[start : start + batch_size]
        document_batches.extend(client._encode_batch(batch, input_type="passage"))
    document_embeddings = np.asarray(document_batches, dtype=np.float32)
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        query_dimension = query_embeddings.shape[1]
        document_dimension = document_embeddings.shape[1]
        raise ValueError(f"Query dimension {query_dimension} does not match corpus dimension {document_dimension}")

    scores = query_embeddings @ document_embeddings.T
    for row_index, record in enumerate(records):
        positives = record.get("pos_doc")
        if not isinstance(positives, list) or not positives:
            raise ValueError(f"Training record {row_index} has no positive document")
        try:
            positive_indices = [doc_index[positive["id"]] for positive in positives]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Training record {row_index} references an unknown positive") from error
        positive_scores = [float(scores[row_index, index]) for index in positive_indices]
        for positive, score in zip(positives, positive_scores, strict=True):
            if "score" in positive:
                positive.setdefault("source_score", positive["score"])
            positive["score"] = score

        candidate_scores = scores[row_index].copy()
        candidate_scores[positive_indices] = -np.inf
        candidate_scores[candidate_scores > min(positive_scores) * hard_neg_margin] = -np.inf
        candidate_indices = np.argsort(-candidate_scores, kind="stable")
        selected = [index for index in candidate_indices if np.isfinite(candidate_scores[index])][
            :hard_negatives_to_mine
        ]
        if len(selected) < hard_negatives_to_mine:
            raise ValueError(f"Training record {row_index} yielded only {len(selected)} finite hard negatives")
        record["neg_doc"] = [{"id": doc_ids[index], "score": float(candidate_scores[index])} for index in selected]
        record["negative_scores"] = [float(candidate_scores[index]) for index in selected]
        record["negative_mining_performed"] = True

    payload["corpus"]["path"] = str(corpus_path.resolve())
    payload["mining"] = {
        "backend": "vllm",
        "model": model,
        "api_url": api_url,
        "batch_size": batch_size,
        "hard_negatives_to_mine": hard_negatives_to_mine,
        "hard_neg_margin": hard_neg_margin,
        "uses_document_images": has_images,
        "use_text_in_document": use_text_in_document,
        "embedding_dimension": int(document_embeddings.shape[1]),
        "relevance_semantics": "retrieval-mined candidates remain unjudged",
    }
    if api_truncate is not None:
        payload["mining"]["api_truncate"] = api_truncate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path
