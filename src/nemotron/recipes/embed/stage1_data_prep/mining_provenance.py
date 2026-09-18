# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve source annotations across the AutoModel mining serializer."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def index_queries(rows: list[dict]) -> dict[str, dict]:
    """Index unique source query IDs, rejecting ambiguous joins."""
    result = {}
    for row in rows:
        identifier = row.get("question_id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("Mining provenance requires unique nonempty question IDs")
        result[identifier] = row
    return result


def positive_ids(row: dict) -> set[str]:
    """Validate a query's complete, nonduplicated positive-page identity."""
    values = [doc["id"] for doc in row["pos_doc"]]
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Mining provenance requires positive document IDs")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate positive document IDs")
    return set(values)


def restore_mining_provenance(source_path: Path, mined_path: Path) -> None:
    """Restore source fields only after an exact ID/text/corpus/positive join.

    Mined similarities remain ``pos_doc.score``. An original graded relevance
    score is retained separately as ``relevance_grade``; all other source
    annotations, including query families and localized supports, survive.
    Invalid output remains untouched for diagnosis. This is called on rank zero
    after native mining, before Stage 1 can mark it successful or unroll it.
    """
    source = json.loads(source_path.read_text())
    mined = json.loads(mined_path.read_text())
    original = index_queries(source["data"])
    output = index_queries(mined["data"])
    if original.keys() != output.keys():
        raise ValueError("Mining changed the query ID set")
    restored = []
    for identifier, row in output.items():
        seed = original[identifier]
        if row.get("question") != seed.get("question") or row.get("corpus_id") != seed.get("corpus_id"):
            raise ValueError(f"Mining changed query text or corpus identity: {identifier}")
        positives = positive_ids(seed)
        if positive_ids(row) != positives:
            raise ValueError(f"Mining changed positive pages: {identifier}")
        if positives.intersection(doc["id"] for doc in row.get("neg_doc", [])):
            raise ValueError(f"Mining negative collides with a positive: {identifier}")
        # The source owns annotations, the miner owns only scored retrieval data.
        merged = {**row, **seed, "neg_doc": row.get("neg_doc", [])}
        merged["original_question_id"] = seed.get("original_question_id", identifier)
        merged["all_pos_doc_ids"] = sorted(positives | set(seed.get("all_pos_doc_ids", [])))
        if set(merged["all_pos_doc_ids"]).intersection(doc["id"] for doc in merged["neg_doc"]):
            raise ValueError(f"Mining negative collides with an unrolled source positive: {identifier}")
        source_docs = {doc["id"]: doc for doc in seed["pos_doc"]}
        merged["pos_doc"] = []
        for doc in row["pos_doc"]:
            annotation = source_docs[doc["id"]]
            enriched = {**annotation, **doc}
            if "score" in annotation:
                enriched["relevance_grade"] = annotation.get("relevance_grade", annotation["score"])
            merged["pos_doc"].append(enriched)
        for legacy in ("pos_score", "neg_scores", "negative_scores"):
            merged.pop(legacy, None)
        restored.append(merged)
    mined["data"] = restored
    mined.setdefault("mining", {})["provenance_contract"] = "exact-query-and-positive-join-v1"
    with tempfile.NamedTemporaryFile(mode="w", dir=mined_path.parent, prefix=".provenance-", delete=False) as stream:
        json.dump(mined, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    os.replace(temporary, mined_path)
