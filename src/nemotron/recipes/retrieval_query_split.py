# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate declared query-family separation over a shared retrieval collection.

Families must be constructed by the producer before splitting (including
translations, paraphrases and other known derivations). This validator checks
their assignments and exact normalized duplicates, not semantic equivalence.
"""

from __future__ import annotations

import re
import unicodedata

from nemotron.recipes.retrieval_vl import RetrievalVLBundleError, VLBundlePaths, _load_json


def validate_grouped_query_bundle(paths: VLBundlePaths, units: dict, corpus: set[str], view: str) -> None:
    """Reject family leakage, incomplete corpora and contradictory provenance."""
    from nemotron.recipes.retrieval_bundle_v2 import _jsonl

    manifest = _load_json(paths.root / "split_manifest.json")
    required = {"split_protocol": "grouped_query_disjoint", "corpus_scope": "full_collection"}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RetrievalVLBundleError("Split manifest contradicts grouped query protocol")
    if manifest.get("document_disjoint") is not False or manifest.get("assignments"):
        raise RetrievalVLBundleError("Grouped query protocol must not claim document-disjoint assignments")
    # A view may exclude units with an unavailable modality, but never omit an
    # eligible page merely because it has no training or held-out positive.
    eligible = {
        key
        for key, unit in units.items()
        if (bool(unit["text"].strip()) if view == "text" else len(unit["images"]) == 1)
    }
    if corpus != eligible:
        raise RetrievalVLBundleError("Shared corpus must contain every view-eligible retrieval unit")
    by_view = manifest.get("query_assignments")
    if not isinstance(by_view, dict):
        raise RetrievalVLBundleError("Missing per-view query_assignments")
    assignments = by_view.get(view)
    if not isinstance(assignments, dict) or not assignments:
        raise RetrievalVLBundleError("Missing per-view query_assignments")
    partitions = {
        "train": _load_json(paths.train)["data"],
        "validation": _load_json(paths.validation)["data"],
        "evaluation": list(_jsonl(paths.synthetic_eval / "queries.jsonl", "_id").values()),
    }
    seen: set[str] = set()
    families: dict[str, str] = {}
    texts: dict[str, str] = {}
    for split, rows in partitions.items():
        for row in rows:
            identifier = row.get("_id") if split == "evaluation" else row.get("question_id")
            group = row.get("query_group_id")
            text = row.get("text") if split == "evaluation" else row.get("question")
            if not isinstance(group, str) or not group.strip():
                raise RetrievalVLBundleError(f"Missing query_group_id: {identifier}")
            if not isinstance(text, str) or not text.strip():
                raise RetrievalVLBundleError(f"Missing query text: {identifier}")
            if assignments.get(identifier) != {"split": split, "query_group_id": group}:
                raise RetrievalVLBundleError(f"Query assignment mismatch: {identifier}")
            if identifier in seen:
                raise RetrievalVLBundleError(f"Duplicate query identifier: {identifier}")
            seen.add(identifier)
            normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
            if not normalized:
                raise RetrievalVLBundleError(f"Query must contain words or numbers: {identifier}")
            for key, mapping in ((group, families), (normalized, texts)):
                if key in mapping and mapping[key] != split:
                    raise RetrievalVLBundleError(f"Query family or normalized text crosses partitions: {identifier}")
                mapping[key] = split
    if seen != set(assignments):
        raise RetrievalVLBundleError("Query assignments differ from exported view queries")
