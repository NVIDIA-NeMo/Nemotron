# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain source annotations across AutoModel's normalized mining dataset."""

from __future__ import annotations

import json
from pathlib import Path


def restore_mining_provenance(source: dict, output_file: Path) -> None:
    """Join mined scores to original queries without losing their provenance.

    AutoModel loads only the five training fields and writes those normalized
    rows. Restore query families, source references, language, and partial-label
    semantics from the original records. Keep original positive
    scores separately from the miner's similarity scores. The caller snapshots
    the source before writing output so in-place re-mining remains supported.
    """
    output = json.loads(output_file.read_text())

    def index_rows(rows: list[dict]) -> dict[str | int, dict]:
        indexed = {}
        for row in rows:
            identifier = row.get("question_id")
            if type(identifier) not in (str, int) or identifier == "" or identifier in indexed:
                raise ValueError("Mining requires unique nonempty string or integer question identifiers")
            indexed[identifier] = row
        return indexed

    originals = index_rows(source["data"])
    mined = index_rows(output["data"])
    if originals.keys() != mined.keys():
        raise ValueError("Mined query identifiers differ from the input queries")
    restored = []
    for identifier, record in originals.items():
        result = mined[identifier]
        if any(result.get(key) != record.get(key) for key in ("question", "corpus_id")):
            raise ValueError(f"Mined query identity differs from the input: {identifier}")
        # AutoModel normalizes document identifiers to strings when loading.
        source_positives = [doc if isinstance(doc, dict) else {"id": doc} for doc in record["pos_doc"]]
        positives = {str(doc["id"]): doc for doc in source_positives}
        result_positives = {str(doc["id"]): doc for doc in result["pos_doc"]}
        if (
            positives.keys() != result_positives.keys()
            or len(positives) != len(record["pos_doc"])
            or len(result_positives) != len(result["pos_doc"])
        ):
            raise ValueError(f"Mined positives differ from the input: {identifier}")
        row = {**record, "neg_doc": result["neg_doc"], "pos_doc": []}
        # Match AutoModel's removal of obsolete per-row score arrays.
        row.pop("pos_score", None)
        row.pop("neg_scores", None)
        for doc_id, positive in positives.items():
            annotated = {**positive, **result_positives[doc_id]}
            if "score" in positive:
                annotated.setdefault("source_score", positive["score"])
            row["pos_doc"].append(annotated)
        if "negative_scores" in row:
            row["negative_scores"] = [doc["score"] for doc in row["neg_doc"]]
        if "negative_mining_performed" in row:
            row["negative_mining_performed"] = True
        restored.append(row)
    output["data"] = restored
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output_file)
