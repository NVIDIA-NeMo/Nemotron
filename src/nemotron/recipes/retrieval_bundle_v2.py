# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Consumer validation for the unified retrieval-SDG schema-v2 export.

The producer does not emit legacy consumer attestations. This adapter checks
local completeness and integrity directly without modifying its manifest.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from nemotron.recipes.retrieval_vl import (
    RetrievalVLBundleError,
    ViewName,
    VLBundlePaths,
    _contained_path,
    _load_json,
    _sha256,
)


def _jsonl(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    """Read indexed objects, rejecting duplicate or missing identifiers."""
    records = {}
    try:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                identifier = row.get(id_field) if isinstance(row, dict) else None
                if not isinstance(identifier, str) or not identifier:
                    raise RetrievalVLBundleError(f"Missing {id_field} in {path}")
                if identifier in records:
                    raise RetrievalVLBundleError(f"Duplicate {id_field} in {path}: {identifier}")
                records[identifier] = row
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalVLBundleError(f"Invalid JSONL {path}: {error}") from error
    return records


def _inventory(root: Path, manifest: dict[str, Any], verify: bool) -> dict[str, dict[str, Any]]:
    """Verify the producer's complete file inventory, including image assets."""
    records = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RetrievalVLBundleError("Bundle manifest must contain an artifacts list")
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RetrievalVLBundleError("Unified artifacts must have a path")
        relative = record["path"]
        if Path(relative).is_absolute() or Path(relative).as_posix() != relative:
            raise RetrievalVLBundleError(f"Invalid portable artifact path: {relative!r}")
        path = _contained_path(root, relative)
        if relative in records:
            raise RetrievalVLBundleError(f"Duplicate artifact path: {relative}")
        checksum = record.get("sha256")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise RetrievalVLBundleError(f"Invalid declared size for {relative}")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise RetrievalVLBundleError(f"Invalid declared checksum for {relative}")
        records[relative] = record
        if not verify:
            continue
        if not path.is_file():
            raise RetrievalVLBundleError(f"Bundle missing artifact: {relative}")
        if path.stat().st_size != record["bytes"]:
            raise RetrievalVLBundleError(f"Size mismatch for {relative}")
        if _sha256(path) != record.get("sha256"):
            raise RetrievalVLBundleError(f"Checksum mismatch for {relative}")
    return records


def _asset(root: Path, value: str, artifacts: dict) -> Path:
    """Resolve only explicitly inventoried portable assets."""
    path = _contained_path(root, value)
    if value not in artifacts:
        raise RetrievalVLBundleError(f"Unlisted image asset: {value}")
    return path


def _units(root: Path, artifacts: dict, *, shared: bool = False) -> dict[str, dict[str, Any]]:
    """Validate source-document assignments and source asset references."""
    units = _jsonl(root / "retrieval_units.jsonl", "unit_id")
    assignments = _load_json(root / "split_manifest.json").get("assignments")
    if not shared and not isinstance(assignments, dict):
        raise RetrievalVLBundleError("Split manifest requires document assignments")
    for unit in units.values():
        split = unit.get("split")
        if shared:
            valid_split = split == "shared" and isinstance(unit.get("document_id"), str) and bool(unit["document_id"])
        else:
            valid_split = (
                split in {"train", "validation", "evaluation"} and assignments.get(unit.get("document_id")) == split
            )
        if not valid_split:
            raise RetrievalVLBundleError(f"Source document split mismatch for {unit['unit_id']}")
        images = unit.get("images")
        if not isinstance(images, list) or not isinstance(unit.get("text"), str):
            raise RetrievalVLBundleError(f"Invalid source unit: {unit['unit_id']}")
        for value in images:
            _asset(root, value, artifacts)
    return units


def _corpus(root: Path, directory: Path, view: str, split: str, units: dict, artifacts: dict) -> set[str]:
    """Validate Parquet IDs and preserve authoritative source text/images."""
    import pyarrow.parquet as pq

    id_column = {"text": "id", "image": "image_filename", "image_and_text": "docid"}[view]
    identifiers = set()
    for path in sorted(directory.iterdir()):
        relative = path.relative_to(root).as_posix()
        if relative not in artifacts:
            raise RetrievalVLBundleError(f"Unlisted corpus artifact: {relative}")
        if path.suffix != ".parquet":
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                identifier = row.get(id_column)
                if identifier in identifiers:
                    raise RetrievalVLBundleError(f"Duplicate corpus identifier: {identifier}")
                unit = units.get(identifier)
                if unit is None or unit["split"] != split:
                    raise RetrievalVLBundleError(f"Unknown or wrong-split corpus identifier: {identifier}")
                identifiers.add(identifier)
                if view != "image" and row.get("text") != unit["text"]:
                    raise RetrievalVLBundleError(f"Corpus text differs from source text: {identifier}")
                if view != "text":
                    if (
                        len(unit["images"]) != 1
                        or row.get("image") != _asset(root, unit["images"][0], artifacts).read_bytes()
                    ):
                        raise RetrievalVLBundleError(f"Corpus image differs from source asset: {identifier}")
    return identifiers


def _corpus_identity(directory: Path, view: str) -> str:
    """Validate the producer's supported loader metadata and return its identity."""
    metadata = _load_json(directory / "merlin_metadata.json")
    classes = {"text": "TextQADataset", "image": "ColPaliDataset", "image_and_text": "WikiSSNQDataset"}
    if metadata.get("class") != classes[view] or metadata.get("format") != "parquet":
        raise RetrievalVLBundleError(f"Unsupported corpus metadata for {view}: {directory}")
    corpus_id = metadata.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise RetrievalVLBundleError(f"Missing or invalid corpus identity: {directory}")
    return corpus_id


def _training(path: Path, corpus: set[str], units: dict, query_ids: set[str], split: str, corpus_id: str) -> None:
    """Check positive references and partial-judgment semantics without rewriting."""
    payload = _load_json(path)
    if payload.get("corpus") != {"path": f"corpus/{split}"} or not isinstance(payload.get("data"), list):
        raise RetrievalVLBundleError(f"Invalid training corpus reference or data list: {path}")
    for row in payload["data"]:
        identifier = row.get("question_id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier:
            raise RetrievalVLBundleError(f"Missing query identifier in {path}")
        if identifier in query_ids:
            raise RetrievalVLBundleError(f"Duplicate query identifier: {identifier}")
        query_ids.add(identifier)
        if row.get("corpus_id") != corpus_id:
            raise RetrievalVLBundleError(f"Query corpus identity mismatch: {identifier}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise RetrievalVLBundleError(f"Missing or invalid query text: {identifier}")
        positives = row.get("pos_doc")
        if not isinstance(positives, list) or not positives:
            raise RetrievalVLBundleError(f"Missing positive documents: {identifier}")
        positive_ids = [item.get("id") if isinstance(item, dict) else None for item in positives]
        if any(value not in corpus for value in positive_ids) or len(set(positive_ids)) != len(positive_ids):
            raise RetrievalVLBundleError(f"Unknown or duplicate positive document: {identifier}")
        if set(row.get("source_document_ids", [])) != {units[value]["document_id"] for value in positive_ids}:
            raise RetrievalVLBundleError(f"Positive source document mismatch: {identifier}")
        if set(row.get("positive_unit_ids", [])) != set(positive_ids):
            raise RetrievalVLBundleError(f"Positive unit mismatch: {identifier}")
        expected = {
            "negative_mining_performed": False,
            "relevance_judgements_complete": False,
            "unlisted_document_disposition": "unjudged",
        }
        if row.get("neg_doc") != [] or row.get("negative_scores") != []:
            raise RetrievalVLBundleError(f"Unified producer must contain positive-only records: {identifier}")
        if any(key in row and row[key] != value for key, value in expected.items()):
            raise RetrievalVLBundleError(f"Unsupported relevance or mining claim: {identifier}")


def _evaluation(
    root: Path, directory: Path, corpus: set[str], units: dict, artifacts: dict, query_ids: set[str], view: str
) -> None:
    """Validate synthetic BEIR IDs/qrels without asserting independent labels."""
    queries = _jsonl(directory / "queries.jsonl", "_id")
    documents = _jsonl(directory / "corpus.jsonl", "_id")
    if query_ids.intersection(queries):
        raise RetrievalVLBundleError("Duplicate query identifier across partitions")
    if set(documents) != corpus:
        raise RetrievalVLBundleError("Synthetic evaluation corpus differs from evaluation partition")
    for identifier, row in documents.items():
        unit = units[identifier]
        if row.get("metadata", {}).get("source_document_id") != unit["document_id"]:
            raise RetrievalVLBundleError(f"Evaluation source document mismatch: {identifier}")
        if view != "image" and row.get("text") != unit["text"]:
            raise RetrievalVLBundleError(f"Evaluation text differs from source text: {identifier}")
        if view != "text":
            image = row.get("image_path")
            _asset(root, image, artifacts)
            if image not in unit["images"]:
                raise RetrievalVLBundleError(f"Evaluation image differs from source asset: {identifier}")
    seen = set()
    with (directory / "qrels/test.tsv").open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise RetrievalVLBundleError("Invalid qrels header")
        for row in reader:
            pair = (row.get("query-id"), row.get("corpus-id"))
            try:
                score = float(row["score"])
            except (ValueError, TypeError) as error:
                raise RetrievalVLBundleError("Invalid qrel score") from error
            if (
                pair in seen
                or pair[0] not in queries
                or pair[1] not in corpus
                or not math.isfinite(score)
                or score <= 0
            ):
                raise RetrievalVLBundleError(f"Invalid or dangling qrel: {pair}")
            seen.add(pair)
    if {pair[0] for pair in seen} != set(queries):
        raise RetrievalVLBundleError("Synthetic query has no positive qrel")


def inspect_unified_bundle(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    view: ViewName,
    require_local_artifacts: bool,
    verify_checksums: bool,
) -> VLBundlePaths:
    """Resolve v2 paths; local mode always checks inventory, hashes and identities."""
    root = manifest_path.parent
    producer_view = "image_and_text" if view == "text_image" else view
    protocol = manifest.get("split_protocol", "document_disjoint")
    if protocol not in {"document_disjoint", "grouped_query_disjoint"}:
        raise RetrievalVLBundleError(f"Unknown split protocol: {protocol!r}")
    shared = protocol == "grouped_query_disjoint"
    scope = "full_collection" if shared else "partition"
    if manifest.get("corpus_scope", "partition") != scope:
        raise RetrievalVLBundleError(f"Split protocol {protocol} requires corpus_scope={scope}")
    optional_claims = {
        "status": "completed",
        "portable": True,
        "negative_mining_performed": False,
        "relevance_judgements_complete": False,
        "unlisted_document_disposition": "unjudged",
        "relevance_annotation_scope": "known_positives_only",
    }
    for key, expected in optional_claims.items():
        if key in manifest and manifest[key] != expected:
            raise RetrievalVLBundleError(f"Unsupported unified bundle {key}: {manifest[key]!r}")
    if manifest.get("gold_evaluation_status") == "available":
        raise RetrievalVLBundleError("Unified v2 has no independent gold evaluation contract")
    verify = require_local_artifacts or verify_checksums
    artifacts = _inventory(root, manifest, verify)
    required = [
        "retrieval_units.jsonl",
        "split_manifest.json",
        "source_records.jsonl",
        "candidate_diagnostics.jsonl",
        f"views/{producer_view}/train.json",
        f"views/{producer_view}/validation.json",
        f"synthetic_eval/{producer_view}/queries.jsonl",
        f"synthetic_eval/{producer_view}/corpus.jsonl",
        f"synthetic_eval/{producer_view}/qrels/test.tsv",
    ]
    corpus_splits = ("shared",) * 3 if shared else ("train", "validation", "evaluation")
    corpus_paths = [root / f"views/{producer_view}/corpus/{split}" for split in corpus_splits]
    for directory in corpus_paths:
        prefix = directory.relative_to(root).as_posix() + "/"
        required.append(prefix + "merlin_metadata.json")
        if not any(name.startswith(prefix) and name.endswith(".parquet") for name in artifacts):
            raise RetrievalVLBundleError(f"Manifest missing required corpus shard: {prefix}")
    missing = set(required) - artifacts.keys()
    if missing:
        raise RetrievalVLBundleError(f"Manifest missing required artifacts: {sorted(missing)}")
    paths = VLBundlePaths(
        root,
        manifest_path,
        view,
        root / f"views/{producer_view}/train.json",
        root / f"views/{producer_view}/validation.json",
        *corpus_paths,
        root / f"synthetic_eval/{producer_view}",
        None,
        verify,
        protocol,
        scope,
    )
    if verify:
        corpus_ids = [_corpus_identity(directory, producer_view) for directory in corpus_paths]
        units = _units(root, artifacts, shared=shared)
        if shared:
            corpora = [_corpus(root, corpus_paths[0], producer_view, "shared", units, artifacts)] * 3
        else:
            corpora = [
                _corpus(root, directory, producer_view, split, units, artifacts)
                for directory, split in zip(corpus_paths, corpus_splits, strict=True)
            ]
        query_ids: set[str] = set()
        _training(paths.train, corpora[0], units, query_ids, corpus_splits[0], corpus_ids[0])
        _training(paths.validation, corpora[1], units, query_ids, corpus_splits[1], corpus_ids[1])
        _evaluation(root, paths.synthetic_eval, corpora[2], units, artifacts, query_ids, producer_view)
        if shared:
            from nemotron.recipes.retrieval_query_split import validate_grouped_query_bundle

            validate_grouped_query_bundle(paths, units, corpora[0], producer_view)
    return paths
