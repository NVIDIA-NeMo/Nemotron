# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Audit a Data Designer export: file integrity, source content, and query splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from nemotron.recipes.embed.sdg_manifest import (
    BundlePaths,
    _contained_path,
    _file_sha256,
    _load_json,
    resolve_bundle_view,
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
                    raise ValueError(f"Missing {id_field} in {path}")
                if identifier in records:
                    raise ValueError(f"Duplicate {id_field} in {path}: {identifier}")
                records[identifier] = row
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL {path}: {error}") from error
    return records


def _inventory(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify the producer's complete file inventory, including image assets."""
    records = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Bundle manifest must contain an artifacts list")
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Unified artifacts must have a path")
        relative = record["path"]
        if Path(relative).is_absolute() or Path(relative).as_posix() != relative:
            raise ValueError(f"Invalid portable artifact path: {relative!r}")
        path = _contained_path(root, relative)
        if relative in records:
            raise ValueError(f"Duplicate artifact path: {relative}")
        checksum = record.get("sha256")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError(f"Invalid declared size for {relative}")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"Invalid declared checksum for {relative}")
        records[relative] = record
        if not path.is_file():
            raise ValueError(f"Bundle missing artifact: {relative}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch for {relative}")
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"Checksum mismatch for {relative}")
    return records


def _asset(root: Path, value: str, artifacts: dict) -> Path:
    """Resolve only explicitly inventoried portable assets."""
    path = _contained_path(root, value)
    if value not in artifacts:
        raise ValueError(f"Unlisted image asset: {value}")
    return path


def _units(root: Path, artifacts: dict) -> dict[str, dict[str, Any]]:
    """Validate shared retrieval units and source asset references."""
    units = _jsonl(root / "retrieval_units.jsonl", "unit_id")
    for unit in units.values():
        split = unit.get("split")
        valid_split = split == "shared" and isinstance(unit.get("document_id"), str) and bool(unit["document_id"])
        if not valid_split:
            raise ValueError(f"Source document split mismatch for {unit['unit_id']}")
        images = unit.get("images")
        if not isinstance(images, list) or not isinstance(unit.get("text"), str):
            raise ValueError(f"Invalid source unit: {unit['unit_id']}")
        for value in images:
            _asset(root, value, artifacts)
    return units


def _corpus(root: Path, directory: Path, view: str, split: str, units: dict, artifacts: dict) -> set[str]:
    """Validate Parquet IDs and preserve authoritative source text/images."""
    import pyarrow.parquet as pq

    id_column = {"text": "id", "image": "image_filename", "image_and_text": "docid", "mixed": "docid"}[view]
    identifiers = set()
    for path in sorted(directory.iterdir()):
        relative = path.relative_to(root).as_posix()
        if relative not in artifacts:
            raise ValueError(f"Unlisted corpus artifact: {relative}")
        if path.suffix != ".parquet":
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                identifier = row.get(id_column)
                if identifier in identifiers:
                    raise ValueError(f"Duplicate corpus identifier: {identifier}")
                unit = units.get(identifier)
                if unit is None or unit["split"] != split:
                    raise ValueError(f"Unknown or wrong-split corpus identifier: {identifier}")
                identifiers.add(identifier)
                if view != "image" and row.get("text") != unit["text"]:
                    raise ValueError(f"Corpus text differs from source text: {identifier}")
                if view == "mixed" and not unit["images"]:
                    if row.get("image") is not None:
                        raise ValueError(f"Unexpected corpus image: {identifier}")
                elif view != "text":
                    if (
                        len(unit["images"]) != 1
                        or row.get("image") != _asset(root, unit["images"][0], artifacts).read_bytes()
                    ):
                        raise ValueError(f"Corpus image differs from source asset: {identifier}")
    return identifiers


def _training(path: Path, corpus: set[str], units: dict, query_ids: set[str], split: str, corpus_id: str) -> None:
    """Check positive references and partial-judgment semantics without rewriting."""
    payload = _load_json(path)
    if payload.get("corpus") != {"path": f"corpus/{split}"} or not isinstance(payload.get("data"), list):
        raise ValueError(f"Invalid training corpus reference or data list: {path}")
    for row in payload["data"]:
        identifier = row.get("question_id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Missing query identifier in {path}")
        if identifier in query_ids:
            raise ValueError(f"Duplicate query identifier: {identifier}")
        query_ids.add(identifier)
        if row.get("corpus_id") != corpus_id:
            raise ValueError(f"Query corpus identity mismatch: {identifier}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing or invalid query text: {identifier}")
        positives = row.get("pos_doc")
        if not isinstance(positives, list) or not positives:
            raise ValueError(f"Missing positive documents: {identifier}")
        positive_ids = [item.get("id") if isinstance(item, dict) else None for item in positives]
        if any(value not in corpus for value in positive_ids) or len(set(positive_ids)) != len(positive_ids):
            raise ValueError(f"Unknown or duplicate positive document: {identifier}")
        if set(row.get("source_document_ids", [])) != {units[value]["document_id"] for value in positive_ids}:
            raise ValueError(f"Positive source document mismatch: {identifier}")
        if set(row.get("positive_unit_ids", [])) != set(positive_ids):
            raise ValueError(f"Positive unit mismatch: {identifier}")
        expected = {
            "negative_mining_performed": False,
            "relevance_judgements_complete": False,
            "unlisted_document_disposition": "unjudged",
        }
        if row.get("neg_doc") != [] or row.get("negative_scores") != []:
            raise ValueError(f"Unified producer must contain positive-only records: {identifier}")
        if any(key in row and row[key] != value for key, value in expected.items()):
            raise ValueError(f"Unsupported relevance or mining claim: {identifier}")


def _evaluation(
    root: Path, directory: Path, corpus: set[str], units: dict, artifacts: dict, query_ids: set[str], view: str
) -> None:
    """Validate synthetic BEIR IDs/qrels without asserting independent labels."""
    queries = _jsonl(directory / "queries.jsonl", "_id")
    documents = _jsonl(directory / "corpus.jsonl", "_id")
    if query_ids.intersection(queries):
        raise ValueError("Duplicate query identifier across partitions")
    if set(documents) != corpus:
        raise ValueError("Synthetic evaluation corpus differs from shared full collection")
    for identifier, row in documents.items():
        unit = units[identifier]
        if row.get("metadata", {}).get("source_document_id") != unit["document_id"]:
            raise ValueError(f"Evaluation source document mismatch: {identifier}")
        if view != "image" and row.get("text") != unit["text"]:
            raise ValueError(f"Evaluation text differs from source text: {identifier}")
        if view == "mixed" and not unit["images"]:
            if row.get("image_path") is not None:
                raise ValueError(f"Unexpected evaluation image: {identifier}")
        elif view != "text":
            image = row.get("image_path")
            _asset(root, image, artifacts)
            if image not in unit["images"]:
                raise ValueError(f"Evaluation image differs from source asset: {identifier}")
    seen = set()
    with (directory / "qrels/test.tsv").open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Invalid qrels header")
        for row in reader:
            pair = (row.get("query-id"), row.get("corpus-id"))
            try:
                score = float(row["score"])
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid qrel score") from error
            if (
                pair in seen
                or pair[0] not in queries
                or pair[1] not in corpus
                or not math.isfinite(score)
                or score <= 0
            ):
                raise ValueError(f"Invalid or dangling qrel: {pair}")
            seen.add(pair)
    if {pair[0] for pair in seen} != set(queries):
        raise ValueError("Synthetic query has no positive qrel")


def validate_bundle(
    manifest_path: Path,
    *,
    view: str = "image_and_text",
) -> BundlePaths:
    """Audit an exported view without changing files or query partitions."""
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    root = manifest_path.parent
    artifacts = _inventory(root, manifest)
    paths = resolve_bundle_view(manifest_path, view=view)
    required = [
        "retrieval_units.jsonl",
        "split_manifest.json",
        "source_records.jsonl",
        "candidate_diagnostics.jsonl",
        f"views/{view}/train.json",
        f"views/{view}/validation.json",
        f"synthetic_eval/{view}/queries.jsonl",
        f"synthetic_eval/{view}/corpus.jsonl",
        f"synthetic_eval/{view}/qrels/test.tsv",
    ]
    corpus_path = root / f"views/{view}/corpus/shared"
    prefix = corpus_path.relative_to(root).as_posix() + "/"
    required.append(prefix + "merlin_metadata.json")
    if not any(name.startswith(prefix) and name.endswith(".parquet") for name in artifacts):
        raise ValueError(f"Manifest missing required corpus shard: {prefix}")
    missing = set(required) - artifacts.keys()
    if missing:
        raise ValueError(f"Manifest missing required artifacts: {sorted(missing)}")
    corpus_id = _load_json(paths.train_corpus / "merlin_metadata.json")["corpus_id"]
    units = _units(root, artifacts)
    corpus = _corpus(root, paths.train_corpus, view, "shared", units, artifacts)
    query_ids: set[str] = set()
    _training(paths.train, corpus, units, query_ids, "shared", corpus_id)
    _training(paths.validation, corpus, units, query_ids, "shared", corpus_id)
    _evaluation(root, paths.synthetic_eval, corpus, units, artifacts, query_ids, view)
    _validate_query_groups(paths, units, corpus, view)
    return paths


def _validate_query_groups(paths: BundlePaths, units: dict, corpus: set[str], view: str) -> None:
    """Reject family leakage, incomplete corpora and contradictory provenance."""
    manifest = _load_json(paths.root / "split_manifest.json")
    required = {"split_protocol": "grouped_query_disjoint", "corpus_scope": "full_collection"}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Split manifest contradicts grouped query protocol")
    if manifest.get("document_disjoint") is not False or manifest.get("assignments"):
        raise ValueError("Grouped query protocol must not claim document-disjoint assignments")
    # A view may exclude units with an unavailable modality, but never omit an
    # eligible page merely because it has no training or held-out positive.
    eligible = {
        key
        for key, unit in units.items()
        if view == "mixed" or (bool(unit["text"].strip()) if view == "text" else len(unit["images"]) == 1)
    }
    if corpus != eligible:
        raise ValueError("Shared corpus must contain every view-eligible retrieval unit")
    by_view = manifest.get("query_assignments")
    if not isinstance(by_view, dict):
        raise ValueError("Missing per-view query_assignments")
    assignments = by_view.get(view)
    if not isinstance(assignments, dict):
        raise ValueError("Missing per-view query_assignments")
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
                raise ValueError(f"Missing query_group_id: {identifier}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing query text: {identifier}")
            if assignments.get(identifier) != {"split": split, "query_group_id": group}:
                raise ValueError(f"Query assignment mismatch: {identifier}")
            if identifier in seen:
                raise ValueError(f"Duplicate query identifier: {identifier}")
            seen.add(identifier)
            normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
            if not normalized:
                raise ValueError(f"Query must contain words or numbers: {identifier}")
            for key, mapping in ((group, families), (normalized, texts)):
                if key in mapping and mapping[key] != split:
                    raise ValueError(f"Query family or normalized text crosses partitions: {identifier}")
                mapping[key] = split
    if seen != set(assignments):
        raise ValueError("Query assignments differ from exported view queries")


def main() -> None:
    """Run the optional export audit from the SDG environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the plugin run_manifest.json")
    parser.add_argument("--view", required=True, choices=("text", "image", "image_and_text", "mixed"))
    args = parser.parse_args()
    try:
        paths = validate_bundle(args.manifest, view=args.view)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    print(f"Validated {paths.view}: {paths.root}")


if __name__ == "__main__":
    main()
