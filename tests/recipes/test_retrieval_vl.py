# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from nemotron.recipes.retrieval_vl import (
    EXPECTED_CONSUMER_CONTRACT,
    RetrievalVLBundleError,
    _decode_document_image,
    inspect_vl_bundle,
    validate_vl_training_data,
)


def _write(path: Path, content: str = "{}\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _bundle(tmp_path: Path, *, gold: bool = True) -> Path:
    required = [
        "views/text_image/train.json",
        "views/text_image/validation.json",
        "synthetic_eval/text_image/queries.jsonl",
        "synthetic_eval/text_image/corpus.jsonl",
        "synthetic_eval/text_image/qrels/test.tsv",
    ]
    if gold:
        required.extend(
            [
                "gold_eval/text_image/queries.jsonl",
                "gold_eval/text_image/corpus.jsonl",
                "gold_eval/text_image/qrels/test.tsv",
            ]
        )
    artifacts = []
    for relative_path in required:
        checksum = _write(tmp_path / relative_path)
        artifacts.append({"relative_path": relative_path, "checksum": checksum})
    for split in ("train", "validation", "evaluation"):
        (tmp_path / f"views/text_image/corpus/{split}").mkdir(parents=True)

    manifest = {
        "schema_version": 3,
        "status": "completed",
        "portable": True,
        "consumer_contract_identity": EXPECTED_CONSUMER_CONTRACT,
        "consumer_contract_verified": True,
        "relevance_annotation_scope": "known_positives_only",
        "negative_mining_performed": False,
        "unlisted_document_disposition": "unjudged",
        "relevance_judgements_complete": False,
        "gold_evaluation_status": "available" if gold else "not_requested",
        "gold_eval_integrity_verified": gold,
        "artifacts": artifacts,
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _training_file(tmp_path: Path, **record_overrides: object) -> Path:
    record = {
        "question_id": "q1",
        "question": "What is shown?",
        "pos_doc": [{"id": "p1"}],
        "neg_doc": [],
        "negative_mining_performed": False,
        "relevance_judgements_complete": False,
        "unlisted_document_disposition": "unjudged",
    }
    record.update(record_overrides)
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"corpus": {"path": "corpus/train"}, "data": [record]}))
    return path


def test_decode_document_image_handles_plugin_parquet_bytes() -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 3), color="red").save(buffer, format="PNG")

    document = _decode_document_image({"image": buffer.getvalue(), "text": "caption"})

    assert isinstance(document["image"], Image.Image)
    assert document["image"].size == (2, 3)
    assert document["text"] == "caption"


def test_inspect_vl_bundle_resolves_view_and_checksums(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)

    paths = inspect_vl_bundle(manifest, verify_checksums=True)

    assert paths.train == tmp_path / "views/text_image/train.json"
    assert paths.gold_eval == tmp_path / "gold_eval/text_image"
    assert paths.evaluation_corpus == tmp_path / "views/text_image/corpus/evaluation"


def test_inspect_vl_bundle_rejects_contract_drift(tmp_path: Path) -> None:
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["consumer_contract_identity"] = "future-contract"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RetrievalVLBundleError, match="consumer_contract_identity"):
        inspect_vl_bundle(manifest_path)


def test_validate_vl_training_data_rejects_positive_only_export(tmp_path: Path) -> None:
    path = _training_file(tmp_path)

    with pytest.raises(RetrievalVLBundleError, match="0 negatives; 3 are required"):
        validate_vl_training_data(path, required_negatives=3, require_mined_negatives=True)


def test_validate_vl_training_data_accepts_scored_mined_negatives(tmp_path: Path) -> None:
    path = _training_file(
        tmp_path,
        neg_doc=[{"id": "n1", "score": 0.3}, {"id": "n2", "score": 0.2}, {"id": "n3", "score": 0.1}],
        negative_mining_performed=True,
    )

    count = validate_vl_training_data(path, required_negatives=3, require_mined_negatives=True)

    assert count == 1
    record = json.loads(path.read_text())["data"][0]
    assert record["relevance_judgements_complete"] is False
    assert record["unlisted_document_disposition"] == "unjudged"


def test_validate_vl_training_data_rejects_unscored_mined_negatives(tmp_path: Path) -> None:
    path = _training_file(
        tmp_path,
        neg_doc=[{"id": "n1"}, {"id": "n2", "score": 0.2}, {"id": "n3", "score": 0.1}],
        negative_mining_performed=True,
    )

    with pytest.raises(RetrievalVLBundleError, match="no finite mining score"):
        validate_vl_training_data(path, required_negatives=3, require_mined_negatives=True)
