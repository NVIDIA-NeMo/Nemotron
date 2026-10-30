# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.recipes._retrieval import validate_training_data


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


def test_validate_training_data_rejects_positive_only_export(tmp_path: Path) -> None:
    path = _training_file(tmp_path)

    with pytest.raises(ValueError, match="0 negatives; 3 are required"):
        validate_training_data(path, required_negatives=3, require_mined_negatives=True)


def test_validate_training_data_accepts_scored_mined_negatives(tmp_path: Path) -> None:
    path = _training_file(
        tmp_path,
        neg_doc=[{"id": "n1", "score": 0.3}, {"id": "n2", "score": 0.2}, {"id": "n3", "score": 0.1}],
        negative_mining_performed=True,
    )

    count = validate_training_data(path, required_negatives=3, require_mined_negatives=True)

    assert count == 1
    record = json.loads(path.read_text())["data"][0]
    assert record["relevance_judgements_complete"] is False
    assert record["unlisted_document_disposition"] == "unjudged"


def test_validate_training_data_rejects_unscored_mined_negatives(tmp_path: Path) -> None:
    path = _training_file(
        tmp_path,
        neg_doc=[{"id": "n1"}, {"id": "n2", "score": 0.2}, {"id": "n3", "score": 0.1}],
        negative_mining_performed=True,
    )

    with pytest.raises(ValueError, match="no finite mining score"):
        validate_training_data(path, required_negatives=3, require_mined_negatives=True)


@pytest.mark.parametrize("score", [None, True, float("inf"), float("nan"), "0.2"])
def test_training_preflight_rejects_nonfinite_or_invalid_mining_scores(tmp_path: Path, score) -> None:
    path = _training_file(tmp_path, neg_doc=[{"id": "n1", "score": score}], negative_mining_performed=True)
    with pytest.raises(ValueError, match="finite mining score"):
        validate_training_data(path, required_negatives=1, require_mined_negatives=True)


def test_training_preflight_requires_completed_mining(tmp_path: Path) -> None:
    path = _training_file(tmp_path, neg_doc=[{"id": "n1", "score": 0.2}])
    with pytest.raises(ValueError, match="negative_mining_performed=True"):
        validate_training_data(path, required_negatives=1, require_mined_negatives=True)
