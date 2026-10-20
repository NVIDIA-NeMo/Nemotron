# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mining must preserve annotations discarded by AutoModel's training loader."""

import copy
import json

import pytest

from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig, _record_native_vl_mining
from nemotron.recipes.embed.stage1_data_prep.mining_output import restore_mining_provenance
from nemotron.recipes.embed.stage1_data_prep.scripts.unroll_pos_docs import unroll_training_data


def _inputs(tmp_path):
    rows = [
        {
            "question_id": f"q-{index}",
            "question": f"Which operating limit applies to component {index}?",
            "corpus_id": "manual",
            "pos_doc": [{"id": "page-1", "score": 2}, {"id": "page-2", "score": 1}],
            "neg_doc": [],
            "negative_scores": [],
            "query_group_id": f"family-{index}",
            "positive_unit_ids": ["page-1", "page-2"],
            "source_document_ids": ["manual-1"],
            "language": "en",
            "negative_mining_performed": False,
            "relevance_judgements_complete": False,
            "unlisted_document_disposition": "unjudged",
        }
        for index in range(2)
    ]
    source = {"corpus": {"path": "corpus/shared"}, "data": rows}
    # AutoModel's load_datasets retains only these five fields; its writer
    # replaces document scores with similarity scores and serializes those rows.
    normalized = [
        {
            **{key: row[key] for key in ("question_id", "question", "corpus_id")},
            "pos_doc": [{"id": "page-2", "score": 0.7}, {"id": "page-1", "score": 0.9}],
            "neg_doc": [{"id": f"negative-{index}", "score": 0.1 + index / 10}],
        }
        for index, row in enumerate(reversed(rows))
    ]
    args = {"multimodal_encoder": {"p_max_length": 4096, "use_images": True}, "cache_reuse": False}
    output = {"corpus": source["corpus"], "data": normalized, "mining": {"args": args}}
    train = tmp_path / "train.json"
    mined = tmp_path / "mined.json"
    train.write_text(json.dumps(source))
    mined.write_text(json.dumps(output))
    return train, mined, source, output


def test_native_mining_restores_annotations_by_identity_and_retains_args(tmp_path):
    train, mined, source, output = _inputs(tmp_path)
    original_input = train.read_bytes()
    restore_mining_provenance(json.loads(train.read_text()), mined)
    _record_native_vl_mining(mined, DataPrepConfig(model_family="nemotron_3_5", hard_negatives_to_mine=1))
    result = json.loads(mined.read_text())
    assert result["mining"]["args"] == output["mining"]["args"]
    assert result["mining"]["backend"] == "automodel"
    assert train.read_bytes() == original_input
    assert [row["question_id"] for row in result["data"]] == ["q-0", "q-1"]
    for original, restored in zip(source["data"], result["data"], strict=True):
        for key in (
            "query_group_id",
            "positive_unit_ids",
            "source_document_ids",
            "language",
            "relevance_judgements_complete",
            "unlisted_document_disposition",
        ):
            assert restored[key] == original[key]
        assert restored["negative_mining_performed"] is True
        assert restored["negative_scores"] == [restored["neg_doc"][0]["score"]]
        assert restored["pos_doc"] == [
            {"id": "page-1", "score": 0.9, "source_score": 2},
            {"id": "page-2", "score": 0.7, "source_score": 1},
        ]
    unrolled = unroll_training_data(result["data"])
    assert len(unrolled) == 4
    assert [row["query_group_id"] for row in unrolled] == ["family-0", "family-0", "family-1", "family-1"]
    assert all(row["relevance_judgements_complete"] is False for row in unrolled)


@pytest.mark.parametrize("mutation", ["missing-query", "duplicate-query", "query-text", "corpus", "positive"])
def test_native_mining_cannot_relabel_mismatched_results(tmp_path, mutation):
    train, mined, _, output = _inputs(tmp_path)
    if mutation == "missing-query":
        output["data"].pop()
    elif mutation == "duplicate-query":
        output["data"].append(copy.deepcopy(output["data"][0]))
    elif mutation == "query-text":
        output["data"][0]["question"] = "Another question"
    elif mutation == "corpus":
        output["data"][0]["corpus_id"] = "another-corpus"
    else:
        output["data"][0]["pos_doc"][0]["id"] = "unrelated-page"
    mined.write_text(json.dumps(output))
    before = mined.read_bytes()
    with pytest.raises(ValueError, match="identif|identity|positives"):
        restore_mining_provenance(json.loads(train.read_text()), mined)
    assert mined.read_bytes() == before


def test_native_mining_preserves_supported_scalar_positive_inputs(tmp_path):
    train, mined, source, _ = _inputs(tmp_path)
    for row in source["data"]:
        row["pos_doc"] = ["page-1", "page-2"]
    train.write_text(json.dumps(source))
    restore_mining_provenance(json.loads(train.read_text()), mined)
    assert json.loads(mined.read_text())["data"][0]["pos_doc"] == [
        {"id": "page-1", "score": 0.9},
        {"id": "page-2", "score": 0.7},
    ]


def test_native_mining_preserves_integer_question_identifiers(tmp_path):
    train, mined, source, output = _inputs(tmp_path)
    for row in [*source["data"], *output["data"]]:
        row["question_id"] = int(row["question_id"].removeprefix("q-"))
    train.write_text(json.dumps(source))
    mined.write_text(json.dumps(output))
    restore_mining_provenance(json.loads(train.read_text()), mined)
    assert [row["question_id"] for row in json.loads(mined.read_text())["data"]] == [0, 1]


def test_native_mining_does_not_restore_obsolete_score_arrays(tmp_path):
    train, mined, source, _ = _inputs(tmp_path)
    for row in source["data"]:
        row.update(pos_score=9.0, neg_scores=[-1.0])
    restore_mining_provenance(source, mined)
    for row in json.loads(mined.read_text())["data"]:
        assert "pos_score" not in row
        assert "neg_scores" not in row
        assert row["negative_scores"] == [row["neg_doc"][0]["score"]]


def test_native_mining_restores_an_in_place_source_snapshot(tmp_path):
    train, mined, source, _ = _inputs(tmp_path)
    train.write_bytes(mined.read_bytes())
    restore_mining_provenance(source, train)
    assert json.loads(train.read_text())["data"][0]["query_group_id"] == "family-0"
