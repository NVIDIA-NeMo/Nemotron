# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint mining with real corpus files and a deterministic embedding client."""

import json
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemotron.recipes.embed.stage1_data_prep.mining import mine_vllm_hard_negatives
from nemotron.recipes.embed.stage3_eval import eval as embedding_eval


@pytest.mark.parametrize("absolute_reference", [False, True])
def test_prepared_corpus_reference_survives_endpoint_remining(tmp_path, monkeypatch, absolute_reference):
    """Native relative paths and endpoint absolute paths can both be mined again."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pq.write_table(
        pa.table({"docid": ["positive", "negative"], "text": ["Supported answer.", "Unrelated passage."]}),
        corpus / "part-00000.parquet",
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    train = prepared / "train.json"
    train.write_text(
        json.dumps(
            {
                "corpus": {"path": str(corpus) if absolute_reference else "../corpus"},
                "data": [
                    {
                        "question_id": "q1",
                        "question": "What is supported?",
                        "corpus_id": "example",
                        "pos_doc": [{"id": "positive", "score": 2}],
                        "neg_doc": [],
                    }
                ],
            }
        )
    )
    client = Mock()
    client.encode_queries.return_value = np.asarray([[1.0, 0.0]], dtype=np.float32)
    client._encode_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]
    monkeypatch.setattr(embedding_eval, "NIMEmbeddingModel", Mock(return_value=client))
    options = dict(
        api_url="http://unused",
        model="test-model",
        batch_size=2,
        hard_negatives_to_mine=1,
        hard_neg_margin=0.95,
    )
    first = tmp_path / "first" / "mined.json"
    second = tmp_path / "second" / "mined.json"

    mine_vllm_hard_negatives(train, first, **options)
    mine_vllm_hard_negatives(first, second, **options)

    for output in (first, second):
        payload = json.loads(output.read_text())
        assert payload["corpus"]["path"] == str(corpus)
        row = payload["data"][0]
        assert row["pos_doc"] == [{"id": "positive", "score": 1.0, "source_score": 2}]
        assert row["neg_doc"] == [{"id": "negative", "score": 0.0}]
        assert row["negative_mining_performed"] is True
