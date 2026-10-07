# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_FIELD, INPUT_ID_FIELD, SEQ_ORDER_FIELD
from nemo_curator.tasks import DocumentBatch

from nemotron.steps.byob.runtime.deduplication import CallableTokenizerStage, TextSemanticDeduplication


def test_callable_tokenizer_stage_does_not_require_batch_encode_plus() -> None:
    calls: list[tuple[list[str], dict]] = []

    class CallableOnlyTokenizer:
        def __call__(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return SimpleNamespace(
                input_ids=np.asarray([[1, 2, 0], [3, 0, 0]]),
                attention_mask=np.asarray([[1, 1, 0], [1, 0, 0]]),
            )

    stage = CallableTokenizerStage(model_identifier="test/model", max_seq_length=3)
    stage.tokenizer = CallableOnlyTokenizer()
    batch = DocumentBatch(dataset_name="test", data=pd.DataFrame({"text": ["long", "short"]}))

    output = stage.process(batch).to_pandas()

    assert calls[0][0] == ["long", "short"]
    assert calls[0][1] == {
        "max_length": 3,
        "padding": "max_length",
        "return_tensors": "np",
        "truncation": True,
        "add_special_tokens": True,
        "return_token_type_ids": False,
    }
    assert list(output.columns) == ["text", INPUT_ID_FIELD, ATTENTION_MASK_FIELD, SEQ_ORDER_FIELD]
    assert output["text"].tolist() == ["short", "long"]


def test_embedding_pipeline_replaces_only_curators_tokenizer_stage(monkeypatch) -> None:
    captured: list = []

    class Adapter(TextSemanticDeduplication):
        def prepare_input_data(self, dataset):
            return dataset

    class Pipeline:
        def __init__(self, *, stages, name) -> None:
            captured.extend(stages)
            self.name = name

        def run(self, executor) -> None:
            assert executor == "executor"

    adapter = object.__new__(Adapter)
    adapter.config = SimpleNamespace(semantic_deduplication_config={"model_identifier": "sentence-transformers/test"})
    adapter.embeddings_path = "/tmp/byob-test-embeddings"
    adapter.executor = "executor"
    monkeypatch.setattr("nemotron.steps.byob.runtime.deduplication.Pipeline", Pipeline)

    adapter._compute_embeddings("/tmp/byob-test-input.parquet")

    embedding_stage = captured[1]
    tokenizer_stage = embedding_stage.stages[0]
    assert isinstance(tokenizer_stage, CallableTokenizerStage)
    assert len(embedding_stage.stages) == 2
    # The replacement inherits every setting Curator gave its own tokenizer stage.
    assert tokenizer_stage.model_identifier == "sentence-transformers/test"
    assert tokenizer_stage.text_field == "text"
    assert tokenizer_stage.sort_by_length is True
    assert tokenizer_stage.padding_side == "right"
