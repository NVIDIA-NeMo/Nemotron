# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
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


@pytest.mark.parametrize("kind", ["coverage", "outlier"])
@pytest.mark.parametrize("use_sentence_transformer", [False, True])
def test_optional_mcq_embedding_paths_use_callable_tokenizer(
    monkeypatch: pytest.MonkeyPatch, kind: str, use_sentence_transformer: bool
) -> None:
    from nemotron.steps.byob.runtime import text_coverage
    from nemotron.steps.byob.runtime.benchmark_families.mcq import semantic_outlier

    class Coverage(text_coverage.TextCoverage):
        def prepare_input_data(self, dataset):
            return dataset

    module = text_coverage if kind == "coverage" else semantic_outlier
    cls = Coverage if kind == "coverage" else semantic_outlier.TextSemanticOutlierDetectionMCQ
    adapter = object.__new__(cls)
    config_key = "coverage_check_config" if kind == "coverage" else "semantic_outlier_detection_config"
    adapter.config = SimpleNamespace(**{config_key: {"model_identifier": "test/multilingual-model"}})
    captured = []

    class Pipeline:
        def __init__(self, *, stages, name):
            captured.extend(stages)

        def run(self):
            pass

    native = EmbeddingCreatorStage(
        model_identifier="test/multilingual-model",
        use_sentence_transformer=use_sentence_transformer,
        text_field="question",
        embedding_field="vectors",
        cache_dir="/tmp/byob-model-cache",
        hf_token="synthetic-test-token",
        max_chars=80,
        max_seq_length=24,
        padding_side="left",
        sort_by_length=False,
        model_inference_batch_size=8,
        autocast=False,
        transformers_init_kwargs={"trust_remote_code": False},
    )
    original_tokenizer, original_model = native.stages
    original_tokenizer.unk_token = True

    def create_native(**kwargs) -> EmbeddingCreatorStage:
        assert kwargs == {
            "model_identifier": "test/multilingual-model",
            "text_field": "question",
            "embedding_field": "vectors",
        }
        return native

    monkeypatch.setattr(module, "EmbeddingCreatorStage", create_native)
    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module.glob, "glob", lambda path: ["output.parquet"])
    monkeypatch.setattr(module.pd, "read_parquet", lambda path: pd.DataFrame({"id": [1], "vectors": [[1.0]]}))
    result = adapter._compute_embeddings("input.parquet", "output", "question", "vectors", ["id", "vectors"])
    assert len(result) == 1
    stage = captured[1]
    assert isinstance(stage.stages[0], CallableTokenizerStage)
    assert stage.stages[0].model_identifier == "test/multilingual-model"
    assert stage.stages[0].text_field == "question"
    assert stage is native
    assert len(stage.stages) == 2
    assert stage.stages[1] is original_model
    for key in inspect.signature(TokenizerStage).parameters:
        assert getattr(stage.stages[0], key) == getattr(original_tokenizer, key)
