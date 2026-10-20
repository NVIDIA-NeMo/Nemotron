"""Bounded-memory endpoint evaluation regressions without network or GPU access."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from nemotron.recipes.embed.stage3_eval import eval as evaluation


class Response:
    def __init__(self, vectors):
        self.vectors = vectors

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"embeddings": {"float": self.vectors}}).encode()


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_images_are_not_read_ahead_of_current_request(monkeypatch, tmp_path, as_mapping, batch_size):
    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
    image = tmp_path / "page.png"
    image.write_bytes(b"synthetic-image")
    documents = [{"text": "text"}, {"image_path": "page.png"}, {"text": "mixed", "image_path": "page.png"}] * 2
    reads = []
    completed = []
    original_read = Path.read_bytes

    def read(path):
        reads.append(path)
        return original_read(path)

    def endpoint(request, timeout):
        payload = json.loads(request.data)
        count = len(payload.get("inputs", payload.get("texts", [])))
        expected = documents[: len(completed) + count]
        assert len(reads) == sum(bool(doc.get("image_path")) for doc in expected)
        vectors = [[float(index), 1.0] for index in range(len(completed), len(completed) + count)]
        completed.extend(vectors)
        return Response(vectors)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    client = evaluation.NIMEmbeddingModel(
        api_url="http://example.invalid", model="fixture/model", api_backend="vllm", dataset_path=tmp_path
    )
    corpus = {str(index): doc for index, doc in enumerate(documents)} if as_mapping else documents
    actual = client.encode_corpus(corpus, batch_size=batch_size)
    np.testing.assert_array_equal(actual, np.asarray([[float(index), 1.0] for index in range(6)], dtype=np.float32))


class SearchConfiguredError(Exception):
    """Stop before any retrieval work after inspecting its construction."""


@pytest.mark.parametrize("configured_chunk_size", [7, 1024, 50000])
def test_endpoint_evaluation_receives_recipe_chunk_size(monkeypatch, tmp_path, configured_chunk_size):
    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
    observed = []

    def search(model, *, corpus_chunk_size, batch_size):
        observed.append(corpus_chunk_size)
        raise SearchConfiguredError

    modules = {
        "beir.datasets.data_loader": {"GenericDataLoader": object},
        "beir.retrieval.evaluation": {"EvaluateRetrieval": object},
        "beir.retrieval.search.dense.exact_search": {"DenseRetrievalExactSearch": search},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    cfg = evaluation.EvalConfig(
        eval_data_path=tmp_path,
        output_dir=tmp_path / "output",
        eval_base=False,
        eval_finetuned=False,
        eval_nim=True,
        corpus_chunk_size=configured_chunk_size,
    )
    with pytest.raises(SearchConfiguredError):
        evaluation.run_eval(cfg)
    assert observed == [configured_chunk_size]
