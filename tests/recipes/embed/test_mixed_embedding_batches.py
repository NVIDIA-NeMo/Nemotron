"""Public-client regressions for heterogeneous retrieval corpus batches."""

from __future__ import annotations

import base64
import copy
import itertools
import json

import numpy as np
import pytest

from nemotron.recipes.embed.stage3_eval.eval import NIMEmbeddingModel


class Response:
    """Minimal HTTP response containing ordered vectors."""

    def __init__(self, vectors: list) -> None:
        self.vectors = vectors

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps({"embeddings": {"float": self.vectors}}).encode()


class MixedBatchEndpoint:
    """Reject the malformed request observed in a real mixed corpus run."""

    def __init__(self, expected_inputs: list[dict], retry_index: int | None) -> None:
        self.expected_inputs = expected_inputs
        self.retry_index = retry_index
        self.calls = 0
        self.vectors = [[float(index + 1), 1.0] for index in range(len(expected_inputs))]

    def __call__(self, request, timeout) -> Response:
        payload = json.loads(request.data)
        assert request.full_url == "http://example.invalid/v2/embed"
        assert payload["input_type"] == "document"
        assert payload["model"] == "fixture/model"
        if self.calls == 0:
            assert "texts" not in payload
            assert all(isinstance(item, dict) for item in payload["inputs"]), "Cohere inputs must all be objects"
            assert payload["inputs"] == self.expected_inputs
            vectors = copy.deepcopy(self.vectors)
            if self.retry_index is not None:
                vectors[self.retry_index] = [None, None]
        else:
            assert self.calls == 1 and self.retry_index is not None
            item = self.expected_inputs[self.retry_index]
            if item["content"][0]["type"] == "text":
                assert payload["texts"] == [item["content"][0]["text"]]
                assert "inputs" not in payload
            else:
                assert payload["inputs"] == [item]
                assert "texts" not in payload
            vectors = [self.vectors[self.retry_index]]
        self.calls += 1
        return Response(vectors)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("retry_index", [None, 0, 1, 2])
def test_mixed_corpus_batch_preserves_content_order_and_single_item_retries(
    monkeypatch, tmp_path, order, retry_index
) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    image_bytes = b"invented-image-fixture-no-network-decoding"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(image_bytes)
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode()},
    }
    sources = [
        {"title": "Source title", "text": "Standalone source text"},
        {"image_path": "page.png"},
        {"text": "Image source text", "image_path": "page.png"},
    ]
    expected = [
        {"content": [{"type": "text", "text": "Source title Standalone source text"}]},
        {"content": [image]},
        {"content": [image, {"type": "text", "text": "Image source text"}]},
    ]
    corpus = [sources[index] for index in order]
    original = copy.deepcopy(corpus)
    endpoint = MixedBatchEndpoint([expected[index] for index in order], retry_index)
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    client = NIMEmbeddingModel(
        api_url="http://example.invalid",
        model="fixture/model",
        api_backend="vllm",
        dataset_path=tmp_path,
        invalid_embedding_retries=1,
    )

    encoded = client.encode_corpus(corpus, batch_size=3)

    np.testing.assert_array_equal(encoded, np.asarray(endpoint.vectors, dtype=np.float32))
    assert corpus == original
    assert endpoint.calls == (1 if retry_index is None else 2)
    assert client.invalid_embedding_retry_requests == (0 if retry_index is None else 1)
