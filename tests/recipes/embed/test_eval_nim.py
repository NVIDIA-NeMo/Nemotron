"""Regression tests for NIM evaluation response handling."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from nemotron.recipes.embed.stage3_eval.eval import NIMEmbeddingModel


class _Response:
    status = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_local_eval_uses_transformers_dtype_keyword() -> None:
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    source = textwrap.dedent(inspect.getsource(eval_module.evaluate_model))
    tree = ast.parse(source)
    model_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "HuggingFace"
    )
    keywords = {keyword.arg: keyword.value for keyword in model_call.keywords}

    assert ast.literal_eval(keywords["dtype"]) == "bfloat16"
    assert "torch_dtype" not in keywords


def test_beir_tokenizer_remote_code_is_trusted_and_loader_is_restored() -> None:
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return object()

    beir_huggingface = SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)

    with eval_module._allow_beir_tokenizer_remote_code(beir_huggingface):
        beir_huggingface.AutoTokenizer.from_pretrained("custom/model", use_fast=True)

    assert calls == [(("custom/model",), {"use_fast": True, "trust_remote_code": True})]
    assert beir_huggingface.AutoTokenizer is FakeAutoTokenizer


def test_invalid_nim_embedding_is_retried(monkeypatch) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    responses = iter(
        [
            _Response(
                {
                    "data": [
                        {"index": 0, "embedding": [None, None]},
                        {"index": 1, "embedding": [None, None]},
                    ]
                }
            ),
            _Response({"data": [{"index": 0, "embedding": [0.25, 0.75]}]}),
            _Response({"data": [{"index": 0, "embedding": [0.5, 0.5]}]}),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: next(responses))

    model = NIMEmbeddingModel(api_url="http://nim", model="test/model")
    embeddings = model.encode_queries(["first query", "second query"])

    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        embeddings,
        np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32),
    )
    assert model.diagnostics() == {
        "requested_model": "test/model",
        "embedding_dimension": 2,
        "invalid_embedding_retry_requests": 2,
    }


def test_persistent_invalid_nim_embedding_raises(monkeypatch) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    response = _Response({"data": [{"index": 0, "embedding": [None, None]}]})
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: response)

    model = NIMEmbeddingModel(api_url="http://nim", model="test/model")

    try:
        model.encode_queries(["query"])
    except RuntimeError as error:
        assert "invalid embeddings after 3 retries" in str(error)
    else:
        raise AssertionError("persistent invalid embeddings should fail")


def test_nim_eval_failure_propagates(monkeypatch, tmp_path) -> None:
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    eval_data = tmp_path / "eval"
    eval_data.mkdir()
    cfg = eval_module.EvalConfig(
        eval_data_path=eval_data,
        output_dir=tmp_path / "output",
        eval_base=False,
        eval_finetuned=False,
        eval_nim=True,
    )
    monkeypatch.setattr(
        eval_module,
        "evaluate_nim",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("NIM failed")),
    )

    with pytest.raises(RuntimeError, match="NIM failed"):
        eval_module.run_eval(cfg)


def test_zero_retries_accepts_valid_nim_embedding(monkeypatch) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    response = _Response({"model": "test/model", "data": [{"index": 0, "embedding": [1.0, 2.0]}]})
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: response)

    model = NIMEmbeddingModel(
        api_url="http://nim",
        model="test/model",
        invalid_embedding_retries=0,
        expected_dimension=2,
    )

    np.testing.assert_array_equal(model.encode_queries(["query"]), np.array([[1.0, 2.0]], dtype=np.float32))


def test_duplicate_nim_indices_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    response = _Response(
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ]
        }
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: response)

    model = NIMEmbeddingModel(api_url="http://nim", model="test/model")

    with pytest.raises(RuntimeError, match="returned indices"):
        model.encode_queries(["first", "second"])


def test_unexpected_nim_model_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    response = _Response({"model": "wrong/model", "data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: response)

    model = NIMEmbeddingModel(api_url="http://nim", model="test/model")

    with pytest.raises(RuntimeError, match="wrong/model"):
        model.encode_queries(["query"])


def test_unexpected_nim_dimension_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(NIMEmbeddingModel, "_check_connection", lambda self: None)
    response = _Response({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.5]}]})
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: response)

    model = NIMEmbeddingModel(api_url="http://nim", model="test/model", expected_dimension=2)

    with pytest.raises(RuntimeError, match="dimension 3; expected 2"):
        model.encode_queries(["query"])


def _metrics(ndcg: float, recall: float) -> tuple[dict, dict, dict, dict]:
    return (
        {"NDCG@1": ndcg, "NDCG@10": ndcg},
        {"MAP@1": ndcg, "MAP@10": ndcg},
        {"Recall@1": recall, "Recall@10": recall},
        {"P@1": recall, "P@10": recall},
    )


def test_nim_comparison_serializes_provenance_and_deltas(monkeypatch, tmp_path) -> None:
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    eval_data = tmp_path / "eval"
    finetuned_model = tmp_path / "checkpoint"
    eval_data.mkdir()
    finetuned_model.mkdir()
    output_dir = tmp_path / "nim-comparison"
    checkpoint_metrics = _metrics(0.5, 0.7)
    nim_metrics = _metrics(0.505, 0.695)

    monkeypatch.setattr(eval_module, "evaluate_model", lambda **kwargs: (checkpoint_metrics, {}))
    monkeypatch.setattr(
        eval_module,
        "evaluate_nim",
        lambda **kwargs: (
            nim_metrics,
            {},
            {
                "requested_model": "nvidia/nemotron-3-embed-1b",
                "embedding_dimension": 2048,
                "invalid_embedding_retry_requests": 2,
            },
        ),
    )
    monkeypatch.setattr(eval_module, "_release_cuda_memory", lambda: None)

    cfg = eval_module.EvalConfig(
        eval_data_path=eval_data,
        finetuned_model_path=finetuned_model,
        output_dir=output_dir,
        eval_base=False,
        eval_finetuned=True,
        eval_nim=True,
        nim_metric_tolerance=0.01,
        nim_metric_low_k_tolerance=0.03,
    )
    eval_module.run_eval(cfg)

    saved = json.loads((output_dir / "eval_results.json").read_text())
    metadata = saved["_metadata"]
    comparison = metadata["nim_metric_comparison"]

    assert metadata["eval_data_path"] == str(eval_data.resolve())
    assert metadata["finetuned_model_path"] == str(finetuned_model.resolve())
    assert metadata["nim_model"] == "nvidia/nemotron-3-embed-1b"
    assert metadata["nim_diagnostics"]["embedding_dimension"] == 2048
    assert metadata["nim_diagnostics"]["invalid_embedding_retry_requests"] == 2
    assert comparison["kind"] == "aggregate_behavioral_metric_drift"
    assert comparison["model_identity_proof"] is False
    assert comparison["within_tolerance"] is True
    assert comparison["deltas"]["NDCG"]["NDCG@10"]["delta"] == pytest.approx(0.005)


def test_nim_metric_drift_gate_writes_evidence_before_failing(monkeypatch, tmp_path) -> None:
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    eval_data = tmp_path / "eval"
    finetuned_model = tmp_path / "checkpoint"
    eval_data.mkdir()
    finetuned_model.mkdir()
    output_dir = tmp_path / "nim-comparison"

    monkeypatch.setattr(
        eval_module,
        "evaluate_model",
        lambda **kwargs: (_metrics(0.5, 0.7), {}),
    )
    monkeypatch.setattr(
        eval_module,
        "evaluate_nim",
        lambda **kwargs: (
            _metrics(0.2, 0.3),
            {},
            {
                "requested_model": "nvidia/nemotron-3-embed-1b",
                "embedding_dimension": 2048,
                "invalid_embedding_retry_requests": 0,
            },
        ),
    )
    monkeypatch.setattr(eval_module, "_release_cuda_memory", lambda: None)

    cfg = eval_module.EvalConfig(
        eval_data_path=eval_data,
        finetuned_model_path=finetuned_model,
        output_dir=output_dir,
        eval_base=False,
        eval_finetuned=True,
        eval_nim=True,
        fail_on_nim_metric_drift=True,
    )

    with pytest.raises(RuntimeError, match="metric drift exceeds"):
        eval_module.run_eval(cfg)

    saved = json.loads((output_dir / "eval_results.json").read_text())
    comparison = saved["_metadata"]["nim_metric_comparison"]
    assert comparison["within_tolerance"] is False
    assert comparison["fail_on_drift"] is True
