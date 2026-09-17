"""Public boundary coverage for optional Cohere truncation policy."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_runspec.config import load_pydantic_config
from nemotron.recipes.embed.stage1_data_prep import data_prep
from nemotron.recipes.embed.stage3_eval import eval as evaluation


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


@pytest.mark.parametrize("policy", [None, "NONE", "START", "END"])
def test_policy_reaches_query_image_document_and_retry_requests(monkeypatch, tmp_path, policy):
    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
    (tmp_path / "image.png").write_bytes(b"invented-image")
    requests = []

    def endpoint(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        if policy is None:
            assert "truncate" not in payload
        else:
            assert payload["truncate"] == policy
        vectors = [[None, None]] if len(requests) == 1 else [[1.0, 0.0]]
        return Response({"embeddings": {"float": vectors}})

    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    client = evaluation.NIMEmbeddingModel(
        api_backend="vllm", api_truncate=policy, dataset_path=tmp_path, invalid_embedding_retries=1
    )
    client.encode_queries(["Standalone query"])
    client.encode_corpus([{"text": "Native text", "image_path": "image.png"}])
    assert len(requests) == 3
    assert [row["input_type"] for row in requests] == ["query", "query", "document"]
    assert client.diagnostics().get("api_truncate") == policy


def test_default_nim_request_unchanged(monkeypatch):
    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
    requests = []

    def endpoint(request, timeout):
        requests.append(json.loads(request.data))
        return Response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    client = evaluation.NIMEmbeddingModel(model="fixture/model", api_truncate=None)
    client.encode_queries(["Query"])
    assert requests == [{"input": ["Query"], "model": "fixture/model", "input_type": "query"}]
    assert "api_truncate" not in client.diagnostics()


@pytest.mark.parametrize("invalid", ["INVALID", "none", "", 0, True, []])
def test_shared_client_rejects_invalid_policy_before_connection(monkeypatch, invalid):
    def connection(self):
        pytest.fail("Invalid policy reached the connection check")

    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", connection)
    with pytest.raises(ValueError, match="truncate"):
        evaluation.NIMEmbeddingModel(api_backend="vllm", api_truncate=invalid)


@pytest.mark.parametrize("policy", ["NONE", "START", "END"])
def test_shared_client_rejects_policy_on_nim_before_connection(monkeypatch, policy):
    def connection(self):
        pytest.fail("Incompatible policy reached the connection check")

    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", connection)
    with pytest.raises(ValueError, match="vllm"):
        evaluation.NIMEmbeddingModel(api_backend="nim", api_truncate=policy)


@pytest.mark.parametrize("stage", ["prep", "eval"])
@pytest.mark.parametrize("profile", ["default", "llama", "mistral3-vl"])
def test_cli_policy_is_explicit_and_defaults_are_preserved(stage, profile):
    module = data_prep if stage == "prep" else evaluation
    model = data_prep.DataPrepConfig if stage == "prep" else evaluation.EvalConfig
    field = "mining_api_truncate" if stage == "prep" else "embedding_api_truncate"
    backend = "mining_backend" if stage == "prep" else "embedding_api_backend"
    config = module.STAGE_PATH / "config" / f"{profile}.yaml"
    assert getattr(load_pydantic_config(config, [], model), field) is None
    assert getattr(load_pydantic_config(config, [f"{backend}=vllm", f"{field}=NONE"], model), field) == "NONE"


@pytest.mark.parametrize(
    "model,field,backend",
    [
        (data_prep.DataPrepConfig, "mining_api_truncate", "mining_backend"),
        (evaluation.EvalConfig, "embedding_api_truncate", "embedding_api_backend"),
    ],
)
def test_config_rejects_incompatible_backend(model, field, backend):
    with pytest.raises(ValueError, match="requires.*vllm"):
        model(**{field: "NONE"})
    with pytest.raises(ValueError, match=field):
        model(**{field: "INVALID", backend: "vllm"})


@pytest.mark.parametrize("policy", [None, "NONE", "START", "END"])
def test_stage1_policy_reaches_real_mining_requests_and_provenance(monkeypatch, tmp_path, policy):
    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"docid": "positive", "text": "Positive document"}, {"docid": "negative", "text": "Negative document"}]
        ),
        corpus / "part.parquet",
    )
    training = tmp_path / "train.json"
    training.write_text(
        json.dumps(
            {
                "corpus": {"path": "corpus"},
                "data": [{"question_id": "query", "question": "Plausible query", "pos_doc": [{"id": "positive"}]}],
            }
        )
    )
    requests = []

    def endpoint(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        vectors = [[1.0, 0.0]] if payload["input_type"] == "query" else [[1.0, 0.0], [0.1, 1.0]]
        return Response({"embeddings": {"float": vectors}})

    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    cfg = data_prep.DataPrepConfig(
        sdg_input_path=None,
        train_input_file=training,
        output_dir=tmp_path / "output",
        mining_backend="vllm",
        mining_api_truncate=policy,
        hard_negatives_to_mine=1,
        mining_batch_size=2,
    )
    result = json.loads(data_prep.run_mining(cfg, training).read_text())
    assert len(requests) == 2
    assert all(row.get("truncate") == policy for row in requests)
    assert result["mining"].get("api_truncate") == policy
    assert result["data"][0]["neg_doc"][0]["id"] == "negative"


class SearchConfiguredError(Exception):
    pass


@pytest.mark.parametrize("policy", [None, "NONE", "START", "END"])
def test_stage3_policy_reaches_actual_shared_client(monkeypatch, tmp_path, policy):
    monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
    observed = []

    def search(model, **kwargs):
        observed.append(model.api_truncate)
        assert model.diagnostics().get("api_truncate") == policy
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
        embedding_api_backend="vllm",
        embedding_api_truncate=policy,
    )
    with pytest.raises(SearchConfiguredError):
        evaluation.run_eval(cfg)
    assert observed == [policy]
