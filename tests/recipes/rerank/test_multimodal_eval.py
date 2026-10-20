# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rerank evaluation preserves modalities, learned logits, and candidate identity."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from nemo_runspec.config import load_pydantic_config
from nemotron.recipes.rerank.stage3_eval import eval as eval_module
from nemotron.recipes.rerank.stage3_eval import multimodal


@pytest.fixture
def native_backend(monkeypatch):
    import torch

    captured = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Model:
        config = SimpleNamespace(pooling="avg", _commit_hash="resolved-revision")
        is_causal = False
        model = SimpleNamespace(dtype=torch.bfloat16)

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["model_options"] = kwargs
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            captured["inputs"] = inputs
            return SimpleNamespace(logits=captured.get("logits", torch.tensor([[-2.25], [1.5]])))

    class Processor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            captured["processor_path"] = model_path
            captured["processor_options"] = kwargs
            return cls()

        def process_queries_documents_crossencoder(self, features, **kwargs):
            captured["features"] = features
            return {
                "input_ids": torch.ones((len(features), 2), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2)),
                "image_sizes": None,
            }

    root = ModuleType("nemo_automodel")
    root.NeMoAutoModelCrossEncoder = Model
    processor = ModuleType("nemo_automodel.components.models.ministral_bidirectional.processor")
    processor.Mistral3BiEncoderProcessor = Processor
    monkeypatch.setitem(sys.modules, root.__name__, root)
    monkeypatch.setitem(sys.modules, processor.__name__, processor)
    return captured


def test_native_scoring_preserves_head_revision_processor_and_raw_scores(native_backend):
    scorer = multimodal.AutoModelReranker("nvidia/rerank", revision="frozen-revision", image_longest_edge=1120)
    scores = scorer.score_pairs(
        [("Query", {"title": "Title", "text": "Text"}), ("Query", {"image_path": "/source/page.png"})], batch_size=2
    )
    assert scores == [-2.25, 1.5]
    assert native_backend["model_options"]["revision"] == "frozen-revision"
    assert native_backend["model_options"]["key_mapping"] == {r"^language_model\.score\.weight$": "score.weight"}
    assert "temperature" not in native_backend["model_options"]
    assert native_backend["processor_options"] == {
        "rerank_max_length": 4096,
        "use_prompt_template": True,
        "pad_to_multiple_of": 8,
        "image_longest_edge": 1120,
        "revision": "resolved-revision",
    }
    assert native_backend["features"] == [
        {"question": "Query", "doc_text": "Title Text", "doc_image": ""},
        {"question": "Query", "doc_text": "", "doc_image": "/source/page.png"},
    ]
    assert str(native_backend["inputs"]["pixel_values"].dtype) == "torch.bfloat16"
    assert str(native_backend["inputs"]["input_ids"].dtype) == "torch.int64"


def test_local_checkpoint_does_not_receive_remote_revision(tmp_path, native_backend):
    multimodal.AutoModelReranker(tmp_path, revision="remote-base-only")
    assert "revision" not in native_backend["model_options"]
    assert "revision" not in native_backend["processor_options"]


@pytest.mark.parametrize("logits", [[[float("nan")]], [[1.0, 2.0]]])
def test_native_invalid_scores_fail(native_backend, logits):
    import torch

    native_backend["logits"] = torch.tensor(logits)
    with pytest.raises(ValueError, match="logit"):
        multimodal.AutoModelReranker("nvidia/rerank").score_pairs([("Query", {"text": "Text"})])


@pytest.fixture
def endpoint(monkeypatch):
    state = {"requests": [], "response": {"data": [{"probs": [-1.25]}]}}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(state["response"]).encode()

    def urlopen(request, timeout):
        state["requests"].append((request.full_url, json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr(multimodal.urllib.request, "urlopen", urlopen)
    return state


@pytest.mark.parametrize("modality", ["text", "image", "image_and_text"])
def test_vllm_classify_sends_real_modalities_and_requests_raw_logits(tmp_path, endpoint, modality):
    image = tmp_path / "page.png"
    Image.new("RGB", (2, 3), "red").save(image)
    document = {}
    if modality != "image":
        document["text"] = "Supplied text"
    if modality != "text":
        document["image_path"] = str(image)
    scores = multimodal.VLLMReranker("http://localhost:8000/", "served", timeout=90).score_pairs([("Query", document)])
    assert scores == [-1.25]
    url, payload, timeout = endpoint["requests"][0]
    assert url == "http://localhost:8000/classify"
    assert timeout == 90
    assert payload["model"] == "served"
    assert payload["add_special_tokens"] is False
    assert payload["use_activation"] is False
    assert payload["messages"][0] == {"role": "query", "content": [{"type": "text", "text": "Query"}]}
    content = payload["messages"][1]["content"]
    assert [item["type"] for item in content] == (
        {"text": ["text"], "image": ["image_url"], "image_and_text": ["text", "image_url"]}[modality]
    )
    if modality != "text":
        assert content[-1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "response",
    [
        {"data": []},
        {"data": [{"probs": [True]}]},
        {"data": [{"probs": [float("inf")]}]},
        {"data": [{"probs": [1, 2]}]},
    ],
)
def test_vllm_rejects_missing_multiple_or_nonfinite_scores(endpoint, response):
    endpoint["response"] = response
    with pytest.raises(ValueError, match="one finite raw classification score"):
        multimodal.VLLMReranker("http://localhost:8000", "served").score_pairs([("q", {"text": "d"})])


def test_image_only_policy_keeps_text_only_documents(tmp_path, endpoint):
    image = tmp_path / "page.png"
    Image.new("RGB", (2, 3)).save(image)
    scorer = multimodal.VLLMReranker("http://localhost:8000", "served", use_text_in_document=False)
    scorer.score_pairs([("q", {"text": "caption", "image_path": str(image)}), ("q", {"text": "text only"})])
    assert [p["type"] for p in endpoint["requests"][0][1]["messages"][1]["content"]] == ["image_url"]
    assert endpoint["requests"][1][1]["messages"][1]["content"] == [{"type": "text", "text": "text only"}]


def test_profile_selects_native_family_and_training_revision():
    config = load_pydantic_config(eval_module.STAGE_PATH / "config/nemotron-3.5.yaml", [], eval_module.EvalConfig)
    assert config.model_family == "nemotron_3_5"
    assert config.model_revision == "39aa01525864f736e63f1d5742fc4cf59e754abd"
    assert config.max_length == 4096
    assert config.batch_size == 1
    assert config.query_prefix is None


def test_saved_and_dense_candidates_preserve_matching_query_document_ids(tmp_path, monkeypatch):
    pytest.importorskip("beir")
    image = tmp_path / "page.png"
    Image.new("RGB", (2, 3)).save(image)
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"_id": "good", "title": "", "text": "", "image_path": "page.png"},
                {"_id": "bad", "title": "", "text": "unrelated"},
            ]
        )
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"good", "text":"What is shown?"}\n')
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\ngood\tgood\t1\n")
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"good": {"bad": 0.9, "good": 0.1}}))
    corpus, queries, qrels, results = eval_module._get_first_stage_results(
        "unused", tmp_path, model_family="nemotron_3_5", first_stage_results_path=candidates
    )
    assert corpus["good"]["image_path"] == str(image)

    class Scorer:
        def score_pairs(self, pairs, *, batch_size):
            assert pairs[0][1]["text"] == "unrelated"
            return [-5.0, 2.0]

    metrics, rankings = multimodal.evaluate_multimodal_reranker(
        Scorer(),
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        first_stage_results=results,
        top_k=2,
        batch_size=1,
        k_values=[1],
    )
    assert rankings == {"good": {"bad": -5.0, "good": 2.0}}
    assert metrics[0]["NDCG@1"] == 1.0

    # Exercise BEIR's real dense-search boundary: it normally drops matching IDs.
    import numpy as np

    from nemotron.recipes.embed.stage3_eval import automodel_backend

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_queries(self, queries, **kwargs):
            return np.array([[1.0, 0.0] for _ in queries])

        def encode_corpus(self, documents, **kwargs):
            return np.array([[1.0, 0.0] if document.get("image_path") else [0.0, 1.0] for document in documents])

    monkeypatch.setattr(automodel_backend, "AutoModelBEIREncoder", Encoder)
    _, _, _, retrieved = eval_module._get_first_stage_results(
        "unused", tmp_path, model_family="nemotron_3_5", k_values=[1]
    )
    assert retrieved["good"]["good"] == 1.0
    assert set(retrieved) == {"good"}
    candidates.write_text(json.dumps({"other-query": {"bad": 0.9}}))
    with pytest.raises(ValueError, match="exactly the evaluation query IDs"):
        eval_module._get_first_stage_results(
            "unused", tmp_path, model_family="nemotron_3_5", first_stage_results_path=candidates
        )


@pytest.mark.parametrize("profile,extra", [("default", "text"), ("nemotron-3.5", "vl")])
def test_local_cli_selects_matching_dependency_extra(monkeypatch, profile, extra):
    from nemo_runspec import execution
    from nemotron.cli.commands.rerank import eval as cli_eval

    calls = []
    monkeypatch.setattr(execution, "execute_uv_local_from_spec", lambda **kwargs: calls.append(kwargs))
    cli_eval._execute_uv_local(eval_module.STAGE_PATH / f"config/{profile}.yaml", [])
    assert calls[0]["extras"] == [extra]


def test_multimodal_missing_checkpoint_fails_before_retrieval(tmp_path, monkeypatch):
    config = eval_module.EvalConfig(
        model_family="nemotron_3_5", eval_data_path=tmp_path, finetuned_model_path=tmp_path / "missing"
    )
    monkeypatch.setattr(eval_module, "_get_first_stage_results", lambda **kwargs: pytest.fail("retrieval ran"))
    with pytest.raises(FileNotFoundError, match="Fine-tuned checkpoint"):
        eval_module.run_eval(config)


@pytest.mark.parametrize("source", [{"sdg_input_path": "generation_result.json"}, {"retrieval_view": "mixed"}])
def test_portable_evaluation_requires_manifest_and_view_together(source):
    with pytest.raises(ValueError, match="must be set together"):
        eval_module.EvalConfig(model_family="nemotron_3_5", **source)
