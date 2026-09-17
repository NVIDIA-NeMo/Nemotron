# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for native AutoModel BEIR evaluation."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemotron.recipes.embed.stage3_eval import eval as evaluation
from nemotron.recipes.embed.stage3_eval.automodel_backend import AutoModelBEIREncoder


def test_automodel_backend_is_additive_default() -> None:
    """Existing local evaluation keeps its Hugging Face backend by default."""
    assert evaluation.EvalConfig().local_backend == "huggingface"
    assert evaluation.EvalConfig().tokenizer_force_default is False


def test_evaluate_model_dispatches_native_backend(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The public evaluator forwards its local contract to the native adapter."""
    captured = {}

    class Native:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "nemotron.recipes.embed.stage3_eval.automodel_backend.AutoModelBEIREncoder",
        Native,
    )

    class StopSearchError(Exception):
        pass

    def search(model, **kwargs):
        captured["model"] = model
        raise StopSearchError

    modules = {
        "beir.datasets.data_loader": {"GenericDataLoader": object},
        "beir.retrieval": {"models": SimpleNamespace(HuggingFace=MagicMock())},
        "beir.retrieval.evaluation": {"EvaluateRetrieval": object},
        "beir.retrieval.models": {},
        "beir.retrieval.models.huggingface": {"AutoTokenizer": object},
        "beir.retrieval.search.dense.exact_search": {"DenseRetrievalExactSearch": search},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(StopSearchError):
        evaluation.evaluate_model("checkpoint", tmp_path, local_backend="automodel", tokenizer_force_default=True)
    assert captured["model_path"] == "checkpoint"
    assert captured["query_prefix"] == "query: "
    assert captured["passage_prefix"] == "passage: "
    assert captured["tokenizer_force_default"] is True


def test_native_encoder_restores_attention_and_uses_native_embedding_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter preserves saved attention while pinning pooling, normalization, and tokens."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    load_model = MagicMock()
    model = MagicMock()
    model.is_causal = False
    model.parameters.return_value = iter([torch.zeros(1)])
    model.encode.return_value = torch.tensor([[1.0, 0.0]])
    load_model.return_value = model
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    tokenizer.pad.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    load_tokenizer = MagicMock(return_value=tokenizer)

    auto_model_module = ModuleType("nemo_automodel._transformers.auto_model")
    auto_model_module.NeMoAutoModelBiEncoder = SimpleNamespace(from_pretrained=load_model)
    auto_tokenizer_module = ModuleType("nemo_automodel._transformers.auto_tokenizer")
    auto_tokenizer_module.NeMoAutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
    monkeypatch.setitem(sys.modules, "nemo_automodel._transformers.auto_model", auto_model_module)
    monkeypatch.setitem(sys.modules, "nemo_automodel._transformers.auto_tokenizer", auto_tokenizer_module)

    encoder = AutoModelBEIREncoder("checkpoint", tokenizer_force_default=True)
    encoder.encode_queries(["question"], batch_size=1)

    kwargs = load_model.call_args.kwargs
    assert kwargs["pooling"] == "avg"
    assert kwargs["l2_normalize"] is True
    assert "is_causal" not in kwargs
    assert encoder.model.is_causal is False
    model.to.assert_called_once_with("cpu")
    assert load_tokenizer.call_args.kwargs == {
        "force_default": True,
        "add_bos_token": True,
        "add_eos_token": False,
    }
    assert tokenizer.call_args.args[0] == ["query: question"]


def test_run_eval_records_backend_and_persists_each_local_ranking_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Base and fine-tuned rankings are retained from their existing evaluation passes."""
    eval_data = tmp_path / "eval"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    eval_data.mkdir()
    checkpoint.mkdir()
    metrics = ({"NDCG@1": 1.0}, {"MAP@1": 1.0}, {"Recall@1": 1.0}, {"P@1": 1.0})
    rankings = iter([{"query": {"base-doc": 1.0}}, {"query": {"finetuned-doc": 1.0}}])
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return metrics, next(rankings)

    monkeypatch.setattr(evaluation, "evaluate_model", evaluate)
    monkeypatch.setattr(evaluation, "_release_cuda_memory", lambda: None)
    cfg = evaluation.EvalConfig(
        eval_data_path=eval_data,
        base_model="base",
        finetuned_model_path=checkpoint,
        output_dir=output,
        local_backend="automodel",
        tokenizer_force_default=True,
        eval_nim=False,
    )

    evaluation.run_eval(cfg)

    assert len(calls) == 2
    assert all(call["local_backend"] == "automodel" for call in calls)
    assert json.loads((output / "base_retrieval_results.json").read_text()) == {"query": {"base-doc": 1.0}}
    assert json.loads((output / "finetuned_retrieval_results.json").read_text()) == {"query": {"finetuned-doc": 1.0}}
    saved = json.loads((output / "eval_results.json").read_text())
    assert saved["_metadata"]["local_backend"] == "automodel"
    assert saved["_metadata"]["tokenizer_force_default"] is True
    assert not (output / "retrieval_results.json").exists()


def test_multimodal_encoder_uses_native_mining_loader_and_preserves_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The image path reuses the native mining loader and model-owned encoder."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = MagicMock(pooling="avg", l2_normalize=True, is_causal=False)
    load_model = MagicMock(return_value=model)
    auto_model_module = ModuleType("nemo_automodel._transformers.auto_model")
    auto_model_module.NeMoAutoModelBiEncoder = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "nemo_automodel._transformers.auto_model", auto_model_module)

    native = MagicMock()
    native.encode_queries.return_value = "queries"
    native.encode_documents.return_value = "documents"
    config = MagicMock()
    config.build.return_value = native

    encoder = AutoModelBEIREncoder("merged-checkpoint", multimodal_config=config)
    assert encoder.encode_queries(["question"], batch_size=2) == "queries"
    assert (
        encoder.encode_corpus(
            [{"title": "Title", "text": "Body", "image_path": "/bundle/assets/page.png"}],
            batch_size=1,
        )
        == "documents"
    )

    assert load_model.call_args.args == ("merged-checkpoint",)
    assert load_model.call_args.kwargs == {"use_liger_kernel": False, "use_sdpa_patching": True}
    config.build.assert_called_once_with(model=model, device=torch.device("cpu"))
    native.encode_documents.assert_called_once_with(
        [{"title": "Title", "text": "Body", "image": "/bundle/assets/page.png"}],
        batch_size=1,
    )


def test_evaluate_model_builds_exact_multimodal_processor_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The local checkpoint is also the processor source with the requested limits."""
    captured = {}

    class Native:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "nemotron.recipes.embed.stage3_eval.automodel_backend.AutoModelBEIREncoder",
        Native,
    )

    class StopSearchError(Exception):
        pass

    def search(model, **kwargs):
        raise StopSearchError

    modules = {
        "beir.datasets.data_loader": {"GenericDataLoader": object},
        "beir.retrieval": {"models": SimpleNamespace(HuggingFace=MagicMock())},
        "beir.retrieval.evaluation": {"EvaluateRetrieval": object},
        "beir.retrieval.models": {},
        "beir.retrieval.models.huggingface": {"AutoTokenizer": object},
        "beir.retrieval.search.dense.exact_search": {"DenseRetrievalExactSearch": search},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(StopSearchError):
        evaluation.evaluate_model(
            "merged-checkpoint",
            tmp_path,
            local_backend="automodel",
            model_family="mistral3_vl",
            query_max_length=512,
            passage_max_length=8192,
            image_longest_edge=1120,
        )

    config = captured["multimodal_config"]
    assert config.processor_name_or_path == "merged-checkpoint"
    assert (config.q_max_length, config.p_max_length, config.image_longest_edge) == (512, 8192, 1120)
    assert (config.query_prefix, config.passage_prefix) == ("query:", "passage:")
    assert config.use_images is True
    assert config.use_text_in_document is True


def test_restore_multimodal_images_after_portable_bundle_move(tmp_path) -> None:
    """Relative image records resolve from the moved bundle and become absolute at runtime."""
    original = tmp_path / "original"
    dataset = original / "synthetic_eval" / "text_image"
    image = original / "assets" / "pages" / "page.png"
    dataset.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    (dataset / "corpus.jsonl").write_text(
        json.dumps({"_id": "page", "text": "body", "image_path": "assets/pages/page.png"}) + "\n"
    )
    moved = tmp_path / "moved"
    original.rename(moved)
    moved_dataset = moved / "synthetic_eval" / "text_image"
    corpus = {"page": {"title": "", "text": "body"}}

    evaluation._restore_corpus_image_paths(corpus, moved_dataset, image_root=moved, resolve=True, strict=True)

    assert corpus["page"]["image_path"] == str((moved / "assets/pages/page.png").resolve())


def test_restore_multimodal_images_fails_when_declared_image_is_missing(tmp_path) -> None:
    """A declared image cannot silently fall back to text-only encoding."""
    dataset = tmp_path / "synthetic_eval" / "text_image"
    dataset.mkdir(parents=True)
    (dataset / "corpus.jsonl").write_text(
        json.dumps({"_id": "page", "text": "body", "image_path": "assets/pages/missing.png"}) + "\n"
    )

    with pytest.raises(FileNotFoundError, match="missing.png"):
        evaluation._restore_corpus_image_paths(
            {"page": {"text": "body"}}, dataset, image_root=tmp_path, resolve=True, strict=True
        )


@pytest.mark.parametrize("image_path", ["../outside.png", "/tmp/outside.png", ""])
def test_restore_multimodal_images_rejects_nonportable_or_empty_paths(tmp_path, image_path: str) -> None:
    """Native image declarations must be non-empty paths contained by the explicit root."""
    dataset = tmp_path / "eval"
    dataset.mkdir()
    (dataset / "corpus.jsonl").write_text(json.dumps({"_id": "page", "text": "body", "image_path": image_path}) + "\n")

    with pytest.raises(ValueError):
        evaluation._restore_corpus_image_paths(
            {"page": {"text": "body"}}, dataset, image_root=tmp_path, resolve=True, strict=True
        )


def test_restore_multimodal_images_rejects_symlink_escape(tmp_path) -> None:
    """A symlink cannot redirect an image declaration outside the configured artifact root."""
    root = tmp_path / "artifact"
    dataset = root / "eval"
    outside = tmp_path / "outside.png"
    dataset.mkdir(parents=True)
    outside.write_bytes(b"image")
    (root / "escaped.png").symlink_to(outside)
    (dataset / "corpus.jsonl").write_text(
        json.dumps({"_id": "page", "text": "body", "image_path": "escaped.png"}) + "\n"
    )

    with pytest.raises(ValueError, match="escapes"):
        evaluation._restore_corpus_image_paths(
            {"page": {"text": "body"}}, dataset, image_root=root, resolve=True, strict=True
        )


def test_native_text_evaluation_rejects_declared_images(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The native text family cannot silently flatten a corpus that declares images."""
    dataset = tmp_path / "eval"
    dataset.mkdir()
    (dataset / "corpus.jsonl").write_text(json.dumps({"_id": "page", "text": "body", "image_path": "page.png"}) + "\n")

    class Loader:
        def __init__(self, path):
            pass

        def load(self, split):
            return {"page": {"text": "body"}}, {"query": "question"}, {"query": {"page": 1}}

    class Retriever:
        k_values = [10]

        def __init__(self, *args, **kwargs):
            pass

    modules = {
        "beir.datasets.data_loader": {"GenericDataLoader": Loader},
        "beir.retrieval": {"models": SimpleNamespace(HuggingFace=MagicMock())},
        "beir.retrieval.evaluation": {"EvaluateRetrieval": Retriever},
        "beir.retrieval.models": {},
        "beir.retrieval.models.huggingface": {"AutoTokenizer": object},
        "beir.retrieval.search.dense.exact_search": {"DenseRetrievalExactSearch": MagicMock()},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("nemotron.recipes.embed.stage3_eval.automodel_backend.AutoModelBEIREncoder", MagicMock())

    with pytest.raises(ValueError, match="cannot discard"):
        evaluation.evaluate_model("checkpoint", dataset, local_backend="automodel")


def test_real_beir_preserves_legitimate_query_document_id_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The ViDoRe mode retains relevant pages whose IDs equal their query IDs."""
    torch = pytest.importorskip("torch")
    dataset = tmp_path / "eval"
    (dataset / "qrels").mkdir(parents=True)
    (dataset / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "title": "", "text": "alpha"}),
                json.dumps({"_id": "2", "title": "", "text": "beta"}),
            ]
        )
        + "\n"
    )
    (dataset / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "text": "alpha"}),
                json.dumps({"_id": "2", "text": "beta"}),
            ]
        )
        + "\n"
    )
    (dataset / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\n1\t1\t2\n2\t2\t2\n")

    class Native:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def _vectors(texts):
            return torch.tensor([[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts])

        def encode_queries(self, queries, **kwargs):
            return self._vectors(queries)

        def encode_corpus(self, corpus, **kwargs):
            return self._vectors(
                [(document.get("title", "") + " " + document.get("text", "")).strip() for document in corpus]
            )

    monkeypatch.setattr("nemotron.recipes.embed.stage3_eval.automodel_backend.AutoModelBEIREncoder", Native)
    metrics, results = evaluation.evaluate_model(
        "checkpoint",
        dataset,
        local_backend="automodel",
        k_values=[1],
        ignore_identical_ids=False,
    )

    assert results["1"] == {"1": 1.0}
    assert results["2"] == {"2": 1.0}
    assert metrics[0]["NDCG@1"] == 1.0


def test_multimodal_eval_config_requires_native_limits_and_backend() -> None:
    """Image-capable local evaluation is explicit and fully bounded."""
    with pytest.raises(ValueError, match="requires local_backend=automodel"):
        evaluation.EvalConfig(
            model_family="mistral3_vl",
            query_max_length=512,
            passage_max_length=8192,
            image_longest_edge=1120,
        )
    config = evaluation.EvalConfig(
        model_family="mistral3_vl",
        local_backend="automodel",
        query_max_length=512,
        passage_max_length=8192,
        image_longest_edge=1120,
    )
    assert config.use_text_in_document is True
