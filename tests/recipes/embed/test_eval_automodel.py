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
        model_family="nemotron_3_5",
        query_max_length=512,
        passage_max_length=8192,
        eval_nim=False,
    )

    evaluation.run_eval(cfg)

    assert len(calls) == 2
    assert all(call["model_family"] == "nemotron_3_5" for call in calls)
    assert json.loads((output / "base_retrieval_results.json").read_text()) == {"query": {"base-doc": 1.0}}
    assert json.loads((output / "finetuned_retrieval_results.json").read_text()) == {"query": {"finetuned-doc": 1.0}}
    saved = json.loads((output / "eval_results.json").read_text())
    assert saved["_metadata"]["local_backend"] == "automodel"
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
    assert load_model.call_args.kwargs == {
        "use_liger_kernel": False,
        "use_sdpa_patching": True,
        "attn_implementation": "sdpa",
    }
    config.build.assert_called_once_with(model=model, device=torch.device("cpu"))
    native.encode_documents.assert_called_once_with(
        [{"title": "Title", "text": "Body", "image": "/bundle/assets/page.png"}],
        batch_size=1,
    )


@pytest.mark.parametrize(
    ("query_prefix", "passage_prefix", "expected_prefixes"),
    [(None, None, {}), ("", "", {"query_prefix": "", "passage_prefix": ""})],
)
def test_evaluate_model_builds_exact_multimodal_processor_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    query_prefix: str | None,
    passage_prefix: str | None,
    expected_prefixes: dict[str, str],
) -> None:
    """Checkpoint defaults are omitted while explicit empty prefixes are preserved."""
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

    class CheckpointMiningEncoderConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mining_module = ModuleType("nemotron.recipes.embed.mining_encoder")
    mining_module.CheckpointMiningEncoderConfig = CheckpointMiningEncoderConfig
    monkeypatch.setitem(sys.modules, "nemotron.recipes.embed.mining_encoder", mining_module)

    with pytest.raises(StopSearchError):
        evaluation.evaluate_model(
            "merged-checkpoint",
            tmp_path,
            model_family="nemotron_3_5",
            query_max_length=512,
            passage_max_length=8192,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )

    config = captured["multimodal_config"]
    assert config.kwargs == {
        "q_max_length": 512,
        "p_max_length": 8192,
        "use_images": True,
        "use_text_in_document": True,
        **expected_prefixes,
    }


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

    evaluation._restore_corpus_image_paths(corpus, moved_dataset, image_root=moved, resolve=True)

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
            {"page": {"text": "body"}}, dataset, image_root=tmp_path, resolve=True
        )


@pytest.mark.parametrize("image_path", ["../outside.png", "/tmp/outside.png", ""])
def test_restore_multimodal_images_rejects_nonportable_or_empty_paths(tmp_path, image_path: str) -> None:
    """Native image declarations must be non-empty paths contained by the explicit root."""
    dataset = tmp_path / "eval"
    dataset.mkdir()
    (dataset / "corpus.jsonl").write_text(json.dumps({"_id": "page", "text": "body", "image_path": image_path}) + "\n")

    with pytest.raises(ValueError):
        evaluation._restore_corpus_image_paths(
            {"page": {"text": "body"}}, dataset, image_root=tmp_path, resolve=True
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
            {"page": {"text": "body"}}, dataset, image_root=root, resolve=True
        )


@pytest.mark.parametrize("title_fields", [{}, {"title": None}, {"title": ""}, {"title": "Document"}])
@pytest.mark.parametrize("query_count", [1, 2])
def test_real_beir_preserves_legitimate_query_document_id_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path, title_fields: dict, query_count: int
) -> None:
    """Real BEIR supports optional titles and legitimate query/document ID collisions."""
    torch = pytest.importorskip("torch")
    dataset = tmp_path / "eval"
    (dataset / "qrels").mkdir(parents=True)
    (dataset / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "text": "alpha", **title_fields}),
                json.dumps({"_id": "2", "text": "beta", **title_fields}),
            ]
        )
        + "\n"
    )
    (dataset / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "text": "alpha"}),
                json.dumps({"_id": "2", "text": "beta"}),
            ][:query_count]
        )
        + "\n"
    )
    (dataset / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\t1\t2\n" + ("2\t2\t2\n" if query_count == 2 else "")
    )

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
        model_family="nemotron_3_5",
        query_max_length=512,
        passage_max_length=8192,
        k_values=[1],
        ignore_identical_ids=False,
    )

    assert results["1"] == {"1": 1.0}
    assert set(results) == {str(i + 1) for i in range(query_count)}
    if query_count == 2:
        assert results["2"] == {"2": 1.0}
    assert metrics[0]["NDCG@1"] == 1.0
    assert json.loads((dataset / "corpus.jsonl").read_text().splitlines()[0]) == {
        "_id": "1",
        "text": "alpha",
        **title_fields,
    }


def test_multimodal_eval_config_requires_native_limits() -> None:
    """Image-capable local evaluation requires explicit sequence limits."""
    with pytest.raises(ValueError, match="requires query and passage limits"):
        evaluation.EvalConfig(model_family="nemotron_3_5")
    config = evaluation.EvalConfig(
        model_family="nemotron_3_5",
        query_max_length=512,
        passage_max_length=8192,
    )
    assert config.image_longest_edge is None
    assert config.use_text_in_document is True


@pytest.mark.parametrize("backend", ["native", "api"])
@pytest.mark.parametrize("text_fields", [{}, {"text": None}])
def test_real_beir_evaluates_image_only_corpus_with_restored_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path, backend: str, text_fields: dict
) -> None:
    """Image-only exports survive BEIR loading/search on local and API paths."""
    import base64
    from pathlib import Path

    torch = pytest.importorskip("torch")
    dataset = tmp_path / "bundle/synthetic_eval/image"
    root = tmp_path / "bundle"
    (dataset / "qrels").mkdir(parents=True)
    (root / "assets").mkdir()
    for name in ("positive", "negative"):
        (root / f"assets/{name}.png").write_bytes(name.encode())
    corpus_rows = [
        {"_id": name, "title": "", "image_path": f"assets/{name}.png", **text_fields}
        for name in ("positive", "negative")
    ]
    (dataset / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in corpus_rows))
    (dataset / "queries.jsonl").write_text(json.dumps({"_id": "q", "text": "Which image matches?"}) + "\n")
    (dataset / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\tpositive\t2\n")
    seen_images = []

    def image_vector(content):
        seen_images.append(content)
        return [1.0, 0.0] if content == b"positive" else [0.0, 1.0]

    if backend == "native":
        class Native:
            def __init__(self, **kwargs):
                pass

            def encode_queries(self, queries, **kwargs):
                return torch.tensor([[1.0, 0.0] for _ in queries])

            def encode_corpus(self, corpus, **kwargs):
                assert all(document["text"] == "" for document in corpus)
                paths = [Path(document["image_path"]) for document in corpus]
                assert all(path.is_absolute() and path.is_relative_to(root) for path in paths)
                return torch.tensor([image_vector(path.read_bytes()) for path in paths])

        monkeypatch.setattr("nemotron.recipes.embed.stage3_eval.automodel_backend.AutoModelBEIREncoder", Native)
        metrics, rankings = evaluation.evaluate_model(
            "checkpoint",
            dataset,
            model_family="nemotron_3_5",
            query_max_length=512,
            passage_max_length=4096,
            image_root=root,
            k_values=[1],
        )
    else:
        def request(self, inputs, input_type):
            if input_type == "query":
                return [[1.0, 0.0] for _ in inputs]
            vectors = []
            for document in inputs:
                content = document["content"]
                assert len(content) == 1 and content[0]["type"] == "image_url"
                vectors.append(image_vector(base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])))
            return vectors

        monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_check_connection", lambda self: None)
        monkeypatch.setattr(evaluation.NIMEmbeddingModel, "_request_batch", request)
        metrics, rankings, _ = evaluation.evaluate_nim(
            "http://example.invalid", "fixture", dataset, api_backend="vllm", image_root=root, k_values=[1]
        )

    assert set(seen_images) == {b"positive", b"negative"}
    assert rankings["q"] == {"positive": 1.0}
    assert metrics[0]["NDCG@1"] == 1.0
    assert [json.loads(line) for line in (dataset / "corpus.jsonl").read_text().splitlines()] == corpus_rows
