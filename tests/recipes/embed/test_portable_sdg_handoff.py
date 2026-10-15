"""Cross-repository portable export tests with explicitly synthetic judge fixtures."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("data_designer_retrieval_sdg")

from nemotron.recipes.embed.sdg_manifest import (
    resolve_generation_input,
    resolve_portable_evaluation_input,
    resolve_portable_training_input,
    write_generation_manifest,
)
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig, run_sdg
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig, run_data_prep


def write_generated_fixture(root: Path) -> Path:
    """Write authored test judgments, not model-generation or quality evidence."""
    root.mkdir(parents=True)
    image = root / "page.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    candidate = {
        "question": "What storage temperature does Drug X require?",
        "answer": "20-25 C",
        "evidence": "Drug X storage is 20-25 C.",
        "query_surface": "question",
        "positive_unit_ids": ["p1"],
        "question_complexity": 4,
        "query_type": "contextual",
        "reasoning_type": "factual",
        "segment_ids": [1],
        "hop_count": 1,
        "hop_contexts": [],
    }
    criterion = {"score": 9, "justification": "Authored fixture."}
    record = {
        "source_id": "p1",
        "language": "en",
        "retrieval_units": [
            {
                "unit_id": "p1",
                "document_id": "doc",
                "text": "Drug X storage is 20-25 C.",
                "images": [str(image)],
                "segment_id": 1,
            }
        ],
        "deduplicated_qa_pairs": [candidate],
        "query_quality_evaluations": {
            "evaluations": [
                {
                    "candidate_index": 0,
                    "standalone_query": True,
                    "plausible_information_need": True,
                    "retrieval_discriminative": True,
                    "query_surface_correct": True,
                    "source_language_preserved": True,
                    "reason": "Authored fixture.",
                }
            ]
        },
        "qa_evaluations": {
            "evaluations": [
                {
                    "candidate_index": 0,
                    "relevance": criterion,
                    "accuracy": criterion,
                    "context_support": criterion,
                    "clarity": criterion,
                    "overall": {"score": 9, "assessment": "Authored fixture."},
                    "improvements": "None.",
                    "answer_grounded": True,
                    "answer_resolves_query": True,
                    "unsupported_claims": [],
                    "positive_source_relevant": True,
                    "evidence_modality": "text_only",
                    "answer_not_revealed_by_query": True,
                    "source_language_preserved": True,
                    "native_text_evidence": [{"unit_id": "p1", "quote": "Drug X storage is 20-25 C."}],
                }
            ]
        },
        "source_assessments": {
            "assessments": [
                {
                    "candidate_index": 0,
                    "positive_unit_ids": ["p1"],
                    "answerable": True,
                    "independent_answer": "Store Drug X between 20 and 25 C.",
                    "supporting_evidence": "Drug X storage is 20-25 C.",
                    "uncertainty_reasons": [],
                    "positive_source_relevant": True,
                    "evidence_modality": "text_only",
                    "source_language_preserved": True,
                    "native_text_evidence": [{"unit_id": "p1", "quote": "Drug X storage is 20-25 C."}],
                }
            ]
        },
    }
    generated = root / "generated.jsonl"
    generated.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return generated


@pytest.fixture
def portable_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run real recipe export and manifest publication with a mocked generation result."""
    from nemotron.recipes.embed.stage0_sdg import plugin_adapter

    output = tmp_path / "stage0"
    generated = write_generated_fixture(output)
    sources = tmp_path / "sources.jsonl"
    sources.write_text('{"unit_id":"p1","document_id":"doc","text":"Drug X storage is 20-25 C."}')
    result = SimpleNamespace(
        output_path=generated,
        dataset_name="generated",
        num_records=1,
        requested_num_records=1,
        resolved_config_path=None,
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(plugin_adapter, "execute_generation", lambda *_args: result)
    run_sdg(SDGConfig(sources_file=sources, output_dir=output, portable_export=True, export_train_ratio=1))
    return output / "generation_result.json"


@pytest.mark.parametrize("view", ["text", "image", "image_and_text"])
def test_exported_view_survives_relocation(portable_handoff: Path, tmp_path: Path, view: str) -> None:
    copied = tmp_path / "relocated"
    shutil.copytree(portable_handoff.parent, copied)
    train = resolve_portable_training_input(copied / portable_handoff.name, view)
    assert train == copied / "generated.bundle" / "views" / view / "train.json"
    assert len(json.loads(train.read_text())["data"]) == 1


def test_evaluation_follows_suffixed_bundle_in_latest_manifest(portable_handoff: Path) -> None:
    original = portable_handoff.parent / "generated.bundle"
    suffixed = portable_handoff.parent / "generated-1.bundle"
    original.rename(suffixed)
    payload = json.loads(portable_handoff.read_text())
    bundle_manifest = suffixed / "run_manifest.json"
    payload["portable_bundle_manifest"] = "generated-1.bundle/run_manifest.json"
    payload["portable_bundle_sha256"] = hashlib.sha256(bundle_manifest.read_bytes()).hexdigest()
    portable_handoff.write_text(json.dumps(payload))

    evaluation, image_root = resolve_portable_evaluation_input(portable_handoff, "image_and_text")

    assert evaluation == suffixed / "synthetic_eval/image_and_text"
    assert image_root == suffixed


def test_modified_referenced_evaluation_image_fails_before_model_call(
    portable_handoff: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nemotron.recipes.embed.stage3_eval import eval as evaluation

    bundle = portable_handoff.parent / "generated.bundle"
    asset = next((bundle / "assets/units").iterdir())
    corpus = bundle / "synthetic_eval/image_and_text/corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "p1", "text": "source", "image_path": asset.relative_to(bundle).as_posix()}) + "\n"
    )
    manifest_path = bundle / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    corpus_record = next(
        item for item in manifest["artifacts"] if item["path"] == "synthetic_eval/image_and_text/corpus.jsonl"
    )
    corpus_record["sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    corpus_record["bytes"] = corpus.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    handoff = json.loads(portable_handoff.read_text())
    handoff["portable_bundle_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    portable_handoff.write_text(json.dumps(handoff))
    asset.write_bytes(b"changed")

    evaluate_model = Mock()
    monkeypatch.setattr(evaluation, "evaluate_model", evaluate_model)
    cfg = evaluation.EvalConfig(
        sdg_input_path=portable_handoff,
        retrieval_view="image_and_text",
        eval_data_path=tmp_path / "placeholder",
        output_dir=tmp_path / "eval",
        eval_base=True,
        eval_finetuned=False,
        eval_nim=False,
    )

    with pytest.raises(ValueError, match="Size mismatch for assets/units/"):
        evaluation.run_eval(cfg)
    evaluate_model.assert_not_called()


def test_portable_handoff_cannot_fall_back_to_legacy_conversion(portable_handoff: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_view"):
        resolve_generation_input(portable_handoff)


def test_modified_corpus_fails_before_mining(portable_handoff: Path) -> None:
    corpus = portable_handoff.parent / "generated.bundle/views/image/corpus/shared/part-00000.parquet"
    corpus.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Size mismatch"):
        resolve_portable_training_input(portable_handoff, "image")


def test_modified_manifest_fails_before_mining(portable_handoff: Path) -> None:
    manifest = portable_handoff.parent / "generated.bundle/run_manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        resolve_portable_training_input(portable_handoff, "text")


def test_portable_view_flows_to_mining_without_conversion(
    portable_handoff: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nemotron.recipes.embed.stage1_data_prep import data_prep

    calls = []
    train = resolve_portable_training_input(portable_handoff, "image_and_text")
    monkeypatch.setattr(data_prep, "run_convert", unexpected_conversion)
    monkeypatch.setattr(data_prep, "run_mining", lambda cfg, path: calls.append(path))
    monkeypatch.setattr(data_prep, "run_unroll", lambda cfg: train)
    result = run_data_prep(
        DataPrepConfig(sdg_input_path=portable_handoff, retrieval_view="image_and_text", output_dir=tmp_path / "prep")
    )
    assert result == train
    assert calls == [train]
    output = capsys.readouterr().out
    assert str(train.parents[2] / "synthetic_eval/image_and_text") in output
    assert str(tmp_path / "prep/eval_beir") not in output


def unexpected_conversion(*args: object, **kwargs: object) -> None:
    """Reject accidental fallback to the legacy conversion path."""
    pytest.fail("Portable views must not be passed through legacy conversion")


def test_empty_training_view_is_not_replaced_with_another_view(tmp_path: Path) -> None:
    from data_designer_retrieval_sdg import SplitRatios, export_retrieval_data

    source = write_generated_fixture(tmp_path / "output")
    bundle = source.parent / "bundle"
    export_retrieval_data(source, bundle, dataset_id="test", ratios=SplitRatios(train=0, evaluation=1))
    manifest = write_generation_manifest(
        output_dir=source.parent, output_path=source, dataset_name="test", portable_bundle=bundle
    )
    with pytest.raises(ValueError, match="no accepted training examples"):
        resolve_portable_training_input(manifest, "text")


def test_partial_generation_does_not_publish_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemotron.recipes.embed.stage0_sdg import plugin_adapter

    source = write_generated_fixture(tmp_path / "output")
    sources = tmp_path / "sources.jsonl"
    sources.write_text('{"unit_id":"p1","document_id":"doc","text":"source"}')
    result = SimpleNamespace(output_path=source, dataset_name="test", num_records=1, requested_num_records=2)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(plugin_adapter, "execute_generation", lambda *_args: result)
    with pytest.raises(ValueError, match="Incomplete generation"):
        run_sdg(SDGConfig(sources_file=sources, output_dir=source.parent, portable_export=True))
    assert not (source.parent / "generation_result.json").exists()
    assert not (source.parent / "test.bundle").exists()


def test_portable_config_rejects_conflicting_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_view"):
        DataPrepConfig(sdg_input_path=None, train_input_file=tmp_path / "train.json", retrieval_view="image")
    with pytest.raises(ValueError, match="fractions"):
        SDGConfig(export_train_ratio=0.8, export_validation_ratio=0.3)
    with pytest.raises(ValueError, match="strict_visual"):
        SDGConfig(strict_visual=True)
