"""Cross-repository portable export tests with explicitly synthetic judge fixtures."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path
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


def export_fixture(config) -> Path:
    """Export authored multimodal judgments through the actual pinned producer."""
    from data_designer_retrieval_sdg.multimodal.export import export_multimodal_bundle
    from data_designer_retrieval_sdg.multimodal.models import Candidate
    from data_designer_retrieval_sdg.multimodal.source_models import RetrievalSource

    root = config.output_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    image = root / "page.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    sources = [RetrievalSource(unit_id="p1", document_id="doc", text="Store parts at 20 C.", images=[str(image)])]
    candidates = [
        Candidate.model_validate(
            {
                "query_id": f"q{index}",
                "context_id": "page-context",
                "query": query,
                "source_unit_ids": ["p1"],
                "language": "en",
                "requested_instruction": {"name": modality, "instruction": "Ask about this source."},
                "query_judgment": {
                    "self_sufficiency": 5,
                    "has_answer": False,
                    "observed_type": "factual",
                    "observed_format": "question",
                    "reasoning": "Authored fixture.",
                },
                "relevance_judgment": {"relevance": 5, "reasoning": "Authored fixture."},
                "localization": {
                    "supports": [
                        {
                            "unit_id": "p1",
                            "grade": 2,
                            "modality": modality,
                            "quote": "Store parts at 20 C." if modality == "text" else "",
                            "visual_evidence": "Storage chart" if modality != "text" else "",
                            "contribution": "Direct support.",
                        }
                    ]
                },
                "accepted": True,
                "rejection_reasons": [],
                "evidence_modality": modality,
            }
        )
        for index, (modality, query) in enumerate(
            [
                ("text", "What temperature is specified for part storage?"),
                ("image", "Which storage zone is highlighted in the chart?"),
            ]
        )
    ]
    return export_multimodal_bundle(config.output_dir / "bundle", sources, candidates, [], config)


@pytest.fixture
def portable_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Path:
    """Publish a real bundle and recipe handoff; only inference is substituted."""
    import data_designer_retrieval_sdg.multimodal as producer

    output = tmp_path / "stage0"
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(producer, "run_multimodal_sdg", export_fixture)
    run_sdg(
        SDGConfig(
            sdg_workflow="multimodal",
            sources_file=tmp_path / "sources.jsonl",
            output_dir=output,
            export_train_ratio=getattr(request, "param", 1),
        )
    )
    return output / "generation_result.json"


@pytest.mark.parametrize("view", ["text", "image", "image_and_text", "mixed"])
def test_exported_view_survives_relocation(portable_handoff: Path, tmp_path: Path, view: str) -> None:
    copied = tmp_path / "relocated"
    shutil.copytree(portable_handoff.parent, copied)
    train = resolve_portable_training_input(copied / portable_handoff.name, view)
    assert train == copied / "multimodal/bundle" / "views" / view / "train.json"
    assert len(json.loads(train.read_text())["data"]) == (2 if view in {"image_and_text", "mixed"} else 1)


@pytest.mark.parametrize("portable_handoff", [0], indirect=True)
def test_evaluation_follows_suffixed_bundle_in_latest_manifest(portable_handoff: Path) -> None:
    original = portable_handoff.parent / "multimodal/bundle"
    suffixed = portable_handoff.parent / "multimodal/bundle-1"
    original.rename(suffixed)
    payload = json.loads(portable_handoff.read_text())
    bundle_manifest = suffixed / "run_manifest.json"
    payload["output_path"] = "multimodal/bundle-1/source_records.jsonl"
    payload["portable_bundle_manifest"] = "multimodal/bundle-1/run_manifest.json"
    payload["portable_bundle_sha256"] = hashlib.sha256(bundle_manifest.read_bytes()).hexdigest()
    portable_handoff.write_text(json.dumps(payload))

    evaluation, image_root = resolve_portable_evaluation_input(portable_handoff, "image_and_text")

    assert evaluation == suffixed / "synthetic_eval/image_and_text"
    assert image_root == suffixed


def test_explicit_audit_rejects_modified_referenced_evaluation_image(
    portable_handoff: Path,
) -> None:
    from nemotron.recipes.embed.stage0_sdg.validate import validate_bundle

    bundle = portable_handoff.parent / "multimodal/bundle"
    asset = next((bundle / "assets").iterdir())
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

    with pytest.raises(ValueError, match="Size mismatch for assets/"):
        validate_bundle(manifest_path, view="image_and_text")


def test_portable_handoff_cannot_fall_back_to_legacy_conversion(portable_handoff: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_view"):
        resolve_generation_input(portable_handoff)


def test_modified_corpus_fails_before_mining(portable_handoff: Path) -> None:
    corpus = portable_handoff.parent / "multimodal/bundle/views/image/corpus/shared/part-00000.parquet"
    corpus.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Unreadable Parquet"):
        resolve_portable_training_input(portable_handoff, "image")


def test_modified_manifest_fails_before_mining(portable_handoff: Path) -> None:
    manifest = portable_handoff.parent / "multimodal/bundle/run_manifest.json"
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
    from nemotron.recipes.embed.stage0_sdg.plugin_adapter import build_multimodal_config

    cfg = SDGConfig(
        sdg_workflow="multimodal",
        sources_file=tmp_path / "sources.jsonl",
        output_dir=tmp_path / "output",
        export_train_ratio=0,
    )
    handoff = export_fixture(build_multimodal_config(cfg))
    manifest = write_generation_manifest(
        output_dir=cfg.output_dir,
        output_path=handoff.parent / "source_records.jsonl",
        dataset_name="test",
        portable_bundle=handoff.parent,
    )
    with pytest.raises(ValueError, match="no accepted training examples"):
        resolve_portable_training_input(manifest, "text")


def test_failed_generation_does_not_publish_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import data_designer_retrieval_sdg.multimodal as producer

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(producer, "run_multimodal_sdg", Mock(side_effect=RuntimeError("Incomplete generation")))
    with pytest.raises(RuntimeError, match="Incomplete generation"):
        run_sdg(
            SDGConfig(
                sdg_workflow="multimodal",
                sources_file=tmp_path / "sources.jsonl",
                output_dir=tmp_path / "output",
            )
        )
    assert not (tmp_path / "output/generation_result.json").exists()


def test_portable_config_rejects_conflicting_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_view"):
        DataPrepConfig(sdg_input_path=None, train_input_file=tmp_path / "train.json", retrieval_view="image")
    with pytest.raises(ValueError, match="fractions"):
        SDGConfig(export_train_ratio=0.8, export_validation_ratio=0.3)
