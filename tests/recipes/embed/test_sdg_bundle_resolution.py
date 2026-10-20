# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check the consumer boundary separately from the optional export audit."""

import json
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemotron.recipes.embed.sdg_manifest import (
    resolve_bundle_view,
    resolve_portable_evaluation_input,
    resolve_portable_training_input,
    write_generation_manifest,
)
from nemotron.recipes.embed.stage0_sdg.validate import validate_bundle
from tests.recipes.embed.test_sdg_bundle_validation import _bundle, _json


@pytest.mark.parametrize("view", ["text", "image", "image_and_text", "mixed"])
def test_selected_view_keeps_query_splits_corpus_and_qrels(tmp_path, view):
    manifest = _bundle(tmp_path / "bundle", view)
    original = {path: path.read_bytes() for path in manifest.parent.rglob("*") if path.is_file()}
    paths = resolve_bundle_view(manifest, view=view)
    handoff = write_generation_manifest(
        output_dir=tmp_path, output_path=paths.train, dataset_name="fixture", portable_bundle=paths.root
    )
    assert resolve_portable_training_input(handoff, view) == paths.train
    assert resolve_portable_evaluation_input(handoff, view) == (paths.synthetic_eval, paths.root)
    assert json.loads(paths.train.read_text())["corpus"] == {"path": "corpus/shared"}
    assert json.loads(paths.validation.read_text())["corpus"] == {"path": "corpus/shared"}
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("view", ["text", "image", "image_and_text", "mixed"])
def test_normal_handoff_does_not_repeat_full_export_audit(tmp_path, view):
    manifest = _bundle(tmp_path, view)
    paths = resolve_bundle_view(manifest, view=view)
    handoff = write_generation_manifest(
        output_dir=tmp_path / "handoff", output_path=paths.train, dataset_name="fixture", portable_bundle=paths.root
    )
    # Source diagnostics are not consumed by mining/training/evaluation.
    (tmp_path / "source_records.jsonl").write_text("changed source diagnostics\n")
    assert resolve_portable_training_input(handoff, view) == paths.train
    assert resolve_portable_evaluation_input(handoff, view) == (paths.synthetic_eval, paths.root)
    with pytest.raises(ValueError, match="mismatch"):
        validate_bundle(manifest, view=view)


@pytest.mark.parametrize(
    "relative",
    [
        "views/image/train.json",
        "views/image/validation.json",
        "views/image/corpus/shared/merlin_metadata.json",
        "views/image/corpus/shared/part-00000.parquet",
        "synthetic_eval/image/queries.jsonl",
        "synthetic_eval/image/corpus.jsonl",
        "synthetic_eval/image/qrels/test.tsv",
    ],
)
def test_missing_selected_artifacts_fail_at_handoff(tmp_path, relative):
    manifest = _bundle(tmp_path, "image")
    (tmp_path / relative).unlink()
    with pytest.raises(ValueError, match="missing required"):
        resolve_bundle_view(manifest, view="image")


@pytest.mark.parametrize(
    "relative",
    [
        "views/image/train.json",
        "views/image/corpus/shared",
        "views/image/corpus/shared/merlin_metadata.json",
        "views/image/corpus/shared/part-00000.parquet",
        "synthetic_eval/image",
        "synthetic_eval/image/qrels/test.tsv",
    ],
)
def test_selected_artifacts_cannot_escape_through_symlinks(tmp_path, relative):
    manifest = _bundle(tmp_path / "bundle", "image")
    path = manifest.parent / relative
    outside = tmp_path / "outside"
    shutil.move(path, outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(ValueError, match="escapes"):
        resolve_bundle_view(manifest, view="image")


@pytest.mark.parametrize("corpus_path", ["../outside", "/tmp/outside", "corpus/other"])
def test_training_corpus_reference_must_use_selected_shared_corpus(tmp_path, corpus_path):
    manifest = _bundle(tmp_path, "text")
    train = tmp_path / "views/text/train.json"
    payload = json.loads(train.read_text())
    payload["corpus"]["path"] = corpus_path
    _json(train, payload)
    with pytest.raises(ValueError, match="corpus reference"):
        resolve_bundle_view(manifest, view="text")


def test_unavailable_view_never_falls_back(tmp_path):
    manifest = _bundle(tmp_path, "image")
    payload = json.loads(manifest.read_text())
    payload["counts"] = {"views": {"image": {}}}
    _json(manifest, payload)
    with pytest.raises(ValueError, match="'text' is unavailable"):
        resolve_bundle_view(manifest, view="text")


@pytest.mark.parametrize("mutation", ["metadata", "schema", "json"])
def test_unloadable_selected_view_fails_at_handoff(tmp_path, mutation):
    manifest = _bundle(tmp_path, "text")
    corpus = tmp_path / "views/text/corpus/shared"
    if mutation == "metadata":
        _json(
            corpus / "merlin_metadata.json",
            {"class": "UnexpectedDataset", "format": "parquet", "corpus_id": "fixture"},
        )
    elif mutation == "schema":
        pq.write_table(pa.table({"wrong_column": ["text"]}), corpus / "part-00000.parquet")
    else:
        (tmp_path / "views/text/train.json").write_text("[]")
    with pytest.raises(ValueError, match="metadata|columns|JSON object"):
        resolve_bundle_view(manifest, view="text")


def test_validation_command_reports_success_and_corruption(tmp_path):
    manifest = _bundle(tmp_path, "text")
    command = [sys.executable, "-m", "nemotron.recipes.embed.stage0_sdg.validate", str(manifest), "--view", "text"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Validated text:" in result.stdout
    (tmp_path / "source_records.jsonl").write_text("changed\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Validation failed:" in result.stderr


@pytest.mark.parametrize("field,value", [("counts", []), ("views", []), ("corpus_id", 7), ("corpus_id", " ")])
def test_malformed_view_metadata_fails_clearly(tmp_path, field, value):
    manifest = _bundle(tmp_path, "text")
    path = tmp_path / "views/text/corpus/shared/merlin_metadata.json" if field == "corpus_id" else manifest
    payload = json.loads(path.read_text())
    if field == "views":
        payload["counts"] = {"views": value}
    else:
        payload[field] = value
    _json(path, payload)
    with pytest.raises(ValueError, match="object|metadata"):
        resolve_bundle_view(manifest, view="text")


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 99),
        ("split_protocol", "document_disjoint"),
        ("corpus_scope", "partition"),
        ("status", "failed"),
        ("portable", False),
        ("negative_mining_performed", True),
        ("relevance_judgements_complete", True),
        ("gold_evaluation_status", "available"),
    ],
)
def test_incompatible_export_contract_is_rejected_by_consumer(tmp_path, key, value):
    manifest = _bundle(tmp_path, "text")
    payload = json.loads(manifest.read_text())
    payload[key] = value
    _json(manifest, payload)
    with pytest.raises(ValueError):
        resolve_bundle_view(manifest, view="text")
