# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Portable retrieval bundle integrity and consumer compatibility."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemotron.recipes.embed.stage0_sdg import validate as bundle_validation


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _inventory(root: Path) -> Path:
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    ]
    manifest = root / "run_manifest.json"
    _json(
        manifest,
        {
            "schema_version": 2,
            "dataset_id": "fixture",
            "artifacts": artifacts,
            "split_protocol": "grouped_query_disjoint",
            "corpus_scope": "full_collection",
        },
    )
    return manifest


def _document_bundle(root: Path, view: str = "image_and_text") -> Path:
    buffer = BytesIO()
    Image.new("RGB", (2, 3), "blue").save(buffer, format="PNG")
    units = []
    assignments = {}
    for split in ("train", "validation", "evaluation"):
        identifier = f"page-{split}"
        document = f"doc-{split}"
        image_path = f"assets/{split}.png"
        (root / "assets").mkdir(parents=True, exist_ok=True)
        (root / image_path).write_bytes(buffer.getvalue())
        unit = {
            "unit_id": identifier,
            "document_id": document,
            "split": split,
            "text": "Texte source français",
            "images": [image_path],
        }
        units.append(unit)
        assignments[document] = split
        corpus_path = root / f"views/{view}/corpus/{split}"
        corpus_path.mkdir(parents=True)
        if view == "text":
            row = {"id": identifier, "text": unit["text"]}
        elif view == "image":
            row = {"image_filename": identifier, "image": buffer.getvalue()}
        else:
            row = {"docid": identifier, "text": unit["text"], "image": buffer.getvalue()}
        pq.write_table(pa.Table.from_pylist([row]), corpus_path / "part-00000.parquet")
        _json(
            corpus_path / "merlin_metadata.json",
            {
                "class": {
                    "text": "TextQADataset",
                    "image": "ColPaliDataset",
                    "image_and_text": "WikiSSNQDataset",
                    "mixed": "WikiSSNQDataset",
                }[view],
                "corpus_id": "fixture",
                "view": view,
                "format": "parquet",
                "schema_version": 1,
            },
        )
        if split != "evaluation":
            record = {
                "question_id": f"q-{split}",
                "question": "Quel traitement réduit le risque ?",
                "corpus_id": "fixture",
                "pos_doc": [{"id": identifier}],
                "neg_doc": [],
                "negative_scores": [],
                "negative_mining_performed": False,
                "relevance_judgements_complete": False,
                "unlisted_document_disposition": "unjudged",
                "positive_unit_ids": [identifier],
                "source_document_ids": [document],
                "language": "fr",
                "evidence": "Generated explanation, not source text",
                "evidence_modality": "text_only" if view == "text" else "image_grounded",
            }
            _json(root / f"views/{view}/{split}.json", {"corpus": {"path": f"corpus/{split}"}, "data": [record]})
    _jsonl(root / "retrieval_units.jsonl", units)
    _json(root / "split_manifest.json", {"schema_version": 1, "document_disjoint": True, "assignments": assignments})
    _jsonl(root / "source_records.jsonl", [])
    _jsonl(root / "candidate_diagnostics.jsonl", [])
    eval_root = root / f"synthetic_eval/{view}"
    _jsonl(eval_root / "queries.jsonl", [{"_id": "q-evaluation", "text": "Quels risques ?"}])
    document = {"_id": "page-evaluation", "metadata": {"source_document_id": "doc-evaluation"}}
    if view != "image":
        document["text"] = "Texte source français"
    if view != "text":
        document["image_path"] = "assets/evaluation.png"
    _jsonl(eval_root / "corpus.jsonl", [document])
    (eval_root / "qrels").mkdir()
    (eval_root / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq-evaluation\tpage-evaluation\t1\n")
    return _inventory(root)


def _bundle(root: Path, view: str = "image_and_text") -> Path:
    from tests.recipes.embed.test_sdg_query_groups import _grouped

    return _grouped(root, view, validation=True)


@pytest.mark.parametrize(
    "view,producer_view", [("text", "text"), ("image", "image"), ("image_and_text", "image_and_text")]
)
def test_unified_bundle_resolves_view_without_modifying_producer_manifest(tmp_path, view, producer_view):
    manifest = _bundle(tmp_path, producer_view)
    original = manifest.read_bytes()
    paths = bundle_validation.validate_bundle(manifest, view=view)
    assert paths.train == tmp_path / f"views/{producer_view}/train.json"
    assert paths.view == view
    assert manifest.read_bytes() == original


@pytest.mark.parametrize(
    "available,requested,producer_view", [("image", "text", "text"), ("text", "image_and_text", "image_and_text")]
)
def test_unified_bundle_reports_unavailable_view(tmp_path, available, requested, producer_view):
    manifest = _bundle(tmp_path, available)
    payload = json.loads(manifest.read_text())
    payload["counts"] = {"views": {available: {}, "future_view": {}}}
    _json(manifest, payload)
    original = manifest.read_bytes()
    assert bundle_validation.validate_bundle(manifest, view=available).view == available

    with pytest.raises(ValueError) as error:
        bundle_validation.validate_bundle(manifest, view=requested)

    assert str(error.value) == (
        f"Requested retrieval view '{producer_view}' is unavailable in this bundle. "
        f"Available views: {available}. Select an available retrieval_view or re-export the requested view."
    )
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("requested", ["image", "text"])
def test_unified_bundle_checks_integrity_before_view_availability(tmp_path, requested):
    manifest = _bundle(tmp_path, "image")
    payload = json.loads(manifest.read_text())
    payload["counts"] = {"views": {"image": {}}}
    _json(manifest, payload)
    (tmp_path / "assets/train.png").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="Size mismatch for assets/train.png"):
        bundle_validation.validate_bundle(manifest, view=requested)


def test_unified_bundle_declared_view_still_requires_corpus_shards(tmp_path):
    manifest = _bundle(tmp_path, "image")
    payload = json.loads(manifest.read_text())
    payload["counts"] = {"views": {"image": {}}}
    payload["artifacts"] = [
        item for item in payload["artifacts"] if item["path"] != "views/image/corpus/shared/part-00000.parquet"
    ]
    _json(manifest, payload)

    with pytest.raises(ValueError, match="Manifest missing required corpus shard: views/image/corpus/shared/"):
        bundle_validation.validate_bundle(manifest, view="image")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "missing"),
        ("bytes", "[Ss]ize"),
        ("hash", "[Cc]hecksum"),
        ("traversal", "[Pp]ath|escapes"),
        ("symlink", "escapes"),
        ("duplicate-unit", "[Dd]uplicate"),
        ("duplicate-query", "[Dd]uplicate"),
        ("dangling-positive", "positive|unknown"),
        ("dangling-qrel", "qrel"),
        ("document-split", "split|partition"),
        ("missing-asset", "artifact|asset"),
        ("undeclared-shard", "[Uu]nlisted|[Uu]ndeclared"),
        ("evidence-as-text", "source text"),
    ],
)
def test_unified_bundle_rejects_corruption_and_semantic_drift(tmp_path, mutation, match):
    root = tmp_path / "bundle"
    manifest = _bundle(root)
    train = root / "views/image_and_text/train.json"
    units = [json.loads(line) for line in (root / "retrieval_units.jsonl").read_text().splitlines()]
    if mutation == "missing":
        train.unlink()
    elif mutation == "hash":
        train.write_bytes(train.read_bytes().replace(b"q-train", b"z-train"))
    elif mutation == "bytes":
        train.write_text(train.read_text() + " ")
    elif mutation in {"traversal", "symlink", "missing-asset"}:
        if mutation == "traversal":
            units[0]["images"] = ["../outside.png"]
        elif mutation == "missing-asset":
            units[0]["images"] = ["assets/unlisted.png"]
        else:
            target = tmp_path / "outside.png"
            target.write_bytes((root / "assets/train.png").read_bytes())
            (root / "assets/train.png").unlink()
            (root / "assets/train.png").symlink_to(target)
        _jsonl(root / "retrieval_units.jsonl", units)
        _inventory(root)
    elif mutation in {"duplicate-unit", "document-split"}:
        if mutation == "duplicate-unit":
            units.append(units[0])
        else:
            units[1]["split"] = "evaluation"
        _jsonl(root / "retrieval_units.jsonl", units)
        _inventory(root)
    elif mutation in {"duplicate-query", "dangling-positive"}:
        data = json.loads(train.read_text())
        if mutation == "duplicate-query":
            data["data"].append(data["data"][0])
        else:
            data["data"][0]["pos_doc"] = [{"id": "unknown"}]
        _json(train, data)
        _inventory(root)
    elif mutation == "dangling-qrel":
        (root / "synthetic_eval/image_and_text/qrels/test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq-evaluation\tunknown\t1\n"
        )
        _inventory(root)
    else:
        corpus = root / "views/image_and_text/corpus/shared"
        shard = corpus / ("extra.parquet" if mutation == "undeclared-shard" else "part-00000.parquet")
        table = pq.read_table(corpus / "part-00000.parquet")
        if mutation == "evidence-as-text":
            row = table.to_pylist()[0]
            row["text"] = "Generated explanation, not source text"
            table = pa.Table.from_pylist([row])
        pq.write_table(table, shard)
        if mutation != "undeclared-shard":
            _inventory(root)
    with pytest.raises(ValueError, match=match):
        bundle_validation.validate_bundle(manifest)


@pytest.mark.parametrize("view", ["text", "image", "image_and_text"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("class", None),
        ("class", "UnknownDataset"),
        ("format", None),
        ("format", "json"),
        ("corpus_id", None),
        ("corpus_id", ""),
        ("corpus_id", 42),
    ],
)
def test_unified_bundle_rejects_unloadable_corpus_metadata(tmp_path, view, field, value):
    manifest = _bundle(tmp_path, view)
    path = tmp_path / f"views/{view}/corpus/shared/merlin_metadata.json"
    metadata = json.loads(path.read_text())
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    _json(path, metadata)
    _inventory(tmp_path)
    with pytest.raises(ValueError, match="[Mm]etadata|corpus"):
        bundle_validation.validate_bundle(manifest, view=view)


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("corpus_id", None),
        ("corpus_id", "another-corpus"),
        ("question", None),
        ("question", ""),
        ("question", "  "),
        ("question", 42),
    ],
)
def test_unified_bundle_rejects_unloadable_training_queries(tmp_path, split, field, value):
    manifest = _bundle(tmp_path)
    path = tmp_path / f"views/image_and_text/{split}.json"
    payload = json.loads(path.read_text())
    if value is None:
        del payload["data"][0][field]
    else:
        payload["data"][0][field] = value
    _json(path, payload)
    _inventory(tmp_path)
    with pytest.raises(ValueError, match="corpus|[Qq]uery|question"):
        bundle_validation.validate_bundle(manifest)


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "failed"),
        ("portable", False),
        ("negative_mining_performed", True),
        ("relevance_judgements_complete", True),
        ("gold_evaluation_status", "available"),
    ],
)
def test_contradictory_completion_and_label_claims_are_rejected(tmp_path, key, value):
    manifest = _bundle(tmp_path)
    payload = json.loads(manifest.read_text())
    payload[key] = value
    _json(manifest, payload)
    with pytest.raises(ValueError):
        bundle_validation.validate_bundle(manifest)


@pytest.mark.parametrize("version", [3, 99])
def test_unsupported_version_and_missing_view_are_not_fabricated(tmp_path, version):
    manifest = _bundle(tmp_path, "text")
    with pytest.raises(ValueError, match="missing|required"):
        bundle_validation.validate_bundle(manifest, view="image")
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = version
    _json(manifest, payload)
    with pytest.raises(ValueError, match="schema_version"):
        bundle_validation.validate_bundle(manifest)


class _MiningClient:
    """Deterministic CPU transport substitute; never calls an endpoint."""

    document_inputs: list = []

    def __init__(self, **kwargs):
        pass

    def encode_queries(self, queries):
        return np.array([[1.0, 0.0] for _ in queries], dtype=np.float32)

    def _encode_batch(self, documents, *, input_type):
        start = len(self.document_inputs)
        self.document_inputs.extend(documents)
        return [[1.0, 0.0] if start + index == 0 else [0.2, 0.8] for index, _ in enumerate(documents)]


@pytest.mark.parametrize(
    "view,producer_view,use_images",
    [
        ("text", "text", False),
        ("image", "image", True),
        ("image_and_text", "image_and_text", True),
    ],
)
@pytest.mark.parametrize("use_text_in_document", [True, False])
def test_portable_view_reaches_actual_recipe_miner(
    tmp_path, monkeypatch, view, producer_view, use_images, use_text_in_document
):
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    manifest = _bundle(tmp_path / "bundle", producer_view)
    corpus_path = manifest.parent / f"views/{producer_view}/corpus/shared/part-00000.parquet"
    rows = pq.read_table(corpus_path).to_pylist()
    id_key = {"text": "id", "image": "image_filename", "image_and_text": "docid"}[view]
    distractor = dict(rows[0], **{id_key: "page-distractor"})
    pq.write_table(pa.Table.from_pylist([*rows, distractor]), corpus_path)
    units_path = manifest.parent / "retrieval_units.jsonl"
    units = [json.loads(line) for line in units_path.read_text().splitlines()]
    units.append(dict(units[0], unit_id="page-distractor"))
    _jsonl(units_path, units)
    eval_path = manifest.parent / f"synthetic_eval/{producer_view}/corpus.jsonl"
    eval_rows = [json.loads(line) for line in eval_path.read_text().splitlines()]
    eval_rows.append(dict(eval_rows[0], _id="page-distractor"))
    _jsonl(eval_path, eval_rows)
    _inventory(manifest.parent)
    train = bundle_validation.validate_bundle(manifest, view=view).train
    training = json.loads(train.read_text())
    training["data"][0]["pos_doc"][0]["score"] = 2
    _json(train, training)
    _MiningClient.document_inputs = []
    monkeypatch.setattr(eval_module, "NIMEmbeddingModel", _MiningClient)
    from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig, run_mining

    result = run_mining(
        DataPrepConfig(
            sdg_input_path=None,
            train_input_file=train,
            output_dir=tmp_path / "mined",
            mining_backend="vllm",
            mining_api_url="http://unused",
            base_model="fixture",
            mining_batch_size=2,
            hard_negatives_to_mine=1,
            hard_neg_margin=0.95,
            mining_use_images=use_images,
            use_text_in_document=use_text_in_document,
        ),
        train,
    )
    row = json.loads(result.read_text())["data"][0]
    assert row["neg_doc"][0]["id"] != row["pos_doc"][0]["id"]
    assert row["negative_mining_performed"] is True
    assert row["pos_doc"][0]["source_score"] == 2
    assert row["pos_doc"][0]["score"] != 2
    assert row["unlisted_document_disposition"] == "unjudged"
    assert row["relevance_judgements_complete"] is False
    document = _MiningClient.document_inputs[0]
    assert isinstance(document, dict if use_images else str)
    if use_images:
        content_types = [part["type"] for part in document["content"]]
        assert "image_url" in content_types
        assert ("text" in content_types) == (use_text_in_document and view == "image_and_text")
    else:
        assert document == "Texte source français"  # Text-only documents remain encodable.


@pytest.mark.parametrize("view", ["mixed", "image_and_text", "image"])
@pytest.mark.parametrize("mutation", [None, "invented-corpus-image", "invented-evaluation-image"])
def test_mixed_view_absent_images_match_original_sources(tmp_path, view, mutation):
    """Only mixed accepts absent pixels, without inventing source image evidence."""
    root = tmp_path / "bundle"
    manifest = _bundle(root, view)
    units = [json.loads(line) for line in (root / "retrieval_units.jsonl").read_text().splitlines()]
    identifier = units[0]["unit_id"]
    units[0]["images"] = []
    units[1]["text"] = ""
    _jsonl(root / "retrieval_units.jsonl", units)
    shard = root / f"views/{view}/corpus/shared/part-00000.parquet"
    table = pq.read_table(shard)
    rows = table.to_pylist()
    id_key = "image_filename" if view == "image" else "docid"
    for row in rows:
        if row[id_key] == units[1]["unit_id"] and "text" in row:
            row["text"] = ""
        if row[id_key] == identifier and mutation != "invented-corpus-image":
            row["image"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)
    eval_path = root / f"synthetic_eval/{view}/corpus.jsonl"
    documents = [json.loads(line) for line in eval_path.read_text().splitlines()]
    for row in documents:
        if row["_id"] == units[1]["unit_id"] and "text" in row:
            row["text"] = ""
        if row["_id"] == identifier and mutation != "invented-evaluation-image":
            row.pop("image_path", None)
    _jsonl(eval_path, documents)
    qrels = (root / f"synthetic_eval/{view}/qrels/test.tsv").read_bytes()
    manifest = _inventory(root)
    if mutation or view != "mixed":
        with pytest.raises(ValueError, match="image"):
            bundle_validation.validate_bundle(manifest, view=view)
    else:
        paths = bundle_validation.validate_bundle(manifest, view="mixed")
        assert (paths.synthetic_eval / "qrels/test.tsv").read_bytes() == qrels
