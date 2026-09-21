# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Boundary obligations from specs/retrieval_bundle_compatibility.allium."""

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

from nemotron.recipes import retrieval_vl


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
                "class": {"text": "TextQADataset", "image": "ColPaliDataset", "image_and_text": "WikiSSNQDataset"}[
                    view
                ],
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
    from tests.recipes.test_grouped_query_bundle import _grouped

    return _grouped(root, view, validation=True)


@pytest.mark.parametrize(
    "view,producer_view", [("text", "text"), ("image", "image"), ("text_image", "image_and_text")]
)
def test_unified_bundle_explicit_view_mapping_and_producer_preservation(tmp_path, view, producer_view):
    manifest = _bundle(tmp_path, producer_view)
    original = manifest.read_bytes()
    paths = retrieval_vl.inspect_vl_bundle(manifest, view=view)
    assert paths.train == tmp_path / f"views/{producer_view}/train.json"
    assert paths.view == view
    assert paths.gold_eval is None
    assert paths.local_artifacts_verified is True
    assert manifest.read_bytes() == original


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
    with pytest.raises(retrieval_vl.RetrievalVLBundleError, match=match):
        retrieval_vl.inspect_vl_bundle(manifest)


def test_structural_only_inspection_never_attests_local_integrity(tmp_path):
    manifest = _bundle(tmp_path)
    (tmp_path / "assets/train.png").unlink()
    paths = retrieval_vl.inspect_vl_bundle(manifest, require_local_artifacts=False)
    assert paths.local_artifacts_verified is False


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
    recipe_view = "text_image" if view == "image_and_text" else view
    with pytest.raises(retrieval_vl.RetrievalVLBundleError, match="[Mm]etadata|corpus"):
        retrieval_vl.inspect_vl_bundle(manifest, view=recipe_view)


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
    with pytest.raises(retrieval_vl.RetrievalVLBundleError, match="corpus|[Qq]uery|question"):
        retrieval_vl.inspect_vl_bundle(manifest)


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
    with pytest.raises(retrieval_vl.RetrievalVLBundleError):
        retrieval_vl.inspect_vl_bundle(manifest)


def test_preparation_preserves_source_and_partial_relevance_without_overwriting(tmp_path):
    manifest = _bundle(tmp_path / "bundle")
    original = manifest.read_bytes()
    output = tmp_path / "prepared"
    train = retrieval_vl.prepare_vl_training_data(manifest, output_dir=output)
    data = json.loads(train.read_text())
    record = data["data"][0]
    assert record["language"] == "fr"
    assert record["evidence"] == "Generated explanation, not source text"
    assert record["negative_mining_performed"] is False
    assert record["relevance_judgements_complete"] is False
    assert record["unlisted_document_disposition"] == "unjudged"
    assert record["neg_doc"] == []
    corpus = pq.read_table(sorted((train.parent / data["corpus"]["path"]).glob("*.parquet"))).to_pylist()
    assert corpus[0]["text"] == "Texte source français"
    assert corpus[0]["image"]
    assert data["source_bundle"]["schema_version"] == 2
    assert manifest.read_bytes() == original
    with pytest.raises(FileExistsError):
        retrieval_vl.prepare_vl_training_data(manifest, output_dir=output)


def test_unsupported_version_and_missing_view_are_not_fabricated(tmp_path):
    manifest = _bundle(tmp_path, "text")
    with pytest.raises(retrieval_vl.RetrievalVLBundleError, match="missing|required"):
        retrieval_vl.inspect_vl_bundle(manifest, view="image")
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = 99
    _json(manifest, payload)
    with pytest.raises(retrieval_vl.RetrievalVLBundleError, match="schema_version"):
        retrieval_vl.inspect_vl_bundle(manifest)


def test_legacy_checksum_mode_includes_corpus_artifacts(tmp_path):
    manifest = _document_bundle(tmp_path, "text")
    payload = json.loads(manifest.read_text())
    payload.update(
        schema_version=3,
        status="completed",
        portable=True,
        consumer_contract_identity=retrieval_vl.EXPECTED_CONSUMER_CONTRACT,
        consumer_contract_verified=True,
        relevance_annotation_scope="known_positives_only",
        negative_mining_performed=False,
        unlisted_document_disposition="unjudged",
        relevance_judgements_complete=False,
    )
    payload["artifacts"] = [
        {"relative_path": item["path"], "checksum": item["sha256"], "size_bytes": item["bytes"]}
        for item in payload["artifacts"]
    ]
    _json(manifest, payload)
    assert retrieval_vl.inspect_vl_bundle(manifest, view="text", verify_checksums=True).gold_eval is None
    shard = tmp_path / "views/text/corpus/train/part-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(retrieval_vl.RetrievalVLBundleError, match="Checksum"):
        retrieval_vl.inspect_vl_bundle(manifest, view="text", verify_checksums=True)


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
        ("text_image", "image_and_text", True),
    ],
)
def test_prepared_view_reaches_actual_recipe_miner(tmp_path, monkeypatch, view, producer_view, use_images):
    from nemotron.recipes.embed.stage3_eval import eval as eval_module

    manifest = _bundle(tmp_path / "bundle", producer_view)
    corpus_path = manifest.parent / f"views/{producer_view}/corpus/shared/part-00000.parquet"
    rows = pq.read_table(corpus_path).to_pylist()
    id_key = {"text": "id", "image": "image_filename", "text_image": "docid"}[view]
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
    train = retrieval_vl.prepare_vl_training_data(manifest, output_dir=tmp_path / "prepared", view=view)
    _MiningClient.document_inputs = []
    monkeypatch.setattr(eval_module, "NIMEmbeddingModel", _MiningClient)
    result = retrieval_vl.mine_vllm_hard_negatives(
        train,
        tmp_path / "mined.json",
        api_url="http://unused",
        model="fixture",
        batch_size=2,
        hard_negatives_to_mine=1,
        hard_neg_margin=0.95,
        use_images=use_images,
    )
    row = json.loads(result.read_text())["data"][0]
    assert row["neg_doc"][0]["id"] != row["pos_doc"][0]["id"]
    assert row["negative_mining_performed"] is True
    assert row["unlisted_document_disposition"] == "unjudged"
    assert row["relevance_judgements_complete"] is False
    assert isinstance(_MiningClient.document_inputs[0], dict if use_images else str)
