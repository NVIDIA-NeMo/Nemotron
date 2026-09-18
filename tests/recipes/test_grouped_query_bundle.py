# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-corpus query-family separation and public Stage 1/3 handoff tests."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nemotron.recipes import retrieval_vl
from nemotron.recipes.embed.sdg_manifest import (
    resolve_portable_evaluation_input,
    resolve_portable_training_input,
    write_generation_manifest,
)

from tests.recipes.test_retrieval_bundle_v2 import _bundle, _inventory, _json, _jsonl


def _seal(root):
    manifest = _inventory(root)
    payload = json.loads(manifest.read_text())
    payload.update(split_protocol="grouped_query_disjoint", corpus_scope="full_collection")
    _json(manifest, payload)
    return manifest


def _grouped(root, view="image_and_text", validation=False):
    _bundle(root, view)
    units = [json.loads(line) for line in (root / "retrieval_units.jsonl").read_text().splitlines()]
    rows = []
    for split in ("train", "validation", "evaluation"):
        rows.extend(pq.read_table(root / f"views/{view}/corpus/{split}/part-00000.parquet").to_pylist())
    corpus = root / f"views/{view}/corpus/shared"
    corpus.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), corpus / "part-00000.parquet")
    _json(corpus / "merlin_metadata.json", json.loads((corpus.parent / "train/merlin_metadata.json").read_text()))
    for unit in units:
        unit["split"] = "shared"
    _jsonl(root / "retrieval_units.jsonl", units)
    assignments = {}
    for split in ("train", "validation"):
        path = root / f"views/{view}/{split}.json"
        payload = json.loads(path.read_text())
        payload["corpus"] = {"path": "corpus/shared"}
        if split == "validation" and not validation:
            payload["data"] = []
        for row in payload["data"]:
            row["question"] += f" {split}"
            row["query_group_id"] = f"family-{split}"
            assignments[row["question_id"]] = {"split": split, "query_group_id": row["query_group_id"]}
        _json(path, payload)
    directory = root / f"synthetic_eval/{view}"
    _jsonl(
        directory / "queries.jsonl",
        [{"_id": "q-evaluation", "text": "Quels risques ?", "query_group_id": "family-eval"}],
    )
    assignments["q-evaluation"] = {"split": "evaluation", "query_group_id": "family-eval"}
    documents = []
    for unit in units:
        row = {"_id": unit["unit_id"], "metadata": {"source_document_id": unit["document_id"]}}
        if view != "image":
            row["text"] = unit["text"]
        if view != "text":
            row["image_path"] = unit["images"][0]
        documents.append(row)
    _jsonl(directory / "corpus.jsonl", documents)
    # Intentional shared positive page and a second graded positive, plus an
    # unjudged distractor page. All three must remain searchable.
    (directory / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-evaluation\tpage-train\t2\nq-evaluation\tpage-evaluation\t1\n"
    )
    _json(
        root / "split_manifest.json",
        {
            "split_protocol": "grouped_query_disjoint",
            "corpus_scope": "full_collection",
            "document_disjoint": False,
            "query_assignments": {view: assignments},
        },
    )
    return _seal(root)


@pytest.mark.parametrize("view", ["text", "image", "image_and_text"])
@pytest.mark.parametrize("validation", [False, True])
def test_shared_full_corpus_and_public_handoffs(tmp_path, view, validation):
    root = tmp_path / "bundle"
    manifest = _grouped(root, view, validation)
    before = (root / f"synthetic_eval/{view}/qrels/test.tsv").read_bytes()
    paths = retrieval_vl.inspect_vl_bundle(manifest, view="text_image" if view == "image_and_text" else view)
    assert paths.split_protocol == "grouped_query_disjoint"
    assert paths.corpus_scope == "full_collection"
    assert paths.train_corpus == paths.validation_corpus == paths.evaluation_corpus
    handoff = write_generation_manifest(
        output_dir=tmp_path, output_path=paths.train, dataset_name="fixture", portable_bundle=root
    )
    assert resolve_portable_training_input(handoff, view, "grouped_query_disjoint") == paths.train
    assert resolve_portable_evaluation_input(handoff, view, "grouped_query_disjoint") == (paths.synthetic_eval, root)
    assert (root / f"synthetic_eval/{view}/qrels/test.tsv").read_bytes() == before
    with pytest.raises(ValueError, match="differs"):
        resolve_portable_training_input(handoff, view, "document_disjoint")


@pytest.mark.parametrize(
    "mutation",
    [
        "family",
        "text",
        "missing-group",
        "assignment",
        "extra-assignment",
        "missing-page",
        "scope",
        "document-claim",
        "asset",
    ],
)
def test_grouped_protocol_rejects_leakage_and_incomplete_inputs(tmp_path, mutation):
    manifest = _grouped(tmp_path)
    split_path = tmp_path / "split_manifest.json"
    split = json.loads(split_path.read_text())
    query_path = tmp_path / "synthetic_eval/image_and_text/queries.jsonl"
    query = json.loads(query_path.read_text())
    if mutation == "family":
        query["query_group_id"] = "family-train"
        split["query_assignments"]["image_and_text"]["q-evaluation"]["query_group_id"] = "family-train"
    elif mutation == "text":
        train = json.loads((tmp_path / "views/image_and_text/train.json").read_text())["data"][0]
        query["text"] = "  " + train["question"].upper() + "  "
    elif mutation == "missing-group":
        del query["query_group_id"]
    elif mutation == "assignment":
        split["query_assignments"]["image_and_text"]["q-evaluation"]["split"] = "train"
    elif mutation == "extra-assignment":
        split["query_assignments"]["image_and_text"]["unexported"] = {"split": "train", "query_group_id": "extra"}
    elif mutation == "missing-page":
        shard = tmp_path / "views/image_and_text/corpus/shared/part-00000.parquet"
        pq.write_table(
            pa.Table.from_pylist([r for r in pq.read_table(shard).to_pylist() if r["docid"] != "page-validation"]),
            shard,
        )
        path = tmp_path / "synthetic_eval/image_and_text/corpus.jsonl"
        _jsonl(
            path,
            [
                json.loads(line)
                for line in path.read_text().splitlines()
                if json.loads(line)["_id"] != "page-validation"
            ],
        )
    elif mutation == "scope":
        split["corpus_scope"] = "partition"
    elif mutation == "document-claim":
        split["document_disjoint"] = True
    else:
        path = tmp_path / "retrieval_units.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["images"] = ["../outside.png"]
        _jsonl(path, rows)
    _jsonl(query_path, [query])
    _json(split_path, split)
    _seal(tmp_path)
    with pytest.raises(retrieval_vl.RetrievalVLBundleError):
        retrieval_vl.inspect_vl_bundle(manifest)


def test_recipe_protocol_assertion_requires_portable_input():
    from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig
    from nemotron.recipes.embed.stage3_eval.eval import EvalConfig

    for config in (DataPrepConfig, EvalConfig):
        with pytest.raises(ValueError, match="retrieval_split_protocol requires"):
            config(retrieval_split_protocol="grouped_query_disjoint")
        resolved = config(
            sdg_input_path="generation_result.json",
            retrieval_view="image_and_text",
            retrieval_split_protocol="grouped_query_disjoint",
        )
        assert resolved.retrieval_split_protocol == "grouped_query_disjoint"
