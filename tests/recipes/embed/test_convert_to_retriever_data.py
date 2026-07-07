"""Tests for generated SDG JSON/JSONL loading."""

from __future__ import annotations

import json
import random

import pandas as pd

from nemotron.recipes.embed.stage1_data_prep.scripts.convert_to_retriever_data import (
    build_corpus_and_mappings,
    create_train_val_test_split,
    generate_eval_set,
    get_file_identifier,
    load_generated_json_files,
)


def _record(name: str) -> dict:
    return {
        "file_name": name,
        "qa_evaluations": {"evaluations": []},
        "deduplicated_qa_pairs": [],
    }


def test_load_generated_json_file_allows_leading_whitespace_pretty_object(tmp_path):
    input_file = tmp_path / "generated.json"
    input_file.write_text("\n  " + json.dumps(_record("doc.txt"), indent=2), encoding="utf-8")

    df = load_generated_json_files(str(input_file))

    assert len(df) == 1
    assert df.iloc[0]["file_name"] == ["doc.txt"]


def test_load_generated_json_file_falls_back_to_jsonl(tmp_path):
    input_file = tmp_path / "generated.jsonl"
    input_file.write_text(
        "\n".join(json.dumps(_record(name)) for name in ["a.txt", "b.txt"]),
        encoding="utf-8",
    )

    df = load_generated_json_files(str(input_file))

    assert len(df) == 2
    assert df["file_name"].tolist() == [["a.txt"], ["b.txt"]]


def test_train_val_test_split_orders_files_before_seeded_shuffle():
    df = pd.DataFrame(
        [
            {"file_name": ["z.txt"], "query": "z"},
            {"file_name": ["a.txt"], "query": "a"},
            {"file_name": ["m.txt"], "query": "m"},
            {"file_name": ["b.txt"], "query": "b"},
        ]
    )

    train_df, _, test_df = create_train_val_test_split(df, train_ratio=0.5, val_ratio=0.0, seed=13)

    expected_files = [("a.txt",), ("b.txt",), ("m.txt",), ("z.txt",)]
    random.Random(13).shuffle(expected_files)
    expected_train = {item[0] for item in expected_files[:2]}
    expected_test = {item[0] for item in expected_files[2:]}

    assert {row[0] for row in train_df["file_name"]} == expected_train
    assert {row[0] for row in test_df["file_name"]} == expected_test


def test_single_doc_identifier_preserves_full_normalized_path():
    dotted_source = (
        "researchnvidiacom/"
        "research.nvidia.com_publication_2022-11_document-one"
    )

    assert get_file_identifier([dotted_source]) == dotted_source
    assert get_file_identifier(["./" + dotted_source]) == dotted_source
    assert get_file_identifier(["corporateblog/60007"]) != get_file_identifier(
        ["techblog/60007"]
    )


def test_eval_qrels_do_not_collide_for_dotted_source_identifiers(tmp_path):
    first_source = (
        "researchnvidiacom/"
        "research.nvidia.com_publication_2015-03_document-one"
    )
    second_source = (
        "researchnvidiacom/"
        "research.nvidia.com_publication_2016-04_document-two"
    )
    generated_df = pd.DataFrame(
        [
            {
                "file_name": [first_source],
                "chunks": [{"chunk_id": 1, "text": "first source passage"}],
            },
            {
                "file_name": [second_source],
                "chunks": [{"chunk_id": 1, "text": "second source passage"}],
            },
        ]
    )
    eval_df = pd.DataFrame(
        [
            {
                "file_name": [first_source],
                "segment_ids": [1],
                "question": "Question about the first source",
            },
            {
                "file_name": [second_source],
                "segment_ids": [1],
                "question": "Question about the second source",
            },
        ]
    )

    corpus, chunk_mapping = build_corpus_and_mappings(generated_df)
    generate_eval_set(corpus, chunk_mapping, eval_df, str(tmp_path))

    corpus_by_id = {
        record["_id"]: record["text"]
        for record in (
            json.loads(line)
            for line in (tmp_path / "eval_beir" / "corpus.jsonl").read_text().splitlines()
        )
    }
    qrels = {
        query_id: corpus_by_id[corpus_id]
        for query_id, corpus_id, _ in (
            line.split("\t")
            for line in (tmp_path / "eval_beir" / "qrels" / "test.tsv")
            .read_text()
            .splitlines()[1:]
        )
    }

    assert qrels == {
        "q0": "first source passage",
        "q1": "second source passage",
    }
