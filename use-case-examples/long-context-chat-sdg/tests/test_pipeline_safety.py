# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import pytest

from pipeline import (
    _atomic_write_jsonl,
    _positive_limit,
    _prepare_generated_rows,
    _validate_dry_run_scope,
    run_query_gen,
)


def test_generate_dry_run_is_rejected_before_model_work():
    with pytest.raises(SystemExit, match="supported only for query_gen"):
        _validate_dry_run_scope("generate", True, False)
    with pytest.raises(SystemExit, match="needs a configured query_gen source"):
        _validate_dry_run_scope("all", True, False)
    _validate_dry_run_scope("query_gen", True, True)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_limit_must_be_positive(value):
    with pytest.raises(Exception, match="at least 1"):
        _positive_limit(value)
    assert _positive_limit("3") == 3


def test_missing_jsonl_corpus_has_friendly_pre_spend_guard(tmp_path):
    cfg = {"query_gen": {"source": "jsonl", "chunks_path": "missing.jsonl"}}
    with pytest.raises(SystemExit, match=r"\[query_gen\] corpus not found:.*missing.jsonl"):
        run_query_gen(cfg, tmp_path, None)


def test_generated_record_conversion_accounts_for_every_rejection():
    cfg = {"tools": [], "retrieval": {"tools": ["search"]}}
    records = [
        {"conversation_messages": json.dumps([{"role": "assistant", "content": "ok"}]),
         "retrieval_mode": "http", "hops_taken": 1},
        {"conversation_messages": "not-json"},
        {"conversation_messages": "[]", "conversation_status": False},
    ]
    rows, rejected = _prepare_generated_rows(records, cfg, "http")
    assert len(rows) == 1
    assert rows[0]["retrieval_tools"] == ["search"]
    assert [r["reason"] for r in rejected] == [
        "invalid_conversation_messages", "empty_conversation_messages"
    ]


def test_atomic_jsonl_replaces_instead_of_appending(tmp_path):
    path = tmp_path / "rows.jsonl"
    _atomic_write_jsonl(path, [{"old": True}])
    _atomic_write_jsonl(path, [{"new": 1}, {"new": 2}])
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"new": 1}, {"new": 2}
    ]
    assert not list(tmp_path.glob(".rows.jsonl.*.tmp"))
