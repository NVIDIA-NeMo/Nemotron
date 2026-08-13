"""Tests for Stage 0 raw-response logging and QA output validation.

Covers the behavior requested in issue #314:
    - Every raw LLM response is logged before parsing (via the `{col}__trace`
      column produced by `with_trace=TraceType.LAST_MESSAGE`).
    - Parse failures (missing/None generated columns), empty QA results, and
      underfilled QA results (fewer pairs than requested) are all detected
      and logged with their raw response.
    - The returned/printed summary reports requested vs. persisted records,
      omitted records by column, and requested vs. generated QA totals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from retriever_sdg.pipeline import (
    _extract_pairs,
    _merge_stage0_stats,
    _raw_trace_text,
    log_stage0_raw_responses,
)


def _row(
    file_name: str,
    document_artifacts: object = {"key_concepts": []},
    document_artifacts_trace: object = None,
    qa_pairs: list | None = None,
    qa_generation_trace: object = None,
) -> dict:
    """Build one synthetic generated_df row."""
    if qa_pairs is None:
        qa_pairs = [{"question": "q1"}, {"question": "q2"}]
    return {
        "file_name": file_name,
        "document_artifacts": document_artifacts,
        "document_artifacts__trace": document_artifacts_trace or [],
        "qa_generation": {"pairs": qa_pairs} if qa_pairs is not None else None,
        "qa_generation__trace": qa_generation_trace or [],
    }


class TestExtractPairs:
    def test_extracts_from_dict(self):
        assert _extract_pairs({"pairs": [1, 2, 3]}) == [1, 2, 3]

    def test_none_returns_empty_list(self):
        assert _extract_pairs(None) == []

    def test_missing_pairs_key_returns_empty_list(self):
        assert _extract_pairs({}) == []


class TestRawTraceText:
    def test_extracts_last_assistant_message_content(self):
        row = pd.Series({"qa_generation__trace": [{"role": "assistant", "content": "raw text"}]})
        assert _raw_trace_text(row, "qa_generation__trace") == "raw text"

    def test_missing_trace_column_returns_none(self):
        row = pd.Series({"qa_generation": {}})
        assert _raw_trace_text(row, "qa_generation__trace") is None

    def test_empty_trace_returns_none(self):
        row = pd.Series({"qa_generation__trace": []})
        assert _raw_trace_text(row, "qa_generation__trace") is None


class TestLogStage0RawResponses:
    def test_healthy_record_is_not_logged(self, tmp_path: Path):
        df = pd.DataFrame([_row("doc0.txt")])
        stats = log_stage0_raw_responses(df, num_pairs=2, batch_idx=0, output_dir=tmp_path)

        assert stats["requested_records"] == 1
        assert stats["persisted_records"] == 1
        assert stats["empty_qa_records"] == 0
        assert stats["underfilled_qa_records"] == 0
        assert not (tmp_path / "stage0_raw_responses.jsonl").exists()

    def test_parse_failure_is_logged_with_raw_response(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                _row(
                    "doc1.txt",
                    document_artifacts=None,
                    document_artifacts_trace=[{"role": "assistant", "content": "not parsable json"}],
                )
            ]
        )
        stats = log_stage0_raw_responses(df, num_pairs=2, batch_idx=0, output_dir=tmp_path)

        assert stats["omitted_by_column"]["document_artifacts"] == 1
        assert stats["persisted_records"] == 0

        entries = _read_jsonl(tmp_path / "stage0_raw_responses.jsonl")
        assert len(entries) == 1
        assert entries[0]["status"] == "omitted_parse_failure"
        assert entries[0]["column"] == "document_artifacts"
        assert entries[0]["raw_response"] == "not parsable json"

    def test_empty_qa_is_logged_and_counted(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                _row(
                    "doc2.txt",
                    qa_pairs=[],
                    qa_generation_trace=[{"role": "assistant", "content": '{"pairs": []}'}],
                )
            ]
        )
        stats = log_stage0_raw_responses(df, num_pairs=2, batch_idx=0, output_dir=tmp_path)

        assert stats["empty_qa_records"] == 1
        assert stats["requested_qa_total"] == 2
        assert stats["generated_qa_total"] == 0

        entries = _read_jsonl(tmp_path / "stage0_raw_responses.jsonl")
        assert entries[0]["status"] == "empty_qa"
        assert entries[0]["requested_qa_pairs"] == 2
        assert entries[0]["generated_qa_pairs"] == 0

    def test_underfilled_qa_is_logged_and_counted(self, tmp_path: Path):
        df = pd.DataFrame([_row("doc3.txt", qa_pairs=[{"question": "only one"}])])
        stats = log_stage0_raw_responses(df, num_pairs=2, batch_idx=0, output_dir=tmp_path)

        assert stats["underfilled_qa_records"] == 1
        assert stats["empty_qa_records"] == 0
        assert stats["requested_qa_total"] == 2
        assert stats["generated_qa_total"] == 1

        entries = _read_jsonl(tmp_path / "stage0_raw_responses.jsonl")
        assert entries[0]["status"] == "underfilled_qa"

    def test_mixed_batch_reports_correct_aggregate_counts(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                _row("doc0.txt"),  # healthy, full QA count
                _row("doc1.txt", document_artifacts=None),  # parse failure
                _row("doc2.txt", qa_pairs=[]),  # empty QA
                _row("doc3.txt", qa_pairs=[{"question": "q1"}]),  # underfilled QA
            ]
        )
        stats = log_stage0_raw_responses(df, num_pairs=2, batch_idx=0, output_dir=tmp_path)

        assert stats["requested_records"] == 4
        assert stats["persisted_records"] == 3
        assert stats["omitted_by_column"]["document_artifacts"] == 1
        assert stats["empty_qa_records"] == 1
        assert stats["underfilled_qa_records"] == 1
        # requested: 2 pairs x 4 records (qa_generation is present on all 4 rows,
        # even doc1.txt whose *document_artifacts* column failed to parse)
        assert stats["requested_qa_total"] == 8
        # generated: 2 (doc0, healthy) + 2 (doc1, qa_generation itself succeeded)
        #          + 0 (doc2, empty) + 1 (doc3, underfilled)
        assert stats["generated_qa_total"] == 5

        entries = _read_jsonl(tmp_path / "stage0_raw_responses.jsonl")
        assert len(entries) == 3  # only the problematic records are logged
        statuses = {entry["status"] for entry in entries}
        assert statuses == {"omitted_parse_failure", "empty_qa", "underfilled_qa"}

    def test_log_file_is_appended_across_batches(self, tmp_path: Path):
        df_batch0 = pd.DataFrame([_row("doc0.txt", qa_pairs=[])])
        df_batch1 = pd.DataFrame([_row("doc1.txt", document_artifacts=None)])

        log_stage0_raw_responses(df_batch0, num_pairs=2, batch_idx=0, output_dir=tmp_path)
        log_stage0_raw_responses(df_batch1, num_pairs=2, batch_idx=1, output_dir=tmp_path)

        entries = _read_jsonl(tmp_path / "stage0_raw_responses.jsonl")
        assert len(entries) == 2
        assert entries[0]["batch_index"] == 0
        assert entries[1]["batch_index"] == 1


class TestMergeStage0Stats:
    def test_accumulates_across_batches(self):
        total = {
            "requested_records": 0,
            "persisted_records": 0,
            "omitted_by_column": {},
            "empty_qa_records": 0,
            "underfilled_qa_records": 0,
            "requested_qa_total": 0,
            "generated_qa_total": 0,
        }
        batch_stats = {
            "requested_records": 4,
            "persisted_records": 3,
            "omitted_by_column": {"document_artifacts": 1, "qa_generation": 0},
            "empty_qa_records": 1,
            "underfilled_qa_records": 1,
            "requested_qa_total": 6,
            "generated_qa_total": 3,
        }

        merged = _merge_stage0_stats(total, batch_stats)
        merged = _merge_stage0_stats(merged, batch_stats)

        assert merged["requested_records"] == 8
        assert merged["persisted_records"] == 6
        assert merged["omitted_by_column"]["document_artifacts"] == 2
        assert merged["empty_qa_records"] == 2
        assert merged["underfilled_qa_records"] == 2
        assert merged["requested_qa_total"] == 12
        assert merged["generated_qa_total"] == 6


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
