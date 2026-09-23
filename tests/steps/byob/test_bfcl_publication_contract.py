"""Contract tests for what benchmark_raw.parquet and benchmark.parquet mean."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    DedupBalancingDecision,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import BENCHMARK_ROW_FIELDS
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PUBLICATION_BENCHMARK_TABLE,
    PUBLICATION_CONTRACT_VERSION,
    PUBLICATION_RESTATED_FIELDS,
    RAW_BENCHMARK_TABLE,
    PublicationContractError,
    PublicationPlan,
    PublicationSemanticsReport,
    plan_publication,
    publication_manifest_section,
    verify_publication_tables,
    verify_written_benchmarks,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
    encode_arguments,
)

HASH = "sha256:" + "0" * 64
OTHER_HASH = "sha256:" + "1" * 64

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Return one account balance.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    }
]
ARGUMENTS = {"account_id": "1", "limit": 1}


def _row(task_id: str, **overrides: Any) -> dict[str, Any]:
    """Build one benchmark row exactly as Stage 12 writes it to parquet."""
    row: dict[str, Any] = {
        "task_id": task_id,
        "template_id": "tpl",
        "variant_index": 0,
        "messages": [
            {"role": "user", "content": "Balance of 1?", "tool_calls": None, "tool_call_id": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_balance", "arguments": canonical_json(ARGUMENTS)},
                    }
                ],
                "tool_call_id": None,
            },
            {
                "role": "tool",
                "content": canonical_json({"balance": 10}),
                "tool_calls": None,
                "tool_call_id": "call_0",
            },
        ],
        "tools": canonical_json(TOOLS),
        "expected_tool_calls": [
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": encode_arguments(ARGUMENTS),
            }
        ],
        "success_assertions": ["assert_balance_reported"],
        "fixture_refs": ['["accounts","1"]'],
        "intent": "check_balance",
        "category": "accounts",
        "difficulty": "easy",
        "required_tools": ["get_balance"],
        "required_tools_fingerprint": canonical_json(["get_balance"]),
        "tools_present": ["get_balance"],
        "turn_policy": "single_turn",
        "is_multi_turn": False,
        "num_tool_calls": 1,
        "call_order": "strict",
        "call_order_prefix": None,
        "system_prompt_id": "sp-1",
        "tier": "gold",
        "gold_eligible": True,
        "validated_by": ["schema", "replay", "assertions"],
        "pack_id": "pack",
        "pack_version": "1.0.0",
        "seed": 7,
        "paraphrase_model": None,
        "paraphrase_model_canonical": None,
        "held_out_hit": None,
        "src": "pack:tpl",
        "metadata": canonical_json(
            {
                "language": "en",
                "expt_name": "expt",
                "base_task_id": None,
                "surface_source": "template",
                "profile_hash": None,
            }
        ),
    }
    row.update(overrides)
    return row


def _decision(task_id: str, *, selected: bool, rank: int) -> DedupBalancingDecision:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "selected": selected,
        "is_duplicate": not selected,
        "selection_rank": rank,
    }
    if not selected:
        payload["drop_reason"] = "semantic_duplicate"
        payload["duplicate_cluster_id"] = "cluster-0"
        payload["representative_task_id"] = "t1"
    return DedupBalancingDecision.model_validate(payload)


def _write(path: Path, rows: list[dict[str, Any]], *, schema: Any = None) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows, schema=schema or benchmark_schema()), path)
    return path


def _guard_plan(raw_ids: list[str], published: list[str]) -> PublicationPlan:
    return plan_publication(
        raw_task_ids=raw_ids,
        replay_validated_rows=len(raw_ids),
        guard_violations={task_id: task_id not in published for task_id in raw_ids},
    )


def test_the_contract_forbids_publication_from_restating_any_column() -> None:
    # Locking this to empty is central to the publication contract: publication selects
    # rows, and a field listed here would be one the audit table cannot explain.
    assert PUBLICATION_RESTATED_FIELDS == frozenset()


def test_the_compared_field_list_is_the_published_schema() -> None:
    assert tuple(field.name for field in benchmark_schema()) == BENCHMARK_ROW_FIELDS


def test_a_guarded_run_publishes_the_rows_that_passed_in_raw_order() -> None:
    plan = _guard_plan(["t1", "t2", "t3"], ["t1", "t3"])

    assert plan.schema_version == PUBLICATION_CONTRACT_VERSION
    assert plan.raw_task_ids == ("t1", "t2", "t3")
    assert plan.published_task_ids == ("t1", "t3")
    assert plan.surface_gate == "deterministic_guards"
    assert plan.ordering == "raw_order"
    assert not plan.dedup_balancing_applied
    assert not plan.held_out_evaluated


def test_stage_ten_decides_publication_when_surface_quality_ran() -> None:
    plan = plan_publication(
        raw_task_ids=["t1", "t2"],
        replay_validated_rows=2,
        surface_quality_decisions={"t1": "kept", "t2": "dropped"},
    )

    assert plan.surface_gate == "surface_quality"
    assert plan.published_task_ids == ("t1",)


def test_stage_eleven_fixes_publication_order_by_selection_rank() -> None:
    plan = plan_publication(
        raw_task_ids=["t1", "t2", "t3"],
        replay_validated_rows=3,
        surface_quality_decisions={"t1": "kept", "t2": "kept", "t3": "kept"},
        dedup_decisions=[
            _decision("t1", selected=True, rank=1),
            _decision("t2", selected=False, rank=0),
            _decision("t3", selected=True, rank=0),
        ],
    )

    assert plan.published_task_ids == ("t3", "t1")
    assert plan.ordering == "selection_rank"
    assert plan.dedup_balancing_applied


def test_a_raw_table_shorter_than_the_replay_verdicts_is_refused() -> None:
    with pytest.raises(PublicationContractError, match="must precede every publication drop"):
        plan_publication(
            raw_task_ids=["t1"],
            replay_validated_rows=2,
            guard_violations={"t1": False},
        )


def test_publication_needs_exactly_one_surface_gate() -> None:
    with pytest.raises(PublicationContractError, match="exactly one surface gate"):
        plan_publication(raw_task_ids=["t1"], replay_validated_rows=1)
    with pytest.raises(PublicationContractError, match="exactly one surface gate"):
        plan_publication(
            raw_task_ids=["t1"],
            replay_validated_rows=1,
            guard_violations={"t1": False},
            surface_quality_decisions={"t1": "kept"},
        )


def test_a_gate_must_answer_for_every_raw_row() -> None:
    with pytest.raises(PublicationContractError, match=r"Stage 10 decisions.*missing=\['t2'\]"):
        plan_publication(
            raw_task_ids=["t1", "t2"],
            replay_validated_rows=2,
            surface_quality_decisions={"t1": "kept"},
        )


def test_stage_eleven_must_answer_for_every_stage_ten_survivor() -> None:
    with pytest.raises(PublicationContractError, match=r"Stage 11 decisions.*missing=\['t2'\]"):
        plan_publication(
            raw_task_ids=["t1", "t2"],
            replay_validated_rows=2,
            surface_quality_decisions={"t1": "kept", "t2": "kept"},
            dedup_decisions=[_decision("t1", selected=True, rank=0)],
        )


def test_stage_eleven_selection_ranks_must_be_a_total_order() -> None:
    with pytest.raises(PublicationContractError, match="rank its selections"):
        plan_publication(
            raw_task_ids=["t1", "t2"],
            replay_validated_rows=2,
            surface_quality_decisions={"t1": "kept", "t2": "kept"},
            dedup_decisions=[
                _decision("t1", selected=True, rank=0),
                _decision("t2", selected=True, rank=2),
            ],
        )


def test_an_unrecognized_stage_ten_verdict_is_refused() -> None:
    with pytest.raises(PublicationContractError, match="kept or dropped"):
        plan_publication(
            raw_task_ids=["t1"],
            replay_validated_rows=1,
            surface_quality_decisions={"t1": "maybe"},
        )


def test_a_repeated_raw_task_id_is_refused() -> None:
    with pytest.raises(PublicationContractError, match="repeats a task id"):
        plan_publication(
            raw_task_ids=["t1", "t1"],
            replay_validated_rows=2,
            guard_violations={"t1": False},
        )


def test_a_held_out_row_cannot_reach_the_publication_set() -> None:
    with pytest.raises(PublicationContractError, match="bind held-out material"):
        plan_publication(
            raw_task_ids=["t1", "t2"],
            replay_validated_rows=2,
            guard_violations={"t1": False, "t2": False},
            held_out_hits={"t1": False, "t2": True},
        )


def test_a_held_out_row_that_never_reached_publication_does_not_stop_the_run() -> None:
    plan = plan_publication(
        raw_task_ids=["t1", "t2"],
        replay_validated_rows=2,
        guard_violations={"t1": False, "t2": True},
        held_out_hits={"t1": False, "t2": True},
    )

    assert plan.published_task_ids == ("t1",)
    assert plan.held_out_evaluated


def test_a_plan_cannot_publish_a_row_the_raw_table_does_not_carry() -> None:
    with pytest.raises(ValidationError, match="absent from the raw table"):
        PublicationPlan(
            raw_task_ids=("t1",),
            published_task_ids=("t2",),
            surface_gate="deterministic_guards",
            dedup_balancing_applied=False,
            held_out_evaluated=False,
            ordering="raw_order",
        )


def test_a_plan_without_stage_eleven_cannot_claim_selection_rank_order() -> None:
    with pytest.raises(ValidationError, match="exactly when Stage 11 ran"):
        PublicationPlan(
            raw_task_ids=("t1",),
            published_task_ids=("t1",),
            surface_gate="deterministic_guards",
            dedup_balancing_applied=False,
            held_out_evaluated=False,
            ordering="selection_rank",
        )


def test_a_plan_without_stage_eleven_must_keep_the_raw_order() -> None:
    with pytest.raises(ValidationError, match="keep their raw order"):
        PublicationPlan(
            raw_task_ids=("t1", "t2"),
            published_task_ids=("t2", "t1"),
            surface_gate="deterministic_guards",
            dedup_balancing_applied=False,
            held_out_evaluated=False,
            ordering="raw_order",
        )


def test_written_tables_that_match_the_plan_produce_a_report(tmp_path: Path) -> None:
    raw = [_row("t1"), _row("t2"), _row("t3")]
    plan = _guard_plan(["t1", "t2", "t3"], ["t1", "t3"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, raw)
    published_path = _write(tmp_path / PUBLICATION_BENCHMARK_TABLE, [raw[0], raw[2]])

    report = verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)

    assert report.raw_rows == 3
    assert report.published_rows == 2
    assert report.restated_fields == ()
    assert report.raw_content_hash != report.publication_content_hash
    assert report.surface_gate == "deterministic_guards"
    assert report.ordering == "raw_order"


def test_a_published_row_that_rewrites_its_truth_is_refused(tmp_path: Path) -> None:
    raw = [_row("t1")]
    rewritten = _row("t1", success_assertions=[])
    plan = _guard_plan(["t1"], ["t1"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, raw)
    published_path = _write(tmp_path / PUBLICATION_BENCHMARK_TABLE, [rewritten])

    with pytest.raises(PublicationContractError, match=r"restates \['success_assertions'\]"):
        verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)


def test_a_published_row_that_rewrites_an_argument_type_is_refused(tmp_path: Path) -> None:
    raw = [_row("t1")]
    coerced = _row(
        "t1",
        expected_tool_calls=[
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": encode_arguments({"account_id": "1", "limit": "1"}),
            }
        ],
    )
    plan = _guard_plan(["t1"], ["t1"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, raw)
    published_path = _write(tmp_path / PUBLICATION_BENCHMARK_TABLE, [coerced])

    with pytest.raises(PublicationContractError, match=r"restates \['expected_tool_calls'\]"):
        verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)


def test_a_publication_written_out_of_plan_order_is_refused(tmp_path: Path) -> None:
    raw = [_row("t1"), _row("t2")]
    plan = _guard_plan(["t1", "t2"], ["t1", "t2"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, raw)
    published_path = _write(tmp_path / PUBLICATION_BENCHMARK_TABLE, [raw[1], raw[0]])

    with pytest.raises(PublicationContractError, match="in that order"):
        verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)


def test_a_raw_table_filtered_after_the_plan_was_derived_is_refused(tmp_path: Path) -> None:
    plan = _guard_plan(["t1", "t2"], ["t1"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, [_row("t1")])
    published_path = _write(tmp_path / PUBLICATION_BENCHMARK_TABLE, [_row("t1")])

    with pytest.raises(PublicationContractError, match=f"{RAW_BENCHMARK_TABLE} does not carry"):
        verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)


def test_a_table_written_with_another_schema_is_refused(tmp_path: Path) -> None:
    import pyarrow as pa

    plan = _guard_plan(["t1"], ["t1"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, [_row("t1")])
    published_path = tmp_path / PUBLICATION_BENCHMARK_TABLE
    _write(
        published_path,
        [{"task_id": "t1"}],
        schema=pa.schema([("task_id", pa.string())]),
    )

    with pytest.raises(PublicationContractError, match="not the published benchmark schema"):
        verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)


def test_a_missing_table_is_refused(tmp_path: Path) -> None:
    plan = _guard_plan(["t1"], ["t1"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, [_row("t1")])

    with pytest.raises(PublicationContractError, match="was not written"):
        verify_written_benchmarks(
            raw_path=raw_path,
            publication_path=tmp_path / PUBLICATION_BENCHMARK_TABLE,
            plan=plan,
        )


def test_a_scanned_run_requires_a_held_out_verdict_on_every_raw_row() -> None:
    plan = plan_publication(
        raw_task_ids=["t1"],
        replay_validated_rows=1,
        guard_violations={"t1": False},
        held_out_hits={"t1": False},
    )

    with pytest.raises(PublicationContractError, match="records no verdict"):
        verify_publication_tables([_row("t1")], [_row("t1")], plan)


def test_an_unscanned_run_cannot_claim_a_held_out_verdict() -> None:
    plan = _guard_plan(["t1"], ["t1"])
    scanned = _row("t1", held_out_hit=False)

    with pytest.raises(PublicationContractError, match="no held-out policy was evaluated"):
        verify_publication_tables([scanned], [scanned], plan)


def test_a_published_held_out_row_is_refused_even_when_the_plan_missed_it() -> None:
    plan = PublicationPlan(
        raw_task_ids=("t1",),
        published_task_ids=("t1",),
        surface_gate="deterministic_guards",
        dedup_balancing_applied=False,
        held_out_evaluated=True,
        ordering="raw_order",
    )
    leaked = _row("t1", held_out_hit=True)

    with pytest.raises(PublicationContractError, match="held-out row"):
        verify_publication_tables([leaked], [leaked], plan)


def test_an_off_schema_row_is_refused() -> None:
    plan = _guard_plan(["t1"], ["t1"])
    row = _row("t1")
    del row["difficulty"]

    with pytest.raises(PublicationContractError, match=r"off-schema \(missing=\['difficulty'\]"):
        verify_publication_tables([row], [row], plan)


def test_a_report_cannot_publish_more_rows_than_raw_holds() -> None:
    with pytest.raises(ValidationError, match="cannot carry more rows"):
        PublicationSemanticsReport(
            raw_rows=1,
            published_rows=2,
            surface_gate="surface_quality",
            dedup_balancing_applied=False,
            held_out_evaluated=False,
            ordering="raw_order",
            raw_content_hash=HASH,
            publication_content_hash=OTHER_HASH,
        )


def test_a_report_cannot_invent_a_restated_field() -> None:
    with pytest.raises(ValidationError, match="contract's own allowance"):
        PublicationSemanticsReport(
            raw_rows=1,
            published_rows=1,
            surface_gate="surface_quality",
            dedup_balancing_applied=False,
            held_out_evaluated=False,
            ordering="raw_order",
            restated_fields=("gold_eligible",),
            raw_content_hash=HASH,
            publication_content_hash=OTHER_HASH,
        )


def test_two_tables_of_different_size_cannot_share_one_content_hash() -> None:
    with pytest.raises(ValidationError, match="cannot be the same file"):
        PublicationSemanticsReport(
            raw_rows=2,
            published_rows=1,
            surface_gate="surface_quality",
            dedup_balancing_applied=False,
            held_out_evaluated=False,
            ordering="raw_order",
            raw_content_hash=HASH,
            publication_content_hash=HASH,
        )


def test_the_manifest_section_describes_both_tables(tmp_path: Path) -> None:
    raw = [_row("t1"), _row("t2")]
    plan = _guard_plan(["t1", "t2"], ["t1"])
    raw_path = _write(tmp_path / RAW_BENCHMARK_TABLE, raw)
    published_path = _write(tmp_path / PUBLICATION_BENCHMARK_TABLE, [raw[0]])

    section = publication_manifest_section(
        verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)
    )

    assert section["schema_version"] == PUBLICATION_CONTRACT_VERSION
    assert section["raw"] == {
        "file": RAW_BENCHMARK_TABLE,
        "rows": 2,
        "content_hash": section["raw"]["content_hash"],
        "contains": "schema_valid_and_replay_valid_rows",
    }
    assert section["published"]["rows"] == 1
    assert section["published"]["surface_gate"] == "deterministic_guards"
    assert section["published"]["ordering"] == "raw_order"
    assert section["restated_fields"] == []
    assert section["verified"] is True
