"""Contract tests for the single decode path from benchmark.parquet to a writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
    ExportedToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    EXPORT_PROJECTION_VERSION,
    CanonicalCallGroup,
    CanonicalConversationPlan,
    CanonicalExportProjection,
    ExportProjectionError,
    ProjectionProvenance,
    ProjectionSource,
    conversation_plan,
    derive_provenance,
    project_benchmark_rows,
    project_published_benchmark,
    projection_lineage,
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
    },
    {
        "type": "function",
        "function": {"name": "list_cards", "description": "List cards.", "parameters": None},
    },
]


def _system(content: str = "You use tools.") -> dict[str, Any]:
    return {"role": "system", "content": content, "tool_calls": None, "tool_call_id": None}


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content, "tool_calls": None, "tool_call_id": None}


def _assistant_text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": None, "tool_call_id": None}


def _assistant_calls(calls: list[tuple[str, dict[str, Any]]], *, first_id: int = 0) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{first_id + index}",
                "type": "function",
                "function": {"name": name, "arguments": canonical_json(arguments)},
            }
            for index, (name, arguments) in enumerate(calls)
        ],
        "tool_call_id": None,
    }


def _tool_result(call_id: str, payload: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": canonical_json(payload),
        "tool_calls": None,
        "tool_call_id": call_id,
    }


def _expected(
    name: str,
    arguments: dict[str, Any],
    *,
    turn_index: int = 0,
    call_group: int = 0,
    position_in_group: int = 0,
) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "call_group": call_group,
        "position_in_group": position_in_group,
        "function_name": name,
        "arguments": encode_arguments(arguments),
    }


def _row(task_id: str = "pack__tpl__abcdef", **overrides: Any) -> dict[str, Any]:
    """One single-turn benchmark row exactly as Stage 12 writes it to parquet."""
    arguments = {"account_id": "1"}
    row: dict[str, Any] = {
        "task_id": task_id,
        "template_id": "tpl",
        "variant_index": 0,
        "messages": [
            _system(),
            _user("Balance of 1?"),
            _assistant_calls([("get_balance", arguments)]),
            _tool_result("call_0", {"balance": 10}),
            _assistant_text("It is 10."),
        ],
        "tools": canonical_json(TOOLS),
        "expected_tool_calls": [_expected("get_balance", arguments)],
        "success_assertions": ["assert_balance_reported"],
        "fixture_refs": ['["accounts","1"]'],
        "intent": "check_balance",
        "category": "accounts",
        "difficulty": "easy",
        "required_tools": ["get_balance"],
        "required_tools_fingerprint": canonical_json(["get_balance"]),
        "tools_present": ["get_balance", "list_cards"],
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
        "held_out_hit": False,
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


def _parallel_row(task_id: str = "pack__tpl__parallel") -> dict[str, Any]:
    balance = {"account_id": "1"}
    cards: dict[str, Any] = {}
    return _row(
        task_id,
        turn_policy="multi_tool",
        num_tool_calls=2,
        required_tools=["get_balance", "list_cards"],
        required_tools_fingerprint=canonical_json(["get_balance", "list_cards"]),
        messages=[
            _system(),
            _user("Balance and cards?"),
            _assistant_calls([("get_balance", balance), ("list_cards", cards)]),
            _tool_result("call_0", {"balance": 10}),
            _tool_result("call_1", {"cards": []}),
            _assistant_text("Here they are."),
        ],
        expected_tool_calls=[
            _expected("get_balance", balance, position_in_group=0),
            _expected("list_cards", cards, position_in_group=1),
        ],
    )


def _multi_turn_row(task_id: str = "pack__tpl__multiturn") -> dict[str, Any]:
    arguments = {"account_id": "1"}
    return _row(
        task_id,
        turn_policy="missing_slot",
        is_multi_turn=True,
        messages=[
            _system(),
            _user("What is my balance?"),
            _assistant_text("Which account?"),
            _user("Account 1."),
            _assistant_calls([("get_balance", arguments)]),
            _tool_result("call_0", {"balance": 10}),
            _assistant_text("It is 10."),
        ],
        # The call is issued by the second assistant message, so its turn_index is 1.
        expected_tool_calls=[_expected("get_balance", arguments, turn_index=1)],
    )


def _canonical(row: dict[str, Any] | None = None) -> CanonicalExportRow:
    return CanonicalExportRow.from_benchmark_row(row if row is not None else _row())


def _source(rows: int = 1, **overrides: Any) -> ProjectionSource:
    payload: dict[str, Any] = {"file": "benchmark.parquet", "content_hash": HASH, "rows": rows}
    payload.update(overrides)
    return ProjectionSource(**payload)


def _call(**overrides: Any) -> ExportedToolCall:
    payload: dict[str, Any] = {
        "turn_index": 0,
        "call_group": 0,
        "position_in_group": 0,
        "function_name": "get_balance",
        "arguments": {"account_id": "1"},
    }
    payload.update(overrides)
    return ExportedToolCall(**payload)


def _group(**overrides: Any) -> CanonicalCallGroup:
    payload: dict[str, Any] = {
        "turn_index": 0,
        "call_group": 0,
        "user_turn_index": 0,
        "calls": (_call(),),
        "is_parallel": False,
    }
    payload.update(overrides)
    return CanonicalCallGroup(**payload)


def test_a_single_turn_row_projects_to_one_sequential_group() -> None:
    plan = conversation_plan(_canonical())

    assert plan.schema_version == EXPORT_PROJECTION_VERSION
    assert plan.user_turns == 1
    assert plan.assistant_turns == 2
    assert not plan.is_multi_turn
    assert plan.call_count == 1
    assert plan.parallel_groups == ()
    (group,) = plan.groups
    assert group.turn_index == 0
    assert not group.is_parallel


def test_calls_issued_together_stay_in_one_parallel_group() -> None:
    plan = conversation_plan(_canonical(_parallel_row()))

    (group,) = plan.groups
    assert group.is_parallel
    assert [call.function_name for call in group.calls] == ["get_balance", "list_cards"]
    assert plan.parallel_groups == (group,)


def test_a_multi_turn_row_places_its_group_on_the_assistant_turn_that_issues_it() -> None:
    plan = conversation_plan(_canonical(_multi_turn_row()))

    assert plan.user_turns == 2
    assert plan.is_multi_turn
    (group,) = plan.groups
    assert group.turn_index == 1


def test_an_expected_call_on_a_turn_that_issues_none_is_refused() -> None:
    arguments = {"account_id": "1"}
    row = _row(
        messages=[
            _system(),
            _user("Balance of 1?"),
            _assistant_calls([("get_balance", arguments)]),
            _tool_result("call_0", {"balance": 10}),
            _assistant_text("It is 10."),
        ],
        expected_tool_calls=[_expected("get_balance", arguments, turn_index=1)],
    )

    with pytest.raises(ExportProjectionError, match="no expected call claims"):
        conversation_plan(_canonical(row))


def test_a_turn_index_beyond_the_conversation_is_refused() -> None:
    arguments = {"account_id": "1"}
    row = _row(expected_tool_calls=[_expected("get_balance", arguments, turn_index=5)])

    with pytest.raises(ExportProjectionError, match="no expected call claims"):
        conversation_plan(_canonical(row))


def test_one_assistant_message_may_not_mix_call_groups() -> None:
    balance = {"account_id": "1"}
    cards: dict[str, Any] = {}
    row = _parallel_row()
    row["expected_tool_calls"] = [
        _expected("get_balance", balance, call_group=0, position_in_group=0),
        _expected("list_cards", cards, call_group=1, position_in_group=0),
    ]

    with pytest.raises(ExportProjectionError, match="mixes call groups"):
        conversation_plan(_canonical(row))


def test_a_row_whose_multi_turn_flag_contradicts_its_conversation_is_refused() -> None:
    row = _row(is_multi_turn=True)

    with pytest.raises(ExportProjectionError, match="multi-turn exactly when"):
        conversation_plan(_canonical(row))


def test_a_group_is_parallel_exactly_when_it_issues_more_than_one_call() -> None:
    with pytest.raises(ValidationError, match="parallel exactly when"):
        _group(is_parallel=True)


def test_a_group_must_occupy_consecutive_positions() -> None:
    with pytest.raises(ValidationError, match=r"positions 0..n-1"):
        _group(calls=(_call(position_in_group=0), _call(position_in_group=2)), is_parallel=True)


def test_a_group_may_not_span_two_assistant_turns() -> None:
    with pytest.raises(ValidationError, match="same assistant turn"):
        _group(
            calls=(_call(position_in_group=0), _call(turn_index=1, position_in_group=1)),
            is_parallel=True,
        )


def test_a_plan_may_not_place_a_group_past_its_last_assistant_turn() -> None:
    with pytest.raises(ValidationError, match="past its last assistant turn"):
        CanonicalConversationPlan(
            task_id="t1",
            user_turns=1,
            assistant_turns=1,
            is_multi_turn=False,
            groups=(_group(turn_index=3, calls=(_call(turn_index=3),)),),
        )


def test_a_plan_may_not_give_one_assistant_turn_two_groups() -> None:
    with pytest.raises(ValidationError, match="two call groups"):
        CanonicalConversationPlan(
            task_id="t1",
            user_turns=1,
            assistant_turns=2,
            is_multi_turn=False,
            groups=(_group(), _group()),
        )


def test_a_plan_may_not_answer_a_user_turn_the_conversation_never_asks() -> None:
    with pytest.raises(ValidationError, match="never asks"):
        CanonicalConversationPlan(
            task_id="t1",
            user_turns=1,
            assistant_turns=1,
            is_multi_turn=False,
            groups=(_group(user_turn_index=3),),
        )


def test_a_plan_may_not_answer_an_earlier_user_turn_after_a_later_one() -> None:
    with pytest.raises(ValidationError, match="earlier user turn after a later one"):
        CanonicalConversationPlan(
            task_id="t1",
            user_turns=2,
            assistant_turns=2,
            is_multi_turn=True,
            groups=(
                _group(turn_index=0, user_turn_index=1),
                _group(turn_index=1, user_turn_index=0, calls=(_call(turn_index=1),)),
            ),
        )


def test_a_clarifying_user_turn_keeps_its_empty_answer_slot() -> None:
    plan = conversation_plan(_canonical(_multi_turn_row()))

    # The first user turn is answered with a question, not a call. Dropping its
    # empty slot would attribute the call to the wrong request.
    assert plan.calls_by_user_turn == ((), (plan.groups[0].calls[0],))
    assert plan.groups[0].user_turn_index == 1


def test_a_single_turn_row_answers_its_only_user_turn() -> None:
    plan = conversation_plan(_canonical())

    (turn,) = plan.calls_by_user_turn
    assert [call.function_name for call in turn] == ["get_balance"]


def test_a_tool_call_before_the_first_user_turn_is_refused() -> None:
    arguments = {"account_id": "1"}
    row = _row(
        messages=[
            _system(),
            _assistant_calls([("get_balance", arguments)]),
            _tool_result("call_0", {"balance": 10}),
            _user("Was that right?"),
            _assistant_text("Yes."),
        ],
        expected_tool_calls=[_expected("get_balance", arguments)],
    )

    with pytest.raises(ExportProjectionError, match="before it has been asked anything"):
        conversation_plan(_canonical(row))


def test_provenance_is_derived_from_every_row() -> None:
    provenance = derive_provenance([_canonical(), _canonical(_multi_turn_row())])

    assert provenance.pack_id == "pack"
    assert provenance.pack_version == "1.0.0"
    assert provenance.expt_name == "expt"
    assert provenance.tier == "gold"
    assert provenance.gold_eligible
    assert provenance.languages == ("en",)
    assert provenance.turn_policies == ("missing_slot", "single_turn")
    assert provenance.paraphrase_models == ()


def test_one_non_gold_row_makes_the_whole_projection_non_gold() -> None:
    provenance = derive_provenance([_canonical(), _canonical(_row("pack__tpl__other", gold_eligible=False))])

    assert not provenance.gold_eligible


def test_rows_from_two_packs_cannot_share_one_provenance() -> None:
    other = _row("other__tpl__abcdef", pack_id="other", src="other:tpl")

    with pytest.raises(ExportProjectionError, match="rows carry pack_id"):
        derive_provenance([_canonical(), _canonical(other)])


def test_rows_from_two_runs_cannot_share_one_provenance() -> None:
    # An export descriptor cites one run as its lineage, so two runs' rows in one
    # projection would make that citation false for some of them.
    other = _row("pack__tpl__other")
    other["metadata"] = canonical_json(
        {
            "language": "en",
            "expt_name": "another-run",
            "base_task_id": None,
            "surface_source": "template",
            "profile_hash": None,
        }
    )

    with pytest.raises(ExportProjectionError, match="rows carry expt_name"):
        derive_provenance([_canonical(), _canonical(other)])


def test_provenance_lists_must_be_sorted_and_distinct() -> None:
    with pytest.raises(ValidationError, match="sorted distinct values"):
        ProjectionProvenance(
            pack_id="pack",
            pack_version="1.0.0",
            expt_name="expt",
            tier="gold",
            gold_eligible=True,
            system_prompt_ids=("sp-2", "sp-1"),
            languages=("en",),
            turn_policies=("single_turn",),
        )


def test_a_projection_only_describes_the_published_benchmark_file() -> None:
    with pytest.raises(ValidationError, match="an export projects benchmark.parquet"):
        _source(file="benchmark_raw.parquet")


def test_a_projection_must_account_for_every_row_of_its_source() -> None:
    with pytest.raises(ExportProjectionError, match="account for every row"):
        project_benchmark_rows([_row()], source=_source(rows=2))


def test_a_projection_may_not_repeat_a_task() -> None:
    with pytest.raises(ExportProjectionError, match="one row per task"):
        project_benchmark_rows([_row(), _row()], source=_source(rows=2))


def test_a_held_out_hit_can_never_enter_any_compatibility_export() -> None:
    with pytest.raises(ExportProjectionError, match="held-out hit"):
        project_benchmark_rows([_row(held_out_hit=True)], source=_source())


def test_a_projection_reports_which_row_it_could_not_decode() -> None:
    broken = _row("pack__tpl__broken", tools="{not json")

    with pytest.raises(ExportProjectionError, match="benchmark.parquet row 1 cannot be projected"):
        project_benchmark_rows([_row(), broken], source=_source(rows=2))


def test_a_projection_keeps_publication_order() -> None:
    rows = [_row("t1"), _parallel_row("t2"), _multi_turn_row("t3")]

    projection = project_benchmark_rows(rows, source=_source(rows=3))

    assert projection.task_ids == ("t1", "t2", "t3")
    assert [plan.task_id for plan in projection.plans] == ["t1", "t2", "t3"]
    assert projection.row("t2").turn_policy == "multi_tool"
    assert projection.plan("t2").parallel_groups
    with pytest.raises(KeyError):
        projection.row("absent")
    with pytest.raises(KeyError):
        projection.plan("absent")


def test_a_projection_needs_a_plan_for_every_row() -> None:
    projection = project_benchmark_rows([_row("t1")], source=_source())

    with pytest.raises(ValidationError, match="in publication order"):
        CanonicalExportProjection(
            source=_source(rows=2),
            provenance=projection.provenance,
            rows=(projection.rows[0], _canonical(_row("t2"))),
            plans=projection.plans,
        )


def _write(path: Path, rows: list[dict[str, Any]], *, schema: Any = None) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows, schema=schema or benchmark_schema()), path)
    return path


def test_a_published_parquet_projects_and_cites_its_own_hash(tmp_path: Path) -> None:
    path = _write(tmp_path / "benchmark.parquet", [_row("t1"), _parallel_row("t2")])

    projection = project_published_benchmark(path, expected_task_ids=["t1", "t2"])

    assert projection.source.file == "benchmark.parquet"
    assert projection.source.rows == 2
    assert projection.source.content_hash.startswith("sha256:")
    assert project_published_benchmark(path, expected_content_hash=projection.source.content_hash).task_ids == (
        "t1",
        "t2",
    )


def test_a_missing_published_benchmark_cannot_be_projected(tmp_path: Path) -> None:
    with pytest.raises(ExportProjectionError, match="no published benchmark"):
        project_published_benchmark(tmp_path / "benchmark.parquet")


def test_a_parquet_written_with_another_schema_cannot_be_projected(tmp_path: Path) -> None:
    import pyarrow as pa

    path = _write(
        tmp_path / "benchmark.parquet",
        [{"task_id": "t1"}],
        schema=pa.schema([("task_id", pa.string())]),
    )

    with pytest.raises(ExportProjectionError, match="published benchmark schema"):
        project_published_benchmark(path)


def test_a_replaced_parquet_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "benchmark.parquet", [_row("t1")])

    with pytest.raises(ExportProjectionError, match="changed after publication"):
        project_published_benchmark(path, expected_content_hash=OTHER_HASH)


def test_a_parquet_in_another_order_than_publication_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "benchmark.parquet", [_row("t1"), _row("t2")])

    with pytest.raises(ExportProjectionError, match="in publication order"):
        project_published_benchmark(path, expected_task_ids=["t2", "t1"])


def test_the_lineage_describes_the_projection_for_a_bundle_descriptor(tmp_path: Path) -> None:
    path = _write(tmp_path / "benchmark.parquet", [_row("t1"), _parallel_row("t2"), _multi_turn_row("t3")])

    lineage = projection_lineage(project_published_benchmark(path))

    assert lineage["schema_version"] == EXPORT_PROJECTION_VERSION
    assert lineage["source"]["file"] == "benchmark.parquet"
    assert lineage["source"]["rows"] == 3
    assert lineage["pack_id"] == "pack"
    assert lineage["tools_exposed"] == ["get_balance", "list_cards"]
    assert lineage["turn_policies"] == ["missing_slot", "multi_tool", "single_turn"]
    assert lineage["multi_turn_rows"] == 1
    assert lineage["parallel_call_rows"] == 1
