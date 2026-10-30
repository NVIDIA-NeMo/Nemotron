"""Tests for the BFCL Stage 11 text projection."""

from __future__ import annotations

from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
    execution_case_hash,
    mask_slot_literals,
    project_dedup_text,
    project_dedup_texts,
    project_surface_text,
    slot_literals,
    user_turn_texts,
)


def _task(**overrides: Any) -> dict[str, Any]:
    task = {
        "task_id": "pack__tpl__aaa",
        "template_id": "tpl",
        "turn_policy": "single_turn",
        "slots_initial": {"account_id": "ACC-1001"},
        "slots": {"account_id": "ACC-1001"},
    }
    task.update(overrides)
    return task


def _surface(*contents: str, **overrides: Any) -> dict[str, Any]:
    surface = {
        "task_id": "pack__tpl__aaa",
        "language": "en",
        "steps": [{"kind": "user", "content": content} for content in contents],
    }
    surface.update(overrides)
    return surface


def test_projection_keeps_only_user_turns_in_conversation_order() -> None:
    surface = _surface(
        steps=[
            {"kind": "user", "content": "Check the balance of ACC-1001"},
            {"kind": "assistant_text", "content": "Which account should I use?"},
            {"kind": "calls", "call_group": 0},
            {"kind": "user", "content": "Use ACC-1001 please"},
        ],
    )

    assert user_turn_texts(surface) == [
        "Check the balance of ACC-1001",
        "Use ACC-1001 please",
    ]
    text, masked = project_surface_text(_task(), surface)
    assert text == "[user] Check the balance of <account_id>\n[user] Use <account_id> please"
    assert masked == ["account_id"]


def test_projection_omits_tool_payloads_and_oracle_truth() -> None:
    task = _task(
        expected_tool_calls=[{"name": "get_balance", "arguments": {"account_id": "ACC-1001"}}],
        success_assertions=["balance_matches"],
        expected_result={"balance": 4200},
    )
    record = project_dedup_text(task, _surface("Check the balance of ACC-1001"))

    assert "get_balance" not in record["text"]
    assert "balance_matches" not in record["text"]
    assert "4200" not in record["text"]


def test_masking_collapses_tasks_that_differ_only_in_bound_values() -> None:
    first = project_dedup_text(
        _task(slots_initial={"account_id": "ACC-1001"}, slots={"account_id": "ACC-1001"}),
        _surface("Check the balance of ACC-1001"),
    )
    second = project_dedup_text(
        _task(
            task_id="pack__tpl__bbb",
            slots_initial={"account_id": "ACC-2002"},
            slots={"account_id": "ACC-2002"},
        ),
        _surface("Check the balance of ACC-2002"),
    )

    assert first["text"] == second["text"]
    assert first["text_hash"] == second["text_hash"]


def test_projection_is_stable_under_whitespace_only_differences() -> None:
    spaced = project_dedup_text(_task(), _surface("Check   the balance\nof ACC-1001  "))
    plain = project_dedup_text(_task(), _surface("Check the balance of ACC-1001"))

    assert spaced["text"] == plain["text"]


def test_masking_never_corrupts_a_word_that_merely_contains_a_value() -> None:
    task = _task(slots_initial={"code": "an"}, slots={"code": "an"})

    masked, slots = mask_slot_literals("The bank confirmed an entry", task)

    assert masked == "The bank confirmed <code> entry"
    assert slots == ["code"]


def test_masking_prefers_the_longest_literal() -> None:
    task = _task(
        slots_initial={"account_id": "1001", "branch_id": "1001-A"},
        slots={"account_id": "1001", "branch_id": "1001-A"},
    )

    masked, slots = mask_slot_literals("Move from 1001-A to 1001", task)

    assert masked == "Move from <branch_id> to <account_id>"
    assert slots == ["account_id", "branch_id"]


def test_masking_numeric_slot_allows_a_localized_unit_suffix() -> None:
    task = _task(
        slots_initial={"amount": 500000},
        slots={"amount": 500000},
    )

    masked, slots = mask_slot_literals(
        "Chuyển 500000đ, không phải 1500000đ",
        task,
    )

    assert masked == "Chuyển <amount>đ, không phải 1500000đ"
    assert slots == ["amount"]


def test_masking_numeric_slot_accepts_grouped_localized_forms() -> None:
    task = _task(
        slots_initial={"amount": 500000},
        slots={"amount": 500000},
    )

    for rendered in (
        "500.000 đồng",
        "500,000đ",
        "500 000 VND",
        "500\u00a0000 VND",
        "500'000 CHF",
        "500\u2019000 CHF",
    ):
        masked, slots = mask_slot_literals(f"Chuyển {rendered}", task)
        assert masked.startswith("Chuyển <amount>")
        assert slots == ["amount"]

    # A grouping the pattern does not recognize must leave the text untouched rather
    # than mask a value the projection cannot identify.
    masked, slots = mask_slot_literals("Chuyển 5,00,000 INR", task)
    assert masked == "Chuyển 5,00,000 INR"
    assert slots == []


def test_a_shared_literal_masks_under_one_deterministic_slot_name() -> None:
    task = _task(
        slots_initial={"target_id": "ACC-1001", "source_id": "ACC-1001"},
        slots={"target_id": "ACC-1001", "source_id": "ACC-1001"},
    )

    assert slot_literals(task) == {"ACC-1001": "source_id"}
    masked, slots = mask_slot_literals("From ACC-1001 to ACC-1001", task)
    assert masked == "From <source_id> to <source_id>"
    assert slots == ["source_id"]


def test_correction_values_are_masked_on_both_sides_of_the_replacement() -> None:
    task = _task(
        turn_policy="correction",
        slots_initial={"account_id": "ACC-1001"},
        slots={"account_id": "ACC-2002"},
        slot_updates=[
            {"entry_index": 0, "values": {"account_id": "ACC-2002"}, "aliases": {"new_account": "ACC-2002"}}
        ],
    )
    text, masked = project_surface_text(
        task,
        _surface("Use ACC-1001", "Sorry, use ACC-2002 instead"),
    )

    assert text == "[user] Use <account_id>\n[user] Sorry, use <account_id> instead"
    assert masked == ["account_id"]


def test_correction_alias_literals_are_masked() -> None:
    task = _task(
        turn_policy="correction",
        slots_initial={"account_id": "ACC-1001"},
        slots={"account_id": "ACC-2002"},
        slot_updates=[
            {
                "entry_index": 0,
                "values": {"account_id": "ACC-2002"},
                "aliases": {"replacement_account": "SECONDARY"},
            }
        ],
    )

    text, masked = project_surface_text(
        task,
        _surface("Use ACC-1001", "Actually use SECONDARY"),
    )

    assert text == "[user] Use <account_id>\n[user] Actually use <replacement_account>"
    assert masked == ["account_id", "replacement_account"]


def test_projection_is_generic_to_a_pack_without_slots() -> None:
    task = {"task_id": "warehouse__tpl__aaa", "template_id": "tpl"}

    record = project_dedup_text(task, _surface("List every pallet in aisle four"))

    assert record["text"] == "[user] List every pallet in aisle four"
    assert record["masked_slots"] == []
    assert record["num_user_turns"] == 1


def test_execution_case_identity_is_independent_of_surface_wording() -> None:
    task = _task()
    paraphrased = {**task, "task_id": "banking__balance__paraphrase-1"}

    assert execution_case_hash(task) == execution_case_hash(paraphrased)
    assert project_dedup_text(task, _surface("Check ACC-1001"))[
        "execution_case_hash"
    ] == project_dedup_text(
        paraphrased,
        _surface("Please look up ACC-1001"),
    )["execution_case_hash"]

    changed = {
        **task,
        "slots": {"account_id": "ACC-2002"},
        "slots_initial": {"account_id": "ACC-2002"},
    }
    assert execution_case_hash(task) != execution_case_hash(changed)


def test_execution_case_identity_accounts_for_every_published_task_column() -> None:
    """A new task column must be classified, not silently left out of the identity."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
        task_instances_schema,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import dedup_balancing

    # Columns that describe where a row came from, how it was labelled, or which
    # rendering of one case it is — none of which changes what the row executes.
    not_executable_meaning = {
        "task_id",
        "base_task_id",
        "template_id",
        "pack_id",
        "pack_version",
        "difficulty",
        "variant_index",
        "seed",
        "mutates",
    }
    hashed = set(dedup_balancing.EXECUTION_CASE_KEYS)
    columns = set(task_instances_schema().names)

    assert not_executable_meaning <= columns
    assert columns - hashed == not_executable_meaning


def test_projection_rejects_a_surface_without_a_user_turn() -> None:
    surface = _surface(steps=[{"kind": "assistant_text", "content": "Anything else?"}])

    with pytest.raises(ValueError, match="projects no user turn"):
        project_surface_text(_task(), surface)


def test_projection_set_requires_one_surface_per_task_without_duplicates() -> None:
    task = _task()
    surfaces = {"pack__tpl__aaa": _surface("Check the balance of ACC-1001")}

    records = project_dedup_texts([task], surfaces)
    assert [record["task_id"] for record in records] == ["pack__tpl__aaa"]
    normalized = project_dedup_texts(
        [_task(task_id=" pack__tpl__aaa ")],
        surfaces,
    )
    assert normalized[0]["task_id"] == "pack__tpl__aaa"

    with pytest.raises(ValueError, match="without a surface"):
        project_dedup_texts([_task(task_id="pack__tpl__missing")], surfaces)
    with pytest.raises(ValueError, match="duplicate Stage 11 projection input"):
        project_dedup_texts([task, task], surfaces)


def test_projection_rejects_malformed_task_and_surface_shapes() -> None:
    with pytest.raises(ValueError, match="non-empty task_id"):
        project_dedup_text({"task_id": " "}, _surface("Check the balance"))
    with pytest.raises(ValueError, match="slots_initial that is not a mapping"):
        project_dedup_text(_task(slots_initial=["account_id"]), _surface("Check the balance"))
    with pytest.raises(ValueError, match="slot_updates that is not a list"):
        project_dedup_text(_task(slot_updates={"entry_index": 0}), _surface("Check the balance"))
    with pytest.raises(ValueError, match="slot update aliases that is not a mapping"):
        project_dedup_text(
            _task(slot_updates=[{"values": {}, "aliases": []}]),
            _surface("Check the balance"),
        )
    with pytest.raises(ValueError, match="unknown surface step kind"):
        project_dedup_text(_task(), _surface(steps=[{"kind": "tool_result", "content": "4200"}]))
