"""Unit rules of the BFCL generation stages that do not need a full run."""

from __future__ import annotations

import collections
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    apply_declared_defaults,
    validate_function_arguments,
    validate_function_schema,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
    decode_arguments,
    encode_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
    ExpansionError,
    primary_key_for,
    task_id_for,
    task_seed_for,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
    derive_pack_tier,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.schema_validation import (
    validate_task,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
    PlanError,
    build_plan,
)


class _Pack:
    """Minimal stand-in exposing only what the stages read."""

    def __init__(self, tools: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> None:
        self.tools = tools
        self.manifest = manifest or {}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "do_thing",
            "parameters": {
                "type": "object",
                "properties": {
                    "thing_id": {"type": "string"},
                    "count": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["fast", "slow"]},
                },
                "required": ["thing_id"],
                "additionalProperties": False,
            },
        },
    }
]

TASK = {
    "task_id": "pack__tpl__0123456789abcdef",
    "template_id": "tpl",
    "tools_present": ["do_thing"],
    "required_tools": ["do_thing"],
    "call_order": "strict",
    "call_order_prefix": None,
    "turn_policy": "single_turn",
}


def _call(**arguments: Any) -> dict[str, Any]:
    return {
        "turn_index": 0,
        "call_group": 0,
        "position_in_group": 0,
        "function_name": "do_thing",
        "arguments": arguments,
    }


def test_task_id_and_seed_depend_only_on_declared_inputs() -> None:
    kwargs = {
        "pack_id": "p",
        "pack_version": "1.0",
        "template_id": "t",
        "fixture_refs": ["c.B", "c.A"],
        "slot_bindings": {"b": 2, "a": 1},
        "variant_index": 0,
    }
    first = task_id_for(**kwargs)
    reordered = task_id_for(**{**kwargs, "fixture_refs": ["c.A", "c.B"], "slot_bindings": {"a": 1, "b": 2}})

    assert first == reordered
    assert first.startswith("p__t__")
    assert len(first.rsplit("__", 1)[-1]) == 16
    assert task_id_for(**{**kwargs, "variant_index": 1}) != first

    seed_kwargs = {
        "global_seed": 7,
        "pack_id": "p",
        "pack_version": "1.0",
        "template_id": "t",
        "fixture_refs": ["c.B", "c.A"],
        "slot_bindings": {"b": 2, "a": 1},
        "variant_index": 0,
    }
    seed = task_seed_for(**seed_kwargs)
    assert seed == task_seed_for(
        **{
            **seed_kwargs,
            "fixture_refs": ["c.A", "c.B"],
            "slot_bindings": {"a": 1, "b": 2},
        }
    )
    assert 0 <= seed < 2**64
    assert seed != task_seed_for(**{**seed_kwargs, "global_seed": 8})
    assert seed != task_seed_for(**{**seed_kwargs, "slot_bindings": {"a": 2, "b": 2}})


def test_primary_key_prefers_declaration_over_convention() -> None:
    rows = [{"code": "X-1", "other_id": "O-1"}]
    assert primary_key_for({"primary_keys": {"things": "code"}}, "things", rows) == "code"
    assert primary_key_for({}, "things", rows) == "other_id"


def test_primary_key_refuses_to_guess_between_a_key_and_a_foreign_key() -> None:
    """Guessing wrong attributes a task to a record it was not built from."""
    rows = [{"card_id": "C-1", "account_id": "A-1", "limit": 10}]
    assert primary_key_for({}, "cards", rows) == "card_id"

    ambiguous = [{"owner_id": "O-1", "holder_id": "H-1"}]
    with pytest.raises(ExpansionError, match="declare primary_keys.things"):
        primary_key_for({}, "things", ambiguous)

    with pytest.raises(ExpansionError, match="rows do not carry"):
        primary_key_for({"primary_keys": {"things": "absent_field"}}, "things", ambiguous)


def test_arguments_codec_round_trips_types() -> None:
    arguments = {"thing_id": "T-1", "count": 3, "flag": False, "nested": {"a": [1, 2]}}
    assert decode_arguments(encode_arguments(arguments)) == arguments


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"value": float("inf")})


def test_parallel_milestones_share_one_assistant_turn() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "single_turn",
        "assistant_milestones": [
            {
                "type": "tool_call",
                "tool": "do_thing",
                "call_group": 0,
            },
            {
                "type": "tool_call",
                "tool": "do_thing",
                "call_group": 0,
            },
            {"type": "final_answer"},
        ],
    }
    plan = build_plan(template, TASK)

    call_steps = [step for step in plan["steps"] if step["kind"] == "calls"]
    assert len(call_steps) == 1
    assert len(call_steps[0]["milestones"]) == 2
    assert plan["num_tool_calls"] == 2
    assert plan["is_multi_turn"] is False


def test_unmarked_milestones_stay_sequential() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "multi_tool",
        "assistant_milestones": [
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "final_answer"},
        ],
    }
    plan = build_plan(template, TASK)

    assert len([step for step in plan["steps"] if step["kind"] == "calls"]) == 2


def test_ask_confirm_requires_a_deterministic_user_reply() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "confirmation",
        "assistant_milestones": [
            {"type": "ask_confirm"},
            {"type": "tool_call", "tool": "do_thing", "args": {"confirm": True}},
            {"type": "final_answer"},
        ],
    }
    with pytest.raises(PlanError, match="user_simulator_turns"):
        build_plan(template, TASK)


def test_template_requires_an_assistant_milestone() -> None:
    with pytest.raises(PlanError, match="assistant milestone"):
        build_plan(
            {"template_id": "empty", "turn_policy": "single_turn", "assistant_milestones": []},
            TASK,
        )


def test_split_call_group_is_rejected() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "multi_tool",
        "assistant_milestones": [
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
            {"type": "final_answer"},
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
        ],
    }
    with pytest.raises(PlanError, match="splits call_group 0"):
        build_plan(template, TASK)


def test_simulator_turn_can_reference_any_milestone_in_a_batch() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "multi_tool",
        "assistant_milestones": [
            {"type": "tool_call", "tool": "do_thing", "call_group": 0, "id": "first"},
            {"type": "tool_call", "tool": "do_thing", "call_group": 0, "id": "second"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "second", "content_template": {"en": "and also"}}],
    }
    plan = build_plan(template, TASK)

    assert [step.get("source") for step in plan["steps"] if step["kind"] == "user"] == [
        "first_turn",
        "simulator",
    ]


def _dependent_template(**overrides: Any) -> dict[str, Any]:
    consumer = {
        "type": "tool_call",
        "tool": "do_thing",
        "call_group": 1,
        "args": {"thing_id": {"from_result": {"call": "first", "path": "items.0.id"}}},
    }
    consumer.update(overrides)
    return {
        "template_id": "tpl",
        "turn_policy": "dependent_call",
        "assistant_milestones": [
            {"id": "first", "type": "tool_call", "tool": "do_thing", "call_group": 0},
            consumer,
            {"type": "final_answer"},
        ],
    }


DEPENDENT_TASK = {**TASK, "turn_policy": "dependent_call", "slots": {"thing_id": "T-1"}}


def test_dependent_call_binds_its_argument_from_a_prior_result() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        build_expected_calls,
    )

    template = _dependent_template()
    prefixes: list[list[str]] = []

    def resolve(prefix: list[dict[str, Any]]) -> list[Any]:
        prefixes.append([call["function_name"] for call in prefix])
        return [{"items": [{"id": "T-9"}]}]

    calls = build_expected_calls(
        _Pack(TOOLS),
        DEPENDENT_TASK,
        build_plan(template, DEPENDENT_TASK),
        resolve_results=resolve,
    )

    # The marker wins over the same-named slot, which is the whole point: the id
    # exists only in the first call's output.
    assert calls[0]["arguments"] == {"thing_id": "T-1"}
    assert calls[1]["arguments"] == {"thing_id": "T-9"}
    assert calls[1]["turn_index"] == 1
    assert prefixes == [["do_thing"]]


def test_dependent_call_resolves_a_trace_in_one_incremental_episode() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        build_expected_calls,
    )

    invocations = 0

    def resolve_trace(steps):  # type: ignore[no-untyped-def]
        nonlocal invocations
        invocations += 1
        iterator = iter(steps)
        assert next(iterator) == {"op": "reset"}
        producer = iterator.send(None)
        assert producer["name"] == "do_thing"
        consumer = iterator.send({"items": [{"id": "T-9"}]})
        assert consumer["arguments"] == {"thing_id": "T-9"}
        with pytest.raises(StopIteration):
            iterator.send({"thing_id": "T-9", "ok": True})
        return [{"items": [{"id": "T-9"}]}, {"thing_id": "T-9", "ok": True}]

    template = _dependent_template()
    calls = build_expected_calls(
        _Pack(TOOLS),
        DEPENDENT_TASK,
        build_plan(template, DEPENDENT_TASK),
        resolve_trace=resolve_trace,
    )

    assert invocations == 1
    assert calls[1]["arguments"] == {"thing_id": "T-9"}


def test_dependent_call_needs_the_oracle_and_a_later_call_group() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        build_expected_calls,
    )

    def resolve(prefix: list[dict[str, Any]]) -> list[Any]:
        return [{"items": [{"id": "T-9"}]}]

    template = _dependent_template()
    with pytest.raises(ExpectedTraceError, match="needs oracle results"):
        build_expected_calls(_Pack(TOOLS), DEPENDENT_TASK, build_plan(template, DEPENDENT_TASK))

    shared_group = _dependent_template(call_group=0)
    with pytest.raises(ExpectedTraceError, match="call_group"):
        build_expected_calls(
            _Pack(TOOLS),
            DEPENDENT_TASK,
            build_plan(shared_group, DEPENDENT_TASK),
            resolve_results=resolve,
        )

    unknown = _dependent_template(args={"thing_id": {"from_result": {"call": "nope", "path": "items.0.id"}}})
    with pytest.raises(ExpectedTraceError, match="unknown milestone id"):
        build_expected_calls(
            _Pack(TOOLS),
            DEPENDENT_TASK,
            build_plan(unknown, DEPENDENT_TASK),
            resolve_results=resolve,
        )


def test_dependent_policy_and_result_reference_must_agree() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        build_expected_calls,
    )

    def resolve(prefix: list[dict[str, Any]]) -> list[Any]:
        return [{"items": [{"id": "T-9"}]}]

    template = _dependent_template()
    mislabelled = {**DEPENDENT_TASK, "turn_policy": "multi_tool"}
    with pytest.raises(ExpectedTraceError, match="declare dependent_call"):
        build_expected_calls(
            _Pack(TOOLS),
            mislabelled,
            build_plan(template, mislabelled),
            resolve_results=resolve,
        )

    slot_bound = {
        "template_id": "tpl",
        "turn_policy": "dependent_call",
        "assistant_milestones": [
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
            {"type": "tool_call", "tool": "do_thing", "call_group": 1},
            {"type": "final_answer"},
        ],
    }
    with pytest.raises(ExpectedTraceError, match="from_result"):
        build_expected_calls(
            _Pack(TOOLS),
            DEPENDENT_TASK,
            build_plan(slot_bound, DEPENDENT_TASK),
            resolve_results=resolve,
        )


def test_dependent_call_refuses_a_failed_or_missing_producer_value() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        build_expected_calls,
    )

    template = _dependent_template()
    plan = build_plan(template, DEPENDENT_TASK)

    with pytest.raises(ExpectedTraceError, match="returned an error"):
        build_expected_calls(
            _Pack(TOOLS),
            DEPENDENT_TASK,
            plan,
            resolve_results=lambda prefix: [{"error": {"code": "not_found"}}],
        )
    with pytest.raises(ExpectedTraceError, match="out of range"):
        build_expected_calls(
            _Pack(TOOLS),
            DEPENDENT_TASK,
            plan,
            resolve_results=lambda prefix: [{"items": []}],
        )


def test_withheld_slot_may_be_revealed_in_a_later_user_turn() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {
        "template_id": "tpl",
        "slots": {"thing_id": {"visible_in_first_turn": False}},
        "paraphrase": {"allowed": False},
    }
    task = {"slots": {"thing_id": "T-1"}}

    assert check_surface_guards(template, task, ["Check my thing.", "It is T-1."], ["do_thing"]) == []
    assert check_surface_guards(template, task, ["Check thing T-1."], ["do_thing"]) == [
        {"guard": "must_omit", "slot": "thing_id"}
    ]


def test_missing_slot_policy_must_collect_the_slot_it_withholds() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "missing_slot",
        "slots": {
            "secret": {"visible_in_first_turn": False},
            "public": {"visible_in_first_turn": True},
        },
        "assistant_milestones": [
            {"id": "ask", "type": "ask_for_slot", "slot": "public"},
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "ask", "content_template": {"en": "Public is {public}."}}],
    }
    task = {**TASK, "turn_policy": "missing_slot"}

    with pytest.raises(PlanError, match="not a withheld slot"):
        build_plan(template, task)


def _correction_template(**overrides: Any) -> dict[str, Any]:
    template = {
        "template_id": "tpl",
        "turn_policy": "correction",
        "slots": {
            "thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": True},
            "count": {"source": "literal:[10]", "visible_in_first_turn": True},
        },
        "user_turn_templates": {"en": "Do thing {thing_id} {count} times."},
        "assistant_milestones": [
            {"id": "confirm_first", "type": "ask_confirm"},
            {"id": "confirm_again", "type": "ask_confirm"},
            {
                "type": "tool_call",
                "tool": "do_thing",
                "call_group": 0,
                "args": {"count": "{count}"},
            },
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [
            {
                "after": "confirm_first",
                "content_template": {"en": "Make it {count_new} instead."},
                "slot_updates": {
                    "count": {"source": "literal:[20]", "bind_as": "count_new"},
                },
            },
            {"after": "confirm_again", "content_template": {"en": "Yes, go ahead."}},
        ],
    }
    template.update(overrides)
    return template


CORRECTION_TASK = {
    **TASK,
    "turn_policy": "correction",
    "slots": {"thing_id": "T-1", "count": 20},
    "slots_initial": {"thing_id": "T-1", "count": 10},
    "slot_updates": [{"entry_index": 0, "values": {"count": 20}, "aliases": {"count_new": 20}}],
}


def test_correction_binds_the_call_to_the_replacement_value() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        build_expected_calls,
    )

    plan = build_plan(_correction_template(), CORRECTION_TASK)
    calls = build_expected_calls(_Pack(TOOLS), CORRECTION_TASK, plan)

    assert [call["arguments"] for call in calls] == [{"thing_id": "T-1", "count": 20}]


def test_correction_refuses_a_call_that_kept_the_replaced_value() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        build_expected_calls,
    )

    # Issuing the call before the correction turn means it consumes the value the
    # user is about to withdraw.
    early = _correction_template(
        assistant_milestones=[
            {
                "type": "tool_call",
                "tool": "do_thing",
                "call_group": 0,
                "args": {"count": "{count}"},
            },
            {"id": "confirm_first", "type": "ask_confirm"},
            {"id": "confirm_again", "type": "ask_confirm"},
            {"type": "final_answer"},
        ]
    )
    with pytest.raises(ExpectedTraceError, match="the value the user replaced"):
        build_expected_calls(_Pack(TOOLS), CORRECTION_TASK, build_plan(early, CORRECTION_TASK))


def test_correction_withdraws_the_confirmation_it_replaces() -> None:
    plan = build_plan(_correction_template(), CORRECTION_TASK)
    assert plan["has_slot_correction"] is True
    assert plan["has_user_confirmation"] is True

    # Without a second ask_confirm the only confirmation predates the correction, so
    # a confirmed mutation must not pass validation.
    once = _correction_template(
        assistant_milestones=[
            {"id": "confirm_first", "type": "ask_confirm"},
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
            {"type": "final_answer"},
        ],
        user_simulator_turns=[
            {
                "after": "confirm_first",
                "content_template": {"en": "Make it {count_new} instead."},
                "slot_updates": {"count": {"source": "literal:[20]", "bind_as": "count_new"}},
            }
        ],
    )
    stale = build_plan(once, CORRECTION_TASK)
    assert stale["has_user_confirmation"] is False

    pack = SimpleNamespace(
        manifest={},
        tools=[{**TOOLS[0], "x-requires-confirmation": True}],
    )
    task = {**CORRECTION_TASK, "confirmed_call_turns": stale["confirmed_call_turns"]}
    reasons = {
        failure["reason"]
        for failure in validate_task(pack, task, [_call(thing_id="T-1", count=20) | {"turn_index": 1}])
    }
    assert "confirmed_mutation_without_user_confirmation" not in reasons

    confirmed = _call(thing_id="T-1", count=20, mode="fast") | {"turn_index": 1}
    confirmed["arguments"]["confirm"] = True
    reasons = {failure["reason"] for failure in validate_task(pack, task, [confirmed])}
    assert "confirmed_mutation_without_user_confirmation" in reasons


def test_validation_rejects_a_published_trace_holding_the_replaced_value() -> None:
    failures = validate_task(_Pack(TOOLS), CORRECTION_TASK, [_call(thing_id="T-1", count=10)])
    assert {
        "reason": "superseded_slot_value_in_trace",
        "tool": "do_thing",
        "argument": "count",
        "slot": "count",
    } in failures


def test_correction_renders_each_turn_with_the_value_in_force() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import render_task

    template = _correction_template()
    template["assistant_milestones"][0]["content_template"] = {"en": "Confirm {count}?"}
    template["assistant_milestones"][1]["content_template"] = {"en": "Confirm {count} now?"}

    surface = render_task(
        SimpleNamespace(
            manifest={"assistant_turn_templates": {"final_answer": {"en": "Done."}}},
            tools=TOOLS,
        ),
        template,
        CORRECTION_TASK,
        build_plan(template, CORRECTION_TASK),
        language="en",
        prompt_bundle={"system_prompt": "s", "system_prompt_id": "sha256:0"},
        tool_names=["do_thing"],
    )
    contents = [step.get("content") for step in surface["steps"]]

    assert contents[0] == "Do thing T-1 10 times."
    assert contents[1] == "Confirm 10?"
    assert contents[2] == "Make it 20 instead."
    assert contents[3] == "Confirm 20 now?"
    assert surface["guard_violations"] == []


def test_render_rejects_empty_user_facing_turns() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        RenderError,
        render_task,
    )

    pack = SimpleNamespace(
        manifest={"assistant_turn_templates": {"final_answer": {"en": "Done."}}},
        tools=TOOLS,
    )
    prompt_bundle = {
        "system_prompt": "s",
        "system_prompt_id": "sha256:0",
    }

    empty_user = _correction_template()
    empty_user["user_turn_templates"] = {"en": "   "}
    with pytest.raises(RenderError, match="empty user-facing turn"):
        render_task(
            pack,
            empty_user,
            CORRECTION_TASK,
            build_plan(empty_user, CORRECTION_TASK),
            language="en",
            prompt_bundle=prompt_bundle,
            tool_names=["do_thing"],
        )

    empty_assistant = _correction_template()
    empty_assistant["assistant_milestones"][0]["content_template"] = {"en": "   "}
    with pytest.raises(RenderError, match="empty user-facing turn"):
        render_task(
            pack,
            empty_assistant,
            CORRECTION_TASK,
            build_plan(empty_assistant, CORRECTION_TASK),
            language="en",
            prompt_bundle=prompt_bundle,
            tool_names=["do_thing"],
        )


def test_correction_expansion_binds_both_values_and_refuses_a_no_op() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    pack = SimpleNamespace(manifest={"pack_id": "pack", "version": "1.0"}, fixtures={}, tools=TOOLS)
    tasks = expand_template(pack, _correction_template(), 4, 0)

    assert [task["slots"]["count"] for task in tasks] == [20]
    assert [task["slots_initial"]["count"] for task in tasks] == [10]
    assert tasks[0]["slot_updates"] == [{"entry_index": 0, "values": {"count": 20}, "aliases": {"count_new": 20}}]

    no_op = _correction_template(
        user_simulator_turns=[
            {
                "after": "confirm_first",
                "content_template": {"en": "Make it {count_new} instead."},
                "slot_updates": {"count": {"source": "literal:[10]", "bind_as": "count_new"}},
            },
            {"after": "confirm_again", "content_template": {"en": "Yes, go ahead."}},
        ]
    )
    with pytest.raises(ExpansionError, match="equalled the original"):
        expand_template(pack, no_op, 4, 0)


def test_correction_declaration_must_match_its_policy_and_slots() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    pack = SimpleNamespace(manifest={"pack_id": "pack", "version": "1.0"}, fixtures={}, tools=TOOLS)

    mislabelled = _correction_template(turn_policy="confirmation")
    with pytest.raises(ExpansionError, match="declare correction"):
        expand_template(pack, mislabelled, 4, 0)

    with pytest.raises(ExpansionError, match="declares no slot_updates"):
        expand_template(pack, _correction_template(user_simulator_turns=[]), 4, 0)

    hidden = _correction_template()
    hidden["slots"]["count"]["visible_in_first_turn"] = False
    with pytest.raises(ExpansionError, match="replaces hidden slot"):
        expand_template(pack, hidden, 4, 0)

    other_kind = _correction_template()
    other_kind["user_simulator_turns"][0]["slot_updates"]["count"]["source"] = "enum:do_thing.mode"
    with pytest.raises(ExpansionError, match="same way"):
        expand_template(pack, other_kind, 4, 0)


def test_expansion_shares_one_budget_across_a_category(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import run_expand

    def template(template_id: str, category: str) -> dict[str, Any]:
        return {
            "template_id": template_id,
            "category": category,
            "slots": {"thing_id": {"source": "fixture:things.thing_id"}},
        }

    pack = SimpleNamespace(
        manifest={"pack_id": "p", "version": "1.0", "primary_keys": {"things": "thing_id"}},
        fixtures={"things": [{"thing_id": f"T-{index}"} for index in range(5)]},
        tools=TOOLS,
        templates=[
            template("a", "shared"),
            template("b", "shared"),
            template("c", "other"),
        ],
        held_out=None,
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 3},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    tasks = run_expand(config, pack)

    by_template = collections.Counter(str(task["template_id"]) for task in tasks)
    assert by_template == {"a": 2, "b": 1, "c": 3}

    config.task_generation = {"tasks_per_category": 1}
    with pytest.raises(ExpansionError, match="tasks_per_category"):
        run_expand(config, pack)


def test_narrow_expansion_budget_spreads_across_multiple_slots() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    pack = SimpleNamespace(
        manifest={"pack_id": "pack", "version": "1.0"},
        fixtures={},
        tools=TOOLS,
    )
    template = {
        "template_id": "two-wide-slots",
        "category": "coverage",
        "slots": {
            "a": {"source": "literal:[1, 2, 3, 4]"},
            "b": {"source": 'literal:["x", "y", "z", "w"]'},
        },
    }

    tasks = expand_template(pack, template, 4, 17)

    assert len(tasks) == 4
    assert len({task["slots"]["a"] for task in tasks}) > 1
    assert len({task["slots"]["b"] for task in tasks}) > 1
    assert [task["task_id"] for task in tasks] == [
        task["task_id"] for task in expand_template(pack, template, 4, 17)
    ]


def test_confirmed_mutation_is_gold_after_a_user_confirmation() -> None:
    tools = [
        {
            **TOOLS[0],
            "x-requires-confirmation": True,
            "function": {
                **TOOLS[0]["function"],
                "parameters": {
                    **TOOLS[0]["function"]["parameters"],
                    "properties": {
                        **TOOLS[0]["function"]["parameters"]["properties"],
                        "confirm": {"type": "boolean"},
                    },
                },
            },
        }
    ]
    # A negative_path template may also confirm, so the gate reads the planned
    # conversation rather than the policy label.
    task = {**TASK, "turn_policy": "negative_path", "confirmed_call_turns": [1]}
    call = {**_call(thing_id="T-1", confirm=True), "turn_index": 1}

    assert validate_task(_Pack(tools), task, [call]) == []

    # The same call one turn earlier is not covered by that confirmation.
    early = {**call, "turn_index": 0}
    assert [failure["reason"] for failure in validate_task(_Pack(tools), task, [early])] == [
        "confirmed_mutation_without_user_confirmation"
    ]


def test_irrelevant_template_must_end_in_decline() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "irrelevant",
        "assistant_milestones": [{"type": "decline"}, {"type": "final_answer"}],
    }
    with pytest.raises(PlanError, match="must end in 'decline'"):
        build_plan(template, TASK)


@pytest.mark.parametrize(
    ("label", "template", "complaint"),
    [
        (
            "single_turn that waits for a reply",
            {
                "turn_policy": "single_turn",
                "assistant_milestones": [
                    {"type": "ask_confirm"},
                    {"type": "tool_call", "tool": "do_thing"},
                    {"type": "final_answer"},
                ],
                "user_simulator_turns": [{"after": "ask_confirm", "content_template": {"en": "Yes."}}],
            },
            "plans 2 user turns",
        ),
        (
            "multi_tool with one call",
            {
                "turn_policy": "multi_tool",
                "assistant_milestones": [
                    {"type": "tool_call", "tool": "do_thing"},
                    {"type": "final_answer"},
                ],
            },
            "plans a single tool call",
        ),
        (
            "missing_slot that withholds nothing",
            {
                "turn_policy": "missing_slot",
                "assistant_milestones": [
                    {"type": "tool_call", "tool": "do_thing"},
                    {"type": "final_answer"},
                ],
            },
            "withholds no slot",
        ),
        (
            "confirmation nobody approved",
            {
                "turn_policy": "confirmation",
                "assistant_milestones": [
                    {"type": "tool_call", "tool": "do_thing"},
                    {"type": "final_answer"},
                ],
            },
            "no call batch a user reply authorized",
        ),
        (
            "correction that replaces nothing",
            {
                "turn_policy": "correction",
                "assistant_milestones": [
                    {"type": "tool_call", "tool": "do_thing"},
                    {"type": "final_answer"},
                ],
            },
            "replaces no slot value",
        ),
        (
            "negative_path nothing pins",
            {
                "turn_policy": "negative_path",
                "assistant_milestones": [
                    {"type": "tool_call", "tool": "do_thing"},
                    {"type": "final_answer"},
                ],
            },
            "declares no success_assertions",
        ),
        (
            "calling policy that calls nothing",
            {
                "turn_policy": "single_turn",
                "assistant_milestones": [{"type": "final_answer"}],
            },
            "plans no tool call",
        ),
    ],
)
def test_a_plan_must_match_the_policy_the_template_declares(
    label: str, template: dict[str, Any], complaint: str
) -> None:
    """turn_policy is what consumers slice rows by, so a mislabeled shape is refused."""
    declared = {"template_id": "tpl", "slots": {}, **template}
    task = {**TASK, "turn_policy": declared["turn_policy"]}
    with pytest.raises(PlanError, match=complaint):
        build_plan(declared, task)


def test_ambiguous_after_reference_is_rejected() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "correction",
        "assistant_milestones": [
            {"type": "ask_for_slot"},
            {"type": "ask_for_slot"},
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "ask_for_slot", "content_template": {"en": "ok"}}],
    }
    with pytest.raises(PlanError):
        build_plan(template, TASK)


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({}, "missing_required_argument"),
        ({"thing_id": "T-1", "extra": 1}, "unknown_argument"),
        ({"thing_id": 5}, "argument_type_mismatch"),
        ({"thing_id": "T-1", "count": True}, "argument_type_mismatch"),
        ({"thing_id": "T-1", "mode": "sideways"}, "argument_not_in_enum"),
    ],
)
def test_schema_validation_rejects_bad_arguments(arguments: dict[str, Any], reason: str) -> None:
    failures = validate_task(_Pack(TOOLS), TASK, [_call(**arguments)])
    assert reason in {failure["reason"] for failure in failures}


def test_schema_validation_accepts_a_bound_call() -> None:
    assert validate_task(_Pack(TOOLS), TASK, [_call(thing_id="T-1", mode="fast")]) == []


def test_schema_validation_checks_nested_objects_and_arrays() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do_thing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "pattern": r"^[A-Z]+$"},
                                "counts": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 1},
                                },
                            },
                            "required": ["label"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    task = {**TASK, "required_tools": ["do_thing"]}
    failures = validate_task(
        _Pack(tools),
        task,
        [_call(payload={"label": "lower", "counts": [1, 0], "extra": True})],
    )
    assert {failure["reason"] for failure in failures} >= {
        "string_pattern_mismatch",
        "number_below_minimum",
        "unknown_argument",
    }


def test_tool_schema_validation_rejects_unsupported_or_malformed_schema() -> None:
    function = {
        "name": "bad",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "imaginary", "oneOf": [{"type": "string"}]},
            },
            "required": ["missing"],
        },
    }
    assert {failure["reason"] for failure in validate_function_schema(function)} == {
        "invalid_schema_type",
        "required_property_not_declared",
        "unsupported_schema_keyword",
    }


@pytest.mark.parametrize(
    ("property_schema", "reason"),
    [
        ({"type": "string", "pattern": 123}, "schema_pattern_not_string"),
        (
            {"type": "integer", "minimum": 10, "maximum": 1},
            "inconsistent_schema_bounds",
        ),
        (
            {"type": "string", "minLength": 5, "maxLength": 1},
            "inconsistent_schema_bounds",
        ),
        # An enum nothing can satisfy, or a fixed value the declared type rejects,
        # leaves the template unable to bind a valid argument at all.
        ({"type": "string", "enum": []}, "schema_enum_empty"),
        ({"type": "string", "enum": ["x", "x"]}, "duplicate_schema_enum_value"),
        ({"type": "string", "enum": [1]}, "schema_value_violates_type"),
        ({"type": "string", "const": 3}, "schema_value_violates_type"),
        ({"type": ["string", "string"]}, "duplicate_schema_type"),
    ],
)
def test_tool_schema_validation_rejects_malformed_constraints(
    property_schema: dict[str, Any],
    reason: str,
) -> None:
    function = {
        "name": "bad",
        "parameters": {
            "type": "object",
            "properties": {"value": property_schema},
        },
    }
    assert reason in {failure["reason"] for failure in validate_function_schema(function)}


def test_tool_schema_validation_rejects_a_repeated_required_property() -> None:
    function = {
        "name": "bad",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value", "value"],
        },
    }
    assert {failure["reason"] for failure in validate_function_schema(function)} == {"duplicate_required_property"}


def test_tool_schema_validation_keeps_distinct_values_of_different_types() -> None:
    """``1`` and ``true`` are separate JSON values, so a union may offer both."""
    function = {
        "name": "fine",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": ["integer", "boolean"], "enum": [1, True]}},
        },
    }
    assert validate_function_schema(function) == []


def test_tool_schema_validation_supports_local_refs_all_of_and_inherited_defaults() -> None:
    function = {
        "name": "bounded",
        "parameters": {
            "type": "object",
            "$defs": {"limit": {"type": "integer", "minimum": 1, "default": 5}},
            "properties": {
                "limit": {
                    "allOf": [
                        {"$ref": "#/$defs/limit"},
                        {"maximum": 10},
                    ]
                }
            },
            "additionalProperties": False,
        },
    }

    assert validate_function_schema(function) == []
    assert apply_declared_defaults({}, function["parameters"]) == {"limit": 5}
    assert validate_function_arguments(function, {"limit": 0}) == [
        {"reason": "number_below_minimum", "argument": "limit"}
    ]
    assert validate_function_arguments(function, {"limit": 11}) == [
        {"reason": "number_above_maximum", "argument": "limit"}
    ]


def test_tool_schema_validation_rejects_an_operational_default_that_violates_its_schema() -> None:
    function = {
        "name": "bad_default",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "default": "many"},
            },
        },
    }

    assert validate_function_schema(function) == [
        {
            "reason": "invalid_schema_default",
            "path": "$.limit",
            "failure": "argument_type_mismatch",
        }
    ]


@pytest.mark.parametrize(
    ("reference", "reason"),
    [
        ("https://example.invalid/schema.json", "schema_ref_not_local"),
        ("#/$defs/missing", "unresolvable_schema_ref"),
        ("#/$defs/loop", "cyclic_schema_ref"),
    ],
)
def test_tool_schema_validation_rejects_unsafe_local_references(reference: str, reason: str) -> None:
    function = {
        "name": "bad_ref",
        "parameters": {
            "type": "object",
            "$defs": {"loop": {"$ref": "#/$defs/loop"}},
            "properties": {"value": {"$ref": reference}},
        },
    }

    assert reason in {failure["reason"] for failure in validate_function_schema(function)}


@pytest.mark.parametrize(
    ("constraint", "reason"),
    [
        ({"enum": [True]}, "argument_not_in_enum"),
        ({"const": True}, "argument_not_equal_const"),
    ],
)
def test_argument_constraints_keep_json_booleans_distinct_from_integers(
    constraint: dict[str, Any], reason: str
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import _check_value

    schema = {"type": ["integer", "boolean"], **constraint}
    assert _check_value(schema, 1, "value") == [{"reason": reason, "argument": "value"}]


def test_message_builder_rejects_trace_surface_mismatches() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        build_messages,
    )

    surface = {
        "system_prompt": "system",
        "steps": [{"kind": "user", "content": "hello"}, {"kind": "calls", "call_group": 0}],
    }
    with pytest.raises(ValueError, match="no expected calls"):
        build_messages(surface, [], [])
    with pytest.raises(ValueError, match="results"):
        build_messages(surface, [_call(thing_id="T-1")], [])


def test_replay_attempts_use_identical_task_context() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.executable_replay import (
        replay_task,
    )

    class RecordingWorker:
        def __init__(self) -> None:
            self.task_ids: list[str] = []

        def run_episode(self, **kwargs: Any) -> list[Any]:
            self.task_ids.append(kwargs["task_id"])
            outputs: list[Any] = []
            for step in kwargs["steps"]:
                if step["op"] == "call_tool":
                    outputs.append({"context_task_id": kwargs["task_id"]})
                elif step["op"] == "get_state":
                    outputs.append({})
                elif step["op"] == "run_assertion":
                    outputs.append({"name": step["name"], "passed": True, "detail": None})
                else:
                    outputs.append(None)
            return outputs

    runtime = SimpleNamespace(
        clock="2026-03-02T09:00:00+07:00",
        import_timeout_s=1.0,
        reset_timeout_s=1.0,
        tool_timeout_s=1.0,
        assertion_timeout_s=1.0,
        episode_timeout_s=5.0,
    )
    config = SimpleNamespace(oracle_runtime=runtime)
    pack = SimpleNamespace(
        manifest={},
        fixtures={},
        held_out=None,
        paths=SimpleNamespace(
            backend_path="backend.py",
            assertions_path="assertions.py",
            pack_root="pack",
        ),
    )
    task = {**TASK, "seed": 7, "success_assertions": []}
    worker = RecordingWorker()

    verdict = replay_task(worker, config, pack, task, [_call(thing_id="T-1")])

    assert verdict["passed"] is True
    assert worker.task_ids == [TASK["task_id"], TASK["task_id"]]


def test_schema_validation_rejects_calls_the_template_did_not_expose() -> None:
    task = {**TASK, "tools_present": [], "required_tools": []}
    failures = validate_task(_Pack(TOOLS), task, [_call(thing_id="T-1")])
    assert "tool_not_exposed" in {failure["reason"] for failure in failures}


def test_confirmed_mutation_requires_confirmation_conversation() -> None:
    tools = [
        {
            **TOOLS[0],
            "x-requires-confirmation": True,
            "function": {
                **TOOLS[0]["function"],
                "parameters": {
                    **TOOLS[0]["function"]["parameters"],
                    "properties": {
                        **TOOLS[0]["function"]["parameters"]["properties"],
                        "confirm": {"type": "boolean"},
                    },
                },
            },
        }
    ]
    failures = validate_task(
        _Pack(tools),
        TASK,
        [_call(thing_id="T-1", confirm=True)],
    )
    assert "confirmed_mutation_without_user_confirmation" in {failure["reason"] for failure in failures}


def test_schema_validation_requires_prefix_count_when_declared() -> None:
    task = {**TASK, "call_order": "prefix"}
    failures = validate_task(_Pack(TOOLS), task, [_call(thing_id="T-1")])
    assert "bad_call_order_prefix" in {failure["reason"] for failure in failures}


def test_validation_and_expansion_share_one_filter_dialect() -> None:
    """A filter accepted by the gold gate must select the same rows at expansion."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import fixture_filter
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        expand,
        oracle_validation,
    )

    assert expand.evaluate_filter is fixture_filter.evaluate_filter
    assert oracle_validation.evaluate_filter is fixture_filter.evaluate_filter

    row = {"status": "available", "copies": 2}
    assert fixture_filter.evaluate_filter(row, "status == 'available' and copies > 0") is True
    assert fixture_filter.evaluate_filter(row, "status in ['available', 'on_loan']") is True
    assert fixture_filter.evaluate_filter(row, "copies <= 1") is False
    with pytest.raises(fixture_filter.FilterError):
        fixture_filter.evaluate_filter(row, "status ~ 'available'")
    with pytest.raises(fixture_filter.FilterError):
        fixture_filter.evaluate_filter(row, "copies > other_field")


def test_a_filter_reads_operators_only_outside_string_literals() -> None:
    """Splitting on operator text mistook ordinary values for syntax."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import fixture_filter

    evaluate = fixture_filter.evaluate_filter

    assert evaluate({"title": "Pride and Prejudice"}, "title == 'Pride and Prejudice'") is True
    assert evaluate({"code": "A==B"}, "code == 'A==B'") is True
    assert evaluate({"status": "open"}, "status not in ['closed', 'void']") is True
    assert evaluate({"status": "closed"}, "status not in ['closed', 'void']") is False
    # A field the row does not carry fails its comparison rather than matching.
    assert evaluate({"status": "open"}, "missing == 'open'") is False
    assert evaluate({"status": "open"}, "missing not in ['closed']") is False
    with pytest.raises(fixture_filter.FilterError, match="only 'and'"):
        evaluate({"status": "open"}, "status == 'open' or status == 'void'")


def test_expansion_error_is_raised_for_unknown_source_kind() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import _candidates

    class _FixturePack(_Pack):
        manifest: dict[str, Any] = {}
        fixtures: dict[str, Any] = {}

    with pytest.raises(ExpansionError, match="unsupported source kind"):
        _candidates(_FixturePack([]), "slot", {"source": "oracle:things.id"})


def test_confirmation_only_covers_the_turns_that_follow_it() -> None:
    """A confirmation asked after the call confirms nothing the call could rely on."""
    template = {
        "template_id": "tpl",
        "turn_policy": "confirmation",
        "slots": {},
        "assistant_milestones": [
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
            {"type": "ask_confirm"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "ask_confirm", "content_template": {"en": "Yes."}}],
    }
    # The policy promises an approved call, so asking afterwards is refused outright.
    with pytest.raises(PlanError, match="no call batch a user reply authorized"):
        build_plan(template, TASK)

    pack = _Pack([{**TOOLS[0], "x-requires-confirmation": True}])
    task = {**TASK, "confirmed_call_turns": []}
    call = _call(thing_id="T-1", confirm=True)
    assert "confirmed_mutation_without_user_confirmation" in {
        failure["reason"] for failure in validate_task(pack, task, [call])
    }

    # Asking first and calling afterwards is the shape that earns the confirmation.
    ordered = build_plan(
        {
            **template,
            "assistant_milestones": [
                {"type": "ask_confirm"},
                {"type": "tool_call", "tool": "do_thing", "call_group": 0},
                {"type": "final_answer"},
            ],
        },
        TASK,
    )
    assert ordered["confirmed_call_turns"] == [1]


def test_confirmation_is_consumed_by_the_next_call_batch() -> None:
    """One approval must not authorize a separate mutation later in the chat."""
    template = {
        "template_id": "third_pack_two_actions",
        "turn_policy": "multi_tool",
        "slots": {},
        "assistant_milestones": [
            {"id": "approve_first", "type": "ask_confirm"},
            {"type": "tool_call", "tool": "mutate_alpha"},
            {"id": "collect_second", "type": "ask_for_slot"},
            {"type": "tool_call", "tool": "mutate_beta"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [
            {"after": "approve_first", "content_template": {"en": "Approve alpha."}},
            {"after": "collect_second", "content_template": {"en": "Use beta too."}},
        ],
    }

    plan = build_plan(template, {"task_id": "third__1", "template_id": template["template_id"]})

    assert plan["confirmed_call_turns"] == [1]


def test_ask_for_slot_withdraws_a_prior_confirmation() -> None:
    """Confirmation must not cover a mutation that depends on a slot collected later."""
    template = {
        "template_id": "confirm_then_collect",
        "turn_policy": "missing_slot",
        "slots": {
            "thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": False},
        },
        "assistant_milestones": [
            {"id": "approve", "type": "ask_confirm"},
            {"id": "ask", "type": "ask_for_slot"},
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [
            {"after": "approve", "content_template": {"en": "Yes."}},
            {"after": "ask", "content_template": {"en": "T-1"}},
        ],
    }

    plan = build_plan(template, {"task_id": "t1", "template_id": template["template_id"]})

    assert plan["confirmed_call_turns"] == []


def test_a_declared_must_not_mention_adds_to_the_tool_name_guard() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {
        "template_id": "tpl",
        "slots": {},
        "paraphrase": {"must_not_mention": ["internal fee table"]},
    }
    violations = check_surface_guards(
        template,
        {"slots": {}},
        ["Just call do_thing using the internal fee table."],
        ["do_thing"],
    )

    assert violations == [
        {"guard": "must_not_mention", "tool": "do_thing"},
        {"guard": "must_not_mention", "phrase": "internal fee table"},
    ]


def test_an_argument_placeholder_inside_a_longer_string_is_substituted() -> None:
    """The gold call and the turn describing it must agree on the same rendered value."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        bind_arguments,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import _substitute

    arguments = bind_arguments(
        _Pack(TOOLS),
        "do_thing",
        {"thing_id": "TXN-{thing_id}", "count": "{count}"},
        {"thing_id": "T-1", "count": 10},
    )

    assert arguments["thing_id"] == _substitute("TXN-{thing_id}", {"thing_id": "T-1"})
    # A lone reference keeps the slot's own type so an integer stays an integer.
    assert arguments["count"] == 10

    with pytest.raises(ExpectedTraceError, match="unbound"):
        bind_arguments(_Pack(TOOLS), "do_thing", {"thing_id": "TXN-{missing}"}, {})


def test_a_value_holding_braces_is_inserted_rather_than_rescanned() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import _substitute

    # Substitution is one pass, so a value naming another slot is not rewritten and a
    # brace in pack data is not mistaken for a placeholder.
    assert _substitute("{memo}", {"memo": "send {mode} now", "mode": "fast"}) == "send {mode} now"
    assert _substitute("{blob} ok", {"blob": '{"a": 1}'}) == '{"a": 1} ok'


def test_a_call_without_a_declared_group_never_joins_another_group() -> None:
    template = {
        "template_id": "tpl",
        "turn_policy": "multi_tool",
        "slots": {},
        "assistant_milestones": [
            {"type": "ask_confirm"},
            {"type": "tool_call", "tool": "do_thing", "call_group": 2},
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "ask_confirm", "content_template": {"en": "Yes."}}],
    }
    plan = build_plan(template, TASK)
    batches = [step for step in plan["steps"] if step["kind"] == "calls"]

    # Two sequential calls stay two assistant turns, and published groups are numbered
    # by the order the turns issue them rather than by the milestone's position.
    assert [len(step["milestones"]) for step in batches] == [1, 1]
    assert [step["call_group"] for step in batches] == [0, 1]


def test_correction_leaves_an_unrelated_argument_sharing_the_value_alone() -> None:
    """Only the corrected slot is withdrawn, not every argument equal to its old value."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        build_expected_calls,
    )

    template = _correction_template(
        assistant_milestones=[
            {"id": "confirm_first", "type": "ask_confirm"},
            {"id": "confirm_again", "type": "ask_confirm"},
            {
                "type": "tool_call",
                "tool": "do_thing",
                "call_group": 0,
                "args": {"thing_id": "{thing_id}", "count": "{count}", "mode": "fast"},
            },
            {"type": "final_answer"},
        ]
    )
    task = {
        **CORRECTION_TASK,
        "slots": {"thing_id": "T-1", "count": 20, "retries": 10},
        "slots_initial": {"thing_id": "T-1", "count": 10, "retries": 10},
    }

    calls = build_expected_calls(_Pack(TOOLS), task, build_plan(template, task))
    assert calls[0]["arguments"] == {"thing_id": "T-1", "count": 20, "mode": "fast"}

    # The replaced value 10 still lives in another slot, and a plain integer argument
    # equal to it is not evidence that the trace kept a withdrawn value.
    padded = _Pack(
        [
            {
                **TOOLS[0],
                "function": {
                    **TOOLS[0]["function"],
                    "parameters": {
                        **TOOLS[0]["function"]["parameters"],
                        "properties": {
                            **TOOLS[0]["function"]["parameters"]["properties"],
                            "page": {"type": "integer"},
                            "flag": {"type": "boolean"},
                        },
                    },
                },
            }
        ]
    )
    call = _call(thing_id="T-1", count=20, page=10, flag=True)
    assert validate_task(padded, task, [call]) == []


def test_a_dependent_path_that_misses_blames_the_instance_not_the_template() -> None:
    """One short fixture row must drop its own task, not end the run."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        TaskDataError,
        _extract,
    )

    assert issubclass(TaskDataError, ExpectedTraceError)

    with pytest.raises(TaskDataError, match="has no field"):
        _extract({"items": [{"id": "A-1"}]}, "total", "step_1")
    with pytest.raises(TaskDataError, match="out of range"):
        _extract({"items": []}, "items.0.id", "step_1")
    with pytest.raises(TaskDataError, match="cannot supply an argument"):
        _extract({"error": {"code": "not_found"}}, "id", "step_1")

    # A path that indexes a list with a non-numeric token is wrong for every instance,
    # so it stays a template fault and stops the run.
    with pytest.raises(ExpectedTraceError, match="indexes a list") as raised:
        _extract({"items": [{"id": "A-1"}]}, "items.first.id", "step_1")
    assert not isinstance(raised.value, TaskDataError)


def test_worker_failure_during_dependency_binding_is_fatal() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        ExpectedTraceError,
        TaskDataError,
        _oracle_trace_resolver,
    )

    class FailingWorker:
        def run_episode(self, **kwargs: Any) -> list[Any]:
            raise TimeoutError("backend exceeded its deadline")

    runtime = SimpleNamespace(
        clock="2026-01-01T00:00:00+00:00",
        import_timeout_s=1.0,
        reset_timeout_s=1.0,
        tool_timeout_s=1.0,
        episode_timeout_s=2.0,
    )
    config = SimpleNamespace(oracle_runtime=runtime)
    pack = SimpleNamespace(
        manifest={},
        fixtures={},
        held_out=None,
        paths=SimpleNamespace(backend_path="third_backend.py", pack_root="."),
    )
    task = {"task_id": "third__timeout", "seed": 1}
    resolver = _oracle_trace_resolver(FailingWorker(), config, pack, task)

    def steps():  # type: ignore[no-untyped-def]
        yield {"op": "reset"}

    with pytest.raises(ExpectedTraceError, match="could not resolve dependent trace") as raised:
        resolver(steps())

    assert not isinstance(raised.value, TaskDataError)


def test_a_confirmation_parameter_may_be_named_by_the_pack() -> None:
    template = _correction_template()
    plan = build_plan(template, CORRECTION_TASK)
    task = {
        **CORRECTION_TASK,
        "confirmed_call_turns": plan["confirmed_call_turns"],
        "tools_present": ["do_thing"],
        "required_tools": [],
    }
    tools = [
        {
            "type": "function",
            "x-requires-confirmation": True,
            "function": {
                "name": "do_thing",
                "parameters": {
                    "type": "object",
                    "properties": {"thing_id": {"type": "string"}, "xac_nhan": {"type": "boolean"}},
                },
            },
        }
    ]
    call = _call(thing_id="T-1", xac_nhan=True)

    renamed = _Pack(tools, manifest={"confirmation": {"parameter": "xac_nhan"}})
    reasons = [failure["reason"] for failure in validate_task(renamed, task, [call])]
    assert "confirmed_mutation_without_user_confirmation" in reasons

    confirmed = {**task, "confirmed_call_turns": [call["turn_index"]]}
    assert validate_task(renamed, confirmed, [call]) == []


def test_corrections_are_applied_in_conversation_order() -> None:
    """Two corrections declared in one order can reach the user in another."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    template = _correction_template(
        user_simulator_turns=[
            # Declared second-to-last but delivered last, so 30 is the value in force.
            {
                "after": "confirm_again",
                "content_template": {"en": "Actually {count_last}."},
                "slot_updates": {"count": {"source": "literal:[30]", "bind_as": "count_last"}},
            },
            {
                "after": "confirm_first",
                "content_template": {"en": "Make it {count_new}."},
                "slot_updates": {"count": {"source": "literal:[20]", "bind_as": "count_new"}},
            },
        ]
    )
    pack = SimpleNamespace(manifest={"pack_id": "pack", "version": "1.0"}, fixtures={}, tools=TOOLS)

    task = expand_template(pack, template, 4, 0)[0]

    assert task["slots"]["count"] == 30
    assert [update["values"]["count"] for update in task["slot_updates"]] == [20, 30]


def test_a_narrow_budget_still_binds_a_correction() -> None:
    """Dropping no-op pairs after capping left a wide pack looking unbindable."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    template = _correction_template(
        slots={
            "thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": True},
            "count": {"source": "literal:[10, 20, 30]", "visible_in_first_turn": True},
        },
        user_simulator_turns=[
            {
                "after": "confirm_first",
                "content_template": {"en": "Make it {count_new}."},
                "slot_updates": {"count": {"source": "literal:[10, 20, 30]", "bind_as": "count_new"}},
            },
            {"after": "confirm_again", "content_template": {"en": "Yes."}},
        ],
    )
    pack = SimpleNamespace(manifest={"pack_id": "pack", "version": "1.0"}, fixtures={}, tools=TOOLS)

    tasks = expand_template(pack, template, 1, 0)

    assert len(tasks) == 1
    assert tasks[0]["slots_initial"]["count"] != tasks[0]["slots"]["count"]


def test_a_narrow_budget_handles_a_chain_of_corrections() -> None:
    """A later no-op candidate must not hide a valid candidate behind the cap."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    template = {
        "template_id": "third_pack_revision_chain",
        "turn_policy": "correction",
        "slots": {
            "quantity": {"source": "literal:[10]", "visible_in_first_turn": True},
        },
        "assistant_milestones": [
            {"id": "revise_once", "type": "ask_confirm"},
            {"id": "revise_twice", "type": "ask_confirm"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [
            {
                "after": "revise_once",
                "content_template": {"en": "Use 20."},
                "slot_updates": {"quantity": {"source": "literal:[20]"}},
            },
            {
                "after": "revise_twice",
                "content_template": {"en": "Use the final quantity."},
                "slot_updates": {"quantity": {"source": "literal:[20,30]"}},
            },
        ],
    }
    pack = SimpleNamespace(
        manifest={"pack_id": "third_pack", "version": "1.0"},
        fixtures={},
        tools=[],
    )

    tasks = expand_template(pack, template, 1, 0)

    assert len(tasks) == 1
    assert [update["values"]["quantity"] for update in tasks[0]["slot_updates"]] == [20, 30]
    assert tasks[0]["slots"]["quantity"] == 30


def test_a_multi_slot_correction_keeps_the_slot_that_changed() -> None:
    """One restated sibling must not discard a real correction in the same turn."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import expand_template

    template = {
        "template_id": "multi_slot_correction",
        "turn_policy": "correction",
        "slots": {
            "count": {"source": "literal:[10]", "visible_in_first_turn": True},
            "mode": {"source": "literal:['fast']", "visible_in_first_turn": True},
        },
        "assistant_milestones": [
            {"id": "revise", "type": "ask_confirm"},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [
            {
                "after": "revise",
                "content_template": {"en": "Use 20, still fast."},
                "slot_updates": {
                    "count": {"source": "literal:[20]"},
                    "mode": {"source": "literal:['fast']"},
                },
            }
        ],
    }
    pack = SimpleNamespace(
        manifest={"pack_id": "pack", "version": "1.0"},
        fixtures={},
        tools=TOOLS,
    )

    tasks = expand_template(pack, template, 1, 0)

    assert len(tasks) == 1
    assert tasks[0]["slots"]["count"] == 20
    assert tasks[0]["slots"]["mode"] == "fast"


def test_strict_call_order_follows_required_tools() -> None:
    task = {
        **TASK,
        "required_tools": ["do_thing", "other_tool"],
        "tools_present": ["do_thing", "other_tool"],
        "call_order": "strict",
    }
    calls = [
        {
            "turn_index": 0,
            "call_group": 0,
            "position_in_group": 0,
            "function_name": "other_tool",
            "arguments": {"thing_id": "T-1"},
        },
        {
            "turn_index": 1,
            "call_group": 1,
            "position_in_group": 0,
            "function_name": "do_thing",
            "arguments": {"thing_id": "T-1"},
        },
    ]
    pack = _Pack(
        [
            TOOLS[0],
            {
                "type": "function",
                "function": {
                    "name": "other_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"thing_id": {"type": "string"}},
                    },
                },
            },
        ]
    )

    reasons = {failure["reason"] for failure in validate_task(pack, task, calls)}
    assert "call_order_mismatch" in reasons


def test_a_bare_literal_slot_keeps_its_type() -> None:
    """An integer-typed parameter cannot accept the string "200000"."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import _candidates

    pack = SimpleNamespace(manifest={}, fixtures={}, tools=TOOLS)

    assert _candidates(pack, "count", {"source": "literal:200000"}) == [(200000, None)]
    assert _candidates(pack, "mode", {"source": "literal:fast"}) == [("fast", None)]


def test_a_question_names_the_slot_it_asks_for() -> None:
    """A withheld-slot question must name that slot, not every slot in force."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        render_task,
    )

    template = {
        "template_id": "tpl",
        "turn_policy": "missing_slot",
        "slots": {
            "thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": True},
            "count": {"source": "literal:[3]", "visible_in_first_turn": False},
        },
        "user_turn_templates": {"en": "Do thing {thing_id}."},
        "assistant_milestones": [
            {"id": "ask", "type": "ask_for_slot"},
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "ask", "content_template": {"en": "{count} times."}}],
    }
    task = {
        **TASK,
        "turn_policy": "missing_slot",
        "slots": {"thing_id": "T-1", "count": 3},
        "slots_initial": {"thing_id": "T-1", "count": 3},
    }
    pack = SimpleNamespace(
        manifest={
            "assistant_turn_templates": {
                "ask_for_slot": {"en": "Which {slot_name}?"},
                "final_answer": {"en": "Done."},
            }
        },
        tools=TOOLS,
    )
    prompt_bundle = {"system_prompt": "S", "system_prompt_id": "sha256:0"}

    surface = render_task(
        pack,
        template,
        task,
        build_plan(template, task),
        language="en",
        prompt_bundle=prompt_bundle,
        tool_names=["do_thing"],
    )
    assert surface["steps"][1]["content"] == "Which count?"

    # Two withheld slots make the question ambiguous, so the template must say which
    # one it asks about rather than the render picking silently.
    two_hidden = {
        **template,
        "slots": {
            "thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": False},
            "count": {"source": "literal:[3]", "visible_in_first_turn": False},
        },
    }
    with pytest.raises(PlanError, match="each ask_for_slot milestone must name"):
        build_plan(two_hidden, task)


def test_the_render_language_is_offered_by_the_opening_turn_and_checked_everywhere() -> None:
    """The first turn decides which languages are on offer; the rest must follow it.

    A pack that states its assistant turns in more languages than its user turns must
    not have one of those extra languages chosen for it, and a block that is missing
    the chosen language must be named before any task renders.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import _language_gaps

    template = {
        "template_id": "tpl",
        "user_turn_templates": {"en": "Do thing.", "vi": "Làm việc."},
        "assistant_milestones": [
            {"id": "ask", "type": "ask_for_slot", "slot": "count"},
            {"type": "tool_call", "tool": "do_thing", "call_group": 0},
            {"type": "final_answer"},
        ],
        "user_simulator_turns": [{"after": "ask", "content_template": {"en": "{count} times."}}],
    }
    pack = SimpleNamespace(
        manifest={
            "assistant_turn_templates": {
                "ask_for_slot": {"en": "How many?", "vi": "Bao nhiêu?"},
                "final_answer": {"en": "Done."},
            }
        },
        tools=TOOLS,
    )

    assert _language_gaps(pack, template, "en") == []
    assert _language_gaps(pack, template, "vi") == [
        "template 'tpl' user_simulator_turns after 'ask'",
        "assistant_turn_templates.final_answer for template 'tpl'",
    ]


def test_a_guard_reads_a_number_as_a_whole_value() -> None:
    """An amount of 4 does not appear in "400000", and 200000 does appear in "200000đ"."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {
        "template_id": "tpl",
        "slots": {
            "amount": {"source": "literal:[200000]", "visible_in_first_turn": True},
            "retries": {"source": "literal:[4]", "visible_in_first_turn": False},
        },
        "paraphrase": {},
    }
    task = {
        "slots": {"amount": 200000, "retries": 4},
        "slots_initial": {"amount": 200000, "retries": 4},
    }

    assert check_surface_guards(template, task, ["Chuyển 200000đ giúp tôi."], []) == []
    assert check_surface_guards(template, task, ["Chuyển 200.000 đồng giúp tôi."], []) == []

    leaked = check_surface_guards(template, task, ["Chuyển 4 lần, 200000đ."], [])
    assert [violation["guard"] for violation in leaked] == ["must_omit"]


def test_a_guard_reads_grouping_but_not_a_list_of_digits() -> None:
    """Only three-digit grouping restates an amount; "1, 2, 3" states no amount at all.

    Reading any punctuated digit run as one number would let an enumeration satisfy
    must_preserve, so a surface that never states the amount would pass the guard.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {
        "template_id": "tpl",
        "slots": {"amount": {"source": "literal:[123]", "visible_in_first_turn": True}},
        "paraphrase": {},
    }
    task = {"slots": {"amount": 123}, "slots_initial": {"amount": 123}}

    assert check_surface_guards(template, task, ["Chuyển 123 đồng."], []) == []
    assert check_surface_guards(template, task, ["Chuyển 1, 2, 3 đồng."], []) == [
        {"guard": "must_preserve", "slot": "amount"}
    ]

    grouped = {
        "template_id": "tpl",
        "slots": {"amount": {"source": "literal:[1234567]", "visible_in_first_turn": True}},
        "paraphrase": {},
    }
    grouped_task = {
        "slots": {"amount": 1234567},
        "slots_initial": {"amount": 1234567},
    }
    for written in ("1.234.567", "1,234,567", "1 234 567"):
        assert check_surface_guards(grouped, grouped_task, [f"Chuyển {written} đồng."], []) == []


def test_a_guard_reads_a_word_value_as_a_whole_token() -> None:
    """A status of "us" is not stated by "status", so neither guard may read it there."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {
        "template_id": "tpl",
        "slots": {
            "rail": {"source": "literal:['us']", "visible_in_first_turn": True},
            "code": {"source": "literal:['vn']", "visible_in_first_turn": False},
        },
        "paraphrase": {},
    }
    task = {
        "slots": {"rail": "us", "code": "vn"},
        "slots_initial": {"rail": "us", "code": "vn"},
    }

    # "status" contains both values as substrings and states neither.
    hidden_inside_words = check_surface_guards(template, task, ["Check the status of my invnoice."], [])
    assert hidden_inside_words == [{"guard": "must_preserve", "slot": "rail"}]

    assert check_surface_guards(template, task, ["Send over rail us please."], []) == []

    leaked = check_surface_guards(template, task, ["Send over rail us to vn."], [])
    assert [violation["guard"] for violation in leaked] == ["must_omit"]


def test_visible_slot_must_appear_in_the_opening_turn() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {
        "template_id": "third_pack_visible_asset",
        "slots": {
            "asset_code": {
                "source": "literal:['ASSET-7']",
                "visible_in_first_turn": True,
            }
        },
        "paraphrase": {},
    }
    task = {
        "slots": {"asset_code": "ASSET-7"},
        "slots_initial": {"asset_code": "ASSET-7"},
    }

    violations = check_surface_guards(
        template,
        task,
        ["Please inspect my asset.", "The code is ASSET-7."],
        [],
    )

    assert violations == [{"guard": "must_preserve", "slot": "asset_code"}]


def test_a_tool_name_inside_a_longer_word_is_not_a_leak() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        check_surface_guards,
    )

    template = {"template_id": "tpl", "slots": {}, "paraphrase": {}}
    task = {"slots": {}, "slots_initial": {}}

    assert check_surface_guards(template, task, ["Tell me about my bookshelf."], ["book"]) == []
    assert check_surface_guards(template, task, ["Call book now."], ["book"]) == [
        {"guard": "must_not_mention", "tool": "book"}
    ]


def test_optional_tool_parameters_are_not_injected_from_same_named_slots() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        bind_arguments,
    )

    consumed: set[str] = set()
    arguments = bind_arguments(
        _Pack(TOOLS),
        "do_thing",
        {"thing_id": "{thing_id}"},
        {"thing_id": "T-1", "count": 999},
        consumed,
    )

    assert arguments == {"thing_id": "T-1"}
    assert consumed == {"thing_id"}


def test_unimplemented_json_schema_constraints_are_rejected() -> None:
    function = {
        "name": "bounded",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                }
            },
        },
    }

    assert validate_function_schema(function) == [
        {
            "reason": "unsupported_schema_keyword",
            "path": "$.amount",
            "keyword": "exclusiveMinimum",
        }
    ]


def test_range_sources_reject_zero_and_include_descending_endpoints() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import _candidates

    pack = _Pack(TOOLS)
    with pytest.raises(ExpansionError, match="must not be zero"):
        _candidates(pack, "count", {"source": "range:{'min': 1, 'max': 3, 'step': 0}"})

    assert _candidates(
        pack,
        "count",
        {"source": "range:{'min': 3, 'max': 1, 'step': -1}"},
    ) == [(3, None), (2, None), (1, None)]


def test_all_trace_drops_are_written_before_the_stage_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import expected_trace

    def fail_one(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise expected_trace.TaskDataError("fixture row has no dependent value")

    monkeypatch.setattr(expected_trace, "build_expected_calls", fail_one)
    config = SimpleNamespace(
        output_dir=tmp_path,
        expt_name="all-dropped",
        oracle_runtime=SimpleNamespace(episode_timeout_s=1.0, worker="thread"),
    )
    task = {
        **TASK,
        "pack_id": "pack",
        "pack_version": "1",
        "variant_index": 0,
        "seed": 0,
        "slots_initial": {},
        "slots": {},
    }

    with pytest.raises(expected_trace.ExpectedTraceError, match="every one"):
        expected_trace.run_expected_trace(
            config,
            _Pack(TOOLS),
            [task],
            {str(task["task_id"]): {"steps": []}},
        )

    rows = pq.read_table(tmp_path / "all-dropped" / "stage_cache" / "expected_traces.parquet").to_pylist()
    assert rows[0]["task_id"] == task["task_id"]
    assert rows[0]["derived"] is False
    assert "fixture row has no dependent value" in rows[0]["drop_reason"]


def test_non_english_gold_surface_requires_a_pack_system_prompt(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        RenderError,
        run_render,
    )

    template = {
        "template_id": "tpl",
        "turn_policy": "single_turn",
        "slots": {},
        "user_turn_templates": {"vi": "Xin chào."},
        "assistant_milestones": [
            {"type": "tool_call", "tool": "do_thing"},
            {"type": "final_answer"},
        ],
        "assistant_turn_templates": {"final_answer": {"vi": "Xong."}},
    }
    task = {**TASK, "slots": {}, "slots_initial": {}}
    plan = build_plan(template, task)
    pack = SimpleNamespace(manifest={}, templates=[template], tools=[])
    config = SimpleNamespace(
        surface_generation={},
        output_dir=tmp_path,
        expt_name="locale",
        lineage=SimpleNamespace(policy="strict_separation"),
    )

    with pytest.raises(RenderError, match="default system prompt"):
        run_render(
            config,
            pack,
            {"tpl": template},
            [task],
            {str(task["task_id"]): plan},
        )

    # A run that publishes nothing may still exercise the pipeline in its own language.
    smoke_config = SimpleNamespace(
        surface_generation={},
        output_dir=tmp_path,
        expt_name="locale-smoke",
        lineage=SimpleNamespace(policy="smoke_no_publication"),
    )
    surfaces, prompt_bundle = run_render(
        smoke_config,
        pack,
        {"tpl": template},
        [task],
        {str(task["task_id"]): plan},
    )
    assert prompt_bundle["origin"].startswith("bfcl/prompts/")
    assert surfaces[str(task["task_id"])]["language"] == "vi"


DATA_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "data"
)
MANUAL_PACKS = ("tiny_oracle_pack", "banking_vn_oracle_pack")
ASSISTED_AUTHORING_ARTIFACTS = frozenset(
    {
        "source_declaration.json",
        "source_declaration.yaml",
        "source_evidence.json",
        "source_observations.json",
        "certification_report.json",
        "intake_provenance.json",
        "resolved_authoring_config.json",
        "domain_brief.md",
        "held_out_redaction_report.json",
    }
)
MANIFEST_ADAPTER_KEYS = frozenset(
    {"adapter_kind", "source_declaration", "certification", "source_evidence"}
)


@pytest.mark.parametrize("pack_name", MANUAL_PACKS)
def test_manual_oracle_packs_require_no_assisted_authoring_adapter_metadata(
    pack_name: str,
) -> None:
    """A manual pack must load from the manual contract alone, with no adapter records."""
    pack_root = DATA_ROOT / pack_name
    present = sorted(
        item.name for item in pack_root.iterdir() if item.name in ASSISTED_AUTHORING_ARTIFACTS
    )
    assert not present, f"{pack_name} ships assisted-authoring artifacts: {present}"

    manifest = yaml.safe_load((pack_root / "manifest.yaml").read_text("utf-8"))
    assert not MANIFEST_ADAPTER_KEYS & set(manifest)

    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=pack_root / "manifest.yaml"),
        (pack_root,),
    )
    assert paths.endpoint_config_path is None
    assert paths.backend_path is not None
    for declared in (
        paths.tools_path,
        paths.templates_path,
        paths.assertions_path,
        paths.validation_cases_path,
        paths.backend_path,
    ):
        assert declared.is_file()
        assert declared.name not in ASSISTED_AUTHORING_ARTIFACTS


def test_manual_gold_tier_ignores_adapter_and_certification_fields() -> None:
    """Assisted-authoring records in a report may not move the manual Gold verdict."""
    gold: dict[str, Any] = {
        "checks": [{"status": "pass", "failures": []}],
        "stats": {
            "has_oracle": True,
            "n_templates": 1,
            "n_assertions": 1,
            "n_tools": 1,
        },
    }
    assert derive_pack_tier(dict(gold)) == (True, "gold")

    decorated = {
        **gold,
        "adapter_kind": "mcp_mode_a",
        "certification": {"attained_tier": "A2", "issuer": "bfcl"},
        "source_evidence": {"schema_version": "bfcl-source-evidence-v2"},
    }
    assert derive_pack_tier(dict(decorated)) == (True, "gold")

    failing = {**decorated, "checks": [{"status": "fail", "failures": ["replay"]}]}
    assert derive_pack_tier(dict(failing)) == (False, "silver")
