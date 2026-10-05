# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn milestones into an ordered, deterministic conversation plan."""

from __future__ import annotations

import logging
from typing import Any, NoReturn

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    CONVERSATION_PLANS,
    conversation_plan_row,
    conversation_plans_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)

TEXT_MILESTONES = ("ask_for_slot", "ask_confirm", "decline", "final_answer")
# A closing ask_for_slot ends the conversation, so only a mid-conversation ask
# needs a deterministic user reply.
REPLY_MILESTONES = ("ask_confirm", "ask_for_slot")
CLOSING_MILESTONE = {"irrelevant": "decline", "clarify_only": "ask_for_slot"}


class PlanError(ValueError):
    """Raised when milestones cannot form a deterministic plan."""


def _check_policy_shape(
    template: dict[str, Any],
    plan: dict[str, Any],
    user_turns: int,
) -> None:
    """Refuse a plan whose shape contradicts the policy the template declares.

    ``turn_policy`` is the label downstream consumers slice the benchmark by, so a
    template that calls itself ``multi_tool`` while planning a single call would
    mislabel every row it produces. Each policy is held to the minimum shape that
    makes its name true; the remaining freedom stays with the pack.
    """
    policy = str(template.get("turn_policy"))
    template_id = template.get("template_id")
    calls = int(plan["num_tool_calls"])

    def refuse(complaint: str) -> NoReturn:
        raise PlanError(f"{policy} template {template_id!r} {complaint}")

    if policy in CLOSING_MILESTONE:
        # A no-call policy is already pinned by its closing milestone.
        return
    if not calls:
        refuse("plans no tool call; a conversation that calls nothing is clarify_only or irrelevant")
    if policy == "single_turn" and user_turns != 1:
        refuse(
            f"plans {user_turns} user turns; a policy whose name promises one opening request "
            "cannot depend on a later reply"
        )
    if policy == "multi_tool" and calls < 2:
        refuse("plans a single tool call")
    if policy == "missing_slot":
        withheld = [
            name
            for name, slot in (template.get("slots") or {}).items()
            if slot.get("visible_in_first_turn") is False
        ]
        if not withheld:
            refuse("withholds no slot; mark the slot it asks for with visible_in_first_turn: false")
        collected: set[str] = set()
        first_call = next(
            (index for index, step in enumerate(plan["steps"]) if step["kind"] == "calls"),
            len(plan["steps"]),
        )
        for index, step in enumerate(plan["steps"]):
            if step["kind"] != "text" or step["milestone_type"] != "ask_for_slot":
                continue
            declared = step["milestone"].get("slot")
            if declared is None:
                if len(withheld) != 1:
                    refuse(
                        "withholds several slots; each ask_for_slot milestone must name the "
                        "slot it collects"
                    )
                declared = withheld[0]
            name = str(declared)
            if name not in withheld:
                refuse(f"asks for {name!r}, which is not a withheld slot")
            if index >= first_call:
                refuse(f"asks for withheld slot {name!r} only after issuing a tool call")
            if index + 1 >= len(plan["steps"]) or plan["steps"][index + 1]["kind"] != "user":
                refuse(f"asks for withheld slot {name!r} but receives no user reply")
            collected.add(name)
        missing = sorted(set(withheld) - collected)
        if missing:
            refuse("never collects withheld slot(s): " + ", ".join(missing))
    if policy == "confirmation" and not plan["has_user_confirmation"]:
        refuse(
            "has no call batch a user reply authorized; pair an ask_confirm milestone with the "
            "user_simulator_turns entry that answers it, before the call it approves"
        )
    if policy == "correction" and not plan["has_slot_correction"]:
        refuse("replaces no slot value; declare slot_updates on the user turn that corrects one")
    if policy == "negative_path" and not (template.get("success_assertions") or []):
        refuse(
            "declares no success_assertions; a refused or failing call is only pinned by an "
            "assertion that states what the backend must have refused"
        )


def _milestone_key(milestone: dict[str, Any], index: int) -> str:
    declared = milestone.get("id")
    return str(declared) if declared else f"{milestone.get('type')}@{index}"


def _simulator_lookup(template: dict[str, Any]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Index user_simulator_turns by the milestone reference in ``after``.

    The declaration index travels with each entry so a turn that replaces a slot can
    be matched to the values expansion bound for it.
    """
    lookup: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, entry in enumerate(template.get("user_simulator_turns") or []):
        after = entry.get("after")
        if after is None:
            raise PlanError("user_simulator_turns entry is missing 'after'")
        lookup.setdefault(str(after), []).append((index, entry))
    return lookup


def simulator_delivery_order(template: dict[str, Any]) -> dict[int, int]:
    """Map each ``user_simulator_turns`` index to its position in the conversation.

    Entries are delivered in milestone order, which is independent of the order they
    happen to be declared in, so anything that depends on "which turn comes first" —
    a slot correction, above all — must read this rather than the declaration index.
    """
    lookup = _simulator_lookup(template)
    milestones = list(template.get("assistant_milestones") or [])
    type_counts: dict[str, int] = {}
    for milestone in milestones:
        kind = str(milestone.get("type"))
        type_counts[kind] = type_counts.get(kind, 0) + 1

    order: dict[int, int] = {}
    consumed: set[int] = set()
    for index, milestone in enumerate(milestones):
        kind = str(milestone.get("type"))
        references = [_milestone_key(milestone, index)]
        if type_counts.get(kind, 0) == 1:
            references.append(kind)
        for reference in references:
            for entry_index, _ in lookup.get(reference, []):
                if entry_index in consumed:
                    continue
                consumed.add(entry_index)
                order[entry_index] = len(order)
    return order


def _group_key(milestone: dict[str, Any], index: int) -> tuple[str, Any]:
    """Identify which batch a call belongs to, declared or standalone.

    A declared ``call_group`` and an omitted one live in separate namespaces: without
    that split, an omitted group defaulting to the milestone index could equal a number
    another call declared, and two sequential calls would silently merge into one
    parallel turn.
    """
    declared = milestone.get("call_group")
    return ("declared", declared) if declared is not None else ("alone", index)


def _confirmed_call_turns(steps: list[dict[str, Any]]) -> list[int]:
    """Return the assistant turn indexes whose calls a live confirmation covers.

    Confirmation is positional, not conversation-wide: it starts when the user answers
    an ``ask_confirm`` and ends when a later turn corrects a value or the assistant
    asks for a new slot — both change what the user would be confirming.
    """
    confirmed = False
    awaiting_reply = False
    turn = -1
    covered: list[int] = []
    for step in steps:
        kind = step["kind"]
        if kind == "text":
            turn += 1
            # Collecting a slot the confirmation did not cover withdraws the approval:
            # the upcoming mutation depends on information the user has not yet given.
            if step["milestone_type"] == "ask_for_slot":
                confirmed = False
            awaiting_reply = step["milestone_type"] == "ask_confirm"
        elif kind == "calls":
            turn += 1
            awaiting_reply = False
            if confirmed:
                covered.append(turn)
                # A confirmation authorizes the next assistant call batch, not every
                # mutation that happens later in the conversation. A later action
                # needs its own ask_confirm/reply pair.
                confirmed = False
        else:
            if step.get("update_index") is not None:
                confirmed = False
            elif awaiting_reply:
                confirmed = True
            awaiting_reply = False
    return covered


def build_plan(template: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Return ordered plan steps plus the derived turn-shape fields."""
    milestones = list(template.get("assistant_milestones") or [])
    if not milestones:
        raise PlanError(
            f"template {template.get('template_id')!r} must declare at least one assistant milestone"
        )
    simulator = _simulator_lookup(template)
    type_counts: dict[str, int] = {}
    for milestone in milestones:
        kind = str(milestone.get("type"))
        type_counts[kind] = type_counts.get(kind, 0) + 1

    steps: list[dict[str, Any]] = [{"kind": "user", "source": "first_turn"}]
    consumed: set[int] = set()
    grouped: set[Any] = set()
    index = 0
    while index < len(milestones):
        start = index
        milestone = milestones[start]
        kind = str(milestone.get("type"))
        batch = [milestone]
        if kind == "tool_call":
            group = _group_key(milestone, start)
            while index + 1 < len(milestones):
                following = milestones[index + 1]
                if str(following.get("type")) != "tool_call":
                    break
                if _group_key(following, index + 1) != group:
                    break
                batch.append(following)
                index += 1
            if group in grouped:
                raise PlanError(
                    f"template {template.get('template_id')!r} splits call_group "
                    f"{milestone.get('call_group')} across assistant turns; calls sharing a group "
                    "must be declared consecutively"
                )
            grouped.add(group)
            steps.append({"kind": "calls", "group_key": group, "milestones": batch})
        elif kind in TEXT_MILESTONES:
            steps.append({"kind": "text", "milestone_type": kind, "milestone": milestone})
        else:
            raise PlanError(f"unsupported milestone type {kind!r}")

        references = [_milestone_key(entry, start + offset) for offset, entry in enumerate(batch)]
        if type_counts.get(kind, 0) == 1:
            references.append(kind)
        replied = False
        for reference in references:
            for entry_index, entry in simulator.get(reference, []):
                if entry_index in consumed:
                    continue
                consumed.add(entry_index)
                updates = entry.get("slot_updates") or None
                steps.append(
                    {
                        "kind": "user",
                        "source": "simulator",
                        "turn": entry,
                        "update_index": entry_index if updates else None,
                    }
                )
                replied = True
        is_last = index == len(milestones) - 1
        if kind in REPLY_MILESTONES and not replied and not is_last:
            raise PlanError(
                f"milestone {kind!r} needs a matching user_simulator_turns entry in template "
                f"{template.get('template_id')!r}"
            )
        index += 1

    declared = sum(len(entries) for entries in simulator.values())
    if len(consumed) != declared:
        raise PlanError(
            f"template {template.get('template_id')!r} has user_simulator_turns whose 'after' does not "
            "resolve to exactly one milestone"
        )

    # Published groups are numbered by the order their assistant turn issues them, so a
    # call that declares no call_group can never collide with one that does.
    for number, step in enumerate(step for step in steps if step["kind"] == "calls"):
        step["call_group"] = number

    user_turns = sum(1 for step in steps if step["kind"] == "user")
    num_tool_calls = sum(len(step["milestones"]) for step in steps if step["kind"] == "calls")
    confirmed_call_turns = _confirmed_call_turns(steps)
    plan = {
        "task_id": task["task_id"],
        "template_id": task["template_id"],
        "steps": steps,
        "is_multi_turn": user_turns > 1,
        "num_tool_calls": num_tool_calls,
        "confirmed_call_turns": confirmed_call_turns,
        "has_user_confirmation": bool(confirmed_call_turns),
        "has_slot_correction": any(step.get("update_index") is not None for step in steps),
    }
    policy = str(template.get("turn_policy"))
    closing = CLOSING_MILESTONE.get(policy)
    if closing is not None:
        if num_tool_calls:
            raise PlanError(f"{policy} template {template.get('template_id')!r} plans a tool call")
        if not milestones or str(milestones[-1].get("type")) != closing:
            raise PlanError(
                f"{policy} template {template.get('template_id')!r} must end in {closing!r}; "
                f"a trailing final_answer would blur declining from answering"
            )
    _check_policy_shape(template, plan, user_turns)
    return plan


def run_state_machine(
    config: BfclConfig,
    templates_by_id: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build one plan per task and cache the derived turn shape."""
    plans: dict[str, dict[str, Any]] = {}
    for task in tasks:
        template = templates_by_id[str(task["template_id"])]
        plan = build_plan(template, task)
        task["is_multi_turn"] = plan["is_multi_turn"]
        task["num_tool_calls"] = plan["num_tool_calls"]
        task["has_user_confirmation"] = plan["has_user_confirmation"]
        task["confirmed_call_turns"] = plan["confirmed_call_turns"]
        plans[str(task["task_id"])] = plan

    write_stage_table(
        stage_cache_dir(config) / CONVERSATION_PLANS,
        [conversation_plan_row(task, plans[str(task["task_id"])]) for task in tasks],
        conversation_plans_schema(),
    )
    logger.info("BFCL state_machine planned %d conversations", len(plans))
    return plans
