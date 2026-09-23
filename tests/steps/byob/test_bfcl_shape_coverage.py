"""Contract coverage for conversation shapes supplied by the tiny oracle pack."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    TURN_POLICIES,
)

PACK_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "data"
    / "tiny_oracle_pack"
)
BANKING_PACK_ROOT = PACK_ROOT.parent / "banking_vn_oracle_pack"


def _templates() -> list[dict]:
    return yaml.safe_load((PACK_ROOT / "task_templates.yaml").read_text(encoding="utf-8"))


def test_tiny_pack_covers_parallel_calls_in_one_group() -> None:
    template = next(item for item in _templates() if item["template_id"] == "lib_status_parallel")
    calls = [
        milestone
        for milestone in template["assistant_milestones"]
        if milestone["type"] == "tool_call"
    ]
    assert len(calls) >= 2
    assert len({call["call_group"] for call in calls}) == 1


def test_tiny_pack_covers_irrelevant_request_without_tools() -> None:
    template = next(item for item in _templates() if item["turn_policy"] == "irrelevant")
    assert template["required_tools"] == []
    assert all(
        milestone["type"] != "tool_call" for milestone in template["assistant_milestones"]
    )
    assert template["assistant_milestones"][-1]["type"] == "decline"


def _banking_templates() -> list[dict]:
    return yaml.safe_load((BANKING_PACK_ROOT / "task_templates.yaml").read_text(encoding="utf-8"))


def test_banking_declares_a_template_for_every_supported_policy() -> None:
    by_policy: dict[str, list[dict]] = {}
    for template in _banking_templates():
        by_policy.setdefault(template["turn_policy"], []).append(template)

    # Bound to the pipeline's own vocabulary rather than a copy of it, so a policy the
    # pipeline gains later cannot leave this pack silently short of the edge it claims.
    assert set(by_policy) == set(TURN_POLICIES)

    chain = by_policy["dependent_call"][0]
    producer, consumer = (
        milestone
        for milestone in chain["assistant_milestones"]
        if milestone["type"] == "tool_call"
    )
    marker = consumer["args"]["transaction_id"]["from_result"]
    assert marker["call"] == producer["id"]
    assert producer["call_group"] < consumer["call_group"]

    withheld = by_policy["missing_slot"][0]
    hidden = [
        name
        for name, slot in withheld["slots"].items()
        if slot.get("visible_in_first_turn") is False
    ]
    assert hidden
    assert withheld["assistant_milestones"][0]["type"] == "ask_for_slot"
    assert withheld["user_simulator_turns"][0]["after"] == "ask_for_slot"

    clarify = by_policy["clarify_only"][0]
    assert clarify["required_tools"] == []
    assert clarify["assistant_milestones"][-1]["type"] == "ask_for_slot"

    declines = by_policy["irrelevant"][0]
    assert declines["required_tools"] == []
    assert declines["assistant_milestones"][-1]["type"] == "decline"

    parallel = by_policy["multi_tool"][0]
    groups = {
        milestone.get("call_group")
        for milestone in parallel["assistant_milestones"]
        if milestone["type"] == "tool_call"
    }
    assert len(groups) == 1

    corrections = by_policy["correction"]
    # Re-confirmation is what a corrected *mutation* needs, so assert it on a mutating
    # correction rather than on whichever correction template happens to come first.
    mutating_corrections = [
        template for template in corrections if template.get("mutates") is True
    ]
    assert mutating_corrections
    corrected = mutating_corrections[0]
    replacing, reconfirming = corrected["user_simulator_turns"]
    updated = replacing["slot_updates"]
    assert set(updated) <= set(corrected["slots"])
    # The correction has to arrive before the call and be confirmed again after it,
    # otherwise the transfer would ride on a withdrawn confirmation.
    confirms = [
        milestone["id"]
        for milestone in corrected["assistant_milestones"]
        if milestone["type"] == "ask_confirm"
    ]
    assert replacing["after"] == confirms[0]
    assert reconfirming["after"] == confirms[1]
    assert "slot_updates" not in reconfirming
    milestone_types = [milestone["type"] for milestone in corrected["assistant_milestones"]]
    last_confirm = max(index for index, kind in enumerate(milestone_types) if kind == "ask_confirm")
    assert milestone_types.index("tool_call") > last_confirm
    for slot_name, update in updated.items():
        assert corrected["slots"][slot_name]["visible_in_first_turn"] is True
        assert update["bind_as"] in replacing["content_template"]["vi"]

    # A read-only correction withdraws nothing, so it corrects once and never asks for
    # a confirmation the policy would then have to re-obtain.
    for read_only in (
        template for template in corrections if template.get("mutates") is not True
    ):
        turns = read_only["user_simulator_turns"]
        assert len(turns) == 1
        assert turns[0]["slot_updates"]
        assert not [
            milestone
            for milestone in read_only["assistant_milestones"]
            if milestone["type"] == "ask_confirm"
        ]


def test_banking_rows_choose_from_the_whole_tool_catalog() -> None:
    # Omitting tools_present is what exposes the full catalog, so narrowing it anywhere
    # would quietly make the row an easier retrieval problem than the pack advertises.
    assert not [
        template["template_id"]
        for template in _banking_templates()
        if "tools_present" in template
    ]
    catalog = json.loads((BANKING_PACK_ROOT / "tools.json").read_text(encoding="utf-8"))
    assert len(catalog) == 9


def test_banking_negative_paths_expect_documented_failures() -> None:
    negatives = [
        template for template in _banking_templates() if template["turn_policy"] == "negative_path"
    ]
    # Pairing each path with its own assertion is what makes the failure documented: a
    # closed set of names would still pass if a path were scored by another path's check.
    assert {
        template["template_id"]: template["success_assertions"] for template in negatives
    } == {
        "bn_balance_unknown_account": ["assert_account_not_found"],
        "bn_txn_status_unknown_id": ["assert_transaction_not_found"],
        "bn_transfer_short_of_funds": ["assert_transfer_rejected_for_funds"],
        "bn_vietqr_status_unknown_ref": ["assert_vietqr_payment_not_found"],
        "bn_dispute_status_unknown_id": ["assert_dispute_not_found"],
        "bn_create_dispute_not_disputable": ["assert_dispute_refused_without_state_change"],
    }


def test_banking_mutations_require_explicit_confirmation_turns() -> None:
    templates = _banking_templates()
    mutators = [
        item
        for item in templates
        if item["template_id"] in {"bn_create_transfer_single", "bn_create_dispute_single"}
    ]
    assert len(mutators) == 2
    for template in mutators:
        assert template["turn_policy"] == "confirmation"
        kinds = [milestone["type"] for milestone in template["assistant_milestones"]]
        assert kinds == ["ask_confirm", "tool_call", "final_answer"]
        assert template["user_simulator_turns"][0]["after"] == "ask_confirm"
