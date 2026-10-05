"""Contract tests for BFCL held-out enforcement."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HELD_OUT_CONTRACT_VERSION,
    HeldOutDecision,
    HeldOutPolicy,
    fixture_ref,
    scan_row,
    validate_complete_scan_set,
)

NORMALIZED = {
    "version": "1",
    "fixtures": {"accounts": ["ACC-HOLD-01", "ACC-HOLD-02"], "cards": []},
    "templates": ["tpl-held"],
    "policy": {"fixtures_in_backend_state": True, "seed": 7},
    "source": "/packs/held_out.yaml",
}


def _policy(**overrides: object) -> HeldOutPolicy:
    return HeldOutPolicy.from_normalized({**NORMALIZED, **overrides})


def test_contract_flattens_the_loader_policy_into_reference_strings() -> None:
    policy = _policy()

    assert policy.contract_version == HELD_OUT_CONTRACT_VERSION
    assert policy.fixture_refs == (
        '["accounts","ACC-HOLD-01"]',
        '["accounts","ACC-HOLD-02"]',
    )
    assert policy.template_ids == ("tpl-held",)
    assert policy.fixtures_in_backend_state is True
    assert policy.seed == 7
    assert policy.reserves_nothing is False
    assert policy.as_lineage() == {
        "contract_version": "1.0",
        "version": "1",
        "source": "/packs/held_out.yaml",
        "policy_hash": policy.as_lineage()["policy_hash"],
        "fixture_ref_count": 2,
        "template_count": 1,
        "fixtures_in_backend_state": True,
        "seed": 7,
    }


def test_a_policy_reserving_nothing_is_still_a_policy() -> None:
    policy = _policy(fixtures={}, templates=[])

    assert policy.reserves_nothing is True
    assert policy.blocks_template("tpl-held") is False
    assert policy.matched_fixture_refs([fixture_ref("accounts", "ACC-HOLD-01")]) == ()


def test_fixture_reference_matches_what_expansion_records() -> None:
    assert fixture_ref("accounts", "ACC-1") == '["accounts","ACC-1"]'
    assert fixture_ref(" accounts ", 42) == '[" accounts ","42"]'
    assert fixture_ref("a", "b.c") != fixture_ref("a.b", "c")

    with pytest.raises(ValueError, match="requires a collection and a primary id"):
        fixture_ref("accounts", " ")


def test_policy_rejects_malformed_or_repeated_references() -> None:
    with pytest.raises(ValidationError, match="canonical JSON pair"):
        HeldOutPolicy(version="1", fixture_refs=("accounts",))
    assert HeldOutPolicy(version="1", template_ids=("tpl", " tpl ")).template_ids == (
        " tpl ",
        "tpl",
    )
    with pytest.raises(ValidationError, match="must be non-empty"):
        HeldOutPolicy(version=" ")
    with pytest.raises(ValueError, match="fixtures.accounts must be a list"):
        HeldOutPolicy.from_normalized({**NORMALIZED, "fixtures": {"accounts": "ACC-HOLD-01"}})


def test_scanning_records_the_evidence_for_every_verdict() -> None:
    policy = _policy()

    clean = scan_row(
        policy,
        task_id="task-clean",
        template_id="tpl-open",
        fixture_refs=[fixture_ref("accounts", "ACC-1")],
    )
    by_fixture = scan_row(
        policy,
        task_id="task-fixture",
        template_id="tpl-open",
        fixture_refs=[
            fixture_ref("accounts", "ACC-1"),
            fixture_ref("accounts", "ACC-HOLD-02"),
        ],
    )
    by_template = scan_row(
        policy,
        task_id="task-template",
        template_id="tpl-held",
        fixture_refs=[fixture_ref("accounts", "ACC-1")],
    )

    assert clean.held_out_hit is False
    assert clean.matched_fixture_refs == ()
    assert by_fixture.held_out_hit is True
    assert by_fixture.matched_fixture_refs == (
        fixture_ref("accounts", "ACC-HOLD-02"),
    )
    assert by_fixture.matched_template_id is None
    assert by_template.held_out_hit is True
    assert by_template.matched_template_id == "tpl-held"


def test_a_verdict_cannot_disagree_with_its_own_evidence() -> None:
    with pytest.raises(ValidationError, match="requires the matched template or fixture"):
        HeldOutDecision(task_id="task-a", held_out_hit=True)
    with pytest.raises(ValidationError, match="requires the matched template or fixture"):
        HeldOutDecision(
            task_id="task-a",
            held_out_hit=False,
            matched_fixture_refs=(fixture_ref("accounts", "ACC-HOLD-01"),),
        )
    with pytest.raises(ValidationError):
        HeldOutDecision(task_id="task-a", held_out_hit="false")


def test_a_partial_scan_is_rejected_before_it_can_publish() -> None:
    clean = HeldOutDecision(task_id="task-a", held_out_hit=False)
    other = HeldOutDecision(task_id="task-b", held_out_hit=False)

    ordered = validate_complete_scan_set([other, clean], expected_task_ids=["task-a", "task-b"])
    assert [decision.task_id for decision in ordered] == ["task-a", "task-b"]

    with pytest.raises(ValueError, match="missing=.*task-b"):
        validate_complete_scan_set([clean], expected_task_ids=["task-a", "task-b"])
    with pytest.raises(ValueError, match="duplicate held-out decision"):
        validate_complete_scan_set([clean, clean], expected_task_ids=["task-a"])
    with pytest.raises(ValueError, match="must be unique"):
        validate_complete_scan_set([clean], expected_task_ids=["task-a", "task-a"])
