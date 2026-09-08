"""Focused contracts for private held-out generalization evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import eval_runner
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.held_out_eval import (
    HeldOutEvalConfig,
    HeldOutEvalError,
    expand_private_held_out_tasks,
    generalization_gap_interval,
    held_out_generalization_report,
    private_runtime_pack,
    verify_held_out_policy,
    wilson_interval,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HeldOutPolicy,
    fixture_ref,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack


def _policy_data(*, fixtures: bool = True, templates: bool = True) -> dict:
    return {
        "version": "held-1",
        "fixtures": {"things": ["T-2"]} if fixtures else {},
        "templates": ["tpl-held"] if templates else [],
        "policy": {"fixtures_in_backend_state": True, "seed": 17},
        "source": "held_out.yaml",
    }


def _config(policy: HeldOutPolicy, **overrides: object) -> HeldOutEvalConfig:
    values = {
        "policy_hash": policy.as_lineage()["policy_hash"],
        "fixture_refs": policy.fixture_refs,
        "template_ids": policy.template_ids,
        "seed": policy.seed,
        "pack_version": "1.2.3",
        "max_tasks_per_template": 4,
    }
    values.update(overrides)
    return HeldOutEvalConfig(**values)


def _pack(policy_data: dict) -> SimpleNamespace:
    templates = [
        {
            "template_id": "tpl-open",
            "category": "things",
            "slots": {"thing_id": {"source": "fixture:things.thing_id"}},
            "required_tools": ["read_thing"],
            "turn_policy": "single_turn",
        },
        {
            "template_id": "tpl-held",
            "category": "things",
            "slots": {"thing_id": {"source": "fixture:things.thing_id"}},
            "required_tools": ["read_thing"],
            "turn_policy": "single_turn",
        },
    ]
    return SimpleNamespace(
        manifest={
            "pack_id": "pack",
            "version": "1.2.3",
            "primary_keys": {"things": "thing_id"},
        },
        fixtures={"things": [{"thing_id": "T-1"}, {"thing_id": "T-2"}]},
        templates=templates,
        tools=[],
        held_out=policy_data,
    )


@pytest.mark.parametrize(
    ("fixtures", "templates", "expected_mode", "expected_count"),
    [
        (True, False, "fixture_only", 2),
        (False, True, "template_only", 2),
        (True, True, "both", 3),
    ],
)
def test_private_expansion_has_explicit_union_semantics(
    fixtures: bool,
    templates: bool,
    expected_mode: str,
    expected_count: int,
) -> None:
    data = _policy_data(fixtures=fixtures, templates=templates)
    policy = HeldOutPolicy.from_normalized(data)

    tasks, content_hash = expand_private_held_out_tasks(_pack(data), _config(policy))

    assert _config(policy).selection_mode == expected_mode
    assert len(tasks) == expected_count
    assert len({task["task_id"] for task in tasks}) == expected_count
    assert all(str(task["task_id"]).startswith("private-heldout-") for task in tasks)
    assert all(len(str(task["task_id"])) == len("private-heldout-") + 64 for task in tasks)
    assert content_hash.startswith("sha256:")
    if expected_mode == "fixture_only":
        assert all(fixture_ref("things", "T-2") in task["fixture_refs"] for task in tasks)
    if expected_mode == "template_only":
        assert all(task["template_id"] == "tpl-held" for task in tasks)


def test_fixture_private_expansion_skips_a_template_whose_filter_misses() -> None:
    data = _policy_data(fixtures=True, templates=False)
    policy = HeldOutPolicy.from_normalized(data)
    pack = _pack(data)
    pack.templates[0]["slots"]["thing_id"]["filter"] = "thing_id == 'T-1'"

    tasks, _content_hash = expand_private_held_out_tasks(pack, _config(policy))

    assert [task["template_id"] for task in tasks] == ["tpl-held"]
    assert tasks[0]["fixture_refs"] == [fixture_ref("things", "T-2")]


def test_policy_pin_verifies_every_frozen_field() -> None:
    policy = HeldOutPolicy.from_normalized(_policy_data())

    assert verify_held_out_policy(_config(policy), policy, pack_version="1.2.3") is policy

    with pytest.raises(HeldOutEvalError, match="policy_hash"):
        verify_held_out_policy(
            _config(policy, policy_hash="sha256:" + "0" * 64),
            policy,
            pack_version="1.2.3",
        )
    with pytest.raises(HeldOutEvalError, match="pack_version"):
        verify_held_out_policy(_config(policy), policy, pack_version="other")


def test_private_runtime_intentionally_opens_isolated_fixture_state() -> None:
    data = _policy_data()
    data["policy"]["fixtures_in_backend_state"] = False
    pack = LoadedPack(
        paths=SimpleNamespace(),
        manifest={"pack_id": "pack", "version": "1.2.3"},
        tools=[],
        fixtures={"things": [{"thing_id": "T-2"}]},
        templates=[],
        validation_cases=[],
        held_out=data,
    )

    runtime = private_runtime_pack(pack)

    assert runtime is not pack
    assert runtime.held_out["policy"]["fixtures_in_backend_state"] is True
    assert pack.held_out["policy"]["fixtures_in_backend_state"] is False


def test_private_candidate_cache_is_scoped_to_the_ephemeral_slice(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "ephemeral-private"
    private_root.mkdir()

    private_cache = eval_runner._private_candidate_cache(private_root)

    assert private_cache.path.parent == private_root
    assert private_cache.path != tmp_path / "candidate_io_cache.jsonl"


def test_report_is_paired_stratified_and_contains_no_private_rows() -> None:
    policy = HeldOutPolicy.from_normalized(_policy_data())
    seen_tasks = {
        "seen-1": {"required_tools": ["read_thing"], "turn_policy": "single_turn"},
        "seen-2": {"required_tools": ["read_thing"], "turn_policy": "single_turn"},
    }
    held_tasks = {
        "private-1": {"required_tools": ["read_thing"], "turn_policy": "single_turn"},
        "private-2": {"required_tools": ["other"], "turn_policy": "single_turn"},
    }
    report = held_out_generalization_report(
        seen_results=[
            {"candidate_alias": "candidate", "task_id": "seen-1", "task_success": True},
            {"candidate_alias": "candidate", "task_id": "seen-2", "task_success": True},
        ],
        held_out_results=[
            {
                "candidate_alias": "candidate",
                "task_id": "private-1",
                "task_success": False,
                "failure_records": [
                    {
                        "layer": "execution",
                        "code": "oracle_assertion_failed",
                        "attribution": "candidate",
                    }
                ],
            },
            {"candidate_alias": "candidate", "task_id": "private-2", "task_success": True},
        ],
        seen_tasks=seen_tasks,
        held_out_tasks=held_tasks,
        policy=policy,
        pack_version="1.2.3",
        slice_content_hash="sha256:" + "9" * 64,
    )

    candidate = report["candidates"][0]
    assert candidate["seen"]["success_rate"] == 1.0
    assert candidate["held_out"]["success_rate"] == 0.5
    assert candidate["held_out"]["failure_taxonomy"] == [
        {
            "layer": "execution",
            "code": "oracle_assertion_failed",
            "attribution": "candidate",
            "count": 1,
        }
    ]
    assert candidate["held_out_generalization_gap"] == 0.5
    assert candidate["held_out_generalization_gap_95"] == generalization_gap_interval(
        seen_successes=2,
        seen_total=2,
        held_out_successes=1,
        held_out_total=2,
    )
    assert candidate["matched_applicable_tool_turn_policy_strata"] == [
        {
            "applicable_tool": "read_thing",
            "turn_policy": "single_turn",
            "seen": {
                "successful_tasks": 2,
                "task_count": 2,
                "success_rate": 1.0,
                "wilson_95": wilson_interval(2, 2),
            },
            "held_out": {
                "successful_tasks": 0,
                "task_count": 1,
                "success_rate": 0.0,
                "wilson_95": wilson_interval(0, 1),
            },
            "held_out_generalization_gap": 1.0,
            "held_out_generalization_gap_95": generalization_gap_interval(
                seen_successes=2,
                seen_total=2,
                held_out_successes=0,
                held_out_total=1,
            ),
        }
    ]
    assert "private-1" not in str(report)
    assert report["privacy"] == {
        "private_tasks_written": False,
        "private_prompts_written": False,
        "private_candidate_caches_written": False,
    }


def test_report_rejects_missing_or_duplicate_candidate_evaluations() -> None:
    policy = HeldOutPolicy.from_normalized(_policy_data())
    task = {"required_tools": ["read_thing"], "turn_policy": "single_turn"}

    with pytest.raises(HeldOutEvalError, match="exactly once"):
        held_out_generalization_report(
            seen_results=[
                {"candidate_alias": "candidate", "task_id": "seen", "task_success": True}
            ],
            held_out_results=[],
            seen_tasks={"seen": task},
            held_out_tasks={"private": task},
            policy=policy,
            pack_version="1.2.3",
            slice_content_hash="sha256:" + "9" * 64,
        )
