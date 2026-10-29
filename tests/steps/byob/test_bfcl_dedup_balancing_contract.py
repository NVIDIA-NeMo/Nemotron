"""Contract tests for BFCL Stage 11 semantic deduplication and balancing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    BALANCING_DIMENSIONS,
    DEDUP_BALANCING_CONTRACT_VERSION,
    DedupBalancingDecision,
    Stage11Coverage,
    validate_complete_decision_set,
)


def _selected(task_id: str, rank: int = 0) -> DedupBalancingDecision:
    return DedupBalancingDecision(
        task_id=task_id,
        selected=True,
        is_duplicate=False,
        selection_rank=rank,
    )


def _coverage(
    task_ids: list[str],
    *,
    language: str = "en",
    turn_policy: str = "single_turn",
    edge_signatures: tuple[str, ...] = (),
) -> dict[str, Stage11Coverage]:
    return {
        task_id: Stage11Coverage(
            language=language,
            turn_policy=turn_policy,
            edge_signatures=edge_signatures,
        )
        for task_id in task_ids
    }


def _representative(task_id: str, cluster_id: str, rank: int) -> DedupBalancingDecision:
    return DedupBalancingDecision(
        task_id=task_id,
        selected=True,
        is_duplicate=False,
        duplicate_cluster_id=cluster_id,
        representative_task_id=task_id,
        selection_rank=rank,
    )


def _duplicate(
    task_id: str,
    cluster_id: str,
    representative_task_id: str,
    rank: int,
) -> DedupBalancingDecision:
    return DedupBalancingDecision(
        task_id=task_id,
        selected=False,
        is_duplicate=True,
        duplicate_cluster_id=cluster_id,
        representative_task_id=representative_task_id,
        drop_reason="semantic_duplicate",
        selection_rank=rank,
    )


def test_contract_locks_version_and_eight_generic_dimensions() -> None:
    assert DEDUP_BALANCING_CONTRACT_VERSION == "1.0"
    assert BALANCING_DIMENSIONS == (
        "intent",
        "category",
        "required_tools",
        "tools_present",
        "difficulty",
        "turn_class",
        "tool_call_count",
        "turn_policy",
    )
    assert len(BALANCING_DIMENSIONS) == 8


def test_selected_row_cannot_carry_drop_detail() -> None:
    assert _selected("pack__template__001").contract_version == "1.0"

    with pytest.raises(ValidationError, match="cannot carry drop detail"):
        DedupBalancingDecision(
            task_id="pack__template__001",
            selected=True,
            is_duplicate=False,
            drop_reason="balance_quota",
            balance_dimension="category",
            selection_rank=0,
        )


def test_semantic_duplicate_requires_a_different_representative() -> None:
    duplicate = DedupBalancingDecision(
        task_id="pack__template__002",
        selected=False,
        is_duplicate=True,
        duplicate_cluster_id="cluster-7",
        representative_task_id="pack__template__001",
        drop_reason="semantic_duplicate",
        selection_rank=1,
    )
    assert duplicate.representative_task_id == "pack__template__001"

    with pytest.raises(ValidationError, match="cannot represent itself"):
        DedupBalancingDecision(
            task_id="pack__template__002",
            selected=False,
            is_duplicate=True,
            duplicate_cluster_id="cluster-7",
            representative_task_id="pack__template__002",
            drop_reason="semantic_duplicate",
            selection_rank=1,
        )
    with pytest.raises(ValidationError, match="valid only for a duplicate"):
        DedupBalancingDecision(
            task_id="pack__template__002",
            selected=False,
            is_duplicate=False,
            drop_reason="semantic_duplicate",
            selection_rank=1,
        )


def test_balance_quota_requires_one_locked_dimension() -> None:
    decision = DedupBalancingDecision(
        task_id="pack__template__003",
        selected=False,
        is_duplicate=False,
        drop_reason="balance_quota",
        balance_dimension="turn_policy",
        selection_rank=2,
    )
    assert decision.balance_dimension == "turn_policy"

    with pytest.raises(ValidationError, match="requires balance_dimension"):
        DedupBalancingDecision(
            task_id="pack__template__003",
            selected=False,
            is_duplicate=False,
            drop_reason="balance_quota",
            selection_rank=2,
        )
    with pytest.raises(ValidationError, match="Input should be"):
        DedupBalancingDecision(
            task_id="pack__template__003",
            selected=False,
            is_duplicate=False,
            drop_reason="balance_quota",
            balance_dimension="banking_product",
            selection_rank=2,
        )


@pytest.mark.parametrize("value", ["true", "false", 0, 1])
def test_decision_booleans_are_strict(value: object) -> None:
    with pytest.raises(ValidationError):
        DedupBalancingDecision(
            task_id="pack__template__001",
            selected=value,
            is_duplicate=False,
            selection_rank=0,
        )


def test_decision_rank_and_cluster_identifiers_are_strict() -> None:
    with pytest.raises(ValidationError):
        DedupBalancingDecision(
            task_id="pack__template__001",
            selected=True,
            is_duplicate=False,
            selection_rank=True,
        )
    with pytest.raises(ValidationError, match="must be non-empty"):
        DedupBalancingDecision(
            task_id="pack__template__002",
            selected=False,
            is_duplicate=True,
            duplicate_cluster_id=" ",
            representative_task_id="pack__template__001",
            drop_reason="semantic_duplicate",
            selection_rank=1,
        )
    with pytest.raises(ValidationError, match="must represent itself"):
        DedupBalancingDecision(
            task_id="pack__template__002",
            selected=True,
            is_duplicate=False,
            duplicate_cluster_id="cluster-7",
            representative_task_id="pack__template__001",
            selection_rank=1,
        )


def test_complete_set_requires_exactly_one_decision_per_stage_ten_survivor() -> None:
    ordered = validate_complete_decision_set(
        [_selected("task-b", 1), _selected("task-a", 0)],
        input_task_ids=["task-a", "task-b"],
        coverage_by_task_id=_coverage(["task-a", "task-b"]),
        remove_duplicates=True,
    )
    assert [decision.task_id for decision in ordered] == ["task-a", "task-b"]

    with pytest.raises(ValueError, match="duplicate Stage 11 decision"):
        validate_complete_decision_set(
            [_selected("task-a"), _selected("task-a")],
            input_task_ids=["task-a"],
            coverage_by_task_id=_coverage(["task-a"]),
            remove_duplicates=True,
        )
    with pytest.raises(ValueError, match="missing=.*task-b"):
        validate_complete_decision_set(
            [_selected("task-a"), _selected("task-extra")],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=True,
        )
    with pytest.raises(ValueError, match="input task_id values must be unique"):
        validate_complete_decision_set(
            [_selected("task-a")],
            input_task_ids=["task-a", "task-a"],
            coverage_by_task_id=_coverage(["task-a"]),
            remove_duplicates=True,
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_complete_decision_set(
            [],
            input_task_ids=[""],
            coverage_by_task_id={},
            remove_duplicates=True,
        )


def test_complete_set_requires_an_in_set_representative() -> None:
    with pytest.raises(ValueError, match="outside Stage 11"):
        validate_complete_decision_set(
            [_duplicate("task-a", "cluster-1", "outside", 1)],
            input_task_ids=["task-a"],
            coverage_by_task_id=_coverage(["task-a"]),
            remove_duplicates=True,
        )

    duplicate = _duplicate("task-a", "cluster-1", "task-b", 1)
    ordered = validate_complete_decision_set(
        [duplicate, _representative("task-b", "cluster-1", 0)],
        input_task_ids=["task-a", "task-b"],
        coverage_by_task_id=_coverage(["task-a", "task-b"]),
        remove_duplicates=True,
    )
    assert [decision.task_id for decision in ordered] == ["task-a", "task-b"]

    with pytest.raises(ValueError, match="same cluster metadata"):
        validate_complete_decision_set(
            [duplicate, _representative("task-b", "cluster-2", 0)],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=True,
        )


def test_balancing_may_drop_a_whole_cluster_including_its_representative() -> None:
    quota_dropped_representative = DedupBalancingDecision(
        task_id="task-b",
        selected=False,
        is_duplicate=False,
        duplicate_cluster_id="cluster-1",
        representative_task_id="task-b",
        drop_reason="balance_quota",
        balance_dimension="category",
        selection_rank=2,
    )
    ordered = validate_complete_decision_set(
        [
            _duplicate("task-a", "cluster-1", "task-b", 1),
            quota_dropped_representative,
            _selected("task-c", 0),
        ],
        input_task_ids=["task-a", "task-b", "task-c"],
        coverage_by_task_id=_coverage(["task-a", "task-b", "task-c"]),
        remove_duplicates=True,
    )

    assert [decision.selected for decision in ordered] == [False, False, True]


def test_selected_cluster_member_requires_a_selected_representative() -> None:
    dropped_representative = DedupBalancingDecision(
        task_id="task-b",
        selected=False,
        is_duplicate=False,
        duplicate_cluster_id="cluster-1",
        representative_task_id="task-b",
        drop_reason="balance_quota",
        balance_dimension="category",
        selection_rank=1,
    )
    selected_duplicate = DedupBalancingDecision(
        task_id="task-a",
        selected=True,
        is_duplicate=True,
        duplicate_cluster_id="cluster-1",
        representative_task_id="task-b",
        selection_rank=0,
    )

    with pytest.raises(ValueError, match="selected members.*representative.*dropped"):
        validate_complete_decision_set(
            [selected_duplicate, dropped_representative],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=False,
        )


def test_a_cluster_carries_exactly_one_representative() -> None:
    with pytest.raises(ValueError, match="exactly one representative, got 2"):
        validate_complete_decision_set(
            [
                _representative("task-a", "cluster-1", 0),
                _representative("task-b", "cluster-1", 1),
            ],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=True,
        )


def test_a_cluster_cannot_span_two_coverage_buckets() -> None:
    decisions = [
        _duplicate("task-a", "cluster-1", "task-b", 1),
        _representative("task-b", "cluster-1", 0),
        _selected("task-c", 1),
    ]
    coverage = _coverage(["task-a", "task-b", "task-c"])
    coverage["task-a"] = Stage11Coverage(language="en", turn_policy="clarify_only")
    coverage["task-c"] = Stage11Coverage(language="en", turn_policy="clarify_only")

    with pytest.raises(ValueError, match="share one coverage bucket"):
        validate_complete_decision_set(
            decisions,
            input_task_ids=["task-a", "task-b", "task-c"],
            coverage_by_task_id=coverage,
            remove_duplicates=True,
        )


def test_complete_set_binds_duplicate_selection_to_policy() -> None:
    representative = _representative("task-b", "cluster-1", 0)
    selected_duplicate = DedupBalancingDecision(
        task_id="task-a",
        selected=True,
        is_duplicate=True,
        duplicate_cluster_id="cluster-1",
        representative_task_id="task-b",
        selection_rank=1,
    )
    validate_complete_decision_set(
        [selected_duplicate, representative],
        input_task_ids=["task-a", "task-b"],
        coverage_by_task_id=_coverage(["task-a", "task-b"]),
        remove_duplicates=False,
    )
    with pytest.raises(
        ValueError,
        match="remove_duplicates=true",
    ):
        validate_complete_decision_set(
            [selected_duplicate, representative],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=True,
        )
    with pytest.raises(
        ValueError,
        match="remove_duplicates=false",
    ):
        validate_complete_decision_set(
            [_duplicate("task-a", "cluster-1", "task-b", 1), representative],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=False,
        )


def test_selected_rows_carry_a_total_publication_order() -> None:
    with pytest.raises(ValueError, match="selection_rank 0..k-1"):
        validate_complete_decision_set(
            [_selected("task-a", 0), _selected("task-b", 0)],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=True,
        )
    with pytest.raises(ValueError, match="selection_rank 0..k-1"):
        validate_complete_decision_set(
            [_selected("task-a", 1), _selected("task-b", 2)],
            input_task_ids=["task-a", "task-b"],
            coverage_by_task_id=_coverage(["task-a", "task-b"]),
            remove_duplicates=True,
        )


def _dropped_on_quota(task_id: str, rank: int) -> DedupBalancingDecision:
    return DedupBalancingDecision(
        task_id=task_id,
        selected=False,
        is_duplicate=False,
        drop_reason="balance_quota",
        balance_dimension="turn_policy",
        selection_rank=rank,
    )


def test_complete_set_preserves_every_locked_coverage_bucket() -> None:
    decisions = [_selected("common", 0), _dropped_on_quota("rare", 1)]
    coverage = {
        "common": Stage11Coverage(language="en", turn_policy="single_turn"),
        "rare": Stage11Coverage(
            language="en",
            turn_policy="clarify_only",
            edge_signatures=("ambiguous_reference",),
        ),
    }
    with pytest.raises(ValueError, match="turn_policy='clarify_only'"):
        validate_complete_decision_set(
            decisions,
            input_task_ids=["common", "rare"],
            coverage_by_task_id=coverage,
            remove_duplicates=True,
        )

    with pytest.raises(ValueError, match="edge_signatures=.*ambiguous_reference"):
        validate_complete_decision_set(
            decisions,
            input_task_ids=["common", "rare"],
            coverage_by_task_id={
                "common": Stage11Coverage(language="en", turn_policy="single_turn"),
                "rare": Stage11Coverage(
                    language="en",
                    turn_policy="single_turn",
                    edge_signatures=("ambiguous_reference",),
                ),
            },
            remove_duplicates=True,
        )

    with pytest.raises(ValueError, match="language='vi'"):
        validate_complete_decision_set(
            decisions,
            input_task_ids=["common", "rare"],
            coverage_by_task_id={
                "common": Stage11Coverage(language="en", turn_policy="single_turn"),
                "rare": Stage11Coverage(language="vi", turn_policy="single_turn"),
            },
            remove_duplicates=True,
        )


def test_complete_set_preserves_composite_coverage_buckets() -> None:
    decisions = [
        _selected("en-single", 0),
        _selected("vi-clarify", 1),
        _dropped_on_quota("en-clarify", 2),
        _dropped_on_quota("vi-single", 3),
    ]
    coverage = {
        "en-single": Stage11Coverage(language="en", turn_policy="single_turn"),
        "vi-clarify": Stage11Coverage(language="vi", turn_policy="clarify_only"),
        "en-clarify": Stage11Coverage(language="en", turn_policy="clarify_only"),
        "vi-single": Stage11Coverage(language="vi", turn_policy="single_turn"),
    }

    with pytest.raises(
        ValueError,
        match="coverage bucket.*language='en'.*turn_policy='clarify_only'",
    ):
        validate_complete_decision_set(
            decisions,
            input_task_ids=list(coverage),
            coverage_by_task_id=coverage,
            remove_duplicates=True,
        )


def test_coverage_buckets_are_normalized_before_comparison() -> None:
    bucket = Stage11Coverage(
        language=" en ",
        turn_policy=" single_turn ",
        edge_signatures=(" missing_slot ", "ambiguous_reference"),
    )

    assert bucket.language == "en"
    assert bucket.turn_policy == "single_turn"
    assert bucket.edge_signatures == ("ambiguous_reference", "missing_slot")
    assert bucket == Stage11Coverage(
        language="en",
        turn_policy="single_turn",
        edge_signatures=("missing_slot", "ambiguous_reference"),
    )
    assert (
        DedupBalancingDecision(
            task_id=" task-a ",
            selected=True,
            is_duplicate=False,
            selection_rank=0,
        ).task_id
        == "task-a"
    )
    ordered = validate_complete_decision_set(
        [_selected("task-a")],
        input_task_ids=[" task-a "],
        coverage_by_task_id={" task-a ": Stage11Coverage(language="en", turn_policy="single_turn")},
        remove_duplicates=True,
    )
    assert [decision.task_id for decision in ordered] == ["task-a"]

    with pytest.raises(ValueError, match="coverage.*after normalization"):
        validate_complete_decision_set(
            [_selected("task-a")],
            input_task_ids=["task-a"],
            coverage_by_task_id={
                "task-a": Stage11Coverage(language="en", turn_policy="single_turn"),
                " task-a ": Stage11Coverage(language="en", turn_policy="single_turn"),
            },
            remove_duplicates=True,
        )
