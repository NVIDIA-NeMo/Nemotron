"""Tests for Stage 11 eight-dimension publication balancing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    BALANCING_DIMENSIONS,
    DedupBalancingDecision,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
    _publication_shortfall_reason,
    balance_publication_set,
    balancing_features,
    largest_remainder_quotas,
)

BFCL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "bfcl" / "config"


def _config(
    tmp_path: Path,
    *,
    task_generation: dict[str, Any] | None = None,
    remove_duplicates: bool = True,
    dedup: dict[str, Any] | None = None,
) -> BfclConfig:
    data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    data["output_dir"] = str(tmp_path / "output")
    data["random_seed"] = 23
    data["surface_quality_validation"] = {
        **(data.get("surface_quality_validation") or {}),
        "enabled": True,
    }
    data["task_generation"] = {
        **(data.get("task_generation") or {}),
        "tasks_per_category": 100,
        **(task_generation or {}),
    }
    data["semantic_deduplication_config"] = {
        "enabled": True,
        "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": 20,
        "eps": 0.08,
        "remove_duplicates": remove_duplicates,
        **(dedup or {}),
    }
    path = tmp_path / "balancing.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return BfclConfig.from_yaml(path)


def _task(task_id: str, **overrides: Any) -> dict[str, Any]:
    task = {
        "task_id": task_id,
        "template_id": "template",
        "intent": "lookup",
        "category": "general",
        "required_tools": ["lookup"],
        "tools_present": ["lookup"],
        "difficulty": "easy",
        "turn_policy": "single_turn",
        "num_tool_calls": 1,
    }
    task.update(overrides)
    return task


def _surface(task_id: str, *, turns: int = 1) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "language": "en",
        "source": "template",
        "steps": [{"kind": "user", "content": f"Request {index}"} for index in range(turns)],
    }


def _selected(task_ids: list[str]) -> list[DedupBalancingDecision]:
    return [
        DedupBalancingDecision(
            task_id=task_id,
            selected=True,
            is_duplicate=False,
            selection_rank=index,
        )
        for index, task_id in enumerate(task_ids)
    ]


def _run(
    config: BfclConfig,
    tasks: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    decisions: list[DedupBalancingDecision] | None = None,
    **kwargs: Any,
) -> tuple[list[DedupBalancingDecision], list[dict[str, Any]], dict[str, Any]]:
    ids = [task["task_id"] for task in tasks]
    return balance_publication_set(
        config,
        tasks,
        surfaces,
        decisions or _selected(ids),
        **kwargs,
    )


def test_largest_remainder_quotas_are_exact_and_deterministic() -> None:
    assert largest_remainder_quotas(
        7,
        {"easy": 0.5, "medium": 0.3, "hard": 0.2},
    ) == {"easy": 4, "medium": 2, "hard": 1}
    assert largest_remainder_quotas(1, {"b": 0.5, "a": 0.5}) == {
        "a": 1,
        "b": 0,
    }
    assert largest_remainder_quotas(0, {"a": 1.0}) == {"a": 0}


def test_features_cover_all_eight_dimensions() -> None:
    task = _task(
        "task-a",
        intent="transfer",
        category="payments",
        required_tools=["b", "a"],
        tools_present=["c", "a"],
        difficulty="hard",
        turn_policy="confirmation",
        num_tool_calls=3,
    )

    features = balancing_features(task, _surface("task-a", turns=2))

    assert set(BALANCING_DIMENSIONS) <= set(features)
    assert features["required_tools"] == '["a","b"]'
    assert features["tools_present"] == '["a","c"]'
    assert features["turn_class"] == "multi_turn"
    assert features["tool_call_count"] == "3+"


def test_balancing_crosses_a_neutral_swap_to_reach_a_feasible_mix(
    tmp_path: Path,
) -> None:
    """A strict local-improvement swap policy gets stuck on this inventory."""
    triples = [
        ("hard", 1, 2),
        ("easy", 2, 2),
        ("easy", 2, 2),
        ("hard", 2, 2),
        ("hard", 1, 1),
        ("easy", 1, 1),
        ("hard", 2, 1),
        ("easy", 1, 2),
    ]
    tasks = [
        _task(task_id, difficulty=difficulty, num_tool_calls=tool_calls)
        for task_id, (difficulty, _, tool_calls) in (
            (f"task-{index}", triple) for index, triple in enumerate(triples)
        )
    ]
    surfaces = {
        task["task_id"]: _surface(
            task["task_id"],
            turns=triples[index][1],
        )
        for index, task in enumerate(tasks)
    }

    _, _, summary = _run(
        _config(
            tmp_path,
            task_generation={
                "tasks_per_category": 4,
                "difficulty_mix": {"easy": 0.5, "hard": 0.5},
                "turn_mix": {"single_turn": 0.5, "multi_turn": 0.5},
                "tool_call_count_mix": {"1": 0.5, "2": 0.5},
            },
        ),
        tasks,
        surfaces,
    )

    assert summary["selected_count"] == 4
    assert summary["unmet_targets"] == []


def test_balancing_finds_the_largest_feasible_difficulty_mix(
    tmp_path: Path,
) -> None:
    difficulties = ["easy"] * 5 + ["medium"] * 3 + ["hard"] * 2
    tasks = [_task(f"task-{index}", difficulty=difficulty) for index, difficulty in enumerate(difficulties)]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}

    decisions, _, summary = _run(
        _config(
            tmp_path,
            task_generation={"difficulty_mix": {"easy": 0.3, "medium": 0.5, "hard": 0.2}},
        ),
        tasks,
        surfaces,
    )

    assert sum(decision.selected for decision in decisions) == 6
    assert summary["target_counts"]["difficulty"] == {
        "easy": 2,
        "hard": 1,
        "medium": 3,
    }
    assert summary["actual_counts"]["difficulty"] == {
        "easy": 2,
        "hard": 1,
        "medium": 3,
    }
    assert summary["unmet_targets"] == []


def test_a_feasible_mix_is_reached_despite_the_greedy_pick_order(
    tmp_path: Path,
) -> None:
    tasks = [
        _task("task-0"),
        _task("task-1"),
        _task("task-2", difficulty="hard"),
    ]
    surfaces = {
        "task-0": _surface("task-0"),
        "task-1": _surface("task-1", turns=2),
        "task-2": _surface("task-2"),
    }

    decisions, records, summary = _run(
        _config(
            tmp_path,
            task_generation={
                "tasks_per_category": 2,
                "difficulty_mix": {"easy": 0.5, "hard": 0.5},
                "turn_mix": {"single_turn": 0.5, "multi_turn": 0.5},
            },
        ),
        tasks,
        surfaces,
    )

    assert {decision.task_id for decision in decisions if decision.selected} == {
        "task-1",
        "task-2",
    }
    assert summary["actual_counts"]["difficulty"] == {"easy": 1, "hard": 1}
    assert summary["actual_counts"]["turn_class"] == {
        "multi_turn": 1,
        "single_turn": 1,
    }
    assert summary["unmet_targets"] == []
    assert sorted(record["selection_rank"] for record in records if record["selected"]) == [0, 1]


def test_category_cap_is_applied_without_cloning_rows(tmp_path: Path) -> None:
    tasks = [
        *[_task(f"a-{index}", category="a") for index in range(3)],
        *[_task(f"b-{index}", category="b") for index in range(3)],
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}

    decisions, _, summary = _run(
        _config(tmp_path, task_generation={"tasks_per_category": 2}),
        tasks,
        surfaces,
    )

    assert sum(decision.selected for decision in decisions) == 4
    assert summary["actual_counts"]["category"] == {"a": 2, "b": 2}
    assert len({decision.task_id for decision in decisions}) == len(tasks)


def test_turn_and_tool_call_targets_are_satisfied_together(
    tmp_path: Path,
) -> None:
    tasks = [
        _task("single-a"),
        _task("single-b"),
        _task("multi-a", turn_policy="multi_tool", num_tool_calls=2),
        _task("multi-b", turn_policy="multi_tool", num_tool_calls=2),
    ]
    surfaces = {
        "single-a": _surface("single-a"),
        "single-b": _surface("single-b"),
        "multi-a": _surface("multi-a", turns=2),
        "multi-b": _surface("multi-b", turns=2),
    }

    _, _, summary = _run(
        _config(
            tmp_path,
            task_generation={
                "tasks_per_category": 2,
                "turn_mix": {"single_turn": 0.5, "multi_turn": 0.5},
                "tool_call_count_mix": {"1": 0.5, "2": 0.5},
            },
        ),
        tasks,
        surfaces,
    )

    assert summary["actual_counts"]["turn_class"] == {
        "multi_turn": 1,
        "single_turn": 1,
    }
    assert summary["actual_counts"]["tool_call_count"] == {"1": 1, "2": 1}
    assert summary["unmet_targets"] == []


def test_policy_mix_targets_the_conversation_shape_directly(tmp_path: Path) -> None:
    # Conversation policy is what a candidate most often fails, and inventory skews
    # towards cheap lookups, so a release has to be able to demand the harder shapes
    # rather than accept whatever expansion produced.
    tasks = [
        *[_task(f"lookup-{index}") for index in range(4)],
        *[
            _task(f"refuse-{index}", turn_policy="clarify_only", num_tool_calls=0)
            for index in range(2)
        ],
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}

    _, _, summary = _run(
        _config(
            tmp_path,
            task_generation={
                "tasks_per_category": 4,
                "policy_mix": {"single_turn": 0.5, "clarify_only": 0.5},
            },
        ),
        tasks,
        surfaces,
    )

    assert summary["target_counts"]["turn_policy"] == {
        "clarify_only": 2,
        "single_turn": 2,
    }
    assert summary["actual_counts"]["turn_policy"] == {
        "clarify_only": 2,
        "single_turn": 2,
    }
    assert summary["unmet_targets"] == []


def test_execution_case_cap_publishes_each_executable_case_once(
    tmp_path: Path,
) -> None:
    # Two rows that call the same tools with the same arguments against the same state
    # score one behavior twice, so a release can require every published row to be a
    # distinct executable case even when the wording differs.
    tasks = [
        _task("a", slots={"account_id": "ACC-1"}),
        _task("b", slots={"account_id": "ACC-1"}),
        _task("c", slots={"account_id": "ACC-2"}),
        _task("d", slots={"account_id": "ACC-2"}),
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}

    decisions, _, summary = _run(
        _config(
            tmp_path,
            task_generation={"tasks_per_category": 4, "target_published_tasks": 4},
            dedup={"max_execution_case_reuse": 1},
        ),
        tasks,
        surfaces,
    )

    assert sum(decision.selected for decision in decisions) == 2
    assert summary["group_diversity"]["execution_case_hash"] == {
        "unique": 2,
        "max_reuse": 1,
        "cap": 1,
    }
    assert summary["unmet_targets"] == [
        {
            "dimension": "publication_count",
            "bucket": "all",
            "target": 4,
            "actual": 2,
            "inventory": 4,
            "reason": "execution_case_cap_limits_inventory",
        }
    ]


def test_intent_cap_stops_one_intent_from_owning_the_release(tmp_path: Path) -> None:
    tasks = [
        *[_task(f"lookup-{index}", slots={"n": index}) for index in range(3)],
        _task("transfer-0", intent="transfer", slots={"n": 9}),
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}

    decisions, records, summary = _run(
        _config(
            tmp_path,
            task_generation={"tasks_per_category": 4},
            dedup={"max_rows_per_intent": 2},
        ),
        tasks,
        surfaces,
    )

    by_id = {decision.task_id: decision for decision in decisions}
    assert sum(decision.selected for decision in decisions) == 3
    assert by_id["transfer-0"].selected is True
    assert summary["group_diversity"]["intent"] == {
        "unique": 2,
        "max_reuse": 2,
        "cap": 2,
    }
    dropped = next(record for record in records if not record["selected"])
    assert dropped["drop_reason"] == "balance_quota"
    assert dropped["balance_dimension"] == "intent"


def test_category_budget_outranks_a_declared_mix(tmp_path: Path) -> None:
    # Coverage locks are the only thing allowed to push a category past its cap, and
    # they force one unavoidable overflow row here. The declared mix could be met
    # exactly by adding a second overflow row, so this pins the ranking: the category
    # budget is honoured before a mix target rather than traded against it at equal
    # weight.
    tasks = [
        *[_task(f"locked-{index}", category="a") for index in range(3)],
        _task("spare-a", category="a"),
        *[_task(f"b-{index}", category="b", difficulty="hard") for index in range(2)],
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}
    edges = {
        task["task_id"]: ([f"edge_{task['task_id']}"] if task["task_id"].startswith("locked-") else [])
        for task in tasks
    }

    decisions, _, summary = _run(
        _config(
            tmp_path,
            task_generation={
                "tasks_per_category": 2,
                "difficulty_mix": {"easy": 1.0},
            },
        ),
        tasks,
        surfaces,
        edge_signatures_by_task_id=edges,
    )

    selected = {decision.task_id for decision in decisions if decision.selected}
    assert {"locked-0", "locked-1", "locked-2"} <= selected
    assert "spare-a" not in selected
    assert summary["actual_counts"]["category"] == {"a": 3, "b": 1}
    assert [
        (entry["dimension"], entry["bucket"], entry["reason"])
        for entry in summary["unmet_targets"]
    ] == [
        ("category", "a", "coverage_constraint"),
        ("difficulty", "easy", "coverage_or_cross_dimension_constraint"),
    ]


def test_coverage_survivor_overrides_a_soft_mix_target(tmp_path: Path) -> None:
    tasks = [
        *[_task(f"easy-{index}") for index in range(4)],
        _task("rare-hard", difficulty="hard"),
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}
    edges = {task["task_id"]: (["rare_edge"] if task["task_id"] == "rare-hard" else []) for task in tasks}

    decisions, records, summary = _run(
        _config(
            tmp_path,
            task_generation={"difficulty_mix": {"easy": 1.0}},
        ),
        tasks,
        surfaces,
        edge_signatures_by_task_id=edges,
    )

    by_id = {decision.task_id: decision for decision in decisions}
    assert by_id["rare-hard"].selected is True
    assert next(record for record in records if record["task_id"] == "rare-hard")["coverage_locked"] is True
    assert summary["actual_counts"]["difficulty"]["hard"] == 1
    assert summary["unmet_targets"][0]["reason"] == ("coverage_or_cross_dimension_constraint")


@pytest.mark.parametrize(
    ("limit_key", "task_overrides", "turns", "reason"),
    [
        (
            "max_turns",
            {},
            3,
            "max_turns_exceeded",
        ),
        (
            "max_tool_calls",
            {"num_tool_calls": 4},
            1,
            "max_tool_calls_exceeded",
        ),
    ],
)
def test_hard_limits_drop_rows_with_controlled_reasons(
    tmp_path: Path,
    limit_key: str,
    task_overrides: dict[str, Any],
    turns: int,
    reason: str,
) -> None:
    tasks = [
        _task("kept"),
        _task("limited", **task_overrides),
    ]
    surfaces = {
        "kept": _surface("kept"),
        "limited": _surface("limited", turns=turns),
    }

    decisions, _, summary = _run(
        _config(tmp_path, task_generation={limit_key: 2}),
        tasks,
        surfaces,
    )

    by_id = {decision.task_id: decision for decision in decisions}
    assert by_id["limited"].drop_reason == reason
    assert summary["hard_limit_drops"] == {reason: 1}


def test_hard_limits_cannot_remove_the_last_coverage_survivor(
    tmp_path: Path,
) -> None:
    tasks = [_task("normal"), _task("rare")]
    surfaces = {
        "normal": _surface("normal"),
        "rare": _surface("rare", turns=3),
    }

    with pytest.raises(ValueError, match="final survivor of coverage bucket"):
        _run(
            _config(tmp_path, task_generation={"max_turns": 2}),
            tasks,
            surfaces,
            edge_signatures_by_task_id={"normal": [], "rare": ["rare_edge"]},
        )


def test_cluster_members_never_survive_without_their_representative(
    tmp_path: Path,
) -> None:
    tasks = [_task("representative"), _task("duplicate")]
    surfaces = {
        "representative": _surface("representative"),
        "duplicate": _surface("duplicate"),
    }
    cluster = "cluster-1"
    decisions = [
        DedupBalancingDecision(
            task_id="representative",
            selected=True,
            is_duplicate=False,
            duplicate_cluster_id=cluster,
            representative_task_id="representative",
            selection_rank=0,
        ),
        DedupBalancingDecision(
            task_id="duplicate",
            selected=True,
            is_duplicate=True,
            duplicate_cluster_id=cluster,
            representative_task_id="representative",
            selection_rank=1,
        ),
    ]

    balanced, _, _ = _run(
        _config(
            tmp_path,
            remove_duplicates=False,
            task_generation={"tasks_per_category": 1},
        ),
        tasks,
        surfaces,
        decisions,
    )

    by_id = {decision.task_id: decision for decision in balanced}
    assert by_id["representative"].selected is True
    assert by_id["duplicate"].selected is False
    assert by_id["duplicate"].drop_reason == "balance_quota"


def test_duplicate_is_ineligible_when_its_representative_hits_a_hard_limit(
    tmp_path: Path,
) -> None:
    tasks = [
        _task("representative", num_tool_calls=4),
        _task("duplicate"),
        _task("other"),
    ]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}
    cluster = "cluster-1"
    decisions = [
        DedupBalancingDecision(
            task_id="representative",
            selected=True,
            is_duplicate=False,
            duplicate_cluster_id=cluster,
            representative_task_id="representative",
            selection_rank=0,
        ),
        DedupBalancingDecision(
            task_id="duplicate",
            selected=True,
            is_duplicate=True,
            duplicate_cluster_id=cluster,
            representative_task_id="representative",
            selection_rank=1,
        ),
        DedupBalancingDecision(
            task_id="other",
            selected=True,
            is_duplicate=False,
            selection_rank=2,
        ),
    ]

    balanced, _, _ = _run(
        _config(
            tmp_path,
            remove_duplicates=False,
            task_generation={"max_tool_calls": 3},
        ),
        tasks,
        surfaces,
        decisions,
    )

    by_id = {decision.task_id: decision for decision in balanced}
    assert by_id["representative"].drop_reason == "max_tool_calls_exceeded"
    assert by_id["duplicate"].selected is False
    assert by_id["other"].selected is True


def test_balancing_is_deterministic_and_reports_every_dimension(
    tmp_path: Path,
) -> None:
    tasks = [_task(f"task-{index}") for index in range(5)]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}
    config = _config(tmp_path, task_generation={"tasks_per_category": 3})

    first, _, first_summary = _run(config, tasks, surfaces)
    second, _, second_summary = _run(config, tasks, surfaces)

    assert [decision.task_id for decision in first if decision.selected] == [
        decision.task_id for decision in second if decision.selected
    ]
    assert first_summary == second_summary
    assert set(first_summary["actual_counts"]) == set(BALANCING_DIMENSIONS)


def test_balancing_enforces_exact_surface_reuse_and_ratio(
    tmp_path: Path,
) -> None:
    tasks = [_task(task_id) for task_id in ("a", "b", "c", "d")]
    surfaces = {
        "a": _surface("a"),
        "b": _surface("b"),
        "c": {
            **_surface("c"),
            "steps": [{"kind": "user", "content": "A different request"}],
        },
        "d": {
            **_surface("d"),
            "steps": [{"kind": "user", "content": "A third request"}],
        },
    }
    config = _config(
        tmp_path,
        task_generation={
            "tasks_per_category": 3,
            "target_published_tasks": 3,
        },
        dedup={
            "max_exact_surface_reuse": 1,
            "min_exact_surface_ratio": 1.0,
        },
    )

    decisions, _, summary = _run(config, tasks, surfaces)

    selected = [decision.task_id for decision in decisions if decision.selected]
    assert len(selected) == 3
    assert len(
        {
            balancing_features(
                next(task for task in tasks if task["task_id"] == task_id),
                surfaces[task_id],
            )["surface_text_hash"]
            for task_id in selected
        }
    ) == 3
    assert summary["exact_surface_diversity"] == {
        "unique": 3,
        "unique_ratio": 1.0,
        "max_reuse": 1,
        "max_exact_surface_reuse": 1,
        "min_exact_surface_ratio": 1.0,
    }
    assert summary["unmet_targets"] == []


def test_publication_target_names_the_bound_that_caused_the_shortfall(
    tmp_path: Path,
) -> None:
    tasks = [_task(task_id) for task_id in ("a", "b", "c")]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}
    config = _config(
        tmp_path,
        task_generation={
            "tasks_per_category": 3,
            "target_published_tasks": 3,
        },
        dedup={"max_exact_surface_reuse": 1},
    )

    _, _, summary = _run(config, tasks, surfaces)

    assert summary["selected_count"] == 1
    assert summary["unmet_targets"] == [
        {
            "dimension": "publication_count",
            "bucket": "all",
            "target": 3,
            "actual": 1,
            "inventory": 3,
            "reason": "insufficient_surface_diversity",
        }
    ]

    # The same shortfall shape has several possible causes, and a run that aborts after
    # hours is only actionable if the report names the bound that actually applied.
    reason = _publication_shortfall_reason
    assert reason(target=10, selected=10, bounds={}, mix_exceeds_inventory=False) == (
        "balancing_constraint"
    )
    assert reason(target=10, selected=11, bounds={}, mix_exceeds_inventory=False) == (
        "coverage_requires_more_than_declared_target"
    )
    assert (
        reason(target=10, selected=4, bounds={}, mix_exceeds_inventory=True)
        == "declared_mix_exceeds_inventory"
    )
    assert (
        reason(
            target=10,
            selected=4,
            bounds={
                "insufficient_candidate_inventory": 12,
                "category_cap_limits_inventory": 6,
                "insufficient_surface_diversity": 4,
            },
            mix_exceeds_inventory=True,
        )
        == "insufficient_surface_diversity"
    )
    assert (
        reason(
            target=10,
            selected=6,
            bounds={
                "insufficient_candidate_inventory": 12,
                "category_cap_limits_inventory": 6,
                "insufficient_surface_diversity": 12,
            },
            mix_exceeds_inventory=False,
        )
        == "category_cap_limits_inventory"
    )
