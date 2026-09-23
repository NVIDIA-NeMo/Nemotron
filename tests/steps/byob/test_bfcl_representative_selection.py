"""Tests for Stage 11 duplicate partitioning and representative selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
    capability_signature,
    derive_stage11_coverage,
    select_duplicate_representatives,
)

BFCL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "bfcl" / "config"


def _config(
    tmp_path: Path,
    *,
    remove_duplicates: bool = True,
    preference: list[str] | None = None,
    seed: int = 17,
    task_generation: dict[str, Any] | None = None,
) -> BfclConfig:
    data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    data["output_dir"] = str(tmp_path / "output")
    data["random_seed"] = seed
    data["surface_quality_validation"] = {
        **(data.get("surface_quality_validation") or {}),
        "enabled": True,
    }
    if task_generation is not None:
        data["task_generation"] = {
            **(data.get("task_generation") or {}),
            **task_generation,
        }
    data["semantic_deduplication_config"] = {
        "enabled": True,
        "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": 20,
        "eps": 0.08,
        "remove_duplicates": remove_duplicates,
    }
    if preference is not None:
        data["semantic_deduplication_config"]["representative_source_preference"] = preference
    path = tmp_path / f"representatives-{remove_duplicates}-{seed}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return BfclConfig.from_yaml(path)


def _task(task_id: str, **overrides: Any) -> dict[str, Any]:
    task = {
        "task_id": task_id,
        "template_id": "template",
        "turn_policy": "single_turn",
        "required_tools": ["lookup"],
        "tools_present": ["lookup"],
        "success_assertions": ["assert_lookup"],
        "mutates": False,
        "call_order": "strict",
    }
    task.update(overrides)
    return task


def _surface(
    task_id: str,
    *,
    language: str = "en",
    source: str = "template",
    turns: int = 1,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "language": language,
        "source": source,
        "steps": [{"kind": "user", "content": f"Look it up {index}"} for index in range(turns)],
    }


def _checks(
    *,
    advisory_failure: bool = False,
    judge_error: bool = False,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {"check": "surface_shape", "status": "passed", "source": "python"},
        {
            "check": "semantic_preservation",
            "status": "passed",
            "source": "python",
        },
        {"check": "leakage", "status": "passed", "source": "python"},
    ]
    judged = (
        [
            {
                "check": name,
                "status": "error",
                "source": "surface_judge",
                "reason_code": "judge_error",
            }
            for name in (
                "language_locale",
                "fluency_naturalness",
                "clarity_coherence",
            )
        ]
        if judge_error
        else [
            {
                "check": name,
                "status": ("failed" if advisory_failure and name == "fluency_naturalness" else "passed"),
                "source": "surface_judge",
                **({"reason_code": "unnatural_wording"} if advisory_failure and name == "fluency_naturalness" else {}),
            }
            for name in (
                "language_locale",
                "fluency_naturalness",
                "clarity_coherence",
            )
        ]
    )
    return [*checks, *judged]


def _quality(
    task_id: str,
    *,
    source: str = "template",
    turn_policy: str = "single_turn",
    advisory_failure: bool = False,
    judge_error: bool = False,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "turn_policy": turn_policy,
        "surface_source": source,
        "checks": _checks(
            advisory_failure=advisory_failure,
            judge_error=judge_error,
        ),
        "decision": "kept",
        "advisory_failures": (["fluency_naturalness:unnatural_wording"] if advisory_failure else []),
        "judge_error": "judge_error" if judge_error else None,
    }


def _semantic(task_ids: list[str], cluster: str = "curator-1") -> dict[str, Any]:
    return {
        "records": [
            {
                "task_id": task_id,
                "cluster_id": cluster,
                "is_duplicate": index > 0,
                "text_hash": f"hash-{task_id}",
            }
            for index, task_id in enumerate(task_ids)
        ]
    }


def _run(
    config: BfclConfig,
    tasks: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    quality: list[dict[str, Any]],
    semantic: dict[str, Any],
    **kwargs: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    return select_duplicate_representatives(
        config,
        tasks,
        surfaces,
        quality,
        semantic,
        **kwargs,
    )


def test_curator_clusters_are_partitioned_by_locked_coverage(
    tmp_path: Path,
) -> None:
    tasks = [
        _task("en-a"),
        _task("en-b"),
        _task("vi-a"),
        _task("vi-b"),
        _task("missing-a", turn_policy="missing_slot"),
        _task("missing-b", turn_policy="missing_slot"),
    ]
    surfaces = {
        "en-a": _surface("en-a"),
        "en-b": _surface("en-b", source="model"),
        "vi-a": _surface("vi-a", language="vi"),
        "vi-b": _surface("vi-b", language="vi", source="model"),
        "missing-a": _surface("missing-a"),
        "missing-b": _surface("missing-b", source="model"),
    }

    decisions, metadata = _run(
        _config(tmp_path),
        tasks,
        surfaces,
        [
            _quality(
                task["task_id"],
                source=surfaces[task["task_id"]]["source"],
                turn_policy=task["turn_policy"],
            )
            for task in tasks
        ],
        _semantic([task["task_id"] for task in tasks]),
    )

    by_id = {decision.task_id: decision for decision in decisions}
    assert by_id["en-a"].representative_task_id == "en-a"
    assert by_id["en-b"].representative_task_id == "en-a"
    assert by_id["vi-a"].representative_task_id == "vi-a"
    assert by_id["vi-b"].representative_task_id == "vi-a"
    assert by_id["missing-a"].representative_task_id == "missing-a"
    assert by_id["missing-b"].representative_task_id == "missing-a"
    assert by_id["en-a"].duplicate_cluster_id != by_id["vi-a"].duplicate_cluster_id
    assert by_id["en-a"].duplicate_cluster_id != by_id["missing-a"].duplicate_cluster_id
    assert {item["curator_cluster_id"] for item in metadata} == {"curator-1"}


def test_edge_and_capability_partitions_cannot_be_collapsed(
    tmp_path: Path,
) -> None:
    tasks = [
        _task("base-a"),
        _task("base-b"),
        _task("edge-a"),
        _task("edge-b"),
        _task("other-tool-a", required_tools=["other"]),
        _task("other-tool-b", required_tools=["other"]),
    ]
    ids = [task["task_id"] for task in tasks]
    surfaces = {task_id: _surface(task_id) for task_id in ids}
    edges = {task_id: (["rare_edge"] if task_id.startswith("edge") else []) for task_id in ids}

    decisions, metadata = _run(
        _config(tmp_path),
        tasks,
        surfaces,
        [_quality(task_id) for task_id in ids],
        _semantic(ids),
        edge_signatures_by_task_id=edges,
    )

    by_id = {decision.task_id: decision for decision in decisions}
    clusters = {
        "base": by_id["base-a"].duplicate_cluster_id,
        "edge": by_id["edge-a"].duplicate_cluster_id,
        "other": by_id["other-tool-a"].duplicate_cluster_id,
    }
    assert len(set(clusters.values())) == 3
    assert {
        by_id["edge-a"].representative_task_id,
        by_id["edge-b"].representative_task_id,
    } <= {"edge-a", "edge-b"}
    assert by_id["edge-a"].representative_task_id == by_id["edge-b"].representative_task_id
    signatures = {item["task_id"]: item["capability_signature"] for item in metadata}
    assert signatures["base-a"] == signatures["edge-a"]
    assert signatures["base-a"] != signatures["other-tool-a"]


def test_representative_ranking_prefers_clean_then_configured_source(
    tmp_path: Path,
) -> None:
    ids = ["error", "advisory", "model", "template"]
    tasks = [_task(task_id) for task_id in ids]
    surfaces = {
        "error": _surface("error"),
        "advisory": _surface("advisory"),
        "model": _surface("model", source="model"),
        "template": _surface("template"),
    }
    quality = [
        _quality("error", judge_error=True),
        _quality("advisory", advisory_failure=True),
        _quality("model", source="model"),
        _quality("template"),
    ]

    decisions, metadata = _run(
        _config(tmp_path),
        tasks,
        surfaces,
        quality,
        _semantic(ids),
    )

    assert {decision.representative_task_id for decision in decisions} == {"template"}
    assert sum(decision.selected for decision in decisions) == 1
    assert sum(decision.is_duplicate for decision in decisions) == len(ids) - 1
    assert {
        decision.drop_reason
        for decision in decisions
        if not decision.selected
    } == {"semantic_duplicate"}
    rank = {item["task_id"]: item["representative_rank"] for item in metadata}
    assert rank["error"]["judge_problem"] is True
    assert rank["advisory"]["applicable_failure_count"] == 1
    assert rank["template"]["source_preference_rank"] == 0

    reversed_decisions, _ = _run(
        _config(tmp_path, preference=["model", "template"]),
        tasks,
        surfaces,
        quality,
        _semantic(ids),
    )
    assert {decision.representative_task_id for decision in reversed_decisions} == {"model"}


def test_a_publishable_member_outranks_a_hard_limited_one(
    tmp_path: Path,
) -> None:
    ids = ["verbose", "concise"]
    tasks = [_task(task_id) for task_id in ids]
    surfaces = {
        "verbose": _surface("verbose", turns=3),
        "concise": _surface("concise"),
    }
    quality = [_quality(task_id) for task_id in ids]

    unlimited, _ = _run(
        _config(tmp_path),
        tasks,
        surfaces,
        quality,
        _semantic(ids),
    )
    assert {decision.representative_task_id for decision in unlimited} == {"verbose"}

    limited, metadata = _run(
        _config(tmp_path, task_generation={"max_turns": 2}),
        tasks,
        surfaces,
        quality,
        _semantic(ids),
    )

    assert {decision.representative_task_id for decision in limited} == {"concise"}
    rank = {item["task_id"]: item["representative_rank"] for item in metadata}
    assert rank["verbose"]["hard_limited"] is True
    assert rank["concise"]["hard_limited"] is False


def test_selection_is_deterministic_under_input_reordering(
    tmp_path: Path,
) -> None:
    ids = ["task-c", "task-a", "task-b"]
    tasks = [_task(task_id) for task_id in ids]
    surfaces = {task_id: _surface(task_id, source="model") for task_id in ids}
    quality = [_quality(task_id, source="model") for task_id in ids]

    decisions, _ = _run(
        _config(tmp_path, preference=["model"]),
        tasks,
        surfaces,
        quality,
        _semantic(ids),
    )
    reversed_ids = list(reversed(ids))
    reversed_decisions, _ = _run(
        _config(tmp_path, preference=["model"]),
        list(reversed(tasks)),
        surfaces,
        list(reversed(quality)),
        _semantic(reversed_ids),
    )

    assert {decision.representative_task_id for decision in decisions} == {
        decision.representative_task_id for decision in reversed_decisions
    }
    assert {decision.duplicate_cluster_id for decision in decisions} == {
        decision.duplicate_cluster_id for decision in reversed_decisions
    }


def test_remove_duplicates_false_keeps_annotations_and_all_rows(
    tmp_path: Path,
) -> None:
    ids = ["task-a", "task-b"]
    tasks = [_task(task_id) for task_id in ids]
    surfaces = {task_id: _surface(task_id) for task_id in ids}

    decisions, _ = _run(
        _config(tmp_path, remove_duplicates=False),
        tasks,
        surfaces,
        [_quality(task_id) for task_id in ids],
        _semantic(ids),
    )

    assert [decision.selected for decision in decisions] == [True, True]
    assert sorted(decision.selection_rank for decision in decisions) == [0, 1]
    assert sum(decision.is_duplicate for decision in decisions) == 1
    assert all(decision.drop_reason is None for decision in decisions)


def test_a_partition_split_to_one_row_is_not_labelled_duplicate(
    tmp_path: Path,
) -> None:
    tasks = [_task("en"), _task("vi")]
    surfaces = {
        "en": _surface("en"),
        "vi": _surface("vi", language="vi"),
    }

    decisions, _ = _run(
        _config(tmp_path),
        tasks,
        surfaces,
        [_quality("en"), _quality("vi")],
        _semantic(["en", "vi"]),
    )

    assert all(not decision.is_duplicate for decision in decisions)
    assert all(decision.duplicate_cluster_id is None for decision in decisions)
    assert all(decision.representative_task_id is None for decision in decisions)


def test_coverage_derivation_is_generic_and_exact() -> None:
    tasks = [_task("task-a")]
    surfaces = {"task-a": _surface("task-a", language=" vi ")}

    coverage = derive_stage11_coverage(
        tasks,
        surfaces,
        edge_signatures_by_task_id={"task-a": [" edge-b ", "edge-a"]},
    )

    assert coverage["task-a"].language == "vi"
    assert coverage["task-a"].edge_signatures == ("edge-a", "edge-b")
    with pytest.raises(ValueError, match="cover inputs exactly"):
        derive_stage11_coverage(
            tasks,
            surfaces,
            edge_signatures_by_task_id={},
        )


def test_capability_signature_is_order_independent_and_sensitive() -> None:
    first = _task(
        "a",
        required_tools=["b", "a"],
        tools_present=["b", "a"],
        success_assertions=["second", "first"],
    )
    reordered = _task(
        "b",
        required_tools=["a", "b"],
        tools_present=["a", "b"],
        success_assertions=["first", "second"],
    )
    changed = _task(
        "c",
        required_tools=["a"],
        tools_present=["a", "b"],
        success_assertions=["first", "second"],
    )

    assert capability_signature(first) == capability_signature(reordered)
    assert capability_signature(first) != capability_signature(changed)


def test_selection_rejects_incomplete_or_pre_stage_ten_inputs(
    tmp_path: Path,
) -> None:
    tasks = [_task("task-a"), _task("task-b")]
    surfaces = {
        "task-a": _surface("task-a"),
        "task-b": _surface("task-b"),
    }
    quality = [_quality("task-a"), _quality("task-b")]

    with pytest.raises(ValueError, match="must cover Stage 11 inputs exactly"):
        _run(
            _config(tmp_path),
            tasks,
            surfaces,
            quality[:1],
            _semantic(["task-a", "task-b"]),
        )

    quality[1]["decision"] = "dropped"
    with pytest.raises(ValueError, match="not a Stage 10 survivor"):
        _run(
            _config(tmp_path),
            tasks,
            surfaces,
            quality,
            _semantic(["task-a", "task-b"]),
        )
