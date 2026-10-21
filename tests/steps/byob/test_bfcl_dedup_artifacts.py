"""Tests for Stage 11 balanced-task and report artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    DedupBalancingDecision,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    BALANCED_TASKS,
    balanced_tasks_schema,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
    DEDUP_BALANCING_REPORT,
    balance_publication_set,
    capability_signature,
    resolve_dedup_settings,
    write_dedup_balancing_artifacts,
)

BFCL_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "bfcl"
    / "config"
)


def _config(tmp_path: Path) -> BfclConfig:
    data = yaml.safe_load(
        (BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8")
    )
    data["output_dir"] = str(tmp_path / "output")
    data["random_seed"] = 31
    data["surface_quality_validation"] = {
        **(data.get("surface_quality_validation") or {}),
        "enabled": True,
    }
    data["task_generation"] = {
        **(data.get("task_generation") or {}),
        "tasks_per_category": 2,
    }
    data["semantic_deduplication_config"] = {
        "enabled": True,
        "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": 20,
        "eps": 0.08,
        "remove_duplicates": True,
    }
    path = tmp_path / "artifacts.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return BfclConfig.from_yaml(path)


def _task(task_id: str, *, edge: bool = False) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "template_id": "rare-template" if edge else "common-template",
        "intent": "lookup",
        "category": "general",
        "required_tools": ["lookup"],
        "tools_present": ["lookup"],
        "success_assertions": ["assert_lookup"],
        "difficulty": "easy",
        "turn_policy": "single_turn",
        "num_tool_calls": 1,
        "mutates": False,
        "call_order": "strict",
    }


def _surface(task_id: str, *, language: str = "en") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "language": language,
        "source": "template",
        "steps": [{"kind": "user", "content": "Look it up"}],
    }


def _inputs(
    tmp_path: Path,
) -> tuple[
    BfclConfig,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[DedupBalancingDecision],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, list[str]],
]:
    config = _config(tmp_path)
    tasks = [_task("task-a"), _task("task-b"), _task("rare", edge=True)]
    surfaces = {task["task_id"]: _surface(task["task_id"]) for task in tasks}
    edge_signatures = {"task-a": [], "task-b": [], "rare": ["rare_edge"]}
    initial = [
        DedupBalancingDecision(
            task_id=task["task_id"],
            selected=True,
            is_duplicate=False,
            selection_rank=index,
        )
        for index, task in enumerate(tasks)
    ]
    decisions, balancing_records, summary = balance_publication_set(
        config,
        tasks,
        surfaces,
        initial,
        edge_signatures_by_task_id=edge_signatures,
    )
    representative_metadata = [
        {
            "task_id": task["task_id"],
            "curator_cluster_id": task["task_id"],
            "curator_is_duplicate": False,
            "curator_predecessor_id": None,
            "curator_similarity_score": None,
            "duplicate_cluster_id": None,
            "representative_task_id": None,
            "capability_signature": capability_signature(task),
            "representative_rank": {
                "judge_problem": False,
                "seeded_tie_break": f"tie-{task['task_id']}",
            },
            "text_hash": f"hash-{task['task_id']}",
        }
        for task in tasks
    ]
    settings = resolve_dedup_settings(config)
    semantic_result = {
        "settings_hash": settings.settings_hash,
        "input_hash": "sha256:projected",
        "input_count": len(tasks),
        "embedding_signature": "sha256:embeddings",
        "effective_n_clusters": 1,
        "records": [
            {
                "task_id": task["task_id"],
                "cluster_id": task["task_id"],
                "is_duplicate": False,
                "text_hash": f"hash-{task['task_id']}",
            }
            for task in tasks
        ],
    }
    return (
        config,
        tasks,
        surfaces,
        decisions,
        representative_metadata,
        balancing_records,
        semantic_result,
        summary,
        edge_signatures,
    )


def test_stage_eleven_writes_complete_parquet_and_report(tmp_path: Path) -> None:
    (
        config,
        tasks,
        surfaces,
        decisions,
        representative_metadata,
        balancing_records,
        semantic_result,
        summary,
        edge_signatures,
    ) = _inputs(tmp_path)

    result = write_dedup_balancing_artifacts(
        config,
        tasks,
        surfaces,
        decisions,
        representative_metadata,
        balancing_records,
        semantic_result,
        summary,
        edge_signatures_by_task_id=edge_signatures,
    )

    assert result["artifact_path"].name == BALANCED_TASKS
    assert result["report_path"].name == DEDUP_BALANCING_REPORT
    assert result["artifact_hash"].startswith("sha256:")
    assert result["report_hash"].startswith("sha256:")
    table = pq.read_table(result["artifact_path"])
    assert table.schema == balanced_tasks_schema()
    rows = table.to_pylist()
    assert [row["task_id"] for row in rows] == [
        "task-a",
        "task-b",
        "rare",
    ]
    assert sum(row["selected"] for row in rows) == 2
    assert all(
        set(
            (
                "intent",
                "category",
                "required_tools",
                "tools_present",
                "difficulty",
                "turn_class",
                "tool_call_count",
                "turn_policy",
            )
        )
        <= set(row)
        for row in rows
    )
    report = json.loads(result["report_path"].read_text(encoding="utf-8"))
    assert report == result["report"]
    assert report["counts"] == {
        "curator_duplicates": 0,
        "dropped": 1,
        "final_duplicates": 0,
        "semantic_duplicate_annotations": 0,
        "semantic_duplicate_drops": 0,
        "selected": 2,
        "stage_ten_survivors": 3,
    }
    assert report["by_template"]["rare-template"]["selected"] == 1
    assert report["rare_edge_preservation"]["rare_edge"] == {
        "input": 1,
        "preserved": True,
        "selected": 1,
    }
    assert (
        report["artifacts"][BALANCED_TASKS]["content_hash"]
        == result["artifact_hash"]
    )
    assert report["lineage"]["projected_input_hash"] == "sha256:projected"
    assert report["lineage"]["embedding_signature"] == "sha256:embeddings"


def test_artifact_writes_are_reproducible_and_leave_no_temporary_files(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    first = write_dedup_balancing_artifacts(
        *inputs[:8],
        edge_signatures_by_task_id=inputs[8],
    )
    second = write_dedup_balancing_artifacts(
        *inputs[:8],
        edge_signatures_by_task_id=inputs[8],
    )

    assert first["artifact_hash"] == second["artifact_hash"]
    assert first["report_hash"] == second["report_hash"]
    assert not list(first["artifact_path"].parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_representative", "representative metadata must cover"),
        ("selection_drift", "balancing metadata disagrees"),
        ("dimension_drift", "must contain all eight dimensions"),
        ("settings_drift", "settings_hash does not match"),
    ],
)
def test_artifact_writer_rejects_inconsistent_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    (
        config,
        tasks,
        surfaces,
        decisions,
        representative_metadata,
        balancing_records,
        semantic_result,
        summary,
        edge_signatures,
    ) = _inputs(tmp_path)
    representative_metadata = copy.deepcopy(representative_metadata)
    balancing_records = copy.deepcopy(balancing_records)
    semantic_result = dict(semantic_result)
    if mutation == "missing_representative":
        representative_metadata.pop()
    elif mutation == "selection_drift":
        balancing_records[0]["selected"] = not balancing_records[0]["selected"]
    elif mutation == "dimension_drift":
        balancing_records[0]["dimensions"].pop("intent")
    else:
        semantic_result["settings_hash"] = "sha256:other"

    with pytest.raises(ValueError, match=message):
        write_dedup_balancing_artifacts(
            config,
            tasks,
            surfaces,
            decisions,
            representative_metadata,
            balancing_records,
            semantic_result,
            summary,
            edge_signatures_by_task_id=edge_signatures,
        )
