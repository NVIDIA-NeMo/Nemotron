"""Tests for the BFCL Stage 11 semantic-dedup adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
    effective_n_clusters,
    reconcile_curator_pairwise_artifacts,
    resolve_dedup_settings,
    run_semantic_dedup,
)

BFCL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "bfcl" / "config"


def _config(tmp_path: Path, **overrides: Any) -> BfclConfig:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["surface_quality_validation"] = {
        **(config_data.get("surface_quality_validation") or {}),
        "enabled": True,
    }
    config_data["semantic_deduplication_config"] = {
        "enabled": True,
        "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": 20,
        "eps": 0.08,
        "remove_duplicates": True,
        **overrides,
    }
    path = tmp_path / "dedup.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return BfclConfig.from_yaml(path)


def _projected(*task_ids: str) -> list[dict[str, Any]]:
    projected = []
    for task_id in task_ids:
        text = f"[user] please handle {task_id}"
        projected.append(
            {
                "task_id": task_id,
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return projected


def _finder(duplicate_ids: list[str], cluster_by_id: dict[str, str]) -> Any:
    def finder(**kwargs: Any) -> dict[str, Any]:
        finder.calls.append(kwargs)
        return {
            "duplicate_ids": duplicate_ids,
            "cluster_by_id": cluster_by_id,
            "embedding_signature": "sha256:fake",
        }

    finder.calls = []
    return finder


def test_settings_come_from_the_locked_config(tmp_path: Path) -> None:
    settings = resolve_dedup_settings(_config(tmp_path))

    assert settings.model_identifier == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.n_clusters == 20
    assert settings.eps == 0.08
    assert settings.remove_duplicates is True
    assert settings.as_lineage()["contract_version"] == "1.0"
    assert settings.settings_hash == resolve_dedup_settings(_config(tmp_path)).settings_hash
    assert settings.settings_hash != resolve_dedup_settings(_config(tmp_path, eps=0.2)).settings_hash


def test_k_is_clamped_to_the_rows_on_hand() -> None:
    assert effective_n_clusters(20, 100) == 20
    assert effective_n_clusters(20, 40) == 20
    assert effective_n_clusters(20, 3) == 1
    assert effective_n_clusters(20, 2) == 1
    assert effective_n_clusters(20, 1) == 1
    assert effective_n_clusters(20, 0) == 0

    with pytest.raises(ValueError, match="n_clusters must be positive"):
        effective_n_clusters(0, 10)
    with pytest.raises(ValueError, match="row_count cannot be negative"):
        effective_n_clusters(20, -1)


def test_adapter_maps_task_ids_and_projected_text_to_the_backend(tmp_path: Path) -> None:
    finder = _finder(["task-b"], {"task-a": "task-a", "task-b": "task-a", "task-c": "task-c"})

    result = run_semantic_dedup(_config(tmp_path), _projected("task-a", "task-b", "task-c"), finder=finder)

    assert finder.calls[0]["rows"] == [
        {"id": "task-a", "text": "[user] please handle task-a"},
        {"id": "task-b", "text": "[user] please handle task-b"},
        {"id": "task-c", "text": "[user] please handle task-c"},
    ]
    assert finder.calls[0]["n_clusters"] == 1
    assert result["records"] == [
        {
            "task_id": "task-a",
            "cluster_id": "task-a",
            "is_duplicate": False,
            "text_hash": _projected("task-a")[0]["text_hash"],
        },
        {
            "task_id": "task-b",
            "cluster_id": "task-a",
            "is_duplicate": True,
            "text_hash": _projected("task-b")[0]["text_hash"],
        },
        {
            "task_id": "task-c",
            "cluster_id": "task-c",
            "is_duplicate": False,
            "text_hash": _projected("task-c")[0]["text_hash"],
        },
    ]
    assert result["embedded"] is True
    assert result["embedding_signature"] == "sha256:fake"


def test_lineage_pins_the_model_settings_and_projected_input(tmp_path: Path) -> None:
    config = _config(tmp_path)
    finder = _finder([], {"task-a": "task-a", "task-b": "task-b"})

    result = run_semantic_dedup(config, _projected("task-a", "task-b"), finder=finder)
    same_input = run_semantic_dedup(config, _projected("task-a", "task-b"), finder=finder)
    other_input = run_semantic_dedup(
        config, _projected("task-a", "task-c"), finder=_finder([], {"task-a": "a", "task-c": "c"})
    )

    assert result["settings"]["model_identifier"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert result["settings"]["eps"] == 0.08
    assert result["input_count"] == 2
    assert result["input_hash"] == same_input["input_hash"]
    assert result["input_hash"] != other_input["input_hash"]


def test_adapter_preserves_curator_pairwise_metadata(tmp_path: Path) -> None:
    def finder(**_: Any) -> dict[str, Any]:
        return {
            "duplicate_ids": ["task-b"],
            "cluster_by_id": {"task-a": "task-a", "task-b": "task-a"},
            "pairwise_by_id": {
                "task-a": {
                    "predecessor_id": "task-a",
                    "similarity_score": 0.0,
                },
                "task-b": {
                    "predecessor_id": "task-a",
                    "similarity_score": 0.97,
                },
            },
            "embedding_signature": "signature",
        }

    result = run_semantic_dedup(
        _config(tmp_path),
        _projected("task-a", "task-b"),
        finder=finder,
    )

    by_id = {record["task_id"]: record for record in result["records"]}
    assert by_id["task-b"]["curator_predecessor_id"] == "task-a"
    assert by_id["task-b"]["curator_similarity_score"] == 0.97


@pytest.mark.parametrize("task_ids", [(), ("task-a",)])
def test_empty_and_single_row_inputs_never_embed(tmp_path: Path, task_ids: tuple[str, ...]) -> None:
    def explode(**_: Any) -> dict[str, Any]:
        raise AssertionError("a set this small must not reach the embedding backend")

    result = run_semantic_dedup(_config(tmp_path), _projected(*task_ids), finder=explode)

    assert result["embedded"] is False
    assert result["embedding_signature"] is None
    assert result["duplicate_ids"] == []
    assert result["clusters"] == {task_id: task_id for task_id in task_ids}
    assert [record["is_duplicate"] for record in result["records"]] == [False] * len(task_ids)


def test_adapter_rejects_a_repeated_task(tmp_path: Path) -> None:
    projected = _projected("task-a")

    with pytest.raises(ValueError, match="cannot embed the same task twice"):
        run_semantic_dedup(_config(tmp_path), [*projected, *projected], finder=_finder([], {}))


def test_adapter_rejects_a_stale_projected_text_hash(tmp_path: Path) -> None:
    projected = _projected("task-a")
    projected[0]["text"] = "[user] changed after hashing"

    with pytest.raises(ValueError, match="text_hash that does not match"):
        run_semantic_dedup(_config(tmp_path), projected, finder=_finder([], {}))


def test_adapter_requires_stage_eleven_to_be_enabled(tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=False)

    with pytest.raises(ValueError, match="requires semantic_deduplication_config.enabled"):
        run_semantic_dedup(config, _projected("task-a", "task-b"), finder=_finder([], {}))


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"duplicate_ids": [], "cluster_by_id": {}}, "missing keys: embedding_signature"),
        (
            {"duplicate_ids": [], "cluster_by_id": {"task-a": "a", "task-b": "b"}, "embedding_signature": " "},
            "non-empty embedding_signature",
        ),
        (
            {"duplicate_ids": [], "cluster_by_id": {"task-a": "a"}, "embedding_signature": "s"},
            "clusters must cover the embedded ids exactly",
        ),
        (
            {"duplicate_ids": [], "cluster_by_id": {"task-a": "a", "task-b": " "}, "embedding_signature": "s"},
            "empty Stage 11 cluster label",
        ),
        (
            {
                "duplicate_ids": ["task-c"],
                "cluster_by_id": {"task-a": "a", "task-b": "b"},
                "embedding_signature": "s",
            },
            "reported unknown ids: task-c",
        ),
        (
            {
                "duplicate_ids": ["task-b", "task-b"],
                "cluster_by_id": {"task-a": "a", "task-b": "b"},
                "embedding_signature": "s",
            },
            "repeated a duplicate id",
        ),
        (
            {
                "duplicate_ids": ["task-b"],
                "cluster_by_id": {"task-a": "a", "task-b": "b"},
                "embedding_signature": "s",
            },
            "sits alone in its cluster",
        ),
        (
            {
                "duplicate_ids": ["task-a", "task-b"],
                "cluster_by_id": {"task-a": "a", "task-b": "a"},
                "embedding_signature": "s",
            },
            "exactly one non-duplicate representative, got 0",
        ),
        (
            {
                "duplicate_ids": [],
                "cluster_by_id": {"task-a": "a", "task-b": "a"},
                "embedding_signature": "s",
            },
            "exactly one non-duplicate representative, got 2",
        ),
        (
            {
                "duplicate_ids": [],
                "cluster_by_id": {"task-a": "a", "task-b": "b"},
                "pairwise_by_id": {"task-a": {}},
                "embedding_signature": "s",
            },
            "pairwise_by_id must cover embedded ids exactly",
        ),
    ],
)
def test_adapter_rejects_a_backend_result_it_cannot_trust(
    tmp_path: Path,
    result: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_semantic_dedup(
            _config(tmp_path),
            _projected("task-a", "task-b"),
            finder=lambda **_: result,
        )


def test_curator_pairs_are_the_single_source_for_duplicate_clusters() -> None:
    clusters = reconcile_curator_pairwise_artifacts(
        task_ids={"task-a", "task-b", "task-c", "far"},
        pairs=[
            ("task-a", "task-a", 0.0),
            ("task-b", "task-a", 0.95),
            ("task-c", "task-b", 0.94),
            ("far", "far", 0.0),
        ],
        duplicate_ids=["task-b", "task-c"],
        eps=0.08,
    )

    assert clusters == {
        "far": "far",
        "task-a": "task-a",
        "task-b": "task-a",
        "task-c": "task-a",
    }


@pytest.mark.parametrize(
    ("pairs", "duplicate_ids", "message"),
    [
        (
            [("a", "a", 0.0)],
            [],
            "cover embedded ids exactly",
        ),
        (
            [("a", "a", 0.0), ("a", "a", 0.0)],
            [],
            "repeated ids",
        ),
        (
            [("a", "a", 0.0), ("b", "outside", 0.99)],
            ["b"],
            "unknown predecessor",
        ),
        (
            [("a", "a", 0.99), ("b", "b", 0.0)],
            ["a"],
            "duplicate of itself",
        ),
        (
            [("a", "a", 0.0), ("b", "a", 0.99)],
            [],
            "artifacts disagree",
        ),
        (
            [("a", "a", 0.0), ("b", "a", float("nan"))],
            [],
            "non-finite score",
        ),
        (
            [("a", "a", 0.0), ("b", "a", 0.99)],
            ["b"],
            "eps must be between 0 and 1",
        ),
    ],
)
def test_curator_pair_artifacts_are_rejected_when_inconsistent(
    pairs: list[tuple[str, str, float]],
    duplicate_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reconcile_curator_pairwise_artifacts(
            task_ids={"a", "b"},
            pairs=pairs,
            duplicate_ids=duplicate_ids,
            eps=1.0 if "eps" in message else 0.08,
        )
