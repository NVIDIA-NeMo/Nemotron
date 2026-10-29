"""End-to-end assets produced by the BFCL generation slice on the tiny pack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.bfcl_json_export import (
    BFCL_JSON_ANSWER_FILE,
    BFCL_JSON_QUESTION_FILE,
    BfclJsonArtifact,
    read_bfcl_json,
    write_bfcl_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    EVAL_CONFIG_SCHEMA_VERSION,
    evaluate_contamination,
    load_eval_config,
    verify_eval_source,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    BfclJsonRecord,
    NemoEvaluatorRecord,
    export_tree_hash,
    validate_export_equivalence,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    ExportProjectionError,
    ProjectionSource,
    project_benchmark_rows,
    project_published_benchmark,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_BUNDLE_FILE,
    NEMO_BUNDLE_FILES,
    NEMO_DATASET_FILE,
    NEMO_EVALUATOR_ROOT,
    NEMO_METADATA_FILE,
    NemoEvaluatorArtifact,
    read_nemo_evaluator_bundle,
    write_nemo_evaluator_bundle,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    decode_arguments,
    decode_tools,
)

BYOB_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
BFCL_CONFIG_DIR = BYOB_ROOT / "bfcl" / "config"
BANKING_PACK_ROOT = BYOB_ROOT / "data" / "banking_vn_oracle_pack"


def _run_tiny(tmp_path: Path) -> tuple[list[dict[str, Any]], Path]:
    import pyarrow.parquet as pq

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    assert benchmark_path is not None
    return pq.read_table(benchmark_path).to_pylist(), benchmark_path.parent


def _write_tiny_quality_config(tmp_path: Path, *, enabled: bool, name: str) -> Path:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["surface_quality_validation"]["enabled"] = enabled
    config_path = tmp_path / name
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path


def _write_tiny_dedup_config(
    tmp_path: Path,
    *,
    enabled: bool,
    name: str,
    unmet_target_policy: str = "abort",
    impossible_mix: bool = False,
) -> Path:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["surface_quality_validation"]["enabled"] = True
    config_data["semantic_deduplication_config"] = {
        "enabled": enabled,
        "model_identifier": "test/semantic-embedding",
        "n_clusters": 4,
        "eps": 0.08,
        "remove_duplicates": True,
        "unmet_target_policy": unmet_target_policy,
    }
    if impossible_mix:
        # Medium exists in the template inventory, but coverage locking keeps
        # required easy-policy rows, so a 100% medium publication is infeasible.
        config_data["task_generation"]["difficulty_mix"] = {"medium": 1.0}
    config_path = tmp_path / name
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path


def _semantic_singletons(config, projected, **_kwargs):  # type: ignore[no-untyped-def]
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    settings = dedup_balancing.resolve_dedup_settings(config)
    return {
        "settings": settings.as_lineage(),
        "settings_hash": settings.settings_hash,
        "input_hash": "sha256:test-projected-input",
        "input_count": len(projected),
        "effective_n_clusters": dedup_balancing.effective_n_clusters(
            settings.n_clusters,
            len(projected),
        ),
        "embedded": True,
        "embedding_signature": "sha256:test-embeddings",
        "duplicate_ids": [],
        "clusters": {str(record["task_id"]): f"curator-{record['task_id']}" for record in projected},
        "records": [
            {
                "task_id": str(record["task_id"]),
                "cluster_id": f"curator-{record['task_id']}",
                "is_duplicate": False,
                "text_hash": str(record["text_hash"]),
            }
            for record in projected
        ],
    }


def _run_tiny_with_surface_quality(
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], Path]:
    import pyarrow.parquet as pq

    benchmark_path = generate_bfcl(_write_tiny_quality_config(tmp_path, enabled=True, name="tiny-quality.yaml"))
    return pq.read_table(benchmark_path).to_pylist(), benchmark_path.parent


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict[str, Any]], Path]:
    return _run_tiny(tmp_path_factory.mktemp("tiny_slice"))


@pytest.fixture(scope="module")
def banking_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    tmp_path = tmp_path_factory.mktemp("banking_slice")
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "smoke.example.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "banking.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    benchmark_path = generate_bfcl(config_path)
    assert benchmark_path is not None
    return pq.read_table(benchmark_path).to_pylist()


@pytest.fixture(scope="module")
def third_pack_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[dict[str, Any]], Path]:
    """Build a non-bundled domain pack and run it through Stages 10 and 11."""
    import pyarrow.parquet as pq

    tmp_path = tmp_path_factory.mktemp("third_pack_slice")
    pack = tmp_path / "asset_oracle"
    pack.mkdir()
    (pack / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "warehouse_assets",
                "version": "1.0",
                "languages": ["en"],
                "paths": {
                    "tools": "tools.json",
                    "backend": "backend.py",
                    "fixtures": "fixtures.json",
                    "templates": "task_templates.yaml",
                    "assertions": "assertions.py",
                    "validation_cases": "validation_cases.yaml",
                },
                "assistant_turn_templates": {"final_answer": {"en": "Inspection complete."}},
                "absent_ids": {"assets": ["ASSET-MISSING"]},
            }
        ),
        encoding="utf-8",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_asset",
                "description": "Read one warehouse asset.",
                "parameters": {
                    "type": "object",
                    "properties": {"asset_id": {"type": "string"}},
                    "required": ["asset_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    (pack / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (pack / "fixtures.json").write_text(
        json.dumps({"assets": [{"asset_id": "ASSET-7", "condition": "ready"}]}),
        encoding="utf-8",
    )
    templates = [
        {
            "template_id": "asset_inspection",
            "intent": "inspect_asset_condition",
            "category": "warehouse",
            "difficulty": "easy",
            "turn_policy": "single_turn",
            "edge_signatures": ["inventory_read"],
            "required_tools": ["inspect_asset"],
            "tools_present": ["inspect_asset"],
            "slots": {
                "asset_id": {
                    "source": "fixture:assets.asset_id",
                    "visible_in_first_turn": True,
                }
            },
            "success_assertions": ["assert_asset_returned"],
            "user_turn_templates": {"en": "What is the condition of asset ASSET-7?"},
            "assistant_milestones": [
                {"type": "tool_call", "tool": "inspect_asset"},
                {"type": "final_answer"},
            ],
        }
    ]
    (pack / "task_templates.yaml").write_text(
        yaml.safe_dump(templates, sort_keys=False),
        encoding="utf-8",
    )
    cases = [
        {
            "id": "asset_success",
            "tool": "inspect_asset",
            "arguments": {"asset_id": "ASSET-7"},
            "expect": {"result_class": "success", "condition": "ready"},
        },
        {
            "id": "asset_missing",
            "tool": "inspect_asset",
            "arguments": {"asset_id": "ASSET-MISSING"},
            "expect": {"result_class": "structured_error", "error_code": "not_found"},
        },
    ]
    (pack / "validation_cases.yaml").write_text(
        yaml.safe_dump(cases, sort_keys=False),
        encoding="utf-8",
    )
    (pack / "backend.py").write_text(
        """
from copy import deepcopy

_state = {}

def list_tools():
    return ["inspect_asset"]

def reset(*, ctx, fixtures=None):
    global _state
    _state = deepcopy(fixtures or {})

def call_tool(name, arguments, *, ctx):
    if name != "inspect_asset":
        return {"error": {"code": "unknown_tool"}}
    asset_id = arguments.get("asset_id")
    for asset in _state.get("assets", []):
        if asset.get("asset_id") == asset_id:
            return dict(asset)
    return {"error": {"code": "not_found"}}

def get_state():
    return deepcopy(_state)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (pack / "assertions.py").write_text(
        """
def assert_asset_returned(*, state, trace, task, ctx):
    assert len(trace) == 1
    assert trace[0]["result"]["asset_id"] == task["slots"]["asset_id"]

ASSERTIONS = {"assert_asset_returned": assert_asset_returned}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["expt_name"] = "third_pack_validation"
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["oracle_pack"] = {"manifest_path": str(pack / "manifest.yaml")}
    config_data["oracle_runtime"]["allowed_roots"] = [str(pack)]
    config_data["task_generation"]["tasks_per_category"] = 1
    config_data["surface_quality_validation"]["enabled"] = True
    config_data["semantic_deduplication_config"] = {
        "enabled": True,
        "model_identifier": "test/single-row-embedding",
        "n_clusters": 4,
        "eps": 0.08,
        "remove_duplicates": True,
    }
    config_path = tmp_path / "third-pack.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    assert benchmark_path is not None
    return pq.read_table(benchmark_path).to_pylist(), benchmark_path.parent


def _row(rows: list[dict[str, Any]], template_id: str) -> dict[str, Any]:
    matches = [row for row in rows if row["template_id"] == template_id]
    assert matches, f"no row for template {template_id}"
    return matches[0]


def test_a_real_published_run_verifies_and_gates_a_candidate(
    tiny_run: tuple[list[dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    """The eval contracts are checked against the manifest the pipeline writes.

    Every other eval test builds its own publication tree, which can only prove
    the verifier is self-consistent. This one reads what Stage 12 actually wrote,
    so a manifest field the eval side expects and the pipeline stopped writing
    fails here rather than in production.
    """
    rows, output_dir = tiny_run
    eval_config = tmp_path / "eval_config.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": EVAL_CONFIG_SCHEMA_VERSION,
                "config_status": "resolved",
                "source_run_manifest": str(output_dir / "run_manifest.json"),
                "source_oracle": None,
                "translation_manifest": None,
                "eval": {"mode": ["trace"]},
                "scoring": {
                    "contract": str(BYOB_ROOT / "references" / "bfcl-eval-scoring-contract.md"),
                    "argument_matching": "schema_then_canonical",
                    "insert_declared_defaults": True,
                    "respect_call_order": True,
                    "respect_call_group": True,
                    "allow_llm_repair": False,
                    "task_success": "all_applicable_gates",
                },
                "limits": {
                    "max_turns": 6,
                    "tool_timeout_s": 30.0,
                    "candidate_timeout_s": 60.0,
                    "episode_timeout_s": 120.0,
                    "max_parallel_tasks": 1,
                    "max_retries": 2,
                },
                "candidates": [
                    {
                        "alias": "candidate_a",
                        "model": "candidate-route",
                        "provider": "nvidia",
                        "provider_api_version": "v1",
                        "api": {
                            "base_url": "https://integrate.example.com/v1",
                            "api_key_env": "NVIDIA_API_KEY",
                        },
                        "model_identity": {
                            "source": "huggingface",
                            "model": "org/candidate",
                            "revision": "a" * 40,
                            "weights_digest": None,
                        },
                        "inference": {
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "max_tokens": 1024,
                            "seed": 42,
                            "tool_choice": "auto",
                        },
                    }
                ],
                "contamination": {
                    "enforce": True,
                    "on_violation": "fail_run",
                    "comparison_set": "common_intersection",
                },
                "publication": {"requested": True, "require_same_task_ids": True},
                "outputs": {
                    "output_dir": str(tmp_path / "eval_out"),
                    "write_task_results": True,
                    "write_eval_manifest": True,
                    "cache_candidate_responses": True,
                    "cache_tool_results": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_eval_config(eval_config)
    source = verify_eval_source(config)
    plan = evaluate_contamination(config, source)

    assert source.task_ids == tuple(row["task_id"] for row in rows)
    # The tiny pack calls no model, so nothing read the rows it published.
    assert source.exposures == ()
    assert plan.common.task_ids == source.task_ids
    assert plan.evaluation_task_ids("candidate_a") == source.task_ids
    assert plan.publication_allowed is True


def test_non_bundled_oracle_pack_runs_end_to_end(third_pack_run) -> None:
    rows, _ = third_pack_run
    assert len(rows) == 1
    row = rows[0]
    assert row["pack_id"] == "warehouse_assets"
    assert row["template_id"] == "asset_inspection"
    assert row["expected_tool_calls"][0]["function_name"] == "inspect_asset"
    assert decode_arguments(row["expected_tool_calls"][0]["arguments"]) == {"asset_id": "ASSET-7"}


def test_surface_quality_is_generic_to_a_non_banking_oracle_pack(
    third_pack_run,
) -> None:
    import pyarrow.parquet as pq

    rows, output_dir = third_pack_run
    cache = output_dir / "stage_cache"
    quality_rows = pq.read_table(cache / "surface_validated_tasks.parquet").to_pylist()
    report = json.loads((cache / "surface_quality_rejections.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert [row["task_id"] for row in quality_rows] == [row["task_id"] for row in rows]
    assert quality_rows[0]["template_id"] == "asset_inspection"
    assert quality_rows[0]["accepted"] is True
    assert quality_rows[0]["surface_source"] == "template"
    assert {
        quality_rows[0]["surface_shape_status"],
        quality_rows[0]["semantic_preservation_status"],
        quality_rows[0]["leakage_status"],
    } == {"passed"}
    assert {
        quality_rows[0]["language_locale_status"],
        quality_rows[0]["fluency_naturalness_status"],
        quality_rows[0]["clarity_coherence_status"],
    } == {"not_run"}
    assert report["by_template"] == {
        "asset_inspection": {
            "evaluated": 1,
            "kept": 1,
            "dropped": 0,
            "drop_reason_counts": {},
            "advisory_failure_counts": {},
            "judge_error_counts": {},
        }
    }
    assert manifest["pack"]["pack_id"] == "warehouse_assets"
    assert manifest["surface_quality_validation"]["enabled"] is True
    assert manifest["stage_counts"]["surface_quality_evaluated"] == 1


def test_stage_eleven_is_generic_to_a_non_banking_oracle_pack(
    third_pack_run,
) -> None:
    import pyarrow.parquet as pq

    rows, output_dir = third_pack_run
    cache = output_dir / "stage_cache"
    balanced = pq.read_table(cache / "balanced_tasks.parquet").to_pylist()
    report = json.loads((cache / "dedup_balancing_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert [row["task_id"] for row in balanced] == [row["task_id"] for row in rows]
    assert balanced[0]["selected"] is True
    assert balanced[0]["category"] == "warehouse"
    assert balanced[0]["required_tools"] == '["inspect_asset"]'
    assert balanced[0]["edge_signatures"] == ["inventory_read"]
    assert report["counts"]["stage_ten_survivors"] == 1
    assert report["counts"]["selected"] == 1
    assert report["rare_edge_preservation"]["inventory_read"]["preserved"] is True
    assert report["lineage"]["embedding_signature"] is None
    assert manifest["pack"]["pack_id"] == "warehouse_assets"
    assert manifest["semantic_deduplication"]["enabled"] is True


def test_tiny_slice_produces_replay_validated_non_publishable_rows(tiny_run) -> None:
    rows, _ = tiny_run
    assert rows
    for row in rows:
        assert row["tier"] == "gold"
        assert row["gold_eligible"] is False
        assert "schema" in row["validated_by"]
        assert "replay" in row["validated_by"]
        # Null, not False: no held-out set was configured, so the row cannot claim it
        # was checked against one.
        assert row["held_out_hit"] is None
        assert row["task_id"].startswith("tiny_library__")
        assert row["messages"][0]["role"] == "system"
        assert row["system_prompt_id"].startswith("sha256:")


def test_every_template_reaches_the_benchmark(tiny_run) -> None:
    rows, _ = tiny_run
    assert {row["template_id"] for row in rows} == {
        "lib_status_single",
        "lib_checkout_confirm",
        "lib_status_parallel",
        "lib_irrelevant_renew",
    }


def test_parallel_group_is_one_assistant_message_with_two_calls(tiny_run) -> None:
    rows, _ = tiny_run
    row = _row(rows, "lib_status_parallel")

    assistant_with_calls = [
        message for message in row["messages"] if message["role"] == "assistant" and message["tool_calls"]
    ]
    assert len(assistant_with_calls) == 1
    assert len(assistant_with_calls[0]["tool_calls"]) == 2

    calls = row["expected_tool_calls"]
    assert [call["call_group"] for call in calls] == [0, 0]
    assert [call["position_in_group"] for call in calls] == [0, 1]
    assert {call["turn_index"] for call in calls} == {0}

    tool_messages = [message for message in row["messages"] if message["role"] == "tool"]
    issued = [call["id"] for call in assistant_with_calls[0]["tool_calls"]]
    assert [message["tool_call_id"] for message in tool_messages] == issued


def test_irrelevant_template_declines_without_calling_a_tool(tiny_run) -> None:
    rows, _ = tiny_run
    row = _row(rows, "lib_irrelevant_renew")

    assert row["expected_tool_calls"] == []
    assert row["num_tool_calls"] == 0
    assert row["required_tools"] == []
    assert all(not message["tool_calls"] for message in row["messages"])
    assert row["messages"][-1]["role"] == "assistant"
    assert row["messages"][-1]["content"]


def test_confirmation_trace_confirms_once_after_a_user_reply(tiny_run) -> None:
    rows, _ = tiny_run
    row = _row(rows, "lib_checkout_confirm")

    calls = row["expected_tool_calls"]
    assert len(calls) == 1
    arguments = decode_arguments(calls[0]["arguments"])
    assert arguments["confirm"] is True
    assert row["is_multi_turn"] is True

    roles = [message["role"] for message in row["messages"]]
    assert roles.count("user") == 2
    # The confirming call is issued by the second assistant message.
    assert calls[0]["turn_index"] == 1


def test_arguments_and_tools_round_trip_without_null_padding(tiny_run) -> None:
    rows, _ = tiny_run
    status_row = _row(rows, "lib_status_single")

    arguments = decode_arguments(status_row["expected_tool_calls"][0]["arguments"])
    assert arguments == {"book_id": "BK-100"}

    tools = decode_tools(status_row["tools"])
    assert {tool["function"]["name"] for tool in tools} == set(status_row["tools_present"])
    status_tool = next(tool for tool in tools if tool["function"]["name"] == "get_book_status")
    assert set(status_tool["function"]["parameters"]["properties"]) == {"book_id"}


def test_banking_rows_expose_only_declared_tools_and_confirm_mutations(banking_run) -> None:
    assert banking_run
    for row in banking_run:
        names = {tool["function"]["name"] for tool in decode_tools(row["tools"])}
        assert names == set(row["tools_present"])
        assert row["gold_eligible"] is False

    for template_id in ("bn_create_transfer_single", "bn_create_dispute_single"):
        row = _row(banking_run, template_id)
        assert row["is_multi_turn"] is True
        assert [message["role"] for message in row["messages"]].count("user") == 2
        assert row["expected_tool_calls"][0]["turn_index"] == 1


def test_banking_categories_share_one_budget(banking_run) -> None:
    config = yaml.safe_load((BFCL_CONFIG_DIR / "smoke.example.yaml").read_text(encoding="utf-8"))
    budget = config["task_generation"]["tasks_per_category"]
    templates = yaml.safe_load((BANKING_PACK_ROOT / "task_templates.yaml").read_text(encoding="utf-8"))

    counts: dict[str, int] = {}
    for row in banking_run:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts
    assert max(counts.values()) <= budget
    # Splitting a budget across templates must not silently drop a template.
    assert {row["template_id"] for row in banking_run} == {template["template_id"] for template in templates}


def test_banking_covers_every_supported_policy_edge(banking_run) -> None:
    policies = {row["turn_policy"] for row in banking_run}
    assert policies == {
        "single_turn",
        "missing_slot",
        "confirmation",
        "correction",
        "multi_tool",
        "dependent_call",
        "negative_path",
        "clarify_only",
        "irrelevant",
    }
    assert any(row["num_tool_calls"] >= 2 for row in banking_run)
    assert all(set(row["required_tools"]) <= set(row["tools_present"]) for row in banking_run)
    assert any(set(row["required_tools"]) < set(row["tools_present"]) for row in banking_run)


def _assert_exports_preserve_the_published_benchmark(rows: list[dict[str, Any]]) -> None:
    # Decoded through the projection, not row by row: the writers below are the
    # first consumers of the single decode path, so a projection that cannot
    # describe a real published row fails here rather than in a later exporter.
    projection = project_benchmark_rows(
        rows,
        source=ProjectionSource(
            file="benchmark.parquet",
            content_hash="sha256:" + "0" * 64,
            rows=len(rows),
        ),
    )
    canonical = list(projection.rows)
    bfcl = [BfclJsonRecord.from_canonical(row) for row in canonical]
    assert projection.task_ids == tuple(row["task_id"] for row in rows)
    for row, plan in zip(canonical, projection.plans, strict=True):
        assert plan.call_count == len(row.expected_tool_calls)
        assert plan.is_multi_turn == row.is_multi_turn
        assert sum(len(group.calls) for group in plan.groups) == row.num_tool_calls
    assert validate_export_equivalence(canonical, bfcl) == []
    assert validate_export_equivalence(canonical, [NemoEvaluatorRecord.from_canonical(row) for row in canonical]) == []
    for source, record, plan in zip(canonical, bfcl, projection.plans, strict=True):
        question = record.question_record(plan.call_group_payload)
        ground_truth = record.ground_truth_record(plan.calls_by_user_turn)
        assert question["id"] == source.task_id == ground_truth["id"]
        assert len(question["question"]) == sum(message.role == "user" for message in source.messages)
        assert len(ground_truth["ground_truth"]) == len(question["question"])
        assert question["x-nemotron"]["success_assertions"] == list(source.success_assertions)


def _assert_publication_only_selects_from_raw(output_dir: Path) -> dict[str, Any]:
    """Hold a real run to the raw/publication semantics, without the contract's help.

    The pipeline already ran :mod:`publication_contract` over these files, so
    re-running it here would only confirm that the check agrees with itself.
    These assertions restate the acceptance criteria directly.
    """
    import pyarrow.parquet as pq

    raw_table = pq.read_table(output_dir / "benchmark_raw.parquet")
    published_table = pq.read_table(output_dir / "benchmark.parquet")
    assert published_table.schema.equals(raw_table.schema)

    raw_rows = raw_table.to_pylist()
    published_rows = published_table.to_pylist()
    raw_by_task = {row["task_id"]: row for row in raw_rows}
    assert len(raw_by_task) == len(raw_rows)
    published_ids = [row["task_id"] for row in published_rows]
    assert len(set(published_ids)) == len(published_ids)
    assert set(published_ids) <= set(raw_by_task)

    for row in published_rows:
        assert json.dumps(row, sort_keys=True, default=str) == json.dumps(
            raw_by_task[row["task_id"]], sort_keys=True, default=str
        )
        assert row["held_out_hit"] in (None, False)

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_counts"]["replay_passed"] == len(raw_rows)
    assert manifest["stage_counts"]["published"] == len(published_rows)
    section = manifest["publication"]
    assert section["raw"]["rows"] == len(raw_rows)
    assert section["published"]["rows"] == len(published_rows)
    assert section["restated_fields"] == []
    assert section["verified"] is True
    return manifest


def test_the_tiny_publication_is_a_selection_of_its_raw_table(tiny_run) -> None:
    _, output_dir = tiny_run

    manifest = _assert_publication_only_selects_from_raw(output_dir)

    assert manifest["publication"]["published"]["ordering"] == "raw_order"


def test_stage_eleven_publishes_in_selection_rank_order(third_pack_run) -> None:
    _, output_dir = third_pack_run
    import pyarrow.parquet as pq

    manifest = _assert_publication_only_selects_from_raw(output_dir)
    published = manifest["publication"]["published"]
    assert published["surface_gate"] == "surface_quality"
    assert published["dedup_balancing_applied"] is True
    assert published["ordering"] == "selection_rank"

    decisions = pq.read_table(output_dir / "stage_cache" / "balanced_tasks.parquet").to_pylist()
    ranked = sorted(
        (decision for decision in decisions if decision["selected"]),
        key=lambda decision: decision["selection_rank"],
    )
    published_ids = [row["task_id"] for row in pq.read_table(output_dir / "benchmark.parquet").to_pylist()]
    assert published_ids == [decision["task_id"] for decision in ranked]


def _assert_projection_is_bound_to_the_published_file(output_dir: Path) -> None:
    """Project the real parquet and check the projection cites what it read."""
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    published_hash = manifest["publication"]["published"]["content_hash"]
    benchmark_path = output_dir / "benchmark.parquet"

    projection = project_published_benchmark(
        benchmark_path,
        expected_content_hash=published_hash,
        expected_task_ids=[row["task_id"] for row in _published_rows(benchmark_path)],
    )

    assert projection.source.content_hash == published_hash
    assert projection.source.rows == len(projection.rows) == manifest["stage_counts"]["published"]
    assert projection.provenance.pack_id == manifest["pack"]["pack_id"]
    assert projection.provenance.pack_version == manifest["pack"]["version"]
    assert projection.provenance.tier == manifest["tier"]
    assert set(projection.provenance.system_prompt_ids) == {row.system_prompt_id for row in projection.rows}
    for row in projection.rows:
        plan = projection.plan(row.task_id)
        assert plan.call_count == len(row.expected_tool_calls)
        # Every group is one assistant message, so the groups account for exactly
        # the calls the conversation issues.
        assert [call.trace_position for group in plan.groups for call in group.calls] == [
            call.trace_position for call in row.expected_tool_calls
        ]


def _published_rows(benchmark_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(benchmark_path).to_pylist()


def test_the_tiny_projection_cites_the_file_it_decoded(tiny_run) -> None:
    _, output_dir = tiny_run

    _assert_projection_is_bound_to_the_published_file(output_dir)


def test_a_projection_refuses_a_benchmark_replaced_after_publication(tiny_run) -> None:
    _, output_dir = tiny_run

    with pytest.raises(ExportProjectionError, match="changed after publication"):
        project_published_benchmark(
            output_dir / "benchmark.parquet",
            expected_content_hash="sha256:" + "0" * 64,
        )


def test_a_projection_refuses_rows_that_are_not_the_published_order(tiny_run) -> None:
    _, output_dir = tiny_run
    benchmark_path = output_dir / "benchmark.parquet"
    reversed_ids = [row["task_id"] for row in reversed(_published_rows(benchmark_path))]

    with pytest.raises(ExportProjectionError, match="in publication order"):
        project_published_benchmark(benchmark_path, expected_task_ids=reversed_ids)


def test_stage_eleven_output_projects_with_its_selection_order(third_pack_run) -> None:
    _, output_dir = third_pack_run

    _assert_projection_is_bound_to_the_published_file(output_dir)


def test_the_banking_benchmark_writes_a_complete_bfcl_json_export(banking_run, tmp_path: Path) -> None:
    # The banking pack covers all nine turn policies and Vietnamese surfaces, so
    # this is the broadest evidence that the writer handles real published rows.
    projection = project_benchmark_rows(
        banking_run,
        source=ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=len(banking_run)),
    )

    artifact = write_bfcl_json(projection, tmp_path)
    questions, answers = read_bfcl_json(tmp_path, artifact)

    assert artifact.rows == len(banking_run)
    assert [record["id"] for record in questions] == [row["task_id"] for row in banking_run]
    assert [record["id"] for record in answers] == [row["task_id"] for row in banking_run]
    for question, answer, row in zip(questions, answers, banking_run, strict=True):
        user_turns = sum(message["role"] == "user" for message in row["messages"])
        assert len(question["question"]) == user_turns
        assert len(answer["ground_truth"]) == user_turns
        assert sum(len(turn) for turn in answer["ground_truth"]) == row["num_tool_calls"]
        assert sum(len(group["calls"]) for group in question["x-nemotron"]["call_groups"]) == row["num_tool_calls"]
    # Vietnamese surfaces must stay readable rather than escaped in the file.
    assert "\\u" not in (tmp_path / artifact.question_file).read_text(encoding="utf-8")


def test_the_bfcl_json_export_is_byte_identical_across_two_writes(tiny_run, tmp_path: Path) -> None:
    _, output_dir = tiny_run
    projection = project_published_benchmark(output_dir / "benchmark.parquet")

    first = write_bfcl_json(projection, tmp_path / "first")
    second = write_bfcl_json(projection, tmp_path / "second")

    assert first.content_hash == second.content_hash
    assert first.source.content_hash == projection.source.content_hash


def test_enabling_bfcl_json_writes_the_export_from_the_pipeline(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["exports"]["bfcl_json"] = True
    config_path = tmp_path / "bfcl-json.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    output_dir = benchmark_path.parent
    projection = project_published_benchmark(benchmark_path)
    artifact = BfclJsonArtifact(
        rows=len(projection.rows),
        content_hash=export_tree_hash(
            output_dir,
            (BFCL_JSON_QUESTION_FILE, BFCL_JSON_ANSWER_FILE),
        ),
        source=projection.source,
    )
    questions, answers = read_bfcl_json(output_dir, artifact)

    assert [record["id"] for record in questions] == list(projection.task_ids)
    assert [record["id"] for record in answers] == list(projection.task_ids)


def test_the_banking_benchmark_writes_a_complete_evaluator_bundle(banking_run, tmp_path: Path) -> None:
    projection = project_benchmark_rows(
        banking_run,
        source=ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=len(banking_run)),
    )

    artifact = write_nemo_evaluator_bundle(projection, tmp_path)
    bundle, records = read_nemo_evaluator_bundle(tmp_path, artifact)

    assert artifact.rows == len(banking_run)
    assert bundle.record_count == len(banking_run)
    assert [record["task_id"] for record in records] == [row["task_id"] for row in banking_run]
    catalog = json.loads((tmp_path / artifact.root / bundle.system_prompt_file).read_text(encoding="utf-8"))
    for record, row in zip(records, banking_run, strict=True):
        # The whole rendered trace, turn for turn: a bundle record is replayed, so a
        # dropped tool result would leave a multi-turn task unable to reach its end.
        assert [(message["role"], message["content"]) for message in record["reference_trace"]] == [
            (message["role"], message["content"]) for message in row["messages"]
        ]
        assert all(message["role"] in {"system", "user"} for message in record["seed_messages"])
        assert len(record["expected_tool_calls"]) == row["num_tool_calls"]
        assert catalog[row["system_prompt_id"]]
    # The banking pack declares a multi_tool template, so ordering is measurable.
    assert "call_ordering" in bundle.scoring.metrics
    assert "\\u" not in (tmp_path / artifact.root / bundle.dataset_file).read_text(encoding="utf-8")


def test_the_evaluator_bundle_is_byte_identical_across_two_writes(tiny_run, tmp_path: Path) -> None:
    _, output_dir = tiny_run
    projection = project_published_benchmark(output_dir / "benchmark.parquet")

    first = write_nemo_evaluator_bundle(projection, tmp_path / "first")
    second = write_nemo_evaluator_bundle(projection, tmp_path / "second")

    assert first.content_hash == second.content_hash
    assert first.source.content_hash == projection.source.content_hash


def test_enabling_both_exports_writes_both_trees_from_one_projection(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["exports"]["bfcl_json"] = True
    config_data["exports"]["nemo_evaluator_bundle"] = True
    config_path = tmp_path / "both-exports.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    output_dir = benchmark_path.parent
    projection = project_published_benchmark(benchmark_path)
    artifact = NemoEvaluatorArtifact(
        rows=len(projection.rows),
        # Bundle-relative, like the descriptor: a moved bundle keeps its digest.
        content_hash=export_tree_hash(output_dir / NEMO_EVALUATOR_ROOT, NEMO_BUNDLE_FILES),
        source=projection.source,
    )
    bundle, records = read_nemo_evaluator_bundle(output_dir, artifact)

    assert [record["task_id"] for record in records] == list(projection.task_ids)
    assert bundle.source.benchmark_content_hash == projection.source.content_hash
    assert (output_dir / BFCL_JSON_QUESTION_FILE).is_file()
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    report_path = output_dir / manifest["exports"]["validation_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert manifest["exports"]["status"] == report["status"] == "passed"
    assert manifest["exports"]["benchmark_rows"] == report["benchmark_rows"] == len(projection.rows)
    assert set(manifest["exports"]["formats"]) == {"bfcl_json", "nemo_evaluator_bundle"}
    assert all(manifest["exports"]["formats"][name]["enabled"] for name in manifest["exports"]["formats"])
    assert not list(output_dir.glob(".stage12-*"))

    first_hashes = {name: details["content_hash"] for name, details in manifest["exports"]["formats"].items()}
    second_path = generate_bfcl(config_path)
    second_manifest = json.loads((second_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert {
        name: details["content_hash"] for name, details in second_manifest["exports"]["formats"].items()
    } == first_hashes


def test_a_disabled_export_does_not_inherit_a_previous_runs_tree(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["exports"]["nemo_evaluator_bundle"] = True
    enabled_path = tmp_path / "bundle-on.yaml"
    enabled_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    benchmark_path = generate_bfcl(enabled_path)
    output_dir = benchmark_path.parent
    assert (output_dir / NEMO_EVALUATOR_ROOT / NEMO_BUNDLE_FILE).is_file()

    config_data["exports"]["nemo_evaluator_bundle"] = False
    disabled_path = tmp_path / "bundle-off.yaml"
    disabled_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    generate_bfcl(disabled_path)

    # A stale bundle beside a manifest that never mentions it would be scored as
    # though this run had produced it.
    assert not (output_dir / NEMO_EVALUATOR_ROOT).exists()


@pytest.mark.parametrize(
    ("format_name", "partial_file"),
    [
        ("bfcl_json", Path(BFCL_JSON_QUESTION_FILE)),
        ("nemo_evaluator_bundle", Path(NEMO_EVALUATOR_ROOT) / NEMO_DATASET_FILE),
    ],
)
def test_export_failure_never_publishes_a_partial_stage_twelve_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    partial_file: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["exports"][format_name] = True
    config_path = tmp_path / "failing-export.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    output_dir = tmp_path / "output" / "bfcl_tiny_library_validation"
    output_dir.mkdir(parents=True)
    for name in ("run_manifest.json", "benchmark_raw.parquet", "benchmark.parquet"):
        (output_dir / name).write_text("stale", encoding="utf-8")

    def fail_after_one_file(_projection, stage: Path):  # type: ignore[no-untyped-def]
        partial = stage / partial_file
        partial.parent.mkdir(parents=True)
        partial.write_text("partial", encoding="utf-8")
        raise RuntimeError("forced export failure")

    monkeypatch.setitem(final_output.EXPORT_WRITERS, format_name, fail_after_one_file)
    with pytest.raises(RuntimeError, match="forced export failure"):
        generate_bfcl(config_path)

    assert not (output_dir / "run_manifest.json").exists()
    assert not (output_dir / "benchmark_raw.parquet").exists()
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "exports").exists()
    assert not list(output_dir.glob(".stage12-*"))


def test_tampered_export_is_rejected_before_the_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["exports"]["nemo_evaluator_bundle"] = True
    config_path = tmp_path / "tampered-export.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    def write_then_tamper(projection, stage: Path):  # type: ignore[no-untyped-def]
        artifact = write_nemo_evaluator_bundle(projection, stage)
        metadata = stage / artifact.root / NEMO_METADATA_FILE
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["tampered"] = True
        metadata.write_text(json.dumps(payload), encoding="utf-8")
        return artifact

    monkeypatch.setitem(final_output.EXPORT_WRITERS, "nemo_evaluator_bundle", write_then_tamper)
    with pytest.raises(RuntimeError, match="tree hash|no longer matches"):
        generate_bfcl(config_path)

    output_dir = tmp_path / "output" / "bfcl_tiny_library_validation"
    assert not (output_dir / "run_manifest.json").exists()
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "exports").exists()
    assert not list(output_dir.glob(".stage12-*"))


def test_promotion_failure_removes_every_final_payload_and_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    output_dir = tmp_path / "output"
    staging_dir = output_dir / ".stage12-test"
    staging_dir.mkdir(parents=True)
    output_dir.mkdir(exist_ok=True)
    for name in ("benchmark_raw.parquet", "benchmark.parquet", "run_manifest.json"):
        (staging_dir / name).write_text(f"new {name}", encoding="utf-8")
        (output_dir / name).write_text(f"old {name}", encoding="utf-8")
    (staging_dir / "exports").mkdir()
    (staging_dir / "exports" / "file").write_text("new export", encoding="utf-8")
    (output_dir / "exports").mkdir()
    (output_dir / "exports" / "file").write_text("old export", encoding="utf-8")

    replace = Path.replace

    def fail_second_promotion(path: Path, target: Path):  # type: ignore[no-untyped-def]
        if path == staging_dir / "benchmark.parquet":
            raise OSError("forced promotion failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_promotion)
    with pytest.raises(OSError, match="forced promotion failure"):
        final_output._commit_staged_publication(staging_dir, output_dir)

    assert not staging_dir.exists()
    assert not (output_dir / "run_manifest.json").exists()
    assert not (output_dir / "benchmark_raw.parquet").exists()
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "exports").exists()


def test_pack_and_endpoint_drift_remove_staged_payloads_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import isolation
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    pack_stage = tmp_path / "pack-stage"
    pack_stage.mkdir()
    (pack_stage / "payload").write_text("partial", encoding="utf-8")
    pack = SimpleNamespace(paths={})
    monkeypatch.setattr(final_output, "pack_fingerprint", lambda _paths: "changed")
    with pytest.raises(RuntimeError, match="oracle pack changed"):
        final_output._require_pack_fingerprint(
            pack,  # type: ignore[arg-type]
            "validated",
            phase="before commit",
            cleanup=(pack_stage,),
        )
    assert not pack_stage.exists()

    endpoint_stage = tmp_path / "endpoint-stage"
    endpoint_stage.mkdir()
    (endpoint_stage / "payload").write_text("partial", encoding="utf-8")
    runtime = SimpleNamespace(
        episode_timeout_s=1.0,
        worker="process",
        clock="2026-01-01T00:00:00Z",
        import_timeout_s=1.0,
        tool_timeout_s=1.0,
    )
    config = SimpleNamespace(oracle_runtime=runtime, random_seed=0)
    endpoint_pack = SimpleNamespace(endpoint_config={"protocol_version": "bfcl-oracle-http-v1"})

    class ChangedEndpointWorker:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            pass

        def run_episode(self, **_kwargs):  # type: ignore[no-untyped-def]
            return [{"oracle_id": "changed"}]

    monkeypatch.setattr(isolation, "ProcessWorker", ChangedEndpointWorker)
    with pytest.raises(RuntimeError, match="endpoint identity changed"):
        final_output._require_endpoint_identity(
            config,  # type: ignore[arg-type]
            endpoint_pack,  # type: ignore[arg-type]
            {"oracle_id": "validated"},
            cleanup=(endpoint_stage,),
        )
    assert not endpoint_stage.exists()


def test_every_published_tiny_row_projects_through_the_export_contract(tiny_run) -> None:
    rows, _ = tiny_run

    _assert_exports_preserve_the_published_benchmark(rows)


def test_the_banking_projection_keeps_parallel_calls_in_one_group(banking_run) -> None:
    projection = project_benchmark_rows(
        banking_run,
        source=ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=len(banking_run)),
    )

    parallel = [plan for plan in projection.plans if plan.parallel_groups]
    assert parallel, "the banking pack declares a multi_tool template, so a parallel group must survive projection"
    for plan in parallel:
        for group in plan.parallel_groups:
            # One assistant message issues them together; splitting the group into
            # consecutive turns would turn parallel calling into sequential calling.
            assert len({call.turn_index for call in group.calls}) == 1
            assert len({call.call_group for call in group.calls}) == 1


def test_every_published_banking_row_projects_through_the_export_contract(banking_run) -> None:
    # The banking pack exercises all nine turn policies, so this is the broadest
    # evidence available that the contract describes real published rows rather
    # than the shape the tests happen to build.
    _assert_exports_preserve_the_published_benchmark(banking_run)


def test_banking_withheld_slot_stays_out_of_the_first_user_turn(banking_run) -> None:
    row = _row(banking_run, "bn_balance_withheld_account")
    account_id = decode_arguments(row["expected_tool_calls"][0]["arguments"])["account_id"]
    user_turns = [message["content"] for message in row["messages"] if message["role"] == "user"]

    assert len(user_turns) == 2
    assert account_id not in user_turns[0]
    assert account_id in user_turns[1]
    assert row["expected_tool_calls"][0]["turn_index"] == 1


def test_banking_dependent_call_reads_its_id_from_the_first_result(banking_run) -> None:
    row = _row(banking_run, "bn_latest_txn_status_dependent")
    first, second = row["expected_tool_calls"]

    assert first["function_name"] == "list_recent_transactions"
    assert second["function_name"] == "get_transaction_status"
    assert first["call_group"] < second["call_group"]
    assert (first["turn_index"], second["turn_index"]) == (0, 1)

    listed = json.loads(next(message for message in row["messages"] if message["role"] == "tool")["content"])
    checked_id = decode_arguments(second["arguments"])["transaction_id"]
    assert checked_id == listed["transactions"][0]["transaction_id"]
    assert checked_id not in decode_arguments(first["arguments"]).values()


def test_banking_correction_transfers_only_the_replacement_amount(banking_run) -> None:
    row = _row(banking_run, "bn_transfer_amount_corrected")
    (call,) = row["expected_tool_calls"]
    arguments = decode_arguments(call["arguments"])
    user_turns = [message["content"] for message in row["messages"] if message["role"] == "user"]

    assert arguments["confirm"] is True
    # The user stated one amount, replaced it, and confirmed again; only the
    # replacement may reach the mutating call.
    assert str(arguments["amount_vnd"]) in user_turns[1]
    assert str(arguments["amount_vnd"]) not in user_turns[0]
    assert len(user_turns) == 3
    assert call["turn_index"] == 2

    committed = json.loads(next(message for message in row["messages"] if message["role"] == "tool")["content"])
    assert committed["status"] == "succeeded"
    assert committed["amount_vnd"] == arguments["amount_vnd"]


def test_banking_negative_paths_keep_the_failure_in_the_trace(banking_run) -> None:
    not_found = _row(banking_run, "bn_txn_status_unknown_id")
    tool_message = next(message for message in not_found["messages"] if message["role"] == "tool")
    assert json.loads(tool_message["content"])["error"]["code"] == "not_found"

    rejected = _row(banking_run, "bn_transfer_short_of_funds")
    tool_message = next(message for message in rejected["messages"] if message["role"] == "tool")
    assert json.loads(tool_message["content"])["status"] == "rejected_insufficient_funds"
    assert decode_arguments(rejected["expected_tool_calls"][0]["arguments"])["confirm"] is True
    assert rejected["expected_tool_calls"][0]["turn_index"] == 1


def test_each_generation_stage_leaves_a_joinable_artifact(tiny_run) -> None:
    import pyarrow.parquet as pq

    rows, output_dir = tiny_run
    cache = output_dir / "stage_cache"
    tables = {
        name: pq.read_table(cache / f"{name}.parquet").to_pylist()
        for name in (
            "task_instances",
            "conversation_plans",
            "rendered_conversations",
            "expected_traces",
            "schema_validated_traces",
            "replay_validated_tasks",
        )
    }

    # Every stage keeps one row per task under the same key, so the intermediates can
    # be joined and a dropped row can be traced to the stage that dropped it.
    task_ids = {row["task_id"] for row in tables["task_instances"]}
    assert task_ids >= {row["task_id"] for row in rows}
    for name, table in tables.items():
        assert {row["task_id"] for row in table} == task_ids, name
        assert len(table) == len(task_ids), name

    plans = {row["task_id"]: row for row in tables["conversation_plans"]}
    traces = {row["task_id"]: row for row in tables["expected_traces"]}
    for row in rows:
        assert plans[row["task_id"]]["num_tool_calls"] == row["num_tool_calls"]
        assert traces[row["task_id"]]["num_tool_calls"] == len(row["expected_tool_calls"])
        assert json.loads(plans[row["task_id"]]["steps"])[0]["source"] == "first_turn"
    assert all(row["valid"] for row in tables["schema_validated_traces"])
    assert all(row["valid"] and row["reason"] is None for row in tables["replay_validated_tasks"])


def test_run_manifest_records_smoke_lineage_and_artifact_hashes(tiny_run) -> None:
    _, output_dir = tiny_run
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["generation_mode"] == "smoke_no_publication"
    assert manifest["lineage_policy"] == "smoke_no_publication"
    assert manifest["tier"] == "gold"
    assert manifest["gold_eligible"] is False
    assert manifest["gold_ineligibility_reasons"] == ["smoke_no_publication"]
    assert manifest["pack"]["pack_id"] == "tiny_library"
    assert manifest["models"]["paraphrase"]["enabled"] is False
    assert manifest["models"]["paraphrase"]["canonical_id"] is None
    assert manifest["models"]["paraphrase"]["config_hash"] is None
    assert manifest["bias_targets"] == {"tasks_per_category": 4}
    assert manifest["bias_applicability"]["B1"]["status"] == "applicable"
    assert manifest["bias_applicability"]["B7"] == {
        "status": "na",
        "reason": "pack declares no held_out policy",
    }
    assert manifest["generation_config_hash"] != manifest["resolved_config_hash"]
    assert manifest["runtime"]["pipeline_source_hash"].startswith("sha256:")
    assert manifest["runtime"]["dependency_lock_hash"].startswith("sha256:")
    checkpoint = json.loads(
        (output_dir / "stage_cache" / "checkpoints" / "final_output" / "manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint["identity"]["pipeline"] == {
        "git_sha": manifest["runtime"]["pipeline_git_sha"],
        "source_hash": manifest["runtime"]["pipeline_source_hash"],
        "dependency_lock_hash": manifest["runtime"]["dependency_lock_hash"],
        "worker_image_digest": manifest["runtime"]["worker_image_digest"],
    }
    assert manifest["stage_counts"]["expanded"] >= manifest["stage_counts"]["replay_passed"]
    assert manifest["stage_counts"]["trace_dropped"] == 0
    assert manifest["paraphrase_rejections"]["requested_candidates"] == 0
    assert manifest["paraphrase_rejections"]["rejected_candidates"] == 0
    assert manifest["trace_drop_rejections"] == {"count": 0, "by_reason": {}}
    assert manifest["exports"]["evaluated"] is False
    assert manifest["exports"]["status"] is None
    assert all(not item["enabled"] for item in manifest["exports"]["formats"].values())
    for artifact in (
        "benchmark_raw_parquet",
        "benchmark_parquet",
        "tools_model_facing",
        "oracle_validation_report",
        "pack_manifest",
        "fixtures_normalized",
        "task_templates_normalized",
        "reference_profile",
        "reference_samples",
        "task_instances",
        "conversation_plans",
        "rendered_conversations",
        "expected_traces",
        "schema_validated_traces",
        "replay_validated_tasks",
    ):
        assert manifest["artifacts"][artifact]["content_hash"].startswith("sha256:")
    assert manifest["stage_counts"]["replay_passed"] == manifest["stage_counts"]["published"]
    assert (output_dir / "benchmark_raw.parquet").exists()


def test_prepare_invalidates_a_previous_publication_before_rewriting_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pipeline
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = BfclConfig.from_yaml(config_path)
    output = Path(config.output_dir) / config.expt_name
    output.mkdir(parents=True)
    stale = [output / "run_manifest.json", output / "benchmark.parquet"]
    for path in stale:
        path.write_text("stale", encoding="utf-8")

    def stop_after_invalidation(_config, *, force=False):  # type: ignore[no-untyped-def]
        assert all(not path.exists() for path in stale)
        raise RuntimeError("invalidation observed")

    monkeypatch.setattr(pipeline, "_validate_pack", stop_after_invalidation)

    with pytest.raises(RuntimeError, match="invalidation observed"):
        pipeline.prepare_bfcl(config_path)


def test_enabled_surface_quality_is_integrated_into_pipeline_and_manifest(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    rows, output_dir = _run_tiny_with_surface_quality(tmp_path)
    cache = output_dir / "stage_cache"
    quality_rows = pq.read_table(cache / "surface_validated_tasks.parquet").to_pylist()
    raw_rows = pq.read_table(output_dir / "benchmark_raw.parquet").to_pylist()
    report = json.loads((cache / "surface_quality_rejections.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert {row["task_id"] for row in quality_rows} == {row["task_id"] for row in raw_rows}
    assert {row["task_id"] for row in rows} == {row["task_id"] for row in quality_rows if row["accepted"]}
    assert report["evaluated"] == len(raw_rows)
    assert report["kept"] == len(rows)
    assert manifest["surface_quality_validation"] == {
        "contract_version": "1.1",
        "enabled": True,
        "drop_authority": False,
        "report": report,
    }
    assert manifest["stage_counts"]["surface_quality_evaluated"] == len(raw_rows)
    assert manifest["stage_counts"]["surface_quality_kept"] == len(rows)
    assert manifest["stage_counts"]["surface_quality_judge_errors"] == 0
    assert manifest["artifacts"]["surface_validated_tasks"]["content_hash"].startswith("sha256:")
    assert manifest["artifacts"]["surface_quality_rejections"]["content_hash"].startswith("sha256:")


def test_enabled_stage_eleven_filters_publication_and_is_in_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    benchmark_path = generate_bfcl(_write_tiny_dedup_config(tmp_path, enabled=True, name="tiny-dedup.yaml"))
    output_dir = benchmark_path.parent
    cache = output_dir / "stage_cache"
    published = pq.read_table(benchmark_path).to_pylist()
    balanced = pq.read_table(cache / "balanced_tasks.parquet").to_pylist()
    report = json.loads((cache / "dedup_balancing_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    selected = sorted(
        (row for row in balanced if row["selected"]),
        key=lambda row: row["selection_rank"],
    )
    assert [row["task_id"] for row in published] == [row["task_id"] for row in selected]
    assert len(balanced) == manifest["stage_counts"]["surface_quality_kept"]
    assert report["counts"]["selected"] == len(published)
    assert manifest["semantic_deduplication"]["enabled"] is True
    assert manifest["semantic_deduplication"]["contract_version"] == "1.0"
    assert manifest["semantic_deduplication"]["model_identifier"] == "test/semantic-embedding"
    assert manifest["semantic_deduplication"]["embedding_signature"] == "sha256:test-embeddings"
    assert manifest["semantic_deduplication"]["report"] == report
    assert manifest["stage_counts"]["dedup_balancing_input"] == len(balanced)
    assert manifest["stage_counts"]["dedup_balancing_selected"] == len(published)
    assert manifest["artifacts"]["balanced_tasks"]["content_hash"].startswith("sha256:")
    assert manifest["artifacts"]["dedup_balancing_report"]["content_hash"].startswith("sha256:")
    assert manifest["artifacts"]["balanced_tasks"]["content_hash"] == (
        "sha256:" + hashlib.sha256((cache / "balanced_tasks.parquet").read_bytes()).hexdigest()
    )
    assert manifest["artifacts"]["dedup_balancing_report"]["content_hash"] == (
        "sha256:" + hashlib.sha256((cache / "dedup_balancing_report.json").read_bytes()).hexdigest()
    )
    assert (
        report["artifacts"]["balanced_tasks.parquet"]["content_hash"]
        == (manifest["artifacts"]["balanced_tasks"]["content_hash"])
    )


def test_disabling_stage_eleven_does_not_republish_stale_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    enabled_path = _write_tiny_dedup_config(
        tmp_path,
        enabled=True,
        name="tiny-dedup-enabled.yaml",
    )
    benchmark_path = generate_bfcl(enabled_path)
    output_dir = benchmark_path.parent
    cache = output_dir / "stage_cache"
    assert (cache / "balanced_tasks.parquet").is_file()
    assert (cache / "dedup_balancing_report.json").is_file()

    generate_bfcl(
        _write_tiny_dedup_config(
            tmp_path,
            enabled=False,
            name="tiny-dedup-disabled.yaml",
        )
    )
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["semantic_deduplication"]["enabled"] is False
    assert manifest["semantic_deduplication"]["report"] is None
    assert "balanced_tasks" not in manifest["artifacts"]
    assert "dedup_balancing_report" not in manifest["artifacts"]
    assert not (cache / "balanced_tasks.parquet").exists()
    assert not (cache / "dedup_balancing_report.json").exists()


def test_stage_eleven_selected_ids_are_deterministic_across_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    config_path = _write_tiny_dedup_config(
        tmp_path,
        enabled=True,
        name="tiny-dedup-repeat.yaml",
    )
    first_path = generate_bfcl(config_path)
    first_rows = pq.read_table(first_path).to_pylist()
    first_balanced = pq.read_table(first_path.parent / "stage_cache" / "balanced_tasks.parquet").to_pylist()

    second_path = generate_bfcl(config_path)
    second_rows = pq.read_table(second_path).to_pylist()
    second_balanced = pq.read_table(second_path.parent / "stage_cache" / "balanced_tasks.parquet").to_pylist()

    assert [row["task_id"] for row in first_rows] == [row["task_id"] for row in second_rows]
    assert [(row["task_id"], row["selected"], row["selection_rank"]) for row in first_balanced] == [
        (row["task_id"], row["selected"], row["selection_rank"]) for row in second_balanced
    ]


def test_stage_eleven_aborts_on_unmet_targets_under_abort_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    config_path = _write_tiny_dedup_config(
        tmp_path,
        enabled=True,
        name="tiny-unmet-abort.yaml",
        impossible_mix=True,
    )

    with pytest.raises(
        dedup_balancing.DedupBalancingPolicyError,
        match="unmet_target_policy='abort'",
    ):
        generate_bfcl(config_path)

    output_dir = tmp_path / "output" / "bfcl_tiny_library_validation"
    report = json.loads((output_dir / "stage_cache" / "dedup_balancing_report.json").read_text(encoding="utf-8"))
    assert report["unmet_targets"]
    assert report["release_policy"] == {
        "gold_eligible": False,
        "unmet_target_action": "abort",
        "unmet_target_policy": "abort",
    }
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "run_manifest.json").exists()


def test_stage_eleven_may_publish_unmet_targets_only_as_non_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    benchmark_path = generate_bfcl(
        _write_tiny_dedup_config(
            tmp_path,
            enabled=True,
            name="tiny-unmet-non-gold.yaml",
            unmet_target_policy="publish_non_gold",
            impossible_mix=True,
        )
    )
    rows = pq.read_table(benchmark_path).to_pylist()
    manifest = json.loads((benchmark_path.parent / "run_manifest.json").read_text(encoding="utf-8"))

    assert rows
    assert all(row["gold_eligible"] is False for row in rows)
    assert manifest["gold_eligible"] is False
    assert manifest["gold_ineligibility_reasons"] == [
        "smoke_no_publication",
        "stage_eleven_unmet_targets",
    ]
    assert manifest["semantic_deduplication"]["report"]["unmet_targets"]
    assert manifest["semantic_deduplication"]["report"]["release_policy"] == {
        "gold_eligible": False,
        "unmet_target_action": "publish_non_gold",
        "unmet_target_policy": "publish_non_gold",
    }


def test_stage_eleven_embedding_failure_aborts_without_final_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    def fail_embedding(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("forced embedding failure")

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", fail_embedding)
    with pytest.raises(RuntimeError, match="forced embedding failure"):
        generate_bfcl(
            _write_tiny_dedup_config(
                tmp_path,
                enabled=True,
                name="tiny-embedding-failure.yaml",
            )
        )

    output_dir = tmp_path / "output" / "bfcl_tiny_library_validation"
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "run_manifest.json").exists()


def test_missing_stage_eleven_artifact_blocks_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    run_stage = dedup_balancing.run_dedup_balancing_stage

    def delete_balanced_tasks(config, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = run_stage(config, *args, **kwargs)
        result["artifacts"]["artifact_path"].unlink()
        return result

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    monkeypatch.setattr(
        dedup_balancing,
        "run_dedup_balancing_stage",
        delete_balanced_tasks,
    )
    with pytest.raises(FileNotFoundError):
        generate_bfcl(
            _write_tiny_dedup_config(
                tmp_path,
                enabled=True,
                name="tiny-missing-dedup-artifact.yaml",
            )
        )

    output_dir = tmp_path / "output" / "bfcl_tiny_library_validation"
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "run_manifest.json").exists()


def test_modified_stage_eleven_artifact_blocks_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        dedup_balancing,
    )

    run_stage = dedup_balancing.run_dedup_balancing_stage

    def modify_balanced_tasks(config, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = run_stage(config, *args, **kwargs)
        artifact_path = result["artifacts"]["artifact_path"]
        artifact_path.write_bytes(artifact_path.read_bytes() + b"modified")
        return result

    monkeypatch.setattr(dedup_balancing, "run_semantic_dedup", _semantic_singletons)
    monkeypatch.setattr(
        dedup_balancing,
        "run_dedup_balancing_stage",
        modify_balanced_tasks,
    )
    with pytest.raises(ValueError, match="content hash does not match"):
        generate_bfcl(
            _write_tiny_dedup_config(
                tmp_path,
                enabled=True,
                name="tiny-modified-dedup-artifact.yaml",
            )
        )

    output_dir = tmp_path / "output" / "bfcl_tiny_library_validation"
    assert not (output_dir / "benchmark.parquet").exists()
    assert not (output_dir / "run_manifest.json").exists()


def test_disabling_stage_ten_does_not_republish_its_previous_artifacts(
    tmp_path: Path,
) -> None:
    """A rerun that skips Stage 10 must not inherit the last run's quality verdict."""
    _, output_dir = _run_tiny_with_surface_quality(tmp_path)
    cache = output_dir / "stage_cache"
    assert (cache / "surface_validated_tasks.parquet").is_file()

    generate_bfcl(_write_tiny_quality_config(tmp_path, enabled=False, name="tiny-plain.yaml"))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["surface_quality_validation"]["enabled"] is False
    assert manifest["surface_quality_validation"]["report"] is None
    assert "surface_validated_tasks" not in manifest["artifacts"]
    assert "surface_quality_rejections" not in manifest["artifacts"]
    assert not (cache / "surface_validated_tasks.parquet").exists()
    assert not (cache / "surface_quality_rejections.json").exists()


def test_a_deleted_stage_ten_artifact_blocks_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enabled quality stage may not publish without the evidence it produced."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import surface_quality

    run_validation = surface_quality.run_surface_quality_validation

    def delete_report(config, tasks, surfaces, **kwargs):  # type: ignore[no-untyped-def]
        decided, report = run_validation(config, tasks, surfaces, **kwargs)
        (Path(config.output_dir) / config.expt_name / "stage_cache" / "surface_quality_rejections.json").unlink()
        return decided, report

    monkeypatch.setattr(surface_quality, "run_surface_quality_validation", delete_report)

    with pytest.raises(FileNotFoundError):
        _run_tiny_with_surface_quality(tmp_path)
    # The run must stop before a published parquet exists without its manifest.
    assert not list((tmp_path / "output").glob("**/run_manifest.json"))
    assert not list((tmp_path / "output").glob("**/benchmark.parquet"))


def test_a_stage_artifact_changed_after_completion_blocks_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    real_final_output = final_output.run_final_output

    def mutate_then_publish(config, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = Path(config.output_dir) / config.expt_name / "stage_cache" / "task_instances.parquet"
        with path.open("ab") as handle:
            handle.write(b"changed-after-stage")
        return real_final_output(config, *args, **kwargs)

    monkeypatch.setattr(final_output, "run_final_output", mutate_then_publish)

    with pytest.raises(RuntimeError, match="changed after its producing stage"):
        _run_tiny(tmp_path)
    assert not list((tmp_path / "output").glob("**/run_manifest.json"))
    assert not list((tmp_path / "output").glob("**/benchmark.parquet"))


def test_stage_ten_cannot_publish_an_empty_gold_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import surface_quality

    run_validation = surface_quality.run_surface_quality_validation

    def reject_every_surface(config, tasks, surfaces, **kwargs):  # type: ignore[no-untyped-def]
        for task in tasks:
            surfaces[str(task["task_id"])]["guard_violations"].append(
                {"guard": "must_not_mention", "phrase": "forced-quality-rejection"}
            )
        return run_validation(config, tasks, surfaces, **kwargs)

    monkeypatch.setattr(
        surface_quality,
        "run_surface_quality_validation",
        reject_every_surface,
    )

    with pytest.raises(RuntimeError, match="no publication rows after Stage 10"):
        _run_tiny_with_surface_quality(tmp_path)
    assert not list((tmp_path / "output").glob("**/run_manifest.json"))
    assert not list((tmp_path / "output").glob("**/benchmark.parquet"))


def test_slice_is_deterministic_across_runs(tmp_path: Path, tiny_run) -> None:
    rows, _ = tiny_run
    rerun_rows, _ = _run_tiny(tmp_path)

    def fingerprint(items: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return sorted(
            (row["task_id"], row["seed"], json.dumps(row["messages"], sort_keys=True, default=str)) for row in items
        )

    assert fingerprint(rows) == fingerprint(rerun_rows)
