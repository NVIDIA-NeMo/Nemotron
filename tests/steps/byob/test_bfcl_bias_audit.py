from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from nemotron.steps.byob.runtime.benchmark_families.bfcl.bias_audit import (
    BIAS_IDS,
    EDGE_IDS,
    AuditInputs,
    BiasAuditError,
    _gini,
    build_bias_audit_report,
    render_bias_audit_markdown,
    validate_bias_audit_report,
    write_bias_audit_reports,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    export_content_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    rendered_conversations_schema,
    write_stage_table,
)
from nemotron.steps.byob.scripts.restore_bfcl_rendered_conversations import (
    _rendered_row,
    restore_rendered_conversations,
)


def _row(
    index: int,
    policy: str,
    *,
    category: str,
    difficulty: str,
    required: list[str],
    call_count: int,
    multi_turn: bool,
    negative_families: tuple[str, ...] = (),
) -> dict[str, Any]:
    task_id = f"task-{index:02d}"
    calls = (
        [
            {
                "function_name": required[position % len(required)],
                "arguments": {"record_id": f"REC-{index:02d}"},
            }
            for position in range(call_count)
        ]
        if required
        else []
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"Request {index}"}]
    for family in negative_families:
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "error": {
                            "code": family,
                            "entity": "records",
                            "id": f"REC-{index:02d}",
                            "field": "record_id",
                            "message": family,
                        }
                    }
                ),
            }
        )
    if multi_turn:
        messages.append({"role": "user", "content": f"Follow-up {index}"})
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
            },
        }
        for name in ("tool_a", "tool_b")
    ]
    return {
        "id": task_id,
        "x-nemotron": {
            "schema_version": "1.0",
            "messages": messages,
            "expected_tool_calls": calls,
            "tools": tools,
            "metadata": {
                "category": category,
                "difficulty": difficulty,
                "intent": f"intent_{index % 3}",
                "turn_policy": policy,
                "required_tools": required,
                "tools_present": ["tool_a", "tool_b"],
                "num_tool_calls": call_count,
                "is_multi_turn": multi_turn,
                "fixture_refs": [],
                "held_out_hit": None,
                "variant_index": 0,
                "surface": {"base_task_id": task_id},
                "paraphrase_model": None,
            },
        },
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    release = root / "release"
    export = release / "exports" / "bfcl_json"
    answers = export / "possible_answer"
    answers.mkdir(parents=True)
    policies = (
        "single_turn",
        "missing_slot",
        "correction",
        "confirmation",
        "dependent_call",
        "multi_tool",
        "negative_path",
        "clarify_only",
        "irrelevant",
    )
    rows = [
        _row(
            index,
            policy,
            category=f"category_{index % 3}",
            difficulty=("easy", "medium", "hard")[index % 3],
            required=(
                ["tool_a", "tool_b"]
                if policy in {"dependent_call", "multi_tool"}
                else []
                if policy in {"clarify_only", "irrelevant"}
                else ["tool_a" if index % 2 == 0 else "tool_b"]
            ),
            call_count=(
                2
                if policy in {"dependent_call", "multi_tool", "negative_path"}
                else 0
                if policy in {"clarify_only", "irrelevant"}
                else 1
            ),
            multi_turn=policy
            in {
                "missing_slot",
                "correction",
                "confirmation",
                "dependent_call",
                "multi_tool",
            },
            negative_families=(("not_found", "business_rejection") if policy == "negative_path" else ()),
        )
        for index, policy in enumerate(policies)
    ]
    question_bytes = ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode()
    answer_bytes = (
        "".join(json.dumps({"id": row["id"], "possible_answer": []}, sort_keys=True) + "\n" for row in rows)
    ).encode()
    question = export / "BFCL_v4_multi_turn.jsonl"
    answer = answers / "BFCL_v4_multi_turn.jsonl"
    question.write_bytes(question_bytes)
    answer.write_bytes(answer_bytes)
    export_hash = export_content_hash(
        {
            "exports/bfcl_json/BFCL_v4_multi_turn.jsonl": question_bytes,
            ("exports/bfcl_json/possible_answer/BFCL_v4_multi_turn.jsonl"): answer_bytes,
        }
    )
    applicability = {bias_id: {"status": "applicable"} for bias_id in BIAS_IDS}
    for bias_id in ("B6", "B7", "B8", "B9", "B10", "B12", "B15"):
        applicability[bias_id] = {
            "status": "na",
            "reason": "synthetic fixture does not exercise this evidence",
        }
    applicability.update({edge: {"status": "applicable"} for edge in EDGE_IDS})
    manifest = {
        "run_id": "synthetic-bias-audit",
        "pack": {"pack_id": "synthetic", "version": "1.0.0"},
        "publication": {"published": {"rows": len(rows)}},
        "artifacts": {},
        "exports": {
            "formats": {
                "bfcl_json": {
                    "path": "exports/bfcl_json",
                    "content_hash": export_hash,
                }
            }
        },
        "bias_applicability": applicability,
        "bias_targets": {
            "tasks_per_category": 3,
            "difficulty_mix": {
                "easy": 1 / 3,
                "medium": 1 / 3,
                "hard": 1 / 3,
            },
            "tool_call_count_mix": {
                "1": 4 / 7,
                "2": 3 / 7,
                "3+": 0.0,
            },
            "max_intent_share": 0.50,
            "turn_mix": {"single_turn": 4 / 9, "multi_turn": 5 / 9},
        },
        "models": {"surface_judge": {"enabled": False}},
        "semantic_deduplication": {
            "enabled": True,
            "report": {
                "counts": {
                    "stage_ten_survivors": len(rows),
                    "semantic_duplicate_drops": 0,
                }
            },
        },
    }
    manifest_path = release / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, export


def _build(root: Path) -> dict[str, Any]:
    manifest, export = _write_fixture(root)
    return build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=root / "audit",
            published=export,
        )
    )


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _published_rows(export: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (export / "BFCL_v4_multi_turn.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_gini_uses_the_frozen_population_formula() -> None:
    assert _gini([3, 3, 3]) == 0.0
    assert _gini([5, 4]) == pytest.approx(1 / 18)
    with pytest.raises(BiasAuditError):
        _gini([])


def test_report_computes_b1_through_b16_once_and_matches_golden(
    tmp_path: Path,
) -> None:
    report = _build(tmp_path)

    projection = {
        "summary": report["summary"],
        "metrics": [
            {
                "bias_id": metric["bias_id"],
                "name": metric["primary_metric"]["name"],
                "value": metric["primary_metric"]["value"],
                "passed": metric["primary_metric"]["passed"],
                "evidence_complete": metric["evidence_complete"],
            }
            for metric in report["metrics"]
        ],
    }
    golden = json.loads(
        (Path(__file__).with_name("golden") / "bfcl_bias_audit_projection.json").read_text(encoding="utf-8")
    )
    assert projection == golden


def test_report_and_markdown_are_byte_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    assert first == second
    first_paths = write_bias_audit_reports(first, tmp_path / "out-1")
    second_paths = write_bias_audit_reports(second, tmp_path / "out-2")
    assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
    assert first_paths[1].read_bytes() == second_paths[1].read_bytes()
    assert render_bias_audit_markdown(first).startswith("# BFCL B1-B16 bias audit\n")


def test_report_hash_detects_semantic_tampering(tmp_path: Path) -> None:
    report = _build(tmp_path)
    tampered = copy.deepcopy(report)
    tampered["metrics"][0]["primary_metric"]["value"] = 0.0

    with pytest.raises(BiasAuditError, match="report_hash"):
        validate_bias_audit_report(tampered)


def test_source_hash_detects_published_artifact_tampering(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    with (export / "BFCL_v4_multi_turn.jsonl").open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write("{}\n")

    with pytest.raises(BiasAuditError, match="export hash"):
        build_bias_audit_report(
            AuditInputs(
                run_manifest=manifest,
                output_dir=tmp_path / "audit",
                published=export,
            )
        )


def test_missing_applicability_fails_closed(tmp_path: Path) -> None:
    manifest, export = _write_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    del document["bias_applicability"]["B16"]
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(BiasAuditError, match="bias_applicability keys mismatch"):
        build_bias_audit_report(
            AuditInputs(
                run_manifest=manifest,
                output_dir=tmp_path / "audit",
                published=export,
            )
        )


def test_applicable_b5_without_manifest_target_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    del document["bias_targets"]["tool_call_count_mix"]
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit",
            published=export,
        )
    )

    b5 = report["metrics"][4]
    assert b5["bias_id"] == "B5"
    assert b5["primary_metric"]["value"] is None
    assert b5["primary_metric"]["passed"] is False
    assert b5["evidence_complete"] is False
    assert b5["supporting_diagnostics"]["missing_evidence"] == ("run_manifest.bias_targets.tool_call_count_mix")
    assert report["summary"]["status"] == "failed"


def test_publication_examples_pin_explicit_balance_and_intent_targets() -> None:
    config_root = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "bfcl" / "config"
    expected = {"1": 0.60, "2": 0.30, "3+": 0.10}

    for name in ("publication.example.yaml", "publication.paraphrase.example.yaml"):
        document = yaml.safe_load((config_root / name).read_text(encoding="utf-8"))
        assert document["task_generation"]["tool_call_count_mix"] == expected
        assert document["task_generation"]["max_intent_share"] == 0.50


def test_b15_uses_manifest_target_with_five_point_tolerance_and_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["bias_applicability"]["B15"] = {"status": "applicable"}
    document["bias_targets"]["max_intent_share"] = 0.95
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "tools": "tools.json",
                    "templates": "task_templates.yaml",
                }
            }
        ),
        encoding="utf-8",
    )
    (pack / "tools.json").write_text("[]\n", encoding="utf-8")
    (pack / "task_templates.yaml").write_text(
        yaml.safe_dump(
            [
                {"category": f"category_{index}", "intent": f"intent_{index}"}
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )

    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit",
            published=export,
            pack_manifest=pack / "manifest.yaml",
        )
    )
    b15 = report["metrics"][14]
    assert b15["primary_metric"]["passed"] is True
    assert b15["supporting_diagnostics"]["allowed_max_intent_share"] == 1.0

    del document["bias_targets"]["max_intent_share"]
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit-missing",
            published=export,
            pack_manifest=pack / "manifest.yaml",
        )
    )
    b15 = report["metrics"][14]
    assert b15["primary_metric"]["passed"] is False
    assert b15["evidence_complete"] is False
    assert b15["supporting_diagnostics"]["missing_evidence"] == [
        "run_manifest.bias_targets.max_intent_share"
    ]


def test_reviewed_b10_and_b13_evidence_is_run_bound_and_seeded(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["bias_applicability"]["B10"] = {"status": "applicable"}
    document["models"]["surface_judge"] = {"enabled": True}
    document["surface_quality_validation"] = {"report": {"judge_prompt_hash": "sha256:" + "a" * 64}}
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_hash = _sha256(manifest)
    rows = _published_rows(export)
    canonical = [row["x-nemotron"] for row in rows]

    seed = 17
    eligible = [
        (row["id"], item["metadata"]["required_tools"])
        for row, item in zip(rows, canonical, strict=True)
        if set(item["metadata"]["tools_present"]) > set(item["metadata"]["required_tools"])
    ]
    sample = sorted(
        eligible,
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest(),
            item[0],
        ),
    )[:30]
    b10 = tmp_path / "b10.json"
    b10.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "distractor_gold_agreement",
                "run_manifest_hash": manifest_hash,
                "reviewers": ["reviewer-a", "reviewer-b"],
                "seed": seed,
                "rows": [
                    {
                        "task_id": task_id,
                        "annotations": [
                            {
                                "reviewer": "reviewer-a",
                                "selected_tools": required,
                            },
                            {
                                "reviewer": "reviewer-b",
                                "selected_tools": required,
                            },
                        ],
                    }
                    for task_id, required in sample
                ],
            }
        ),
        encoding="utf-8",
    )

    judge_seed = 23
    judge_sample = sorted(
        [row["id"] for row in rows],
        key=lambda task_id: (
            hashlib.sha256(f"{judge_seed}\0{task_id}".encode()).hexdigest(),
            task_id,
        ),
    )[:3]
    b13 = tmp_path / "b13.json"
    b13.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "judge_truth_creep",
                "run_manifest_hash": manifest_hash,
                "reviewers": ["reviewer-a", "reviewer-b"],
                "prompt_hashes": ["sha256:" + "a" * 64],
                "seed": judge_seed,
                "sample_size": 3,
                "rows": [{"task_id": task_id, "incidents": []} for task_id in judge_sample],
            }
        ),
        encoding="utf-8",
    )

    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit",
            published=export,
            distractor_evidence=b10,
            judge_evidence=b13,
        )
    )

    metrics = {metric["bias_id"]: metric for metric in report["metrics"]}
    assert metrics["B10"]["primary_metric"]["value"] == 1.0
    assert metrics["B10"]["evidence_complete"] is True
    assert metrics["B13"]["primary_metric"]["value"] == 0
    assert metrics["B13"]["evidence_complete"] is True


def test_approved_exception_preserves_failed_verdict_and_changes_status(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["bias_targets"]["difficulty_mix"] = {
        "easy": 1.0,
        "medium": 0.0,
        "hard": 0.0,
    }
    del document["bias_targets"]["tool_call_count_mix"]
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "exceptions": [
                    {
                        "affected_metric": "B2",
                        "owner": "bias-review-board",
                        "rationale": "Approved synthetic golden-fixture deviation.",
                        "approval_date": "2026-09-03",
                    },
                    {
                        "affected_metric": "B5",
                        "owner": "bias-review-board",
                        "rationale": "Approved synthetic missing-target fixture.",
                        "approval_date": "2026-09-03",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit",
            published=export,
            exceptions=exceptions,
        )
    )

    b2 = report["metrics"][1]
    assert b2["bias_id"] == "B2"
    assert b2["primary_metric"]["passed"] is False
    assert b2["exceptions"][0]["owner"] == "bias-review-board"
    assert report["summary"]["status"] == "passed_with_exceptions"
    assert report["summary"]["approved_exception_bias_ids"] == ["B2", "B5"]


def test_contamination_evidence_must_match_run_and_published_task_set(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    baseline = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "baseline",
            published=export,
        )
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["bias_applicability"]["B9"] = {"status": "applicable"}
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contamination = tmp_path / "contamination_report.json"
    contamination.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_run_id": "synthetic-bias-audit",
                "source_task_ids_hash": baseline["source"]["published_task_ids_hash"],
                "publication_allowed": True,
                "candidates": [
                    {
                        "alias": "candidate",
                        "collisions": [],
                        "unresolved": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit",
            published=export,
            contamination_reports=(contamination,),
        )
    )
    b9 = report["metrics"][8]
    assert b9["bias_id"] == "B9"
    assert b9["primary_metric"]["passed"] is True

    bad = json.loads(contamination.read_text(encoding="utf-8"))
    bad["source_task_ids_hash"] = "sha256:" + "f" * 64
    contamination.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BiasAuditError, match="another source task set"):
        build_bias_audit_report(
            AuditInputs(
                run_manifest=manifest,
                output_dir=tmp_path / "bad",
                published=export,
                contamination_reports=(contamination,),
            )
        )


def test_held_out_and_paraphrase_leaks_are_rescanned_in_every_layer(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    rows = _published_rows(export)
    raw_rows = copy.deepcopy(rows)
    raw_rows[0]["x-nemotron"]["metadata"]["fixture_refs"] = [json.dumps(["records", "HOLD-1"])]
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )

    expanded_rows = copy.deepcopy(rows)
    variant = copy.deepcopy(rows[0])
    variant["id"] = "task-paraphrased"
    variant["x-nemotron"]["metadata"].update(
        {
            "variant_index": 1,
            "paraphrase_model": "registry:translator@sha256:" + "b" * 64,
            "surface": {"base_task_id": rows[0]["id"]},
        }
    )
    variant["x-nemotron"]["messages"][0]["content"] = "Please call tool_a."
    expanded_rows.append(variant)
    expanded = tmp_path / "expanded.jsonl"
    expanded.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in expanded_rows),
        encoding="utf-8",
    )

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "tools.json").write_text("[]\n", encoding="utf-8")
    (pack / "templates.yaml").write_text("[]\n", encoding="utf-8")
    (pack / "held_out.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "fixtures": {"records": ["HOLD-1"]},
                "templates": [],
                "policy": {
                    "fixtures_in_backend_state": False,
                    "seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    pack_manifest = pack / "manifest.yaml"
    pack_manifest.write_text(
        yaml.safe_dump(
            {
                "pack_id": "synthetic",
                "version": "1.0.0",
                "paths": {
                    "tools": "tools.json",
                    "templates": "templates.yaml",
                },
                "held_out": "held_out.yaml",
            }
        ),
        encoding="utf-8",
    )

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["bias_applicability"]["B7"] = {"status": "applicable"}
    document["bias_applicability"]["B8"] = {"status": "applicable"}
    document["artifacts"].update(
        {
            "benchmark_raw_parquet": {"content_hash": _sha256(raw)},
            "rendered_conversations": {"content_hash": _sha256(expanded)},
        }
    )
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_bias_audit_report(
        AuditInputs(
            run_manifest=manifest,
            output_dir=tmp_path / "audit",
            published=export,
            raw=raw,
            expanded=expanded,
            pack_manifest=pack_manifest,
        )
    )

    metrics = {metric["bias_id"]: metric for metric in report["metrics"]}
    assert metrics["B7"]["primary_metric"]["value"] == 1
    assert metrics["B7"]["primary_metric"]["passed"] is False
    assert metrics["B7"]["supporting_diagnostics"]["hits_by_layer"]["raw"] == [rows[0]["id"]]
    assert metrics["B8"]["primary_metric"]["value"] >= 1
    assert metrics["B8"]["primary_metric"]["passed"] is False
    assert metrics["B8"]["supporting_diagnostics"]["leaks_by_layer"]["expanded"]


def test_read_only_cli_writes_both_content_addressed_reports(
    tmp_path: Path,
) -> None:
    manifest, export = _write_fixture(tmp_path)
    source_before = {
        path: path.read_bytes()
        for path in (
            manifest,
            export / "BFCL_v4_multi_turn.jsonl",
            export / "possible_answer" / "BFCL_v4_multi_turn.jsonl",
        )
    }
    output = tmp_path / "audit"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemotron.steps.byob.scripts.audit_bfcl_bias",
            "--run-manifest",
            str(manifest),
            "--published",
            str(export),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    report = json.loads((output / "bias_audit_report.json").read_text(encoding="utf-8"))
    assert result["report_hash"] == report["report_hash"]
    assert (output / "bias_audit_report.md").is_file()
    assert {path: path.read_bytes() for path in source_before} == source_before


def test_rendered_cache_restores_only_when_it_matches_manifest(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_row = {
        "task_id": "task-restored",
        "template_id": "template-restored",
        "variant_index": 1,
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "A paraphrased request"},
            {"role": "assistant", "content": "A response"},
        ],
        "expected_tool_calls": [],
        "system_prompt_id": "sha256:system",
        "paraphrase_model": "registry:model@sha256:" + "a" * 64,
        "paraphrase_model_canonical": "registry:model@sha256:" + "a" * 64,
        "metadata": json.dumps(
            {
                "language": "vi",
                "base_task_id": "task-base",
                "surface_source": "model",
                "profile_hash": "sha256:" + "b" * 64,
            }
        ),
    }
    raw = tmp_path / "benchmark_raw.parquet"
    pq.write_table(
        pa.Table.from_pylist([raw_row], schema=benchmark_schema()),
        raw,
    )
    expected_path = tmp_path / "expected.parquet"
    write_stage_table(
        expected_path,
        [_rendered_row(pq.read_table(raw).to_pylist()[0])],
        rendered_conversations_schema(),
    )
    rendered_hash = _sha256(expected_path)
    expected_path.unlink()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "benchmark_raw_parquet": {"content_hash": _sha256(raw)},
                    "rendered_conversations": {"content_hash": rendered_hash},
                }
            }
        ),
        encoding="utf-8",
    )
    restored = tmp_path / "stage_cache" / "rendered_conversations.parquet"

    assert (
        restore_rendered_conversations(
            run_manifest=manifest,
            raw_benchmark=raw,
            output=restored,
        )
        == restored
    )
    assert _sha256(restored) == rendered_hash
    restored_row = pq.read_table(restored).to_pylist()[0]
    assert json.loads(restored_row["turns"]) == [
        {"content": "A paraphrased request", "kind": "user"},
        {"content": "A response", "kind": "assistant_text"},
    ]
