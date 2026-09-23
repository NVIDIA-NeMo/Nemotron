"""Deterministic Stage 10 mapping from recorded Python guards to six-check records."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import ImmutableModelIOCache
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import TURN_POLICIES
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.surface_quality import (
    FORBIDDEN_JUDGE_INPUT_KEYS,
    JUDGE_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    KNOWN_SURFACE_GUARDS,
    SurfaceQualityAuthority,
    apply_surface_quality_policy,
    evaluate_deterministic_checks,
    evaluate_surfaces,
    evaluate_task_surface_quality,
    judge_model_input,
    resolve_surface_quality_authority,
    run_surface_judge,
    run_surface_quality_validation,
    summarize_surface_quality,
    surface_quality_decision,
    surface_quality_report,
    write_surface_quality_artifact,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_quality_contract import (
    DETERMINISTIC_SURFACE_CHECKS,
    JUDGED_SURFACE_CHECKS,
    SURFACE_QUALITY_CHECKS,
    SurfaceQualityCheckResult,
)

BFCL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "bfcl" / "config"


def _task(**overrides: Any) -> dict[str, Any]:
    task = {
        "task_id": "pack__tpl__aaa",
        "base_task_id": "pack__tpl__aaa",
        "template_id": "tpl",
        "variant_index": 0,
        "seed": 7,
        "turn_policy": "single_turn",
    }
    task.update(overrides)
    return task


def _surface(**overrides: Any) -> dict[str, Any]:
    surface = {
        "task_id": "pack__tpl__aaa",
        "base_task_id": "pack__tpl__aaa",
        "source": "template",
        "language": "en",
        "steps": [{"kind": "user", "content": "Please look this up."}],
        "guard_violations": [],
    }
    surface.update(overrides)
    return surface


def _by_check(checks: list) -> dict[str, Any]:
    return {result.check: result for result in checks}


def _write_tiny_config(tmp_path: Path, name: str, **overrides: Any) -> Path:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config_data.get(key), dict):
            config_data[key].update(value)
        else:
            config_data[key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return path


def _judge_config(tmp_path: Path) -> BfclConfig:
    return BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "judge.yaml",
            lineage={
                "judge_advisory": True,
                "roles": {
                    "profile": {"enabled": False, "model_config": None},
                    "paraphrase": {"enabled": False, "model_config": None},
                    "surface_judge": {
                        "enabled": True,
                        "model_config": {
                            "alias": "surface-judge",
                            "provider": "nvidia",
                            "model": "judge-model",
                            "canonical_id": "source::judge-model@revision",
                            "inference_parameters": {"temperature": 0.0},
                        },
                    },
                },
            },
            surface_quality_validation={"enabled": True, "drop_authority": False},
        )
    )


def _quality_config(tmp_path: Path) -> BfclConfig:
    return BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "quality.yaml",
            surface_quality_validation={"enabled": True, "drop_authority": False},
        )
    )


def _authoritative_judge_config(tmp_path: Path) -> BfclConfig:
    config = _judge_config(tmp_path)
    object.__setattr__(config.lineage, "judge_advisory", False)
    config.surface_quality_validation["drop_authority"] = True
    return config


ADVISORY = SurfaceQualityAuthority(quality_enabled=True, judge_enabled=True, drop_authority=False)
AUTHORITATIVE = SurfaceQualityAuthority(quality_enabled=True, judge_enabled=True, drop_authority=True)
PYTHON_ONLY = SurfaceQualityAuthority(quality_enabled=True, judge_enabled=False, drop_authority=False)


def _quality_checks(
    *,
    python_failure: tuple[str, str] | None = None,
    judged: dict[str, dict[str, Any]] | None = None,
) -> list[SurfaceQualityCheckResult]:
    results: list[SurfaceQualityCheckResult] = []
    for check in SURFACE_QUALITY_CHECKS:
        if check in DETERMINISTIC_SURFACE_CHECKS:
            failed = python_failure is not None and python_failure[0] == check
            results.append(
                SurfaceQualityCheckResult(
                    check=check,
                    status="failed" if failed else "passed",
                    source="python",
                    reason_code=python_failure[1] if failed else None,
                )
            )
            continue
        spec = (judged or {}).get(check) or {"status": "not_run"}
        results.append(
            SurfaceQualityCheckResult(
                check=check,
                status=spec["status"],
                source="surface_judge",
                reason_code=spec.get("reason_code"),
            )
        )
    return results


def _judged(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    judged = {check: {"status": "passed"} for check in JUDGED_SURFACE_CHECKS}
    judged.update(overrides)
    return judged


def _judge_errors(reason_code: str) -> dict[str, dict[str, Any]]:
    return {check: {"status": "error", "reason_code": reason_code} for check in JUDGED_SURFACE_CHECKS}


def _quality_record(task_id: str = "pack__tpl__aaa", **kwargs: Any) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "template_id": "tpl",
        "turn_policy": "single_turn",
        "checks": _quality_checks(**kwargs),
    }


def _passing_judge_response() -> dict[str, Any]:
    return {
        "language_locale": {"passed": True},
        "fluency_naturalness": {"passed": True},
        "clarity_coherence": {"passed": True},
    }


def test_clean_template_surface_passes_python_checks_and_skips_the_judge() -> None:
    checks = evaluate_task_surface_quality(_task(), _surface())

    assert [result.check for result in checks] == list(SURFACE_QUALITY_CHECKS)
    python = [result for result in checks if result.check in DETERMINISTIC_SURFACE_CHECKS]
    judged = [result for result in checks if result.check in JUDGED_SURFACE_CHECKS]
    assert all(result.status == "passed" and result.source == "python" for result in python)
    assert all(result.status == "not_run" and result.source == "surface_judge" for result in judged)
    assert all("passed" not in result.model_dump() for result in judged)


def test_each_python_guard_maps_to_the_owned_check_and_reason() -> None:
    cases = [
        (
            {"guard": "semantic_shape", "reason": "empty_user_turn"},
            "surface_shape",
            "empty_user_turn",
            None,
        ),
        (
            {
                "guard": "semantic_shape",
                "reason": "user_turn_count_changed",
                "expected": 2,
                "actual": 1,
            },
            "surface_shape",
            "user_turn_count_changed",
            None,
        ),
        (
            {"guard": "must_preserve", "slot": "account_id"},
            "semantic_preservation",
            "must_preserve",
            "account_id",
        ),
        (
            {"guard": "must_omit", "slot": "pin"},
            "semantic_preservation",
            "must_omit",
            "pin",
        ),
        (
            {"guard": "novel_literal", "value": "ACCT-99"},
            "semantic_preservation",
            "novel_literal",
            None,
        ),
        (
            {"guard": "must_not_mention", "tool": "lookup_book"},
            "leakage",
            "tool_name_leakage",
            "lookup_book",
        ),
        (
            {"guard": "must_not_mention", "phrase": "oracle"},
            "leakage",
            "forbidden_mention",
            "oracle",
        ),
        (
            {"guard": "expected_result_leakage", "value": "500000"},
            "leakage",
            "expected_result_leakage",
            None,
        ),
    ]
    for violation, check, reason, evidence in cases:
        result = _by_check(evaluate_deterministic_checks([violation], surface_source="model"))[check]
        assert result.status == "failed"
        assert result.source == "python"
        assert result.reason_code == reason
        assert result.evidence == evidence


def test_first_failure_per_check_is_authoritative() -> None:
    checks = evaluate_deterministic_checks(
        [
            {"guard": "must_omit", "slot": "pin"},
            {"guard": "must_preserve", "slot": "account_id"},
            {"guard": "must_not_mention", "tool": "lookup_book"},
            {"guard": "expected_result_leakage", "value": "500000"},
        ],
        surface_source="model",
    )
    by_check = _by_check(checks)

    assert by_check["semantic_preservation"].reason_code == "must_omit"
    assert by_check["semantic_preservation"].evidence == "pin"
    assert by_check["leakage"].reason_code == "tool_name_leakage"
    assert by_check["leakage"].evidence == "lookup_book"
    assert by_check["surface_shape"].status == "passed"


def test_unchanged_surface_fails_only_for_model_rewrites() -> None:
    violation = {"guard": "semantic_shape", "reason": "unchanged_surface"}

    template = _by_check(evaluate_deterministic_checks([violation], surface_source="template"))["surface_shape"]
    model = _by_check(evaluate_deterministic_checks([violation], surface_source="model"))["surface_shape"]

    assert template.status == "passed"
    assert model.status == "failed"
    assert model.reason_code == "unchanged_surface"


def test_unknown_surface_source_cannot_bypass_unchanged_check() -> None:
    with pytest.raises(ValueError, match="unknown surface source"):
        evaluate_deterministic_checks(
            [{"guard": "semantic_shape", "reason": "unchanged_surface"}],
            surface_source="modle",
        )


def test_expected_result_truth_is_not_copied_into_quality_evidence() -> None:
    result = _by_check(
        evaluate_deterministic_checks(
            [{"guard": "expected_result_leakage", "value": "secret-result"}],
            surface_source="model",
        )
    )["leakage"]

    assert result.status == "failed"
    assert result.reason_code == "expected_result_leakage"
    assert result.evidence is None


def test_novel_literal_is_not_copied_into_quality_evidence() -> None:
    result = _by_check(
        evaluate_deterministic_checks(
            [{"guard": "novel_literal", "value": "SECRET-RESULT"}],
            surface_source="model",
        )
    )["semantic_preservation"]

    assert result.status == "failed"
    assert result.reason_code == "novel_literal"
    assert result.evidence is None


def test_evaluate_surfaces_writes_one_complete_record_per_task() -> None:
    canonical = _task()
    variant = _task(
        task_id="pack__tpl__bbb",
        variant_index=1,
        turn_policy="clarify_only",
    )
    records = evaluate_surfaces(
        [canonical, variant],
        {
            canonical["task_id"]: _surface(),
            variant["task_id"]: _surface(
                task_id=variant["task_id"],
                source="model",
                guard_violations=[
                    {"guard": "must_not_mention", "tool": "create_transfer"},
                ],
            ),
        },
    )

    assert [record["task_id"] for record in records] == [
        canonical["task_id"],
        variant["task_id"],
    ]
    assert records[0]["surface_source"] == "template"
    assert records[1]["turn_policy"] == "clarify_only"
    canonical_checks = {item["check"]: item for item in records[0]["checks"]}
    variant_checks = {item["check"]: item for item in records[1]["checks"]}
    assert canonical_checks["leakage"]["status"] == "passed"
    assert variant_checks["leakage"] == {
        "check": "leakage",
        "status": "failed",
        "source": "python",
        "reason_code": "tool_name_leakage",
        "evidence": "create_transfer",
    }
    assert variant_checks["clarity_coherence"]["status"] == "not_run"


def test_unknown_turn_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown turn_policy"):
        evaluate_task_surface_quality(_task(turn_policy="clarify-only"), _surface())
    for policy in TURN_POLICIES:
        evaluate_task_surface_quality(_task(turn_policy=policy), _surface())


def test_unmapped_or_malformed_guards_are_rejected() -> None:
    with pytest.raises(ValueError, match="unmapped surface guard"):
        evaluate_deterministic_checks(
            [{"guard": "underspecified_request"}],
            surface_source="model",
        )
    with pytest.raises(ValueError, match="missing a guard name"):
        evaluate_deterministic_checks([{"slot": "pin"}], surface_source="model")
    with pytest.raises(ValueError, match="must be mappings"):
        evaluate_deterministic_checks(["must_omit"], surface_source="model")
    with pytest.raises(ValueError, match="unknown reason"):
        evaluate_deterministic_checks(
            [{"guard": "semantic_shape", "reason": "reordered_turns"}],
            surface_source="model",
        )
    with pytest.raises(ValueError, match="tool or phrase"):
        evaluate_deterministic_checks(
            [{"guard": "must_not_mention"}],
            surface_source="model",
        )
    with pytest.raises(ValueError, match="guard_violations must be a list"):
        evaluate_task_surface_quality(
            _task(),
            _surface(guard_violations={}),
        )


def test_enabled_judge_requires_supplied_checks() -> None:
    with pytest.raises(ValueError, match="requires judge_checks"):
        evaluate_task_surface_quality(_task(), _surface(), judge_enabled=True)
    with pytest.raises(ValueError, match="requires judge_checks_by_task"):
        evaluate_surfaces([_task()], {_task()["task_id"]: _surface()}, judge_enabled=True)


def test_clarify_only_ambiguous_reference_is_remapped_before_assembly() -> None:
    checks = evaluate_task_surface_quality(
        _task(turn_policy="clarify_only"),
        _surface(),
        judge_enabled=True,
        judge_checks=[
            SurfaceQualityCheckResult(
                check="language_locale",
                status="passed",
                source="surface_judge",
            ),
            SurfaceQualityCheckResult(
                check="fluency_naturalness",
                status="passed",
                source="surface_judge",
            ),
            SurfaceQualityCheckResult(
                check="clarity_coherence",
                status="failed",
                source="surface_judge",
                reason_code="ambiguous_reference",
            ),
        ],
    )
    remapped = _by_check(checks)["clarity_coherence"]
    assert remapped.status == "not_applicable"
    assert remapped.reason_code == "ambiguous_reference"


def test_judge_model_input_is_surface_only() -> None:
    payload = judge_model_input(
        _surface(
            steps=[
                {"kind": "user", "content": "Please look this up."},
                {
                    "kind": "calls",
                    "call_group": 0,
                    "function_name": "lookup_book",
                    "tools": ["lookup_book"],
                },
                {"kind": "assistant_text", "content": "Which title should I use?"},
            ]
        )
    )

    assert set(payload) == {
        "language",
        "turns",
        "surface_hash",
        "style_hints",
        "style_avoid",
        "rubric",
    }
    assert payload["turns"] == [
        {"role": "user", "content": "Please look this up."},
        {"role": "assistant", "content": "Which title should I use?"},
    ]

    def keys(value: Any) -> list[str]:
        if isinstance(value, dict):
            nested = [item for child in value.values() for item in keys(child)]
            return [str(key) for key in value] + nested
        if isinstance(value, list):
            return [item for child in value for item in keys(child)]
        return []

    assert FORBIDDEN_JUDGE_INPUT_KEYS.isdisjoint(keys(payload))
    assert "do not identify tools" in JUDGE_SYSTEM_PROMPT.lower()
    assert "{{ model_input }}" in JUDGE_PROMPT


def test_run_surface_judge_skips_the_model_when_disabled(tmp_path: Path) -> None:
    config = BfclConfig.from_yaml(_write_tiny_config(tmp_path, "no-judge.yaml"))
    results = run_surface_judge(
        config,
        [_task()],
        {_task()["task_id"]: _surface()},
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disabled judge called the model")),
    )
    judged = results[_task()["task_id"]]
    assert all(result.status == "not_run" for result in judged)


def test_run_surface_judge_never_sees_deterministically_rejected_truth(
    tmp_path: Path,
) -> None:
    config = _judge_config(tmp_path)
    task = _task()
    surface = _surface(
        steps=[{"kind": "user", "content": "The result is SECRET-RESULT."}],
        guard_violations=[
            {
                "guard": "expected_result_leakage",
                "value": "SECRET-RESULT",
            }
        ],
    )
    stale_cache = ImmutableModelIOCache(tmp_path / "shared-judge-cache.jsonl")
    stale_cache.put(
        "sha256:unrelated-request",
        _passing_judge_response(),
        model_canonical="source::old-judge@revision",
        input_hash="sha256:unrelated-input",
    )

    results = run_surface_judge(
        config,
        [task],
        {task["task_id"]: surface},
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deterministically rejected surface reached the judge")
        ),
        io_cache=stale_cache,
    )

    assert {result.status for result in results[task["task_id"]]} == {"not_run"}
    stage_cache = Path(config.output_dir) / config.expt_name / "stage_cache"
    cache_path = stage_cache / "surface_judge_io_cache.jsonl"
    assert not cache_path.exists()
    usage = json.loads((stage_cache / "surface_judge_cache_usage.json").read_text(encoding="utf-8"))
    assert usage["requests"] == []
    assert usage["model_canonical"] == "source::judge-model@revision"
    assert stale_cache.path.is_file()


def test_run_surface_judge_rejects_profile_language_mismatch(
    tmp_path: Path,
) -> None:
    config = _judge_config(tmp_path)
    object.__setattr__(
        config.lineage,
        "profile_influenced_surface",
        True,
    )
    profile = {
        "status": "completed",
        "languages": ["vi"],
        "style_hints": ["Use concise wording."],
        "avoid": [],
    }

    with pytest.raises(ValueError, match="profile language must match"):
        run_surface_judge(
            config,
            [_task()],
            {_task()["task_id"]: _surface(language="en")},
            profile=profile,
            model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("language-mismatched surface reached the judge")
            ),
        )


def test_run_surface_judge_contains_schema_invalid_cache_hits(
    tmp_path: Path,
) -> None:
    config = _judge_config(tmp_path)
    task = _task()

    class InvalidCache:
        def get(self, key: str) -> dict[str, Any]:
            del key
            return {"tool_correctness": {"passed": True}}

        def put(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("invalid cache hit was replaced")

    results = run_surface_judge(
        config,
        [task],
        {task["task_id"]: _surface()},
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid cache hit called the model")
        ),
        io_cache=InvalidCache(),  # type: ignore[arg-type]
    )

    judged = results[task["task_id"]]
    assert {result.status for result in judged} == {"error"}
    assert {result.reason_code for result in judged} == {"invalid_response"}


def test_run_surface_judge_caches_valid_responses_only(tmp_path: Path) -> None:
    config = _judge_config(tmp_path)
    cache = ImmutableModelIOCache(tmp_path / "surface_judge_io_cache.jsonl")
    task = _task()
    surfaces = {task["task_id"]: _surface()}
    calls = {"count": 0}

    def invalid_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        return {
            request["request_id"]: {
                "language_locale": {"passed": True},
                "fluency_naturalness": {"passed": True},
                "clarity_coherence": {"passed": True},
                "tool_correctness": {"passed": True},
            }
            for request in kwargs["requests"]
        }

    invalid = run_surface_judge(
        config,
        [task],
        surfaces,
        model_runner=invalid_runner,
        io_cache=cache,
    )
    assert {result.status for result in invalid[task["task_id"]]} == {"error"}
    assert {result.reason_code for result in invalid[task["task_id"]]} == {"invalid_response"}
    assert not cache.path.exists() or not cache.path.read_text(encoding="utf-8").strip()

    def fake_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        calls["count"] += 1
        payload = _passing_judge_response()
        assert "do not identify tools" in kwargs["system_prompt"].lower()
        for request in kwargs["requests"]:
            model_input = request["model_input"]
            assert "expected_tool_calls" not in model_input
            assert "success_assertions" not in model_input
        return {request["request_id"]: payload for request in kwargs["requests"]}

    first = run_surface_judge(
        config,
        [task],
        surfaces,
        model_runner=fake_runner,
        io_cache=cache,
    )
    assert all(result.status == "passed" for result in first[task["task_id"]])
    cached_entry = json.loads(cache.path.read_text(encoding="utf-8").strip())
    assert cached_entry["response"] == {
        "language_locale": {"passed": True, "reason_code": None},
        "fluency_naturalness": {"passed": True, "reason_code": None},
        "clarity_coherence": {"passed": True, "reason_code": None},
    }
    cached = run_surface_judge(
        config,
        [task],
        surfaces,
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit called the model")),
        io_cache=cache,
    )
    assert [result.status for result in cached[task["task_id"]]] == [
        result.status for result in first[task["task_id"]]
    ]
    assert calls["count"] == 1
    usage_path = Path(config.output_dir) / config.expt_name / "stage_cache" / "surface_judge_cache_usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    assert len(usage["requests"]) == 1
    assert usage["requests"][0]["source"] == "cache"
    assert usage["requests"][0]["status"] == "completed"
    assert usage["requests"][0]["observed_response_hash"].startswith("sha256:")


def test_run_surface_judge_records_errors_without_poisoning_the_cache(tmp_path: Path) -> None:
    config = _judge_config(tmp_path)
    cache = ImmutableModelIOCache(tmp_path / "judge-errors.jsonl")
    task = _task()

    def exploding_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args, kwargs
        raise RuntimeError("endpoint down")

    results = run_surface_judge(
        config,
        [task],
        {_task()["task_id"]: _surface()},
        model_runner=exploding_runner,
        io_cache=cache,
    )
    judged = results[task["task_id"]]
    assert {result.status for result in judged} == {"error"}
    assert {result.reason_code for result in judged} == {"judge_error"}
    assert not cache.path.exists() or not cache.path.read_text(encoding="utf-8").strip()


def test_evaluate_surfaces_assembles_judge_results() -> None:
    task = _task()
    passing = [
        SurfaceQualityCheckResult(check=check, status="passed", source="surface_judge")
        for check in SURFACE_QUALITY_CHECKS
        if check in JUDGED_SURFACE_CHECKS
    ]
    records = evaluate_surfaces(
        [task],
        {task["task_id"]: _surface()},
        judge_enabled=True,
        judge_checks_by_task={task["task_id"]: passing},
    )
    by_check = {item["check"]: item for item in records[0]["checks"]}
    assert by_check["fluency_naturalness"]["status"] == "passed"
    assert by_check["leakage"]["status"] == "passed"


def test_missing_surface_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing a surface"):
        evaluate_surfaces([_task()], {})


def test_authority_is_read_from_the_config(tmp_path: Path) -> None:
    template_only = resolve_surface_quality_authority(BfclConfig.from_yaml(_write_tiny_config(tmp_path, "plain.yaml")))
    assert template_only == SurfaceQualityAuthority(
        quality_enabled=False,
        judge_enabled=False,
        drop_authority=False,
    )
    assert template_only.judge_advisory is None

    advisory = resolve_surface_quality_authority(_judge_config(tmp_path))
    assert advisory == ADVISORY
    assert advisory.judge_advisory is True

    authoritative = resolve_surface_quality_authority(_authoritative_judge_config(tmp_path))
    assert authoritative == AUTHORITATIVE
    assert authoritative.judge_advisory is False


def test_authority_refuses_a_judge_that_cannot_be_honored(tmp_path: Path) -> None:
    orphan_drop = _judge_config(tmp_path)
    object.__setattr__(orphan_drop.lineage.roles["surface_judge"], "enabled", False)
    orphan_drop.surface_quality_validation["drop_authority"] = True
    with pytest.raises(ValueError, match="drop_authority requires an enabled surface judge"):
        resolve_surface_quality_authority(orphan_drop)

    unrecorded_stage = _judge_config(tmp_path)
    unrecorded_stage.surface_quality_validation["enabled"] = False
    with pytest.raises(ValueError, match="enabled must be true when the surface judge is enabled"):
        resolve_surface_quality_authority(unrecorded_stage)

    mislabeled = _judge_config(tmp_path)
    mislabeled.surface_quality_validation["drop_authority"] = True
    with pytest.raises(ValueError, match="judge_advisory must equal the inverse"):
        resolve_surface_quality_authority(mislabeled)


def test_python_failures_drop_the_row_under_every_authority() -> None:
    for authority in (PYTHON_ONLY, ADVISORY, AUTHORITATIVE):
        # An enabled judge never scores a surface Python already rejected, so the
        # judged checks stay not_run under every authority here.
        decision = surface_quality_decision(
            _quality_checks(python_failure=("leakage", "tool_name_leakage")),
            authority=authority,
            turn_policy="single_turn",
        )
        assert decision["decision"] == "dropped"
        assert decision["drop_source"] == "python"
        assert decision["drop_reasons"] == ["leakage:tool_name_leakage"]
        assert decision["advisory_failures"] == []


def test_an_advisory_judge_failure_is_recorded_but_keeps_the_row() -> None:
    decision = surface_quality_decision(
        _quality_checks(judged=_judged(fluency_naturalness={"status": "failed", "reason_code": "unnatural_wording"})),
        authority=ADVISORY,
        turn_policy="single_turn",
    )

    assert decision["decision"] == "kept"
    assert decision["drop_source"] is None
    assert decision["drop_reasons"] == []
    assert decision["advisory_failures"] == ["fluency_naturalness:unnatural_wording"]


def test_an_authoritative_judge_failure_drops_the_row() -> None:
    decision = surface_quality_decision(
        _quality_checks(judged=_judged(fluency_naturalness={"status": "failed", "reason_code": "unnatural_wording"})),
        authority=AUTHORITATIVE,
        turn_policy="single_turn",
    )

    assert decision["decision"] == "dropped"
    assert decision["drop_source"] == "surface_judge"
    assert decision["drop_reasons"] == ["fluency_naturalness:unnatural_wording"]
    assert decision["advisory_failures"] == []


def test_a_not_applicable_judgement_neither_passes_nor_drops() -> None:
    decision = surface_quality_decision(
        _quality_checks(
            judged=_judged(clarity_coherence={"status": "not_applicable", "reason_code": "ambiguous_reference"})
        ),
        authority=AUTHORITATIVE,
        turn_policy="clarify_only",
    )

    assert decision["decision"] == "kept"
    assert decision["drop_reasons"] == []
    assert decision["advisory_failures"] == []


def test_not_applicable_cannot_bypass_a_different_turn_policy() -> None:
    checks = _quality_checks(
        judged=_judged(clarity_coherence={"status": "not_applicable", "reason_code": "ambiguous_reference"})
    )

    with pytest.raises(ValueError, match="only when allowed by turn_policy"):
        surface_quality_decision(checks, authority=AUTHORITATIVE, turn_policy="single_turn")
    with pytest.raises(ValueError, match="only when allowed by turn_policy"):
        apply_surface_quality_policy(
            [{**_quality_record(judged=_judged()), "checks": checks}],
            authority=AUTHORITATIVE,
        )


def test_authoritative_judge_failures_are_not_reported_as_advisory_when_python_also_fails() -> None:
    decision = surface_quality_decision(
        _quality_checks(
            python_failure=("leakage", "tool_name_leakage"),
            judged=_judged(fluency_naturalness={"status": "failed", "reason_code": "unnatural_wording"}),
        ),
        authority=AUTHORITATIVE,
        turn_policy="single_turn",
    )

    assert decision["drop_source"] == "python"
    assert decision["advisory_failures"] == []


def test_a_judge_error_never_decides_the_row() -> None:
    for authority in (ADVISORY, AUTHORITATIVE):
        decision = surface_quality_decision(
            _quality_checks(judged=_judge_errors("judge_error")),
            authority=authority,
            turn_policy="single_turn",
        )
        assert decision["decision"] == "kept"
        assert decision["drop_reasons"] == []
        assert decision["advisory_failures"] == []
        assert decision["judge_error"] == "judge_error"


def test_an_authoritative_judge_error_blocks_publication() -> None:
    records = [
        _quality_record("pack__tpl__aaa", judged=_judged()),
        _quality_record("pack__tpl__bbb", judged=_judge_errors("missing_response")),
    ]

    advisory = apply_surface_quality_policy(records, authority=ADVISORY)
    assert [record["decision"] for record in advisory] == ["kept", "kept"]
    assert advisory[1]["judge_error"] == "missing_response"

    with pytest.raises(RuntimeError, match="must not publish"):
        apply_surface_quality_policy(records, authority=AUTHORITATIVE)


def test_a_decision_requires_a_complete_and_consistent_check_set() -> None:
    with pytest.raises(ValueError, match="missing checks"):
        surface_quality_decision(_quality_checks()[:-1], authority=PYTHON_ONLY, turn_policy="single_turn")
    with pytest.raises(ValueError, match="require surface_quality_validation.enabled"):
        surface_quality_decision(
            _quality_checks(),
            authority=SurfaceQualityAuthority(quality_enabled=False, judge_enabled=False, drop_authority=False),
            turn_policy="single_turn",
        )
    with pytest.raises(ValueError, match="must be not_run when the surface judge is disabled"):
        surface_quality_decision(
            _quality_checks(judged={"clarity_coherence": {"status": "passed"}}),
            authority=PYTHON_ONLY,
            turn_policy="single_turn",
        )
    with pytest.raises(ValueError, match="left judged checks not_run"):
        surface_quality_decision(_quality_checks(), authority=ADVISORY, turn_policy="single_turn")


def test_the_policy_summary_separates_advisory_signal_from_drops() -> None:
    decided = apply_surface_quality_policy(
        [
            _quality_record("pack__tpl__aaa", judged=_judged()),
            _quality_record("pack__tpl__bbb", python_failure=("leakage", "tool_name_leakage")),
            _quality_record(
                "pack__tpl__ccc",
                judged=_judged(fluency_naturalness={"status": "failed", "reason_code": "unnatural_wording"}),
            ),
        ],
        authority=ADVISORY,
    )

    assert summarize_surface_quality(decided, authority=ADVISORY) == {
        "evaluated": 3,
        "kept": 2,
        "dropped_by_python": 1,
        "dropped_by_surface_judge": 0,
        "judge_enabled": True,
        "judge_advisory": True,
        "drop_reason_counts": {"leakage:tool_name_leakage": 1},
        "advisory_failure_counts": {"fluency_naturalness:unnatural_wording": 1},
        "judge_error_counts": {},
    }


def test_disabled_judge_report_uses_null_advisory_state() -> None:
    decided = apply_surface_quality_policy(
        [{**_quality_record("pack__tpl__aaa"), "variant_index": 0}],
        authority=PYTHON_ONLY,
    )

    report = surface_quality_report(decided, authority=PYTHON_ONLY)

    assert report["judge_enabled"] is False
    assert report["judge_advisory"] is None
    assert report["judge_prompt_version"] is None
    assert report["judge_prompt_hash"] is None


def test_surface_quality_report_groups_rejections_by_template_and_variant() -> None:
    decided = apply_surface_quality_policy(
        [
            {
                **_quality_record(
                    "generic__alpha__001",
                    python_failure=("leakage", "tool_name_leakage"),
                ),
                "variant_index": 1,
            },
            {
                **_quality_record("generic__alpha__002", judged=_judged()),
                "variant_index": 2,
            },
        ],
        authority=ADVISORY,
    )

    report = surface_quality_report(decided, authority=ADVISORY)

    assert report["by_template"]["tpl"]["evaluated"] == 2
    assert report["by_template"]["tpl"]["dropped"] == 1
    assert report["by_template"]["tpl"]["drop_reason_counts"] == {"leakage:tool_name_leakage": 1}
    assert report["by_variant_index"]["1"]["dropped"] == 1
    assert report["by_variant_index"]["2"]["kept"] == 1


def test_run_surface_quality_validation_writes_stage_outputs(tmp_path: Path) -> None:
    config = _quality_config(tmp_path)
    task = _task()

    decided, report = run_surface_quality_validation(
        config,
        [task],
        {task["task_id"]: _surface()},
    )

    cache = Path(config.output_dir) / config.expt_name / "stage_cache"
    assert decided[0]["decision"] == "kept"
    assert report["evaluated"] == 1
    assert report["kept"] == 1
    assert (cache / "surface_validated_tasks.parquet").is_file()
    stored_report = json.loads((cache / "surface_quality_rejections.json").read_text(encoding="utf-8"))
    assert stored_report == report


@pytest.mark.parametrize(
    ("authoritative", "expected_decision", "expected_advisory"),
    [
        (False, "kept", 1),
        (True, "dropped", 0),
    ],
)
def test_run_surface_quality_validation_applies_judge_authority(
    tmp_path: Path,
    authoritative: bool,
    expected_decision: str,
    expected_advisory: int,
) -> None:
    config = _authoritative_judge_config(tmp_path) if authoritative else _judge_config(tmp_path)
    task = _task()

    def failing_judge(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        payload = {
            "language_locale": {"passed": True},
            "fluency_naturalness": {
                "passed": False,
                "reason_code": "unnatural_wording",
            },
            "clarity_coherence": {"passed": True},
        }
        return {request["request_id"]: payload for request in kwargs["requests"]}

    decided, report = run_surface_quality_validation(
        config,
        [task],
        {task["task_id"]: _surface()},
        model_runner=failing_judge,
    )

    assert decided[0]["decision"] == expected_decision
    assert report["advisory_failure_counts"].get("fluency_naturalness:unnatural_wording", 0) == expected_advisory
    assert report["dropped_by_surface_judge"] == int(authoritative)


def test_surface_quality_artifact_is_explicit_joinable_and_truth_safe(tmp_path: Path) -> None:
    config = _judge_config(tmp_path)
    decided = apply_surface_quality_policy(
        [
            _quality_record("generic__alpha__001", judged=_judged()),
            _quality_record(
                "generic__beta__002",
                python_failure=("leakage", "expected_result_leakage"),
            ),
        ],
        authority=ADVISORY,
    )

    path = write_surface_quality_artifact(config, decided)

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.to_pylist()
    assert path.name == "surface_validated_tasks.parquet"
    assert table.num_rows == 2
    assert [row["task_id"] for row in rows] == ["generic__alpha__001", "generic__beta__002"]
    assert rows[0]["accepted"] is True
    assert rows[0]["language_locale_status"] == "passed"
    assert rows[1]["accepted"] is False
    assert rows[1]["drop_source"] == "python"
    assert rows[1]["drop_reasons"] == ["leakage:expected_result_leakage"]
    checks = json.loads(rows[1]["checks"])
    assert [item["check"] for item in checks] == list(SURFACE_QUALITY_CHECKS)
    assert next(item for item in checks if item["check"] == "leakage")["evidence"] is None


def test_surface_quality_artifact_rejects_duplicate_or_inconsistent_rows(tmp_path: Path) -> None:
    config = _judge_config(tmp_path)
    record = apply_surface_quality_policy(
        [_quality_record("generic__alpha__001", judged=_judged())],
        authority=ADVISORY,
    )[0]

    with pytest.raises(ValueError, match="task_id values must be unique"):
        write_surface_quality_artifact(config, [record, record])
    with pytest.raises(ValueError, match="kept surface-quality row cannot have drop_source"):
        write_surface_quality_artifact(
            config,
            [{**record, "drop_source": "python"}],
        )


def test_surface_quality_artifact_writes_an_empty_typed_table(tmp_path: Path) -> None:
    config = _judge_config(tmp_path)
    path = write_surface_quality_artifact(config, [])

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.num_rows == 0
    assert table.schema.names == [
        "task_id",
        "base_task_id",
        "template_id",
        "variant_index",
        "surface_source",
        "turn_policy",
        "contract_version",
        "accepted",
        "decision",
        "drop_source",
        "drop_reasons",
        "advisory_failures",
        "judge_error",
        "surface_shape_status",
        "semantic_preservation_status",
        "leakage_status",
        "language_locale_status",
        "fluency_naturalness_status",
        "clarity_coherence_status",
        "checks",
    ]


def test_known_guards_cover_render_and_paraphrase_emitters() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import paraphrase, render

    emitted: set[str] = set()
    for module in (render, paraphrase):
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "guard"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    emitted.add(value.value)
    assert emitted == KNOWN_SURFACE_GUARDS
