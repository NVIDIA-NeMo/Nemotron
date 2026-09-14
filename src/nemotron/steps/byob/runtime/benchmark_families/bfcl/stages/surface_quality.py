# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map existing Python surface guards onto the six-check quality contract.

Stage 10's deterministic half reuses the guards render and paraphrase already
recorded. The optional surface judge owns only language, fluency, and clarity.
Neither half inspects oracle truth or decides tool correctness.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
    request_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import TURN_POLICIES
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    SURFACE_VALIDATED_TASKS,
    surface_validated_task_row,
    surface_validated_tasks_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_quality_contract import (
    DETERMINISTIC_SURFACE_CHECKS,
    INAPPLICABLE_FAILURES_BY_TURN_POLICY,
    JUDGED_SURFACE_CHECKS,
    SURFACE_QUALITY_CHECKS,
    SURFACE_QUALITY_CONTRACT_VERSION,
    SURFACE_QUALITY_REASON_CODES,
    SurfaceJudgeResult,
    SurfaceQualityCheckName,
    SurfaceQualityCheckResult,
    judge_error_checks,
    not_run_judge_checks,
    validate_complete_check_set,
)

logger = logging.getLogger(__name__)

# Every guard the render/paraphrase stages currently emit. An unlisted name is a
# contract drift: quality must not silently pass a new failure mode.
KNOWN_SURFACE_GUARDS = frozenset(
    {
        "expected_result_leakage",
        "must_not_mention",
        "must_omit",
        "must_preserve",
        "novel_literal",
        "semantic_shape",
    }
)
SURFACE_SOURCES = frozenset({"model", "template"})
_EVIDENCE_KEYS = ("slot", "tool", "phrase", "value")
_SHAPE_REASONS = frozenset(
    {
        "empty_user_turn",
        "unchanged_surface",
        "user_turn_count_changed",
    }
)
FORBIDDEN_JUDGE_INPUT_KEYS = frozenset(
    {
        "assertion",
        "assertions",
        "assertion_verdict",
        "backend_state",
        "call_group",
        "expected_result",
        "expected_tool_calls",
        "fixture_refs",
        "function_name",
        "oracle_state",
        "required_tools",
        "slots",
        "success_assertions",
        "tool_calls",
        "tools",
    }
)
JUDGE_PROMPT_VERSION = "bfcl-surface-judge-v1"
JUDGE_SYSTEM_PROMPT = """Judge only the language quality of the supplied conversation surface.
Evaluate language_locale, fluency_naturalness, and clarity_coherence.
Do not identify tools, arguments, expected results, backend state, or assertions.
Do not rewrite the conversation. Return only the requested structured verdicts."""
JUDGE_PROMPT = """Score this canonical JSON surface against the supplied rubric:
{{ model_input }}"""
JudgeRunner = Callable[..., dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class SurfaceQualityAuthority:
    """Who may drop a row once the six checks are assembled.

    Python failures always carry drop authority because they protect semantics
    and leakage. The judge drops rows only when the run explicitly grants it
    that authority; otherwise its verdicts are advisory.
    """

    quality_enabled: bool
    judge_enabled: bool
    drop_authority: bool

    @property
    def judge_advisory(self) -> bool | None:
        return None if not self.judge_enabled else not self.drop_authority


def resolve_surface_quality_authority(config: BfclConfig) -> SurfaceQualityAuthority:
    """Read the authority policy, refusing combinations no stage can honor."""
    quality = config.surface_quality_validation or {}
    quality_enabled = bool(quality.get("enabled", False))
    drop_authority = bool(quality.get("drop_authority", False))
    role = (config.lineage.roles or {}).get("surface_judge")
    judge_enabled = bool(role and role.enabled)
    if judge_enabled and not quality_enabled:
        raise ValueError("surface_quality_validation.enabled must be true when the surface judge is enabled")
    if drop_authority and not judge_enabled:
        raise ValueError("surface_quality_validation.drop_authority requires an enabled surface judge")
    if judge_enabled and config.lineage.judge_advisory is not (not drop_authority):
        raise ValueError(
            "lineage.judge_advisory must equal the inverse of "
            "surface_quality_validation.drop_authority when the surface judge is enabled"
        )
    return SurfaceQualityAuthority(
        quality_enabled=quality_enabled,
        judge_enabled=judge_enabled,
        drop_authority=drop_authority,
    )


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


JUDGE_PROMPT_HASH = _sha256(JUDGE_PROMPT_VERSION + "\n" + JUDGE_SYSTEM_PROMPT + "\n" + JUDGE_PROMPT)


def _evidence_token(violation: Mapping[str, Any]) -> str | None:
    """Return the identifier a Python failure may store, or None."""
    for key in _EVIDENCE_KEYS:
        value = violation.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and not isinstance(value, (str, bool)):
            text = str(value)
            if text:
                return text
    return None


def _map_violation(
    violation: Mapping[str, Any],
    *,
    surface_source: str,
) -> tuple[SurfaceQualityCheckName, str] | None:
    """Return ``(check, reason_code)`` or None when the guard does not apply."""
    guard = violation.get("guard")
    if not isinstance(guard, str) or not guard:
        raise ValueError("surface-quality guard violation is missing a guard name")
    if guard not in KNOWN_SURFACE_GUARDS:
        raise ValueError(f"unmapped surface guard {guard!r}")
    if guard == "semantic_shape":
        reason = violation.get("reason")
        if reason not in _SHAPE_REASONS:
            raise ValueError(f"semantic_shape violation has unknown reason {reason!r}")
        # Canonical template surfaces are unchanged by definition; only a model
        # rewrite that copied the source wording is a quality failure.
        if reason == "unchanged_surface" and surface_source != "model":
            return None
        return ("surface_shape", str(reason))
    if guard == "must_preserve":
        return ("semantic_preservation", "must_preserve")
    if guard == "must_omit":
        return ("semantic_preservation", "must_omit")
    if guard == "novel_literal":
        return ("semantic_preservation", "novel_literal")
    if guard == "must_not_mention":
        if "tool" in violation:
            return ("leakage", "tool_name_leakage")
        if "phrase" in violation:
            return ("leakage", "forbidden_mention")
        raise ValueError("must_not_mention requires tool or phrase evidence")
    if guard == "expected_result_leakage":
        return ("leakage", "expected_result_leakage")
    raise AssertionError(f"known surface guard {guard!r} has no quality mapping")


def evaluate_deterministic_checks(
    violations: Sequence[Mapping[str, Any]],
    *,
    surface_source: str,
) -> list[SurfaceQualityCheckResult]:
    """Project recorded guards onto the three Python-owned checks.

    The first applicable failure for a check is authoritative. Later failures on
    the same check stay in the guard list for inspection but do not replace the
    stored reason code.
    """
    if surface_source not in SURFACE_SOURCES:
        raise ValueError(
            f"unknown surface source {surface_source!r}; expected one of " + ", ".join(sorted(SURFACE_SOURCES))
        )
    failures: dict[SurfaceQualityCheckName, SurfaceQualityCheckResult] = {}
    for violation in violations:
        if not isinstance(violation, Mapping):
            raise ValueError(f"surface-quality guard violations must be mappings, got {type(violation).__name__}")
        mapped = _map_violation(violation, surface_source=surface_source)
        if mapped is None:
            continue
        check, reason_code = mapped
        if check in failures:
            continue
        failures[check] = SurfaceQualityCheckResult(
            check=check,
            status="failed",
            source="python",
            reason_code=reason_code,
            # Result and novel literal values may contain oracle truth. The raw
            # guard diagnostic retains them privately; quality records do not.
            evidence=(
                None if reason_code in {"expected_result_leakage", "novel_literal"} else _evidence_token(violation)
            ),
        )
    return [
        failures.get(
            check,
            SurfaceQualityCheckResult(check=check, status="passed", source="python"),
        )
        for check in SURFACE_QUALITY_CHECKS
        if check in DETERMINISTIC_SURFACE_CHECKS
    ]


def surface_guard_violations(
    surface: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return a validated guard list without coercing malformed containers."""
    raw = surface.get("guard_violations")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"surface guard_violations must be a list, got {type(raw).__name__}")
    if any(not isinstance(item, Mapping) for item in raw):
        raise ValueError("surface guard_violations entries must be mappings")
    return raw


def _turn_policy(task: Mapping[str, Any]) -> str:
    policy = task.get("turn_policy")
    if not isinstance(policy, str) or policy not in TURN_POLICIES:
        raise ValueError(f"unknown turn_policy {policy!r}; expected one of " + ", ".join(sorted(TURN_POLICIES)))
    return policy


def apply_turn_policy_applicability(
    checks: Sequence[SurfaceQualityCheckResult],
    turn_policy: str,
) -> list[SurfaceQualityCheckResult]:
    """Drop inapplicable failures so assembly does not abort the run."""
    inapplicable = INAPPLICABLE_FAILURES_BY_TURN_POLICY.get(turn_policy, frozenset())
    remapped: list[SurfaceQualityCheckResult] = []
    for result in checks:
        if result.status == "failed" and (result.check, result.reason_code) in inapplicable:
            remapped.append(
                SurfaceQualityCheckResult(
                    check=result.check,
                    status="not_applicable",
                    source=result.source,
                    reason_code=result.reason_code,
                )
            )
            continue
        remapped.append(result)
    return remapped


def evaluate_task_surface_quality(
    task: Mapping[str, Any],
    surface: Mapping[str, Any],
    *,
    judge_enabled: bool = False,
    judge_checks: Sequence[SurfaceQualityCheckResult] | None = None,
) -> list[SurfaceQualityCheckResult]:
    """Return the complete six-check record for one task."""
    if judge_enabled and judge_checks is None:
        raise ValueError("an enabled surface judge requires judge_checks for each task")
    if not judge_enabled and judge_checks is not None:
        raise ValueError("judge_checks require an enabled surface judge")
    policy = _turn_policy(task)
    deterministic = evaluate_deterministic_checks(
        surface_guard_violations(surface),
        surface_source=str(surface.get("source") or "template"),
    )
    judged = (
        apply_turn_policy_applicability(list(judge_checks), policy)
        if judge_checks is not None
        else not_run_judge_checks()
    )
    return validate_complete_check_set([*deterministic, *judged], turn_policy=policy)


def surface_quality_record(
    task: Mapping[str, Any],
    surface: Mapping[str, Any],
    checks: Sequence[SurfaceQualityCheckResult],
) -> dict[str, Any]:
    """Identity plus the assembled six-check verdict for one task."""
    return {
        "task_id": str(task["task_id"]),
        "base_task_id": str(surface.get("base_task_id") or task.get("base_task_id") or task["task_id"]),
        "template_id": str(task["template_id"]),
        "variant_index": int(task.get("variant_index") or 0),
        "surface_source": str(surface.get("source") or "template"),
        "turn_policy": str(task["turn_policy"]),
        "checks": [check.model_dump() for check in checks],
    }


def evaluate_surfaces(
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    *,
    judge_enabled: bool = False,
    judge_checks_by_task: Mapping[str, Sequence[SurfaceQualityCheckResult]] | None = None,
) -> list[dict[str, Any]]:
    """Return one complete six-check record per task, in task order."""
    if judge_enabled and judge_checks_by_task is None:
        raise ValueError("an enabled surface judge requires judge_checks_by_task")
    if not judge_enabled and judge_checks_by_task is not None:
        raise ValueError("judge_checks_by_task require an enabled surface judge")
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        try:
            surface = surfaces[task_id]
        except KeyError as exc:
            raise ValueError(f"surface-quality evaluation is missing a surface for {task_id!r}") from exc
        judge_checks = None
        if judge_checks_by_task is not None:
            try:
                judge_checks = judge_checks_by_task[task_id]
            except KeyError as exc:
                raise ValueError(f"surface-quality evaluation is missing judge checks for {task_id!r}") from exc
        checks = evaluate_task_surface_quality(
            task,
            surface,
            judge_enabled=judge_enabled,
            judge_checks=judge_checks,
        )
        records.append(surface_quality_record(task, surface, checks))
    return records


def _check_label(result: SurfaceQualityCheckResult) -> str:
    return f"{result.check}:{result.reason_code}"


def surface_quality_decision(
    checks: Sequence[SurfaceQualityCheckResult],
    *,
    authority: SurfaceQualityAuthority,
    turn_policy: str,
) -> dict[str, Any]:
    """Turn one six-check record into a keep/drop verdict under the policy.

    Python failures always drop. Judge failures drop only under an authoritative
    judge; otherwise they are recorded as advisory. A judge error never decides
    anything: it is reported so the run-level policy can act on it.
    """
    if not authority.quality_enabled:
        raise ValueError("surface-quality decisions require surface_quality_validation.enabled")
    validated = validate_complete_check_set(checks, turn_policy=turn_policy)
    by_check = {result.check: result for result in validated}
    python_failures = sorted(
        _check_label(by_check[check]) for check in DETERMINISTIC_SURFACE_CHECKS if by_check[check].status == "failed"
    )
    judge_failures = sorted(
        _check_label(by_check[check]) for check in JUDGED_SURFACE_CHECKS if by_check[check].status == "failed"
    )
    judge_errors = sorted(
        {str(by_check[check].reason_code) for check in JUDGED_SURFACE_CHECKS if by_check[check].status == "error"}
    )
    not_run = [check for check in JUDGED_SURFACE_CHECKS if by_check[check].status == "not_run"]
    if not_run and authority.judge_enabled and not python_failures:
        # An enabled judge only skips a surface that Python already rejected.
        # Anything else must be reported as an error, never as a silent skip.
        raise ValueError("an enabled surface judge left judged checks not_run on a surface it should have scored")
    if not authority.judge_enabled and len(not_run) != len(JUDGED_SURFACE_CHECKS):
        raise ValueError("judged checks must be not_run when the surface judge is disabled")

    if python_failures:
        drop_source: str | None = "python"
        drop_reasons = python_failures
    elif judge_failures and authority.drop_authority:
        drop_source = "surface_judge"
        drop_reasons = judge_failures
    else:
        drop_source = None
        drop_reasons = []
    return {
        "decision": "dropped" if drop_source is not None else "kept",
        "drop_source": drop_source,
        "drop_reasons": drop_reasons,
        # Advisory failures are observations only; they never change the verdict.
        "advisory_failures": judge_failures if authority.judge_advisory else [],
        "judge_error": judge_errors[0] if judge_errors else None,
    }


def apply_surface_quality_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    authority: SurfaceQualityAuthority,
) -> list[dict[str, Any]]:
    """Attach a keep/drop verdict to every record and enforce the failure policy.

    An authoritative judge is a publication gate: if it could not answer for even
    one surface, the gate was not enforced and the run must not publish. An
    advisory judge records the same error and continues.
    """
    decided: list[dict[str, Any]] = []
    errored: list[str] = []
    for record in records:
        checks = [
            item if isinstance(item, SurfaceQualityCheckResult) else SurfaceQualityCheckResult.model_validate(item)
            for item in record["checks"]
        ]
        turn_policy = record.get("turn_policy")
        if not isinstance(turn_policy, str):
            raise ValueError("a surface-quality policy record requires turn_policy")
        decision = surface_quality_decision(checks, authority=authority, turn_policy=turn_policy)
        task_id = str(record["task_id"])
        if decision["judge_error"] is not None:
            errored.append(task_id)
        decided.append({**dict(record), **decision})
    if errored and authority.drop_authority:
        raise RuntimeError(
            "an authoritative surface judge failed to answer for "
            f"{len(errored)} task(s) (first: {errored[0]!r}); the quality gate "
            "was not enforced, so this run must not publish"
        )
    if errored:
        logger.warning(
            "BFCL advisory surface judge could not answer for %d task(s); their judged checks stay unenforced",
            len(errored),
        )
    return decided


def write_surface_quality_artifact(
    config: BfclConfig,
    decided: Sequence[Mapping[str, Any]],
) -> Path:
    """Write Stage 10's one-row-per-task parquet artifact."""
    task_ids = [str(record["task_id"]) for record in decided]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("surface-quality artifact task_id values must be unique")
    rows = [surface_validated_task_row(dict(record)) for record in decided]
    return write_stage_table(
        stage_cache_dir(config) / SURFACE_VALIDATED_TASKS,
        rows,
        surface_validated_tasks_schema(),
    )


def surface_quality_report(
    decided: Sequence[Mapping[str, Any]],
    *,
    authority: SurfaceQualityAuthority,
) -> dict[str, Any]:
    """Build the Stage 10 rejection report by check, template, and variant."""
    report = summarize_surface_quality(decided, authority=authority)

    def grouped(key: str) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for record in decided:
            name = str(record[key])
            group = groups.setdefault(
                name,
                {
                    "evaluated": 0,
                    "kept": 0,
                    "dropped": 0,
                    "drop_reason_counts": {},
                    "advisory_failure_counts": {},
                    "judge_error_counts": {},
                },
            )
            group["evaluated"] += 1
            group["kept" if record["decision"] == "kept" else "dropped"] += 1
            for field, target in (
                ("drop_reasons", "drop_reason_counts"),
                ("advisory_failures", "advisory_failure_counts"),
            ):
                for reason in record[field]:
                    counts = group[target]
                    counts[reason] = counts.get(reason, 0) + 1
            error = record["judge_error"]
            if error is not None:
                counts = group["judge_error_counts"]
                counts[str(error)] = counts.get(str(error), 0) + 1
        for group in groups.values():
            for field in ("drop_reason_counts", "advisory_failure_counts", "judge_error_counts"):
                group[field] = dict(sorted(group[field].items()))
        return dict(sorted(groups.items()))

    return {
        **report,
        "contract_version": SURFACE_QUALITY_CONTRACT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION if authority.judge_enabled else None,
        "judge_prompt_hash": JUDGE_PROMPT_HASH if authority.judge_enabled else None,
        "by_template": grouped("template_id"),
        "by_variant_index": grouped("variant_index"),
    }


def write_surface_quality_report(config: BfclConfig, report: Mapping[str, Any]) -> Path:
    """Write the deterministic Stage 10 report consumed by the manifest."""
    path = stage_cache_dir(config) / "surface_quality_rejections.json"
    return _write_json_atomic(path, report)


def run_surface_quality_validation(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    *,
    profile: Mapping[str, Any] | None = None,
    model_runner: JudgeRunner | None = None,
    io_cache: ImmutableModelIOCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Stage 10 end to end for the replay-validated task set."""
    authority = resolve_surface_quality_authority(config)
    if not authority.quality_enabled:
        raise ValueError("run_surface_quality_validation requires surface_quality_validation.enabled")
    judge_checks = (
        run_surface_judge(
            config,
            tasks,
            surfaces,
            profile=profile,
            model_runner=model_runner,
            io_cache=io_cache,
        )
        if authority.judge_enabled
        else None
    )
    records = evaluate_surfaces(
        tasks,
        surfaces,
        judge_enabled=authority.judge_enabled,
        judge_checks_by_task=judge_checks,
    )
    decided = apply_surface_quality_policy(records, authority=authority)
    write_surface_quality_artifact(config, decided)
    report = surface_quality_report(decided, authority=authority)
    write_surface_quality_report(config, report)
    return decided, report


def summarize_surface_quality(
    decided: Sequence[Mapping[str, Any]],
    *,
    authority: SurfaceQualityAuthority,
) -> dict[str, Any]:
    """Account for what the policy decided, for the manifest and the report."""
    reason_counts: dict[str, int] = {}
    advisory_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    dropped_by_python = 0
    dropped_by_judge = 0
    for record in decided:
        if record["drop_source"] == "python":
            dropped_by_python += 1
        elif record["drop_source"] == "surface_judge":
            dropped_by_judge += 1
        for reason in record["drop_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason in record["advisory_failures"]:
            advisory_counts[reason] = advisory_counts.get(reason, 0) + 1
        error = record["judge_error"]
        if error is not None:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1
    return {
        "evaluated": len(decided),
        "kept": len(decided) - dropped_by_python - dropped_by_judge,
        "dropped_by_python": dropped_by_python,
        "dropped_by_surface_judge": dropped_by_judge,
        "judge_enabled": authority.judge_enabled,
        "judge_advisory": authority.judge_advisory,
        "drop_reason_counts": dict(sorted(reason_counts.items())),
        "advisory_failure_counts": dict(sorted(advisory_counts.items())),
        "judge_error_counts": dict(sorted(error_counts.items())),
    }


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_JUDGE_INPUT_KEYS:
                paths.append(current)
            paths.extend(_forbidden_paths(child, current))
        return paths
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in _forbidden_paths(child, f"{prefix}[{index}]")]
    return []


def user_facing_turns(surface: Mapping[str, Any]) -> list[dict[str, str]]:
    """Project the conversation a person reads, omitting tool-call payloads."""
    turns: list[dict[str, str]] = []
    for step in surface.get("steps") or []:
        if not isinstance(step, Mapping):
            raise ValueError("surface steps must be mappings")
        kind = step.get("kind")
        if kind == "calls":
            continue
        if kind not in {"user", "assistant_text"}:
            raise ValueError(f"unknown surface step kind {kind!r}")
        content = step.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("user-facing surface turns must have non-empty content")
        turns.append(
            {
                "role": "user" if kind == "user" else "assistant",
                "content": content,
            }
        )
    return turns


def judge_model_input(
    surface: Mapping[str, Any],
    *,
    style_hints: Sequence[str] = (),
    style_avoid: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the exact payload the surface judge is allowed to see."""
    language = surface.get("language")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("a surface judge request requires a surface language")
    turns = user_facing_turns(surface)
    payload = {
        "language": language,
        "turns": turns,
        "surface_hash": _sha256(canonical_json(turns)),
        "style_hints": [str(item) for item in style_hints],
        "style_avoid": [str(item) for item in style_avoid],
        "rubric": {
            check: sorted(SURFACE_QUALITY_REASON_CODES[check])
            for check in SURFACE_QUALITY_CHECKS
            if check in JUDGED_SURFACE_CHECKS
        },
    }
    leaked = _forbidden_paths(payload)
    if leaked:
        raise ValueError("surface judge input contains oracle-truth fields: " + ", ".join(sorted(leaked)))
    return payload


def project_judge_response(
    response: Any,
    *,
    turn_policy: str,
) -> list[SurfaceQualityCheckResult]:
    """Validate a model verdict and drop failures that the turn policy allows."""
    return apply_turn_policy_applicability(
        SurfaceJudgeResult.model_validate(response).check_results(),
        turn_policy,
    )


def _style_lists(
    config: BfclConfig,
    profile: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    if not (
        config.lineage.profile_influenced_surface and profile is not None and profile.get("status") == "completed"
    ):
        return [], []
    return list(profile.get("style_hints") or []), list(profile.get("avoid") or [])


def _write_judge_cache_usage(
    config: BfclConfig,
    *,
    model_canonical: str,
    requests: Sequence[Mapping[str, Any]],
) -> Path:
    """Record only cache observations used by this run, never the shared cache."""
    usage = {
        "schema_version": "1.0",
        "model_canonical": model_canonical,
        "prompt_hash": JUDGE_PROMPT_HASH,
        "requests": [
            {
                "task_id": str(item["task"]["task_id"]),
                "request_hash": str(item["key"]),
                "input_hash": _sha256(str(item["input_json"])),
                "source": str(item["source"]),
                "status": (
                    "error"
                    if isinstance(item["response"], dict) and item["response"].get("_judge_error")
                    else "completed"
                ),
                "error_code": (
                    str(item["response"]["_judge_error"])
                    if isinstance(item["response"], dict) and item["response"].get("_judge_error")
                    else None
                ),
                "observed_response_hash": item.get("observed_response_hash"),
            }
            for item in requests
        ],
    }
    path = stage_cache_dir(config) / "surface_judge_cache_usage.json"
    return _write_json_atomic(path, usage)


def run_surface_judge(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    *,
    profile: Mapping[str, Any] | None = None,
    model_runner: JudgeRunner | None = None,
    io_cache: ImmutableModelIOCache | None = None,
) -> dict[str, list[SurfaceQualityCheckResult]]:
    """Call the optional surface judge, reusing the immutable I/O cache."""
    role = (config.lineage.roles or {}).get("surface_judge")
    enabled = bool(role and role.enabled)
    if not enabled:
        return {str(task["task_id"]): not_run_judge_checks() for task in tasks}

    assert role is not None and role.model_config is not None
    model_config = dict(role.model_config)
    canonical_id = str(model_config["canonical_id"]).strip().lower()
    model_config["canonical_id"] = canonical_id
    prompt_hash = JUDGE_PROMPT_HASH
    style_hints, style_avoid = _style_lists(config, profile)
    profile_languages = (
        {str(language).strip().casefold() for language in profile.get("languages") or []}
        if config.lineage.profile_influenced_surface and profile is not None and profile.get("status") == "completed"
        else set()
    )
    if profile_languages and len(profile_languages) != 1:
        raise ValueError(
            "an enabled surface judge requires a completed profile in exactly "
            f"one language; found {', '.join(sorted(profile_languages))}"
        )
    cache = io_cache or ImmutableModelIOCache(stage_cache_dir(config) / "surface_judge_io_cache.jsonl")
    pending: list[dict[str, Any]] = []
    results: dict[str, list[SurfaceQualityCheckResult]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        try:
            surface = surfaces[task_id]
        except KeyError as exc:
            raise ValueError(f"surface judge is missing a surface for {task_id!r}") from exc
        deterministic = evaluate_deterministic_checks(
            surface_guard_violations(surface),
            surface_source=str(surface.get("source") or "template"),
        )
        if any(result.status == "failed" for result in deterministic):
            # Python failures are authoritative and the rejected surface may itself
            # contain leaked oracle truth. Do not expose it to the optional judge.
            results[task_id] = not_run_judge_checks()
            continue
        surface_language = str(surface.get("language") or "").strip().casefold()
        if profile_languages and surface_language not in profile_languages:
            raise ValueError(
                "reference profile language must match every judged surface "
                f"(profile={sorted(profile_languages)}, "
                f"surface={surface_language!r}, task_id={task_id!r})"
            )
        model_input = judge_model_input(
            surface,
            style_hints=style_hints,
            style_avoid=style_avoid,
        )
        key = request_hash(
            model_canonical=canonical_id,
            prompt_hash=prompt_hash,
            model_input=model_input,
            inference_parameters=dict(model_config.get("inference_parameters") or {}),
            output_schema=SurfaceJudgeResult.model_json_schema(),
            seed=int(task.get("seed") or 0),
        )
        cached = cache.get(key)
        cache_hit = cached is not None
        observed_response_hash = _sha256(canonical_json(cached)) if cache_hit else None
        if cached is not None:
            try:
                cached = SurfaceJudgeResult.model_validate(cached).model_dump()
            except ValidationError:
                # A hash-valid entry can still predate stricter schema validation.
                # Preserve cache immutability and report the unusable observation.
                cached = {"_judge_error": "invalid_response"}
        pending.append(
            {
                "key": key,
                "task": task,
                "model_input": model_input,
                "input_json": canonical_json(model_input),
                "response": cached,
                "source": "cache" if cache_hit else "model",
                "observed_response_hash": observed_response_hash,
            }
        )

    missing = [item for item in pending if item["response"] is None]
    if missing:
        if model_runner is None:
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_runner import (
                run_structured_model,
            )

            model_runner = run_structured_model
        batch_size = max(1, int(config.ndd_batch_size))
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            try:
                responses = model_runner(
                    config,
                    stage_name="surface_judge",
                    model_config=model_config,
                    requests=[
                        {
                            "request_id": item["key"],
                            "model_input": item["input_json"],
                        }
                        for item in batch
                    ],
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    prompt=JUDGE_PROMPT,
                    output_format=SurfaceJudgeResult,
                )
            except Exception as exc:  # noqa: BLE001 - quality records must survive
                logger.warning("BFCL surface judge batch failed: %s", type(exc).__name__)
                for item in batch:
                    item["response"] = {"_judge_error": "judge_error"}
                continue
            for item in batch:
                response = responses.get(item["key"])
                if response is None:
                    item["response"] = {"_judge_error": "missing_response"}
                    continue
                item["observed_response_hash"] = _sha256(canonical_json(response))
                try:
                    normalized_response = SurfaceJudgeResult.model_validate(response).model_dump()
                except ValidationError:
                    item["response"] = {"_judge_error": "invalid_response"}
                    continue
                item["response"] = normalized_response
                cache.put(
                    item["key"],
                    normalized_response,
                    model_canonical=canonical_id,
                    input_hash=_sha256(item["input_json"]),
                )

    for item in pending:
        task = item["task"]
        task_id = str(task["task_id"])
        response = item["response"]
        if isinstance(response, dict) and response.get("_judge_error"):
            results[task_id] = judge_error_checks(str(response["_judge_error"]))
            continue
        results[task_id] = project_judge_response(response, turn_policy=_turn_policy(task))
    _write_judge_cache_usage(
        config,
        model_canonical=canonical_id,
        requests=pending,
    )
    return results
