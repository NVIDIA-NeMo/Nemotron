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

"""The A1/A2 probe choreography, written once for every transport that can be reset.

What the certification ladder asks of a source is the same question regardless of how the
source is reached: does a call return what it claims, does an error keep its shape, does a
reset really start over, does one episode leak into the next, does an unconfirmed mutation
wait, and does a call that never returns still leave the source usable. None of those
questions mention a transport.

The transport only decides how one episode is executed, and both supported transports
already run episodes through the same worker — a local backend through a child process, an
HTTP oracle through a session against the endpoint. So this module owns the questions and
the judgements, and the caller supplies the four things that genuinely differ: how to run
an episode, what the reviewed catalog is, what identity means, and how to prove that
identity did not move while the probes ran.

Nothing here derives a tier or signs anything. It returns observations, and certification
decides what they are worth.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterProbeObservation,
    CertificationProbe,
    CertificationRefusalCode,
    ProbeExecutionRecord,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutLeakageError,
    scan_held_out_terms,
)

# The plan was first written for local Python and is transport-neutral in content, so the
# original version stays readable rather than forcing every reviewed plan to be reissued.
ADAPTER_PROBE_PLAN_VERSION: Literal[
    "bfcl-adapter-probe-plan-v1"
] = "bfcl-adapter-probe-plan-v1"
LOCAL_PROBE_PLAN_VERSION: Literal[
    "bfcl-local-probe-plan-v1"
] = "bfcl-local-probe-plan-v1"

_SAFE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class ProbeError(ValueError):
    """A refusal raised before any observation can be trusted."""

    def __init__(self, code: str, detail: str) -> None:
        try:
            self.code = CertificationRefusalCode(code).value
        except ValueError as exc:
            raise ValueError(f"unknown probe refusal code {code!r}") from exc
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class ReviewedProbeTool(Protocol):
    """The reviewed facts about one tool that the probes are entitled to rely on."""

    @property
    def published_name(self) -> str: ...

    @property
    def mutates(self) -> bool: ...

    @property
    def requires_confirmation(self) -> bool: ...


class EpisodeRunner(Protocol):
    """Run one isolated episode of ordered steps and return each step's output."""

    def __call__(
        self,
        task_id: str,
        steps: list[dict[str, Any]],
        *,
        tool_timeout: float | None = None,
    ) -> list[Any]: ...


# Returns whether the live catalog matches the reviewed one, the evidence for that
# judgement, and how many calls it cost. Transports disagree about what a catalog surface
# even is, which is why this is the caller's to answer.
CatalogProbe = Callable[[], "tuple[bool, dict[str, Any], int]"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterProbeCase(_StrictModel):
    case_id: StrictStr
    tool: StrictStr
    arguments: dict[str, Any]
    expectation: Literal["success", "structured_error", "timeout"]
    expected_state_change: StrictBool | None = None
    expected_error_code: StrictStr | None = None

    @field_validator("case_id", "tool")
    @classmethod
    def _safe_name(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("probe case ids and tools must be safe identifiers")
        return value

    @field_validator("arguments")
    @classmethod
    def _canonical_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            sha256_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe arguments must be canonical JSON") from exc
        return value

    @model_validator(mode="after")
    def _expectation_fields(self) -> AdapterProbeCase:
        if self.expectation == "success":
            if self.expected_state_change is None or self.expected_error_code is not None:
                raise ValueError(
                    "success cases require expected_state_change and no error code"
                )
        elif self.expectation == "structured_error":
            if not (self.expected_error_code and self.expected_error_code.strip()):
                raise ValueError("structured-error cases require expected_error_code")
            if self.expected_state_change not in {None, False}:
                raise ValueError("structured-error cases cannot expect state mutation")
        elif (
            self.expected_state_change is not None
            or self.expected_error_code is not None
        ):
            raise ValueError("timeout cases cannot claim an outcome or state change")
        return self


class AdapterProbePlan(_StrictModel):
    schema_version: Literal[
        "bfcl-adapter-probe-plan-v1",
        "bfcl-local-probe-plan-v1",
    ]
    clock: StrictStr
    seed: StrictInt
    fixtures: dict[str, Any] | None
    cases: tuple[AdapterProbeCase, ...]
    confirmation_parameter: StrictStr = "confirm"
    status_field: StrictStr = "status"
    pending_status: StrictStr = "awaiting_confirmation"
    error_path: tuple[StrictStr, ...] = ("error", "code")

    @field_validator("clock")
    @classmethod
    def _clock(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("probe clock must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("probe clock must include an explicit timezone")
        return value

    @field_validator("fixtures")
    @classmethod
    def _fixtures(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        try:
            sha256_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe fixtures must be canonical JSON") from exc
        return value

    @field_validator("cases")
    @classmethod
    def _cases(
        cls,
        value: tuple[AdapterProbeCase, ...],
    ) -> tuple[AdapterProbeCase, ...]:
        if not value:
            raise ValueError("a probe plan requires cases")
        ids = [case.case_id for case in value]
        if len(ids) != len(set(ids)) or ids != sorted(ids):
            raise ValueError("probe cases must have unique, sorted case_id values")
        if sum(case.expectation == "success" for case in value) > 16:
            raise ValueError("a probe plan supports at most 16 success cases")
        if sum(case.expectation == "structured_error" for case in value) > 8:
            raise ValueError("a probe plan supports at most 8 structured-error cases")
        if sum(case.expectation == "timeout" for case in value) > 1:
            raise ValueError("a probe plan supports at most one timeout case")
        return value

    @field_validator(
        "confirmation_parameter",
        "status_field",
        "pending_status",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("probe vocabulary strings must be non-empty")
        return value

    @field_validator("error_path")
    @classmethod
    def _error_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("error_path must contain non-empty path segments")
        return value

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def validate_probe_plan(
    tools: Sequence[ReviewedProbeTool],
    plan: AdapterProbePlan,
) -> None:
    """Refuse a plan that could reach A2 while leaving part of the source unobserved."""
    published = {tool.published_name: tool for tool in tools}
    unknown = sorted({case.tool for case in plan.cases} - set(published))
    if unknown:
        raise ProbeError(
            "reviewed_schema_invalid",
            "probe plan names unknown tools: " + ", ".join(unknown),
        )
    success_by_tool = {
        name: [
            case
            for case in plan.cases
            if case.tool == name and case.expectation == "success"
        ]
        for name in published
    }
    missing = sorted(name for name, cases in success_by_tool.items() if not cases)
    if missing:
        raise ProbeError(
            "result_shape_incomplete",
            "success coverage is missing tools: " + ", ".join(missing),
        )
    mutation_missing = sorted(
        name
        for name, tool in published.items()
        if tool.mutates
        and not any(case.expected_state_change is True for case in success_by_tool[name])
    )
    if mutation_missing:
        raise ProbeError(
            "mutation_declaration_mismatch",
            "mutating tools lack a state-changing case: " + ", ".join(mutation_missing),
        )


def extract_path(value: Any, path: Sequence[str]) -> Any:
    current = value
    for segment in path:
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def json_shape(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return {"type": "depth_limit"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        shapes = {
            sha256_json(json_shape(item, depth=depth + 1)): json_shape(
                item, depth=depth + 1
            )
            for item in value
        }
        return {"type": "array", "items": [shapes[key] for key in sorted(shapes)]}
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "properties": {
                str(key): json_shape(child, depth=depth + 1)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            },
        }
    raise ProbeError(
        "probe_evidence_invalid",
        f"source returned non-JSON value {type(value).__name__}",
    )


def probe_record(
    probe: CertificationProbe,
    *,
    started: float,
    calls: int,
    status: Literal["pass", "fail", "not_applicable"],
    evidence: Any,
    reason: str | None = None,
    cleanup_status: Literal["passed", "failed", "not_required"] = "passed",
) -> ProbeExecutionRecord:
    return ProbeExecutionRecord(
        observation=AdapterProbeObservation(
            probe=probe,
            status=status,
            evidence=evidence,
            reason=reason,
        ),
        observed_calls=calls,
        elapsed_s=float(time.monotonic() - started),
        cleanup_status=cleanup_status,
    )


def run_probe_suite(
    *,
    plan: AdapterProbePlan,
    tools: Sequence[ReviewedProbeTool],
    episode: EpisodeRunner,
    identity_record: ProbeExecutionRecord,
    catalog_probe: CatalogProbe,
    identity_drifted: Callable[[], bool],
    held_out_sensitive_terms: Sequence[str] = (),
    timeout_probe_s: float = 0.25,
) -> tuple[ProbeExecutionRecord, ...]:
    """Observe one source against every probe on the ladder, and judge nothing else.

    Identity arrives already observed, because what pinning an identity costs and whether
    it leaves anything to clean up is the one A0 question only the transport can answer.
    """
    if identity_record.observation.probe is not CertificationProbe.IDENTITY_INTEGRITY:
        raise ProbeError(
            "probe_evidence_invalid",
            "the supplied identity record does not observe identity integrity",
        )
    plan_findings = scan_held_out_terms(
        plan.model_dump(mode="json"),
        sensitive_terms=held_out_sensitive_terms,
        location="$.probe_plan",
    )
    if plan_findings:
        raise HeldOutLeakageError(plan_findings)

    records: dict[CertificationProbe, ProbeExecutionRecord] = {}
    records[CertificationProbe.IDENTITY_INTEGRITY] = identity_record

    catalog_started = time.monotonic()
    try:
        catalog_ok, catalog_evidence, catalog_calls = catalog_probe()
        records[CertificationProbe.CATALOG_INTEGRITY] = probe_record(
            CertificationProbe.CATALOG_INTEGRITY,
            started=catalog_started,
            calls=catalog_calls,
            status="pass" if catalog_ok else "fail",
            evidence=catalog_evidence,
            reason=None if catalog_ok else "catalog_mismatch",
        )
    except TimeoutError:
        records[CertificationProbe.CATALOG_INTEGRITY] = probe_record(
            CertificationProbe.CATALOG_INTEGRITY,
            started=catalog_started,
            calls=1,
            status="fail",
            evidence={"catalog_observed": False},
            reason="probe_timeout",
        )
    except Exception as exc:  # noqa: BLE001 - the source refused; that is the finding
        records[CertificationProbe.CATALOG_INTEGRITY] = probe_record(
            CertificationProbe.CATALOG_INTEGRITY,
            started=catalog_started,
            calls=1,
            status="fail",
            evidence={"error_type": type(exc).__name__},
            reason="probe_failed",
        )

    success_cases = [case for case in plan.cases if case.expectation == "success"]
    error_cases = [
        case for case in plan.cases if case.expectation == "structured_error"
    ]
    timeout_cases = [case for case in plan.cases if case.expectation == "timeout"]

    success_observations: list[dict[str, Any]] = []
    executable_started = time.monotonic()
    executable_reason: str | None = None
    try:
        for case in success_cases:
            output = episode(
                f"probe-success-{case.case_id}",
                [
                    {"op": "reset"},
                    {"op": "get_state"},
                    {
                        "op": "call_tool",
                        "name": case.tool,
                        "arguments": case.arguments,
                    },
                    {"op": "get_state"},
                ],
            )
            before, result, after = output[1], output[2], output[3]
            if scan_held_out_terms(
                {"result": result, "state": after},
                sensitive_terms=held_out_sensitive_terms,
            ):
                raise ProbeError(
                    "probe_evidence_invalid",
                    "observed success output overlaps held-out material",
                )
            success_observations.append(
                {
                    "case_id": case.case_id,
                    "tool": case.tool,
                    "result_digest": sha256_json(result),
                    "result_shape": json_shape(result),
                    "before_state_digest": sha256_json(before),
                    "after_state_digest": sha256_json(after),
                    "state_changed": sha256_json(before) != sha256_json(after),
                }
            )
    except TimeoutError:
        executable_reason = "probe_timeout"
    except ProbeError as exc:
        executable_reason = exc.code
    except Exception:  # noqa: BLE001 - a source that cannot be called is the finding
        executable_reason = "probe_failed"
    records[CertificationProbe.EXECUTABLE_OBSERVATION] = probe_record(
        CertificationProbe.EXECUTABLE_OBSERVATION,
        started=executable_started,
        calls=len(success_observations),
        status="pass" if executable_reason is None else "fail",
        evidence={"observations": success_observations},
        reason=executable_reason,
    )

    error_started = time.monotonic()
    if not error_cases:
        records[CertificationProbe.STRUCTURED_ERROR_SHAPE] = probe_record(
            CertificationProbe.STRUCTURED_ERROR_SHAPE,
            started=error_started,
            calls=0,
            status="not_applicable",
            evidence={"applicable": False, "error_path": list(plan.error_path)},
            reason="no_structured_error_case",
        )
    else:
        error_observations: list[dict[str, Any]] = []
        error_reason: str | None = None
        try:
            for case in error_cases:
                output = episode(
                    f"probe-error-{case.case_id}",
                    [
                        {"op": "reset"},
                        {
                            "op": "call_tool",
                            "name": case.tool,
                            "arguments": case.arguments,
                        },
                    ],
                )
                result = output[1]
                if scan_held_out_terms(
                    result,
                    sensitive_terms=held_out_sensitive_terms,
                ):
                    raise ProbeError(
                        "probe_evidence_invalid",
                        "observed error output overlaps held-out material",
                    )
                observed_code = extract_path(result, plan.error_path)
                if observed_code != case.expected_error_code:
                    error_reason = "structured_error_mismatch"
                error_observations.append(
                    {
                        "case_id": case.case_id,
                        "result_shape": json_shape(result),
                        "error_code_digest": sha256_json({"code": observed_code}),
                        "matched": observed_code == case.expected_error_code,
                    }
                )
        except TimeoutError:
            error_reason = "probe_timeout"
        except ProbeError as exc:
            error_reason = exc.code
        except Exception:  # noqa: BLE001
            error_reason = "probe_failed"
        records[CertificationProbe.STRUCTURED_ERROR_SHAPE] = probe_record(
            CertificationProbe.STRUCTURED_ERROR_SHAPE,
            started=error_started,
            calls=len(error_observations),
            status="pass" if error_reason is None else "fail",
            evidence={"applicable": True, "observations": error_observations},
            reason=error_reason,
        )

    sequence = [
        {
            "op": "call_tool",
            "name": case.tool,
            "arguments": case.arguments,
            "turn_index": index,
        }
        for index, case in enumerate(success_cases)
    ]

    deterministic_started = time.monotonic()
    deterministic_reason: str | None = None
    deterministic_evidence: dict[str, Any] = {}
    try:
        first = episode(
            "probe-determinism-a",
            [{"op": "reset"}, *sequence, {"op": "get_state"}],
        )
        second = episode(
            "probe-determinism-b",
            [{"op": "reset"}, *sequence, {"op": "get_state"}],
        )
        first_digest = sha256_json(first)
        second_digest = sha256_json(second)
        deterministic_evidence = {
            "first_digest": first_digest,
            "second_digest": second_digest,
        }
        if first_digest != second_digest:
            deterministic_reason = "reset_nondeterministic"
    except TimeoutError:
        deterministic_reason = "probe_timeout"
    except Exception:  # noqa: BLE001
        deterministic_reason = "probe_failed"
    records[CertificationProbe.RESET_DETERMINISM] = probe_record(
        CertificationProbe.RESET_DETERMINISM,
        started=deterministic_started,
        calls=2 * len(success_cases),
        status="pass" if deterministic_reason is None else "fail",
        evidence=deterministic_evidence or {"completed": False},
        reason=deterministic_reason,
    )

    isolation_started = time.monotonic()
    isolation_reason: str | None = None
    isolation_evidence: dict[str, Any] = {}
    try:
        baseline = episode(
            "probe-isolation-baseline",
            [{"op": "reset"}, {"op": "get_state"}],
        )[-1]
        episode(
            "probe-isolation-mutation",
            [{"op": "reset"}, *sequence, {"op": "get_state"}],
        )
        fresh = episode(
            "probe-isolation-fresh",
            [{"op": "reset"}, {"op": "get_state"}],
        )[-1]
        baseline_digest = sha256_json(baseline)
        fresh_digest = sha256_json(fresh)
        isolation_evidence = {
            "baseline_digest": baseline_digest,
            "fresh_digest": fresh_digest,
        }
        if baseline_digest != fresh_digest:
            isolation_reason = "episode_state_leakage"
    except TimeoutError:
        isolation_reason = "probe_timeout"
    except Exception:  # noqa: BLE001
        isolation_reason = "probe_failed"
    records[CertificationProbe.EPISODE_ISOLATION] = probe_record(
        CertificationProbe.EPISODE_ISOLATION,
        started=isolation_started,
        calls=len(success_cases),
        status="pass" if isolation_reason is None else "fail",
        evidence=isolation_evidence or {"completed": False},
        reason=isolation_reason,
    )

    confirmation_tools = {
        tool.published_name: tool for tool in tools if tool.requires_confirmation
    }
    confirmation_started = time.monotonic()
    if not confirmation_tools:
        records[CertificationProbe.CONFIRMATION_SAFETY] = probe_record(
            CertificationProbe.CONFIRMATION_SAFETY,
            started=confirmation_started,
            calls=0,
            status="not_applicable",
            evidence={"applicable": False},
            reason="no_confirmation_tools",
        )
    else:
        confirmation_observations: list[dict[str, Any]] = []
        confirmation_reason: str | None = None
        try:
            for name in sorted(confirmation_tools):
                case = next(case for case in success_cases if case.tool == name)
                arguments = {**case.arguments, plan.confirmation_parameter: False}
                output = episode(
                    f"probe-confirmation-{case.case_id}",
                    [
                        {"op": "reset"},
                        {"op": "get_state"},
                        {"op": "call_tool", "name": name, "arguments": arguments},
                        {"op": "get_state"},
                    ],
                )
                before, result, after = output[1], output[2], output[3]
                pending = (
                    isinstance(result, Mapping)
                    and result.get(plan.status_field) == plan.pending_status
                )
                unchanged = sha256_json(before) == sha256_json(after)
                confirmation_observations.append(
                    {
                        "tool": name,
                        "pending": pending,
                        "state_unchanged": unchanged,
                    }
                )
                if not pending or not unchanged:
                    confirmation_reason = "probe_unsafe"
        except TimeoutError:
            confirmation_reason = "probe_timeout"
        except Exception:  # noqa: BLE001
            confirmation_reason = "probe_failed"
        records[CertificationProbe.CONFIRMATION_SAFETY] = probe_record(
            CertificationProbe.CONFIRMATION_SAFETY,
            started=confirmation_started,
            calls=len(confirmation_observations),
            status="pass" if confirmation_reason is None else "fail",
            evidence={
                "applicable": True,
                "observations": confirmation_observations,
            },
            reason=confirmation_reason,
        )

    timeout_started = time.monotonic()
    if not timeout_cases:
        records[CertificationProbe.TIMEOUT_CLEANUP] = probe_record(
            CertificationProbe.TIMEOUT_CLEANUP,
            started=timeout_started,
            calls=0,
            status="fail",
            evidence={"timeout_case_present": False},
            reason="probe_missing",
        )
    else:
        timeout_case = timeout_cases[0]
        timed_out = False
        fresh_ok = False
        # A source that ignores its deadline and a probe that never reached a
        # deadline are different findings, and both would otherwise be recorded as
        # `timeout_observed: false`. Keep the failing exception so the report says
        # which one happened: the first is the source's defect to fix, the second is
        # usually the environment's.
        probe_error: str | None = None
        recovery_error: str | None = None
        try:
            episode(
                f"probe-timeout-{timeout_case.case_id}",
                [
                    {"op": "reset"},
                    {
                        "op": "call_tool",
                        "name": timeout_case.tool,
                        "arguments": timeout_case.arguments,
                    },
                ],
                tool_timeout=timeout_probe_s,
            )
        except TimeoutError:
            timed_out = True
        except Exception as exc:  # noqa: BLE001 - anything else is not a deadline
            probe_error = type(exc).__name__
        try:
            episode(
                "probe-timeout-recovery",
                [{"op": "reset"}, {"op": "get_state"}],
            )
            fresh_ok = True
        except Exception as exc:  # noqa: BLE001
            fresh_ok = False
            recovery_error = type(exc).__name__
        timeout_ok = timed_out and fresh_ok
        evidence = {
            "timeout_observed": timed_out,
            "business_call_attempts": 1,
            "session_cleanup_completed": timed_out,
            "fresh_episode_succeeded": fresh_ok,
            "unknown_commit_state_preserved": timed_out,
        }
        if probe_error is not None:
            evidence["probe_error"] = probe_error
        if recovery_error is not None:
            evidence["recovery_error"] = recovery_error
        if timeout_ok:
            timeout_reason = None
        elif probe_error is not None:
            timeout_reason = "probe_failed"
        else:
            timeout_reason = "cleanup_failed"
        records[CertificationProbe.TIMEOUT_CLEANUP] = probe_record(
            CertificationProbe.TIMEOUT_CLEANUP,
            started=timeout_started,
            calls=1,
            status="pass" if timeout_ok else "fail",
            evidence=evidence,
            reason=timeout_reason,
            cleanup_status="passed" if timeout_ok else "failed",
        )

    mutation_started = time.monotonic()
    mutation_findings = []
    tools_by_name = {tool.published_name: tool for tool in tools}
    for observation, case in zip(success_observations, success_cases, strict=False):
        changed = observation["state_changed"]
        declared = tools_by_name[case.tool].mutates
        if changed != case.expected_state_change or (changed and not declared):
            mutation_findings.append(case.case_id)
    mutation_ok = (
        executable_reason is None
        and len(success_observations) == len(success_cases)
        and not mutation_findings
    )
    records[CertificationProbe.MUTATION_DECLARATION] = probe_record(
        CertificationProbe.MUTATION_DECLARATION,
        started=mutation_started,
        calls=len(success_observations),
        status="pass" if mutation_ok else "fail",
        evidence={
            "checked_cases": [item["case_id"] for item in success_observations],
            "mismatched_cases": mutation_findings,
        },
        reason=None if mutation_ok else "mutation_declaration_mismatch",
    )

    shape_started = time.monotonic()
    covered = sorted({item["tool"] for item in success_observations})
    expected_tools = sorted(tools_by_name)
    shape_ok = executable_reason is None and covered == expected_tools
    records[CertificationProbe.RESULT_SHAPE_COVERAGE] = probe_record(
        CertificationProbe.RESULT_SHAPE_COVERAGE,
        started=shape_started,
        calls=len(success_observations),
        status="pass" if shape_ok else "fail",
        evidence={
            "covered_tools": covered,
            "expected_tools": expected_tools,
            "result_shapes": {
                item["case_id"]: item["result_shape"] for item in success_observations
            },
        },
        reason=None if shape_ok else "result_shape_incomplete",
    )

    if identity_drifted():
        records[CertificationProbe.IDENTITY_INTEGRITY] = probe_record(
            CertificationProbe.IDENTITY_INTEGRITY,
            started=time.monotonic(),
            calls=0,
            status="fail",
            evidence={"identity_changed_during_probes": True},
            reason="identity_drift",
            cleanup_status=identity_record.cleanup_status,
        )
    return tuple(records[probe] for probe in CertificationProbe)
