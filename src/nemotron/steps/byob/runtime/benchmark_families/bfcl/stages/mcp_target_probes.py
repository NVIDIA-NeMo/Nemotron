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

"""BFCL-owned target probe projection for MCP endpoint conformance.

The gateway may advertise these checks, but this module derives them only from work performed by
``run_oracle_validation``. Keeping the projection pure makes a provisional L0 pack useful: BFCL
can validate it, produce canonical probe evidence, and compare that evidence with a subsequently
served attestation without allowing the endpoint to grade itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointOracleClient,
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.run_context import RunContext

GATEWAY_CONFORMANCE_REPORT_PATH = Path(
    ".bfcl/conformance/gateway_conformance_report.json"
)
GATEWAY_SUITE_VERSION = "bfcl-mcp-gateway-conformance-v1"


def _entry(
    entries: Sequence[Mapping[str, Any]],
    identifier: int | str,
) -> Mapping[str, Any] | None:
    return next((entry for entry in entries if entry.get("id") == identifier), None)


def _passed(entry: Mapping[str, Any] | None) -> bool:
    return entry is not None and entry.get("status") == "pass"


def _outcome(identifier: str, passed: bool, reason: str | None = None) -> dict[str, Any]:
    return {
        "id": identifier,
        "requirement": "required",
        "status": "pass" if passed else "fail",
        "reason": None if passed else reason,
    }


def _conditional_outcome(
    identifier: str,
    *,
    applicable: bool,
    passed: bool,
    not_applicable_reason: str,
    failure_reason: str,
) -> dict[str, Any]:
    if not applicable:
        return {
            "id": identifier,
            "requirement": "conditional",
            "status": "not_applicable",
            "reason": not_applicable_reason,
        }
    outcome = _outcome(identifier, passed, failure_reason)
    outcome["requirement"] = "conditional"
    return outcome


def run_endpoint_isolation_probe(
    endpoint_config: EndpointConfig,
    *,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    timeout_s: float,
    activity_case: Mapping[str, Any],
    expect_state_change: bool,
) -> dict[str, Any]:
    """Interleave two live sessions and prove a mutation cannot cross the boundary."""
    first = EndpointOracleClient(
        endpoint_config,
        headers=resolve_endpoint_headers(endpoint_config),
        timeout_s=timeout_s,
    )
    second = EndpointOracleClient(
        endpoint_config,
        headers=resolve_endpoint_headers(endpoint_config),
        timeout_s=timeout_s,
    )
    clock = datetime.fromisoformat(clock_iso)
    first_context = RunContext(
        clock=clock,
        seed=seed,
        timeout_s=timeout_s,
        task_id="mcp-p6:first",
    )
    second_context = RunContext(
        clock=clock,
        seed=seed,
        timeout_s=timeout_s,
        task_id="mcp-p6:second",
    )
    failures: list[dict[str, Any]] = []
    try:
        first.reset(ctx=first_context, fixtures=fixtures)
        second.reset(ctx=second_context, fixtures=fixtures)
        first_before = first.get_state()
        second_before = second.get_state()
        first.call_tool(
            str(activity_case.get("tool")),
            dict(activity_case.get("arguments") or {}),
            ctx=first_context,
        )
        if expect_state_change and first.get_state() == first_before:
            failures.append({"reason": "declared_mutation_did_not_change_first_episode"})
        if second.get_state() != second_before:
            failures.append({"reason": "first_episode_activity_changed_second_episode"})
    except Exception as exc:  # noqa: BLE001 - report the complete probe failure
        failures.append(
            {
                "reason": "isolation_probe_raised",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        first.close(suppress_errors=True)
        second.close(suppress_errors=True)
    return {
        "id": "MP6",
        "name": "mcp_episode_isolation",
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def load_gateway_conformance_report(
    pack_root: Path,
) -> dict[str, Any] | None:
    """Load the immutable BFCL-owned gateway report, rejecting duplicate JSON keys."""
    path = pack_root / GATEWAY_CONFORMANCE_REPORT_PATH
    if not path.is_file():
        return None

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"gateway conformance report repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(document, dict):
        raise ValueError("gateway conformance report must be a JSON object")
    return document


def assess_gateway_timeout_report(
    document: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Accept P9 only when the controlled-fixture report records every safety property."""
    failures: list[dict[str, Any]] = []
    suite = document.get("suite") if isinstance(document, Mapping) else None
    p9 = suite.get("p9") if isinstance(suite, Mapping) else None
    if not isinstance(suite, Mapping) or suite.get("kind") != "gateway":
        failures.append({"reason": "gateway_suite_missing"})
    if not isinstance(suite, Mapping) or suite.get("profile_version") != GATEWAY_SUITE_VERSION:
        failures.append({"reason": "gateway_suite_version_mismatch"})
    required = {
        "timeout_observed": True,
        "business_call_attempts": 1,
        "episode_poisoned": True,
        "transport_cleanup_completed": True,
        "unknown_commit_state_preserved": True,
    }
    if not isinstance(p9, Mapping):
        failures.append({"reason": "gateway_timeout_observation_missing"})
    else:
        for field, expected in required.items():
            if p9.get(field) != expected:
                failures.append(
                    {
                        "reason": "gateway_timeout_property_mismatch",
                        "field": field,
                        "expected": expected,
                        "got": p9.get(field),
                    }
                )
    return {
        "id": "MP9",
        "name": "mcp_gateway_timeout_conformance",
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def build_target_probe_report(
    *,
    checks: Sequence[Mapping[str, Any]],
    extra_checks: Sequence[Mapping[str, Any]],
    endpoint_metadata: Mapping[str, Any] | None,
    observations: Mapping[str, Any] | None,
    tool_names: set[str],
    confirmation_tool_names: set[str],
    structured_error_declared: bool,
) -> dict[str, Any]:
    """Derive target probes from fresh endpoint validation and observed calls."""
    schema_check = _entry(checks, 3)
    validation_check = _entry(checks, 5)
    determinism_check = _entry(extra_checks, "D1")
    isolation_check = _entry(extra_checks, "MP6")
    error_shape_check = _entry(extra_checks, "D2")
    confirmation_check = _entry(checks, 6)
    timeout_check = _entry(extra_checks, "MP9")
    mutation_check = _entry(extra_checks, "M1")
    identity_ok = _passed(schema_check) and endpoint_metadata is not None
    observation_map = observations if isinstance(observations, Mapping) else {}
    calls = observation_map.get("calls")
    complete = (
        observation_map.get("calls_complete") is True
        and observation_map.get("state_deltas_complete") is True
        and isinstance(calls, list)
    )
    successful_tools = {
        str(call.get("tool"))
        for call in calls or []
        if isinstance(call, Mapping) and call.get("result_class") == "success"
    }
    observed_tools = {
        str(call.get("tool"))
        for call in calls or []
        if isinstance(call, Mapping)
    }
    structured_error_observed = any(
        isinstance(call, Mapping) and call.get("result_class") == "structured_error"
        for call in calls or []
    )
    missing_success = sorted(tool_names - successful_tools)
    p4_ok = (
        identity_ok
        and _passed(validation_check)
        and complete
        and not missing_success
    )
    p4_reason_parts: list[str] = []
    if not identity_ok:
        p4_reason_parts.append("live endpoint identity or schema alignment was not verified")
    if not _passed(validation_check):
        p4_reason_parts.append("declared validation cases did not all pass")
    if not complete:
        p4_reason_parts.append("observed call or state-delta log is incomplete")
    if missing_success:
        p4_reason_parts.append(
            "no successful executable case for: " + ", ".join(missing_success)
        )

    return {
        "probes": [
            _outcome(
                "P1",
                identity_ok,
                "live endpoint identity or schema alignment was not verified",
            ),
            _outcome(
                "P2",
                _passed(schema_check),
                "live published tool set did not exactly match the pack",
            ),
            _outcome(
                "P3",
                _passed(schema_check),
                "one or more published tool definitions failed normalization/schema checks",
            ),
            _outcome(
                "P4",
                p4_ok,
                "; ".join(p4_reason_parts) or "executable lifecycle did not pass",
            ),
            _outcome(
                "P5",
                _passed(determinism_check),
                "two fresh replays did not produce identical results and final state",
            ),
            _outcome(
                "P6",
                _passed(isolation_check),
                "activity in one live episode crossed the second episode boundary",
            ),
            _conditional_outcome(
                "P7",
                applicable=structured_error_declared,
                passed=_passed(error_shape_check) and structured_error_observed,
                not_applicable_reason="the pack declares no structured-error validation case",
                failure_reason="an observed structured error did not contain error.code",
            ),
            _conditional_outcome(
                "P8",
                applicable=bool(confirmation_tool_names),
                passed=_passed(confirmation_check),
                not_applicable_reason="the pack declares no confirmation-gated tool",
                failure_reason="an unconfirmed call changed state or did not return pending status",
            ),
            _outcome(
                "P9",
                _passed(timeout_check),
                "the pinned gateway artifact has no independently verified bounded-timeout report",
            ),
            _outcome(
                "P10",
                _passed(mutation_check),
                "observed state changes did not match mutation declarations",
            ),
            _outcome(
                "P11",
                _passed(validation_check)
                and complete
                and tool_names <= observed_tools,
                "one or more declared result paths were unobserved or failed result mapping",
            ),
        ]
    }
