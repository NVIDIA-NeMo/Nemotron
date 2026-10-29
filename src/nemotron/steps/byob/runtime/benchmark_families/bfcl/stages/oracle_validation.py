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

"""Validate an oracle pack and classify its eligibility tier."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.fixture_filter import evaluate_filter
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import HeldOutPolicy
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    validate_function_arguments,
    validate_tool_definition,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    confirmation_protocol,
    oracle_fixture_source_path,
    oracle_runtime_fixtures,
    pack_fingerprint,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.endpoint_conformance import (
    run_endpoint_conformance_check,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.executable_replay import replay_task
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
    ExpansionError,
    _candidates,
    check_category_budgets,
    declared_slot_updates,
    expand_template,
    primary_key_for,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
    _oracle_trace_resolver,
    build_expected_calls,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.held_out import (
    held_out_policy,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.mcp_target_probes import (
    assess_gateway_timeout_report,
    build_target_probe_report,
    load_gateway_conformance_report,
    run_endpoint_isolation_probe,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
    render_task,
    resolve_render_contract,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.schema_validation import (
    validate_task,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
    PlanError,
    build_plan,
)

logger = logging.getLogger(__name__)

# Bump whenever a check is added or tightened: a cached report was produced by the
# older rules and must not stand in for the newer ones.
VALIDATION_LOGIC_VERSION = 13

_PROBES = Path(__file__).resolve().parent.parent / "probes"
SLOW_BACKEND_PATH = _PROBES / "slow_backend.py"
SLOW_BACKEND_TOOL = "sleep_forever"


def derive_pack_tier(report: dict[str, Any]) -> tuple[bool, str]:
    """Derive the only authoritative eligibility result from report details."""
    checks = [*(report.get("checks") or []), *(report.get("extra_checks") or [])]
    all_pass = bool(checks) and all(
        check.get("status") == "pass" and not (check.get("failures") or [])
        for check in checks
    )
    stats = report.get("stats") or {}
    has_oracle = bool(stats.get("has_oracle", stats.get("has_backend")))
    has_templates = int(stats.get("n_templates") or 0) > 0
    has_assertions = int(stats.get("n_assertions") or 0) > 0
    has_tools = int(stats.get("n_tools") or 0) > 0
    if all_pass and has_oracle and has_templates and has_assertions:
        return True, "gold"
    if has_templates and has_tools:
        return False, "silver"
    return False, "prototype"


def validation_config_fingerprint(config: BfclConfig) -> str:
    """Hash every config value that can change an oracle-validation verdict."""
    runtime = config.oracle_runtime
    payload = {
        "validation_logic_version": VALIDATION_LOGIC_VERSION,
        "clock": runtime.clock,
        "worker": runtime.worker,
        "tool_timeout_s": runtime.tool_timeout_s,
        "assertion_timeout_s": runtime.assertion_timeout_s,
        "import_timeout_s": runtime.import_timeout_s,
        "reset_timeout_s": runtime.reset_timeout_s,
        "episode_timeout_s": runtime.episode_timeout_s,
        "allowed_roots": sorted(str(root.resolve()) for root in runtime.allowed_roots),
        "random_seed": int(config.random_seed or 0),
        # The representative-generation check compiles and renders one instance per
        # template, so the budget, the surface settings and the lineage policy that
        # decide those outcomes are part of what a cached verdict was granted under.
        "task_generation": config.task_generation,
        "surface_generation": config.surface_generation,
        "lineage_policy": config.lineage.policy,
        "lineage_roles": {
            name: {
                "enabled": role.enabled,
                "canonical_id": (
                    (role.model_config or {}).get("canonical_id")
                    if role.model_config
                    else None
                ),
            }
            for name, role in sorted((config.lineage.roles or {}).items())
        },
        "reference_content_hash": (
            config.reference_benchmark.content_hash
            if config.reference_benchmark is not None
            else None
        ),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _tool_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name") == name:
            return tool
    return None


def _parse_source(source: str) -> tuple[str, str]:
    if ":" in source:
        kind, rest = source.split(":", 1)
        return kind, rest
    return "fixture", source


def _classify_result(result: dict[str, Any], protocol: dict[str, str]) -> tuple[str, str | None]:
    if "error" in result and isinstance(result["error"], dict):
        return "structured_error", result["error"].get("code")
    if result.get(protocol["status_field"]) == protocol["pending_status"]:
        return "awaiting_confirmation", None
    return "success", None


def _held_out_not_found_context(
    case: dict[str, Any],
    policy: HeldOutPolicy | None,
    error_code: str | None,
) -> list[dict[str, str]]:
    """Identify probe arguments that equal rows removed from oracle state.

    Equality alone cannot prove an argument dereferences a fixture, so this is
    diagnostic context on an observed ``not_found``, never a static pack rejection.
    """
    if (
        policy is None
        or policy.fixtures_in_backend_state
        or error_code != "not_found"
        or not isinstance(case.get("arguments"), dict)
    ):
        return []
    matches: list[dict[str, str]] = []
    for reference in policy.fixture_refs:
        collection, primary_id = json.loads(reference)
        for argument, value in case["arguments"].items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                continue
            if str(value) == str(primary_id):
                matches.append(
                    {
                        "argument": str(argument),
                        "value": str(value),
                        "collection": str(collection),
                        "fixture_ref": reference,
                    }
                )
    return matches


def _representative_held_out_policy(
    policy: HeldOutPolicy | None,
    template_id: str,
) -> HeldOutPolicy | None:
    """Select the reservation contract for one validation representative.

    Public templates follow the generation policy. A reserved template is opened
    for validation; when reserved rows remain in backend state its private fixture
    inventory opens too, otherwise fixture blocking must stay aligned with runtime.
    """
    if policy is None or not policy.blocks_template(template_id):
        return policy
    if policy.fixtures_in_backend_state:
        return None
    return policy.model_copy(update={"template_ids": ()})


def run_oracle_validation(config: BfclConfig, pack: LoadedPack) -> dict[str, Any]:
    """Run contract, determinism, timeout, and isolation checks."""
    checks: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    tool_names = _tool_names(pack.tools)
    backend_path = pack.paths.backend_path
    endpoint_config = pack.endpoint_config
    oracle_available = backend_path is not None or endpoint_config is not None
    endpoint_metadata: dict[str, str] | None = None
    worker = ProcessWorker(
        default_timeout_s=config.oracle_runtime.episode_timeout_s,
        worker=config.oracle_runtime.worker,
    )
    clock_iso = config.oracle_runtime.clock
    seed = int(config.random_seed or 0)
    protocol = confirmation_protocol(pack.manifest)
    held_out = held_out_policy(pack)
    # Validation must probe the state generation will actually run against, or it
    # would grant gold on rows the oracle never sees.
    runtime_fixtures = oracle_runtime_fixtures(
        manifest=pack.manifest,
        fixtures=pack.fixtures,
        held_out=pack.held_out,
    )

    def run_episode(
        *,
        task_id: str,
        steps: list[dict[str, Any]],
        fixtures_override: dict[str, Any] | None = None,
    ) -> list[Any]:
        return worker.run_episode(
            backend_path=backend_path,
            endpoint_config=endpoint_config,
            fixtures=(
                copy.deepcopy(runtime_fixtures) if fixtures_override is None else fixtures_override
            ),
            clock_iso=clock_iso,
            seed=seed,
            task_id=task_id,
            steps=steps,
            import_root=pack.paths.pack_root,
            import_timeout_s=config.oracle_runtime.import_timeout_s,
            reset_timeout_s=config.oracle_runtime.reset_timeout_s,
            tool_timeout_s=config.oracle_runtime.tool_timeout_s,
            assertion_timeout_s=config.oracle_runtime.assertion_timeout_s,
            episode_timeout_s=config.oracle_runtime.episode_timeout_s,
            fixture_source_path=oracle_fixture_source_path(pack),
        )

    # --- Check 1: template tool names ---
    failures_1: list[dict[str, Any]] = []
    for template in pack.templates:
        tid = template.get("template_id")
        for field in ("required_tools", "tools_present"):
            for name in template.get(field) or []:
                if name not in tool_names:
                    failures_1.append({"template_id": tid, "field": field, "tool": name, "reason": "missing_tool"})
        for milestone in template.get("assistant_milestones") or []:
            if milestone.get("type") == "tool_call":
                name = milestone.get("tool")
                if name not in tool_names:
                    failures_1.append(
                        {
                            "template_id": tid,
                            "field": "assistant_milestones",
                            "tool": name,
                            "reason": "missing_tool",
                        }
                    )
        try:
            # Compile the conversation during prepare so a gold report cannot be
            # followed by a generate-time PlanError for the same unchanged template.
            build_plan(
                template,
                {
                    "task_id": f"validation:{tid}",
                    "template_id": tid,
                },
            )
        except PlanError as exc:
            failures_1.append(
                {
                    "template_id": tid,
                    "field": "assistant_milestones",
                    "reason": "invalid_conversation_plan",
                    "detail": str(exc),
                }
            )
    checks.append(
        {
            "id": 1,
            "name": "template_tool_names",
            "status": "pass" if not failures_1 else "fail",
            "failures": failures_1,
        }
    )

    # --- Check 2: slot sources ---
    failures_2: list[dict[str, Any]] = []
    fixtures = pack.fixtures or {}
    for template in pack.templates:
        tid = template.get("template_id")
        slots = template.get("slots") or {}
        # A correction's replacement resolves through a source of its own, so it has to
        # clear the same gate as the value it replaces.
        declared: list[tuple[str, dict[str, Any]]] = list(slots.items())
        try:
            declared.extend(
                (f"{slot_name}:correction", definition)
                for _, slot_name, definition in declared_slot_updates(template)
            )
        except ExpansionError as exc:
            failures_2.append({"template_id": tid, "reason": "invalid_slot_update", "detail": str(exc)})
        for slot_name, slot in declared:
            source = slot.get("source")
            if not source:
                failures_2.append({"template_id": tid, "slot": slot_name, "reason": "missing_source"})
                continue
            kind, rest = _parse_source(source)
            if kind == "fixture":
                if "." not in rest:
                    failures_2.append({"template_id": tid, "slot": slot_name, "reason": "bad_fixture_path"})
                    continue
                collection, field = rest.split(".", 1)
                rows = fixtures.get(collection)
                if not isinstance(rows, list):
                    failures_2.append(
                        {
                            "template_id": tid,
                            "slot": slot_name,
                            "reason": "unknown_collection",
                            "collection": collection,
                        }
                    )
                    continue
                try:
                    matched = [
                        row
                        for row in rows
                        if isinstance(row, dict)
                        and field in row
                        and evaluate_filter(row, slot.get("filter"))
                    ]
                except (ValueError, SyntaxError) as exc:
                    failures_2.append(
                        {
                            "template_id": tid,
                            "slot": slot_name,
                            "reason": "unevaluable_filter",
                            "detail": str(exc),
                        }
                    )
                    continue
                if not matched:
                    failures_2.append(
                        {
                            "template_id": tid,
                            "slot": slot_name,
                            "reason": "filter_matches_zero",
                            "collection": collection,
                        }
                    )
                    continue
                try:
                    key = primary_key_for(pack.manifest, collection, rows)
                except ExpansionError as exc:
                    failures_2.append(
                        {
                            "template_id": tid,
                            "slot": slot_name,
                            "reason": "primary_key_ambiguous",
                            "detail": str(exc),
                        }
                    )
                    continue
                missing_key = [
                    index for index, row in enumerate(matched) if row.get(key) is None
                ]
                if missing_key:
                    failures_2.append(
                        {
                            "template_id": tid,
                            "slot": slot_name,
                            "reason": "fixture_row_missing_primary_key",
                            "collection": collection,
                            "primary_key": key,
                            "rows": missing_key,
                        }
                    )
            elif kind == "enum":
                if "." not in rest:
                    failures_2.append({"template_id": tid, "slot": slot_name, "reason": "bad_enum_path"})
                    continue
                tool_name, param = rest.split(".", 1)
                tool = _tool_by_name(pack.tools, tool_name)
                props = ((tool or {}).get("function") or {}).get("parameters", {}).get("properties", {})
                if param not in props or "enum" not in props[param]:
                    failures_2.append({"template_id": tid, "slot": slot_name, "reason": "enum_missing"})
            elif kind in {"literal", "range", "absent"}:
                try:
                    _candidates(pack, str(slot_name), slot)
                except (ExpansionError, SyntaxError, TypeError, ValueError) as exc:
                    failures_2.append(
                        {
                            "template_id": tid,
                            "slot": slot_name,
                            "reason": "invalid_source",
                            "detail": str(exc),
                        }
                    )
            else:
                failures_2.append(
                    {"template_id": tid, "slot": slot_name, "reason": f"unknown_source_kind:{kind}"}
                )
    checks.append(
        {
            "id": 2,
            "name": "template_slot_sources",
            "status": "pass" if not failures_2 else "fail",
            "failures": failures_2,
        }
    )

    # --- Check 3: executable oracle ↔ schema ---
    failures_3: list[dict[str, Any]] = []
    for tool in pack.tools:
        function = tool.get("function") or {}
        for failure in validate_tool_definition(tool):
            failures_3.append(
                {"tool": function.get("name"), **failure}
            )
    if not oracle_available:
        failures_3.append({"reason": "oracle_missing"})
    else:
        try:
            inspection_steps = [{"op": "inspect_backend"}, {"op": "list_tools"}]
            if endpoint_config is not None:
                inspection_steps.append({"op": "metadata"})
            inspection_outputs = run_episode(
                task_id="inspect-backend",
                steps=inspection_steps,
                fixtures_override={},
            )
            inspection, listed_tools = inspection_outputs[:2]
            if endpoint_config is not None:
                endpoint_metadata = inspection_outputs[2]
            implemented = set(listed_tools)
            for symbol, callable_value in inspection.items():
                if not callable_value:
                    failures_3.append({"symbol": symbol, "reason": "missing_backend_symbol"})
        except Exception as exc:  # noqa: BLE001
            failures_3.append({"reason": "list_tools_failed", "detail": str(exc)})
            implemented = set()
        for name in tool_names:
            if name not in implemented:
                failures_3.append({"tool": name, "reason": "schema_without_implementation"})
        for name in implemented:
            if name not in tool_names:
                failures_3.append({"tool": name, "reason": "implementation_without_schema"})
    checks.append(
        {
            "id": 3,
            "name": "backend_schema_alignment",
            "status": "pass" if not failures_3 else "fail",
            "failures": failures_3,
        }
    )

    # --- Check 4: assertions importable ---
    failures_4: list[dict[str, Any]] = []
    assertion_report: dict[str, dict[str, Any]] = {}
    try:
        assertion_report = worker.inspect_assertions(
            pack.paths.assertions_path,
            import_root=pack.paths.pack_root,
            # Importing the module is what this pays for, not running an assertion.
            timeout_s=config.oracle_runtime.import_timeout_s,
            fixture_source_path=oracle_fixture_source_path(pack),
            fixtures=copy.deepcopy(runtime_fixtures),
        )
    except Exception as exc:  # noqa: BLE001
        failures_4.append({"reason": "assertions_import_failed", "detail": str(exc)})
    for template in pack.templates:
        declared_assertions = template.get("success_assertions") or []
        if not declared_assertions:
            # Without an assertion the task has no statement of what success means, so
            # replay could only ever confirm the trace ran, not that it was right.
            failures_4.append(
                {
                    "template": template.get("template_id"),
                    "reason": "template_without_success_assertion",
                }
            )
        for name in declared_assertions:
            if name not in assertion_report:
                failures_4.append({"assertion": name, "reason": "missing_assertion"})
            elif not assertion_report[name]["valid"]:
                failures_4.append(
                    {
                        "assertion": name,
                        "reason": "invalid_signature",
                        "detail": assertion_report[name]["reason"],
                    }
                )
            elif assertion_report[name]["capabilities"] is None:
                failures_4.append(
                    {
                        "assertion": name,
                        "reason": "invalid_assertion_capability",
                        "detail": assertion_report[name]["capability_reason"],
                    }
                )
            elif not assertion_report[name]["capabilities"]["executable"]:
                failures_4.append(
                    {
                        "assertion": name,
                        "reason": "assertion_not_executable_compatible",
                    }
                )
    checks.append(
        {
            "id": 4,
            "name": "assertions_importable",
            "status": "pass" if not failures_4 else "fail",
            "failures": failures_4,
        }
    )

    # --- Check 5: declared validation cases ---
    failures_5: list[dict[str, Any]] = []
    coverage: dict[str, set[str]] = {name: set() for name in tool_names}
    successful_case_ids: set[str] = set()
    structured_error_case_ids: set[str] = set()
    invalid_case_ids: set[str] = set()
    case_schema_failures: dict[str, list[dict[str, Any]]] = {}
    case_prefixes: dict[str, list[dict[str, Any]]] = {}
    case_by_id = {str(case.get("id")): case for case in pack.validation_cases}
    observed_calls: list[dict[str, Any]] = []
    observed_state_deltas: list[dict[str, Any]] = []
    expected_observation_count = 0
    # Which tools were seen to change state, and in which probe. Compared against the
    # tools.json claim once every case has run.
    observed_mutations: dict[str, list[str]] = {}
    # Probing a backend that disagrees with tools.json would report noise, so the
    # check is recorded as skipped rather than passed — a skip is never gold.
    skipped_5: str | None = None
    if not oracle_available:
        skipped_5 = "pack declares no executable oracle"
    elif failures_3:
        skipped_5 = "backend_schema_alignment failed, so probe results cannot be trusted"
    if skipped_5 is None:
        if not pack.validation_cases:
            failures_5.append({"reason": "validation_cases_empty"})
        for case in pack.validation_cases:
            tool = case.get("tool")
            if tool not in tool_names:
                failures_5.append(
                    {
                        "id": case.get("id"),
                        "reason": "unknown_tool",
                        "tool": tool,
                    }
                )
                invalid_case_ids.add(str(case.get("id")))
                continue
            function = (_tool_by_name(pack.tools, str(tool)) or {}).get("function") or {}
            argument_failures = validate_function_arguments(
                function,
                {} if case.get("arguments") is None else case.get("arguments"),
            )
            if argument_failures:
                # Deliberately malformed calls are legitimate negative probes. They
                # must never prove success coverage, though: success has to be
                # demonstrated through the same schema exposed to the model.
                case_schema_failures[str(case.get("id"))] = argument_failures
        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []
        for case in pack.validation_cases:
            if str(case.get("id")) in invalid_case_ids:
                # Already recorded above; do not invoke an invalid payload on the backend.
                continue
            if case.get("reset_before", True):
                if current_group:
                    groups.append(current_group)
                current_group = [case]
            elif not current_group:
                failures_5.append(
                    {
                        "id": case.get("id"),
                        "reason": "reset_before_false_without_predecessor",
                    }
                )
                # Do not execute an uninitialized episode and add misleading backend
                # failures on top of the structural contract error.
                continue
            else:
                current_group.append(case)
        if current_group:
            groups.append(current_group)
        expected_observation_count = sum(len(group) for group in groups)
        for group in groups:
            for index, case in enumerate(group):
                case_prefixes[str(case.get("id"))] = group[: index + 1]

        for group_index, group in enumerate(groups):
            steps: list[dict[str, Any]] = []
            locations: list[tuple[dict[str, Any], int | None, int, int | None]] = []
            for case in group:
                expect = case.get("expect") or {}
                if case.get("reset_before", True):
                    steps.append({"op": "reset"})
                # Snapshot around every case, not only the ones that declare
                # state_unchanged: the same pair also shows whether a tool mutates,
                # which is what x-mutates claims.
                before_index = len(steps)
                steps.append({"op": "get_state"})
                call_index = len(steps)
                steps.append(
                    {
                        "op": "call_tool",
                        "name": case.get("tool"),
                        "arguments": case.get("arguments") or {},
                    }
                )
                after_index = len(steps)
                steps.append({"op": "get_state"})
                locations.append((case, before_index, call_index, after_index))
            try:
                outputs = run_episode(
                    task_id=f"validation-group:{group_index}",
                    steps=steps,
                )
                for case, before_index, call_index, after_index in locations:
                    case_id = case.get("id")
                    tool = case.get("tool")
                    expect = case.get("expect") or {}
                    result = outputs[call_index]
                    before = outputs[before_index]
                    after = outputs[after_index]
                    if after != before:
                        observed_mutations.setdefault(str(tool), []).append(str(case_id))

                    if not isinstance(result, dict):
                        failures_5.append({"id": case_id, "reason": "non_object_result"})
                        continue
                    canonical_json(result)
                    result_class, error_code = _classify_result(result, protocol)
                    before_json = canonical_json(before)
                    after_json = canonical_json(after)
                    observed_calls.append(
                        {
                            "case_id": str(case_id),
                            "tool": str(tool),
                            "arguments": copy.deepcopy(case.get("arguments") or {}),
                            "result_class": result_class,
                            "error_code": error_code,
                            "result": copy.deepcopy(result),
                        }
                    )
                    observed_state_deltas.append(
                        {
                            "case_id": str(case_id),
                            "tool": str(tool),
                            "before_digest": "sha256:"
                            + hashlib.sha256(before_json.encode("utf-8")).hexdigest(),
                            "after_digest": "sha256:"
                            + hashlib.sha256(after_json.encode("utf-8")).hexdigest(),
                            "changed": before_json != after_json,
                        }
                    )
                    if result_class == "success":
                        argument_failures = case_schema_failures.get(str(case_id), [])
                        if argument_failures:
                            failures_5.append(
                                {
                                    "id": case_id,
                                    "reason": "successful_validation_case_schema_mismatch",
                                    "failures": argument_failures,
                                }
                            )
                        else:
                            coverage.setdefault(tool, set()).add("success")
                            successful_case_ids.add(str(case_id))
                    elif result_class == "structured_error":
                        structured_error_case_ids.add(str(case_id))
                    if result_class in {"structured_error", "awaiting_confirmation"} or case.get(
                        "coverage"
                    ) == "negative":
                        coverage.setdefault(tool, set()).add("negative")

                    held_out_context = _held_out_not_found_context(case, held_out, error_code)
                    if expect.get("result_class") and result_class != expect["result_class"]:
                        failure = {
                            "id": case_id,
                            "reason": "result_class_mismatch",
                            "expected": expect.get("result_class"),
                            "got": result_class,
                            "result": result,
                        }
                        if held_out_context:
                            failure["held_out_fixture_arguments"] = held_out_context
                        failures_5.append(failure)
                    if "error_code" in expect and expect["error_code"] != error_code:
                        failure = {
                            "id": case_id,
                            "reason": "error_code_mismatch",
                            "expected": expect.get("error_code"),
                            "got": error_code,
                        }
                        if held_out_context:
                            failure["held_out_fixture_arguments"] = held_out_context
                        failures_5.append(failure)
                    for field, expected_value in expect.items():
                        if field in {"result_class", "error_code", "state_unchanged"}:
                            continue
                        if result.get(field) != expected_value:
                            failures_5.append(
                                {
                                    "id": case_id,
                                    "reason": "result_field_mismatch",
                                    "field": field,
                                    "expected": expected_value,
                                    "got": result.get(field),
                                }
                            )
                    if expect.get("state_unchanged") and after != before:
                        failures_5.append({"id": case_id, "reason": "state_changed"})
            except Exception as exc:  # noqa: BLE001
                failures_5.append(
                    {
                        "ids": [case.get("id") for case in group],
                        "reason": "raised",
                        "detail": str(exc),
                    }
                )
        for name in tool_names:
            missing = {"success", "negative"} - coverage.get(name, set())
            if missing:
                failures_5.append(
                    {
                        "tool": name,
                        "reason": "incomplete_validation_coverage",
                        "missing": sorted(missing),
                    }
                )
    checks.append(
        {
            "id": 5,
            "name": "declared_validation_cases",
            "status": "skipped" if skipped_5 else ("pass" if not failures_5 else "fail"),
            "failures": failures_5
            if skipped_5 is None
            else [{"reason": "not_run", "detail": skipped_5}],
        }
    )

    def replay_case_with_context(
        case_id: str,
        *,
        task_id: str,
        include_state: bool = False,
    ) -> Any:
        """Replay a validation case with every chained predecessor it depends on."""
        prefix = case_prefixes.get(case_id)
        if not prefix:
            raise LookupError(f"validation case {case_id!r} has no executable prefix")
        steps: list[dict[str, Any]] = []
        last_call_index = -1
        for case in prefix:
            if case.get("reset_before", True):
                steps.append({"op": "reset"})
            last_call_index = len(steps)
            steps.append(
                {
                    "op": "call_tool",
                    "name": case.get("tool"),
                    "arguments": case.get("arguments") or {},
                }
            )
        if include_state:
            steps.append({"op": "get_state"})
        outputs = run_episode(task_id=task_id, steps=steps)
        result = outputs[last_call_index]
        if not include_state:
            return result
        return {"result": result, "state": outputs[-1]}

    # --- Check 6: confirmation policy ---
    failures_6: list[dict[str, Any]] = []
    if oracle_available:
        for tool in pack.tools:
            if not tool.get("x-requires-confirmation"):
                continue
            function = tool.get("function") or {}
            name = function.get("name")
            params = (function.get("parameters") or {}).get("properties") or {}
            if protocol["parameter"] not in params:
                failures_6.append(
                    {
                        "tool": name,
                        "reason": "missing_confirm_param",
                        "parameter": protocol["parameter"],
                    }
                )
                continue
            probe = next(
                (
                    case
                    for case in pack.validation_cases
                    if case.get("tool") == name
                    and (case.get("arguments") or {}).get(protocol["parameter"]) is False
                ),
                None,
            )
            if probe is None:
                failures_6.append({"tool": name, "reason": "missing_confirm_false_case"})
                continue
            try:
                outputs = run_episode(
                    task_id=f"confirm_policy:{name}",
                    steps=[
                        {"op": "reset"},
                        {"op": "get_state"},
                        {"op": "call_tool", "name": name, "arguments": probe.get("arguments") or {}},
                        {"op": "get_state"},
                    ],
                )
                before, result, after = outputs[1], outputs[2], outputs[3]
                if after != before:
                    failures_6.append({"tool": name, "reason": "confirm_false_mutated_state"})
                if (
                    not isinstance(result, dict)
                    or result.get(protocol["status_field"]) != protocol["pending_status"]
                ):
                    failures_6.append({"tool": name, "reason": "confirm_false_bad_status", "result": result})
            except Exception as exc:  # noqa: BLE001
                failures_6.append({"tool": name, "reason": "raised", "detail": str(exc)})
    checks.append(
        {
            "id": 6,
            "name": "confirmation_policy",
            "status": "pass" if not failures_6 else "fail",
            "failures": failures_6,
        }
    )

    # --- Check 7: representative generation contract ---
    # Static source/plan checks do not prove that milestone arguments can actually be
    # bound, that the resulting call satisfies its tool schema, or that the pack can
    # state the conversation in the render language. Compile and render the first
    # deterministic instance of every template through the path generation uses.
    failures_7: list[dict[str, Any]] = []
    skipped_7: str | None = None
    prerequisites = [*failures_1, *failures_2, *failures_3]
    if prerequisites:
        skipped_7 = "template, source, or backend-schema validation failed"
    else:
        reserved_templates = {
            str(template.get("template_id"))
            for template in pack.templates
            if held_out is not None and held_out.blocks_template(template.get("template_id"))
        }
        # Only the budget is settled over what generation may bind: a category a
        # reserved template cannot fill is a shortfall Stage 4 would hit.
        bindable = [
            template
            for template in pack.templates
            if str(template.get("template_id")) not in reserved_templates
        ]
        templates_by_id = {str(template.get("template_id")): template for template in pack.templates}
        render_contract: dict[str, Any] | None = None
        # The budget and the run-wide render inputs govern the whole run rather than one
        # template, so a pack that cannot satisfy them has no template worth compiling.
        try:
            check_category_budgets(
                bindable,
                int(
                    config.task_generation.get(
                        "candidate_tasks_per_category",
                        config.task_generation.get("tasks_per_category", 1),
                    )
                    or 1
                ),
            )
            render_contract = resolve_render_contract(config, pack, templates_by_id)
        except Exception as exc:  # noqa: BLE001 — report the run-wide contract failure
            failures_7.append({"reason": "run_contract_failed", "detail": str(exc)})
        representative = pack.templates if render_contract is not None else []
        for template in representative:
            template_id = str(template.get("template_id"))
            try:
                instances = expand_template(
                    pack,
                    template,
                    1,
                    seed,
                    held_out=_representative_held_out_policy(held_out, template_id),
                )
                if not instances:
                    raise ExpansionError("template produced no representative instance")
                task = instances[0]
                plan = build_plan(template, task)
                task["is_multi_turn"] = plan["is_multi_turn"]
                task["num_tool_calls"] = plan["num_tool_calls"]
                task["has_user_confirmation"] = plan["has_user_confirmation"]
                task["confirmed_call_turns"] = plan["confirmed_call_turns"]
                calls = build_expected_calls(
                    pack,
                    task,
                    plan,
                    resolve_trace=_oracle_trace_resolver(worker, config, pack, task),
                )
                schema_failures = validate_task(pack, task, calls)
                if schema_failures:
                    failures_7.append(
                        {
                            "template_id": template_id,
                            "reason": "representative_trace_schema_mismatch",
                            "failures": schema_failures,
                        }
                    )
                surface = render_task(
                    pack,
                    template,
                    task,
                    plan,
                    language=render_contract["language"],
                    prompt_bundle=render_contract["prompt_bundle"],
                    tool_names=render_contract["tool_names"],
                    preserve_slot_values=bool(
                        config.surface_generation.get("preserve_slot_values", True)
                    ),
                    prevent_tool_name_leakage=bool(
                        config.surface_generation.get("prevent_tool_name_leakage", True)
                    ),
                )
                # Guard violations drop rows during generation. A template whose own
                # representative surface breaks a guard contributes nothing publishable,
                # which is a template defect rather than instance-level noise.
                if surface["guard_violations"]:
                    failures_7.append(
                        {
                            "template_id": template_id,
                            "reason": "representative_surface_guard_violation",
                            "failures": surface["guard_violations"],
                        }
                    )
                if not schema_failures and not surface["guard_violations"]:
                    verdict = replay_task(worker, config, pack, task, calls)
                    if not verdict["passed"]:
                        failures_7.append(
                            {
                                "template_id": template_id,
                                "reason": "representative_replay_failed",
                                "detail": verdict.get("detail"),
                                "replay_reason": verdict.get("reason"),
                            }
                        )
            except Exception as exc:  # noqa: BLE001 — report the template contract failure
                failures_7.append(
                    {
                        "template_id": template_id,
                        "reason": "representative_generation_failed",
                        "detail": str(exc),
                    }
                )
    checks.append(
        {
            "id": 7,
            "name": "representative_generation_contract",
            "status": "skipped" if skipped_7 else ("pass" if not failures_7 else "fail"),
            "failures": (
                failures_7
                if skipped_7 is None
                else [{"reason": "not_run", "detail": skipped_7}]
            ),
        }
    )

    # --- Extra: M1 declared mutation matches observed mutation ---
    failures_m1: list[dict[str, Any]] = []
    mutation_status = "pass"
    if skipped_5 is not None:
        mutation_status = "skipped"
        failures_m1.append({"reason": "not_run", "detail": skipped_5})
    else:
        for tool in pack.tools:
            function = tool.get("function") or {}
            name = str(function.get("name"))
            cases = observed_mutations.get(name) or []
            if cases and not tool.get("x-mutates"):
                failures_m1.append(
                    {
                        "tool": name,
                        "reason": "undeclared_mutation",
                        "cases": cases,
                    }
                )
            successful_mutations = sorted(set(cases) & successful_case_ids)
            if tool.get("x-mutates") and not successful_mutations:
                failures_m1.append(
                    {
                        "tool": name,
                        "reason": "declared_mutation_not_observed",
                        "successful_cases": sorted(
                            case_id
                            for case_id in successful_case_ids
                            if str(case_by_id[case_id].get("tool")) == name
                        ),
                    }
                )
        if failures_m1:
            mutation_status = "fail"
    extras.append(
        {
            "id": "M1",
            "name": "mutation_declaration",
            "status": mutation_status,
            "failures": failures_m1,
        }
    )

    # --- Extra: D1 determinism ---
    failures_d1: list[dict[str, Any]] = []
    if oracle_available:
        success_by_tool: dict[str, dict[str, Any]] = {}
        for case in pack.validation_cases:
            if str(case.get("id")) in successful_case_ids:
                success_by_tool.setdefault(str(case.get("tool")), case)
        missing_determinism = sorted(tool_names - set(success_by_tool))
        for name in missing_determinism:
            failures_d1.append({"tool": name, "reason": "no_success_case_to_repeat"})
        for case in success_by_tool.values():
            observations: list[str] = []
            try:
                for _ in range(2):
                    observations.append(
                        canonical_json(
                            replay_case_with_context(
                                str(case.get("id")),
                                task_id=f"d1:{case.get('id')}",
                                include_state=True,
                            )
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                failures_d1.append({"id": case.get("id"), "reason": "raised", "detail": str(exc)})
                continue
            if observations[0] != observations[1]:
                failures_d1.append({"id": case.get("id"), "reason": "nondeterministic"})
    extras.append(
        {
            "id": "D1",
            "name": "determinism_check",
            "status": "pass" if not failures_d1 else "fail",
            "failures": failures_d1,
        }
    )

    # --- Extra: D2 error shape ---
    failures_d2: list[dict[str, Any]] = []
    for case in pack.validation_cases:
        if str(case.get("id")) not in structured_error_case_ids:
            continue
        if not oracle_available:
            break
        try:
            result = replay_case_with_context(
                str(case.get("id")),
                task_id=f"d2:{case.get('id')}",
            )
        except Exception as exc:  # noqa: BLE001
            failures_d2.append({"id": case.get("id"), "reason": "raised", "detail": str(exc)})
            continue
        err = result.get("error") if isinstance(result, dict) else None
        # A machine-readable code is the part scoring needs; entity, id, and field are
        # optional detail, so a domain whose failures are not about one entity is not
        # forced to pad the envelope with nulls.
        if not isinstance(err, dict):
            failures_d2.append({"id": case.get("id"), "reason": "missing_error_object"})
        elif not isinstance(err.get("code"), str) or not err["code"]:
            failures_d2.append(
                {
                    "id": case.get("id"),
                    "reason": "error_without_code",
                    "detail": sorted(err),
                }
            )
    extras.append(
        {
            "id": "D2",
            "name": "error_shape",
            "status": "pass" if not failures_d2 else "fail",
            "failures": failures_d2,
        }
    )

    # --- Extra: T1 timeout wiring ---
    failures_t1: list[dict[str, Any]] = []
    timeout_status = "pass"
    if config.oracle_runtime.worker == "thread":
        timeout_status = "skipped"
        failures_t1.append(
            {"reason": "not_run", "detail": "thread workers cannot enforce hard timeouts"}
        )
    else:
        # Drive the same episode worker a pack tool uses, with a tool that never
        # returns: this proves the per-operation deadline and the kill that follows it,
        # not merely that a standalone callable can be timed out.
        deadline_s = 0.5
        started_t1 = time.monotonic()
        try:
            worker.run_episode(
                backend_path=SLOW_BACKEND_PATH,
                fixtures=None,
                clock_iso=config.oracle_runtime.clock,
                seed=0,
                task_id="t1:timeout_wiring",
                steps=[{"op": "call_tool", "name": SLOW_BACKEND_TOOL, "arguments": {}}],
                import_timeout_s=config.oracle_runtime.import_timeout_s,
                tool_timeout_s=deadline_s,
                episode_timeout_s=config.oracle_runtime.import_timeout_s + 4 * deadline_s,
            )
            failures_t1.append({"reason": "hung_call_was_not_terminated"})
        except TimeoutError:
            elapsed = time.monotonic() - started_t1
            budget = config.oracle_runtime.import_timeout_s + 4 * deadline_s
            if elapsed > budget:
                failures_t1.append(
                    {"reason": "timeout_overran_its_budget", "detail": f"{elapsed:.2f}s"}
                )
        except Exception as exc:  # noqa: BLE001
            failures_t1.append({"reason": "timeout_worker_failed", "detail": str(exc)})
        if failures_t1:
            timeout_status = "fail"
    extras.append(
        {
            "id": "T1",
            "name": "timeout_wiring",
            "status": timeout_status,
            "failures": failures_t1,
        }
    )

    isolation_failures = []
    if config.oracle_runtime.worker != "process":
        isolation_failures.append(
            {
                "reason": "process_worker_required_for_gold",
                "worker": config.oracle_runtime.worker,
            }
        )
    extras.append(
        {
            "id": "I1",
            "name": "gold_isolation",
            "status": "pass" if not isolation_failures else "fail",
            "failures": isolation_failures,
        }
    )

    mcp_observations: dict[str, Any] | None = None
    mcp_probe_report: dict[str, Any] | None = None
    gateway_conformance_report: dict[str, Any] | None = None
    if endpoint_config is not None and endpoint_config.attestation is not None:
        mutating_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in pack.tools
            if tool.get("x-mutates")
        }
        mutation_case = next(
            (
                case
                for case in pack.validation_cases
                if str(case.get("id")) in successful_case_ids
                and str(case.get("tool")) in mutating_names
                and str(case.get("id"))
                in set(observed_mutations.get(str(case.get("tool"))) or [])
            ),
            None,
        )
        activity_case = mutation_case or next(
            (
                case
                for case in pack.validation_cases
                if str(case.get("id")) in successful_case_ids
            ),
            None,
        )
        if activity_case is None:
            extras.append(
                {
                    "id": "MP6",
                    "name": "mcp_episode_isolation",
                    "status": "fail",
                    "failures": [{"reason": "no_success_case_for_isolation_probe"}],
                }
            )
        else:
            extras.append(
                run_endpoint_isolation_probe(
                    endpoint_config,
                    fixtures=copy.deepcopy(runtime_fixtures),
                    clock_iso=clock_iso,
                    seed=seed,
                    timeout_s=config.oracle_runtime.tool_timeout_s,
                    activity_case=activity_case,
                    expect_state_change=mutation_case is not None,
                )
            )
        try:
            gateway_conformance_report = load_gateway_conformance_report(
                pack.paths.pack_root
            )
            extras.append(
                assess_gateway_timeout_report(gateway_conformance_report)
            )
        except (OSError, UnicodeError, ValueError) as exc:
            extras.append(
                {
                    "id": "MP9",
                    "name": "mcp_gateway_timeout_conformance",
                    "status": "fail",
                    "failures": [
                        {
                            "reason": "gateway_conformance_report_invalid",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                }
            )
        observations_complete = (
            skipped_5 is None
            and len(observed_calls) == expected_observation_count
            and len(observed_state_deltas) == expected_observation_count
        )
        mcp_observations = {
            "calls_complete": observations_complete,
            "calls": observed_calls,
            "state_deltas_complete": observations_complete,
            "state_deltas": observed_state_deltas,
        }
        mcp_probe_report = build_target_probe_report(
            checks=checks,
            extra_checks=extras,
            endpoint_metadata=endpoint_metadata,
            observations=mcp_observations,
            tool_names=tool_names,
            confirmation_tool_names={
                str((tool.get("function") or {}).get("name"))
                for tool in pack.tools
                if tool.get("x-requires-confirmation")
            },
            structured_error_declared=any(
                (case.get("expect") or {}).get("result_class") == "structured_error"
                for case in pack.validation_cases
            ),
        )

    conformance_check = run_endpoint_conformance_check(
        endpoint_config,
        endpoint_metadata,
        probe_report=mcp_probe_report,
        gateway_conformance_report=gateway_conformance_report,
    )
    if conformance_check is not None:
        extras.append(conformance_check)

    report = {
        "pack_id": pack.manifest.get("pack_id"),
        "pack_version": pack.manifest.get("version"),
        "pack_fingerprint": pack_fingerprint(pack.paths),
        "validation_config_fingerprint": validation_config_fingerprint(config),
        "checks": checks,
        "extra_checks": extras,
        "stats": {
            "n_tools": len(tool_names),
            "n_templates": len(pack.templates),
            "n_assertions": len(assertion_report),
            "n_fixture_collections": len(fixtures),
            "n_validation_cases": len(pack.validation_cases),
            "has_backend": backend_path is not None,
            "has_oracle": oracle_available,
            "oracle_kind": "endpoint" if endpoint_config is not None else "python",
        },
        "endpoint_metadata": endpoint_metadata,
    }
    if mcp_observations is not None and mcp_probe_report is not None:
        report["mcp_observations"] = mcp_observations
        report["mcp_probe_report"] = mcp_probe_report
    if gateway_conformance_report is not None:
        report["mcp_gateway_conformance_report"] = gateway_conformance_report
    gold_eligible, tier = derive_pack_tier(report)
    report.update(gold_eligible=gold_eligible, tier=tier)

    cache = stage_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "oracle_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "BFCL oracle_validation pack_id=%s tier=%s gold_eligible=%s",
        report["pack_id"],
        tier,
        gold_eligible,
    )
    return report
