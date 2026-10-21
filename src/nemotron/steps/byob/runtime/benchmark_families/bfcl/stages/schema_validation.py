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

"""Validate expected traces against the pack's tool schemas.

The JSON-Schema subset itself lives in
:mod:`nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema`, so the
gold call a generation run certifies and the candidate call an evaluation run
scores are measured against the same declaration. What stays here is the part
that is specific to a generation trace: which tool was exposed, whether a
mutation was confirmed, and whether the trace respects the row's declared call
order and grouping.
"""

from __future__ import annotations

import logging
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import check_arguments
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    confirmation_protocol,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    SCHEMA_VALIDATED_TRACES,
    schema_validated_row,
    schema_validated_traces_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
    corrected_slot_values,
)

logger = logging.getLogger(__name__)

REJECT_REASON = "expected_trace_schema_mismatch"


def _tool_index(pack: LoadedPack) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for tool in pack.tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str):
            index[name] = function
    return index


def validate_task(
    pack: LoadedPack,
    task: dict[str, Any],
    expected_calls: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return schema failures for one task's expected trace."""
    tool_index = tools if tools is not None else _tool_index(pack)
    confirmation_tools = {
        str((tool.get("function") or {}).get("name"))
        for tool in pack.tools
        if tool.get("x-requires-confirmation")
    }
    confirm_parameter = confirmation_protocol(pack.manifest)["parameter"]
    corrected = corrected_slot_values(task)
    confirmed_turns = task.get("confirmed_call_turns")
    failures: list[dict[str, Any]] = []
    group_turns: dict[int, set[int]] = {}
    group_positions: dict[tuple[int, int], list[int]] = {}

    for call in expected_calls:
        for field in ("turn_index", "call_group", "position_in_group"):
            value = call.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                failures.append({"reason": "bad_struct_field", "field": field, "value": value})
        if not isinstance(call.get("arguments"), dict):
            failures.append({"reason": "arguments_not_object"})
            continue
        turn = call.get("turn_index")
        group = call.get("call_group")
        position = call.get("position_in_group")
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (turn, group, position)):
            group_turns.setdefault(group, set()).add(turn)
            group_positions.setdefault((turn, group), []).append(position)
        name = call.get("function_name")
        function = tool_index.get(str(name))
        if function is None:
            failures.append({"reason": "unknown_tool", "tool": name})
            continue
        if str(name) not in (task.get("tools_present") or []):
            failures.append({"reason": "tool_not_exposed", "tool": name})
        # A confirmed mutation is only gold when this very turn is covered by a
        # confirmation the user already gave and no later turn withdrew, whatever the
        # template's policy label says.
        if (
            str(name) in confirmation_tools
            and call["arguments"].get(confirm_parameter) is True
            and call.get("turn_index") not in (confirmed_turns or [])
        ):
            failures.append({"reason": "confirmed_mutation_without_user_confirmation", "tool": name})
        for argument, value in call["arguments"].items():
            if argument in corrected and value != corrected[argument]:
                failures.append(
                    {
                        "reason": "superseded_slot_value_in_trace",
                        "tool": name,
                        "argument": argument,
                        "slot": argument,
                    }
                )
        failures.extend({**failure, "tool": name} for failure in check_arguments(function, call["arguments"]))

    call_order = task.get("call_order", "strict")
    prefix = task.get("call_order_prefix")
    if task.get("turn_policy") == "dependent_call" and call_order != "strict":
        failures.append(
            {
                "reason": "dependent_call_requires_strict_order",
                "call_order": call_order,
            }
        )
    if call_order not in {"strict", "any", "prefix"}:
        failures.append({"reason": "unknown_call_order", "call_order": call_order})
    # required_tools declares the scoring order. For strict/prefix, the first
    # appearance of each required tool in the trace must follow that sequence.
    required = [str(name) for name in (task.get("required_tools") or [])]
    if call_order == "prefix":
        # The prefix counts required tools: a larger value would silently mean strict.
        if not isinstance(prefix, int) or not 1 <= prefix <= len(required):
            failures.append({"reason": "bad_call_order_prefix", "call_order_prefix": prefix})
    elif prefix is not None:
        failures.append({"reason": "call_order_prefix_without_prefix_order"})

    required_set = set(required)
    seen_required: list[str] = []
    for call in expected_calls:
        name = str(call.get("function_name"))
        if name in required_set and name not in seen_required:
            seen_required.append(name)
    if call_order == "strict" and required and seen_required != required:
        failures.append(
            {
                "reason": "call_order_mismatch",
                "call_order": "strict",
                "expected": required,
                "got": seen_required,
            }
        )
    elif call_order == "prefix" and isinstance(prefix, int) and required:
        expected_prefix = required[:prefix]
        got_prefix = seen_required[:prefix]
        if got_prefix != expected_prefix:
            failures.append(
                {
                    "reason": "call_order_mismatch",
                    "call_order": "prefix",
                    "expected": expected_prefix,
                    "got": got_prefix,
                }
            )
        if sorted(seen_required[prefix:]) != sorted(required[prefix:]):
            failures.append(
                {
                    "reason": "call_order_remainder_mismatch",
                    "expected": sorted(required[prefix:]),
                    "got": sorted(seen_required[prefix:]),
                }
            )

    for group, turns in group_turns.items():
        if len(turns) != 1:
            failures.append({"reason": "call_group_spans_turns", "call_group": group})
    for (turn, group), positions in group_positions.items():
        if sorted(positions) != list(range(len(positions))):
            failures.append(
                {
                    "reason": "non_contiguous_group_positions",
                    "turn_index": turn,
                    "call_group": group,
                    "positions": positions,
                }
            )

    if task.get("turn_policy") == "irrelevant" and expected_calls:
        failures.append({"reason": "irrelevant_task_has_calls"})
    for tool_name in required:
        if tool_name not in {call.get("function_name") for call in expected_calls}:
            failures.append({"reason": "required_tool_not_called", "tool": tool_name})
    return failures


def run_schema_validation(
    config: BfclConfig,
    pack: LoadedPack,
    tasks: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
    *,
    skipped: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Validate every trace and cache the per-task failures.

    ``skipped`` carries task ids dropped before this stage (for example a
    dependent_call bind failure). Those rows stay in the stage table so joins
    across artifacts still show every expanded ``task_id``.
    """
    tool_index = _tool_index(pack)
    skipped = skipped or {}
    failures: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in skipped:
            failures[task_id] = [{"reason": "trace_not_derived", "detail": skipped[task_id]}]
            rows.append(schema_validated_row(task, failures[task_id], "trace_not_derived"))
            continue
        task_failures = validate_task(pack, task, traces[task_id], tool_index)
        failures[task_id] = task_failures
        rows.append(schema_validated_row(task, task_failures, REJECT_REASON))

    write_stage_table(
        stage_cache_dir(config) / SCHEMA_VALIDATED_TRACES,
        rows,
        schema_validated_traces_schema(),
    )
    rejected = sum(1 for items in failures.values() if items)
    logger.info("BFCL schema_validation checked %d tasks (%d rejected)", len(failures), rejected)
    return failures
