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

"""Resolve expected dependent calls only from prior live oracle evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedCall,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    DependencyResolution,
    ExecutableEpisode,
    ExecutedToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableDependency,
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    json_equal,
    thaw_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    declared_function,
    validate_function_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _failure(
    dependency: ExecutableDependency,
    *,
    status: str,
    reason: str,
    detail: str,
    producer_execution_index: int | None = None,
) -> DependencyResolution:
    return DependencyResolution(
        dependency_index=dependency.dependency_index,
        consumer_call_index=dependency.consumer_call_index,
        consumer_turn_index=dependency.consumer_turn_index,
        argument_path=dependency.argument_path,
        producer_call_index=dependency.producer_call_index,
        producer_execution_index=producer_execution_index,
        result_path=dependency.result_path,
        status=status,
        reason_code=reason,
        detail=detail,
    )


def _extract_result(result: Any, path: str) -> Any:
    # A structured rejection is identified the way the driver classifies one, so a
    # pack whose successful result carries a null or descriptive ``error`` field
    # still yields its dependency value.
    if not isinstance(result, dict) or isinstance(result.get("error"), dict):
        raise ValueError("the producer did not return a successful result object")
    current = result
    for token in path.split("."):
        if isinstance(current, list):
            if not token.lstrip("-").isdigit():
                raise KeyError(token)
            index = int(token)
            if not -len(current) <= index < len(current):
                raise KeyError(token)
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        else:
            raise KeyError(token)
    if isinstance(current, (dict, list)):
        raise TypeError("the dependency result path resolved to a container")
    return current


def _has_expected_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return expected == "string" and isinstance(value, str)


def _set_argument_path(
    arguments: dict[str, Any],
    path: Sequence[str | int],
    value: Any,
) -> None:
    current: Any = arguments
    for token in path[:-1]:
        current = current[token]
    current[path[-1]] = value


def _resolved_call(
    call: ScriptedCall,
    values: Sequence[tuple[ExecutableDependency, Any]],
) -> ScriptedCall:
    arguments = thaw_json(call.arguments)
    for dependency, value in values:
        _set_argument_path(arguments, dependency.argument_path, value)
    return call.model_copy(update={"arguments": arguments})


def _resolved_records(
    extracted: Sequence[tuple[ExecutableDependency, int, Any]],
    *,
    before_dependency_index: int | None = None,
) -> tuple[DependencyResolution, ...]:
    return tuple(
        DependencyResolution(
            dependency_index=dependency.dependency_index,
            consumer_call_index=dependency.consumer_call_index,
            consumer_turn_index=dependency.consumer_turn_index,
            argument_path=dependency.argument_path,
            producer_call_index=dependency.producer_call_index,
            producer_execution_index=execution_index,
            result_path=dependency.result_path,
            status="resolved",
            resolved_value=value,
            resolved_value_hash=_sha256_json(value),
            reason_code="dependency.resolved",
            detail="the expected argument was derived from the paired live result",
        )
        for dependency, execution_index, value in extracted
        if before_dependency_index is None
        or dependency.dependency_index < before_dependency_index
    )


def resolve_turn_dependencies(
    *,
    task: ExecutableTaskSpec,
    turn: ScriptedTurn,
    executions: Sequence[ExecutedToolCall],
    producer_executions: Mapping[int, Sequence[int]],
) -> tuple[ScriptedTurn, tuple[DependencyResolution, ...]]:
    """Resolve one expected turn from live results, without changing candidate calls."""

    dependencies = tuple(
        item
        for item in task.dependencies
        if item.consumer_turn_index == turn.turn_index
    )
    if not dependencies:
        return turn, ()

    extracted: list[tuple[ExecutableDependency, int, Any]] = []
    for dependency in dependencies:
        candidates = tuple(producer_executions.get(dependency.producer_call_index, ()))
        if not candidates:
            return turn, (
                *_resolved_records(extracted),
                _failure(
                    dependency,
                    status="producer_missing",
                    reason="dependency.producer_missing",
                    detail="the expected producer has no paired live execution",
                ),
            )
        if len(candidates) != 1:
            return turn, (
                *_resolved_records(extracted),
                _failure(
                    dependency,
                    status="producer_ambiguous",
                    reason="dependency.producer_ambiguous",
                    detail="the expected producer maps to multiple live executions",
                ),
            )
        execution_index = candidates[0]
        if execution_index >= len(executions):
            return turn, (
                *_resolved_records(extracted),
                _failure(
                    dependency,
                    status="producer_missing",
                    reason="dependency.producer_missing",
                    detail="the paired producer execution is absent from episode evidence",
                    producer_execution_index=execution_index,
                ),
            )
        producer = executions[execution_index]
        if producer.status != "completed" or producer.result is None:
            return turn, (
                *_resolved_records(extracted),
                _failure(
                    dependency,
                    status="result_unavailable",
                    reason="dependency.result_unavailable",
                    detail="the live producer did not return a successful canonical result",
                    producer_execution_index=execution_index,
                ),
            )
        try:
            value = _extract_result(thaw_json(producer.result), dependency.result_path)
        except (KeyError, TypeError, ValueError):
            return turn, (
                *_resolved_records(extracted),
                _failure(
                    dependency,
                    status="result_path_missing",
                    reason="dependency.result_path_missing",
                    detail="the declared path does not identify a scalar in the live result",
                    producer_execution_index=execution_index,
                ),
            )
        if not _has_expected_type(value, dependency.expected_value_type):
            return turn, (
                *_resolved_records(extracted),
                _failure(
                    dependency,
                    status="result_type_mismatch",
                    reason="dependency.result_type_mismatch",
                    detail=(
                        "the live dependency value has a different JSON type than "
                        "the verified expected argument"
                    ),
                    producer_execution_index=execution_index,
                ),
            )
        extracted.append((dependency, execution_index, value))

    by_call: dict[int, list[tuple[ExecutableDependency, Any]]] = {}
    for dependency, _, value in extracted:
        by_call.setdefault(dependency.consumer_call_index, []).append(
            (dependency, value)
        )
    resolved_calls: list[ScriptedCall] = []
    for call in turn.calls:
        values = by_call.get(call.call_index, ())
        resolved = _resolved_call(call, values) if values else call
        function = declared_function(task.script.tools, resolved.function_name)
        failures = (
            [{"reason": "unknown_tool"}]
            if function is None
            else validate_function_arguments(
                function,
                thaw_json(resolved.arguments),
            )
        )
        if values and failures:
            dependency = values[0][0]
            execution_index = next(
                index
                for item, index, _ in extracted
                if item.dependency_index == dependency.dependency_index
            )
            return turn, (
                *_resolved_records(
                    extracted,
                    before_dependency_index=dependency.dependency_index,
                ),
                _failure(
                    dependency,
                    status="consumer_schema_invalid",
                    reason="dependency.consumer_schema_invalid",
                    detail="live dependency substitution violates the consumer tool schema",
                    producer_execution_index=execution_index,
                ),
            )
        resolved_calls.append(resolved)

    resolutions = _resolved_records(extracted)
    return turn.model_copy(update={"calls": tuple(resolved_calls)}), resolutions


def dependency_execution_map(
    episode: ExecutableEpisode,
) -> dict[int, list[int]]:
    """Map expected call indexes to the live outcomes paired with them."""

    mapped: dict[int, list[int]] = {}
    for turn in episode.observed:
        if not turn.paired_call_indexes:
            continue
        for expected_index, execution_index in zip(
            turn.paired_call_indexes,
            turn.tool_call_outcome_indexes,
            strict=True,
        ):
            mapped.setdefault(expected_index, []).append(execution_index)
    return mapped


def resolved_script_from_episode(
    *,
    task: ExecutableTaskSpec,
    episode: ExecutableEpisode,
) -> ConversationScript:
    """Verify dependency evidence and rebuild the expected live-value script."""

    producer_executions = dependency_execution_map(episode)
    recorded = {item.dependency_index: item for item in episode.dependencies}
    turns: list[ScriptedTurn] = []
    expected_records: list[DependencyResolution] = []
    for turn in task.script.turns:
        should_resolve = turn.turn_index < len(episode.observed) or (
            episode.status == "dependency_resolution_failed"
            and turn.turn_index == len(episode.observed)
        )
        if should_resolve:
            resolved, outcomes = resolve_turn_dependencies(
                task=task,
                turn=turn,
                executions=episode.executions,
                producer_executions=producer_executions,
            )
            expected_records.extend(outcomes)
            turns.append(resolved)
            if any(item.status != "resolved" for item in outcomes):
                turns.extend(task.script.turns[turn.turn_index + 1 :])
                break
        else:
            turns.append(turn)
    if len(turns) < len(task.script.turns):
        turns.extend(task.script.turns[len(turns) :])
    if set(recorded) != {item.dependency_index for item in expected_records}:
        raise ValueError("episode dependency evidence is not the reached declaration prefix")
    for expected in expected_records:
        actual = recorded[expected.dependency_index]
        if not json_equal(actual.identity_payload(), expected.identity_payload()):
            raise ValueError(
                f"episode dependency {expected.dependency_index} does not match live evidence"
            )
    return task.script.model_copy(update={"turns": tuple(turns)})


__all__ = [
    "dependency_execution_map",
    "resolve_turn_dependencies",
    "resolved_script_from_episode",
]
