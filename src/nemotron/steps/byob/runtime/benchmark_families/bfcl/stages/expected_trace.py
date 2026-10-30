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

"""Derive expected_tool_calls from bound slots and the milestone plan.

Most arguments come from slots the pack bound at expansion. A ``dependent_call``
argument instead names a value that only exists once an earlier call has run, so
this stage asks the oracle for that call's result and locks the extracted value.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Generator
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    oracle_fixture_source_path,
    oracle_reset_fixtures,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    EXPECTED_TRACES,
    expected_trace_row,
    expected_traces_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.templating import (
    PlaceholderError,
    placeholder_names,
    sole_placeholder,
    substitute,
)

logger = logging.getLogger(__name__)

DEPENDENT_KEY = "from_result"
ResultResolver = Callable[[list[dict[str, Any]]], list[Any]]
EpisodeSteps = Generator[dict[str, Any], Any, None]
TraceResolver = Callable[[EpisodeSteps], list[Any]]


class ExpectedTraceError(ValueError):
    """Raised when a template cannot produce a trace at all.

    A template-level fault applies to every instance it would ever bind, so the run
    stops rather than publishing a partial set that hides it.
    """


class TaskDataError(ExpectedTraceError):
    """Raised when this instance's own data cannot produce a trace.

    A sibling instance of the same template may be perfectly fine — one fixture row
    whose list is shorter than the path expects, say — so the instance is dropped and
    the run continues.
    """


def _required_tool_parameters(pack: LoadedPack, tool_name: str) -> list[str]:
    for tool in pack.tools:
        function = tool.get("function") or {}
        if function.get("name") == tool_name:
            parameters = function.get("parameters") or {}
            return [str(name) for name in parameters.get("required") or []]
    return []


def _is_marker(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {DEPENDENT_KEY}


def _resolve_value(value: Any, slots: dict[str, Any], consumed: set[str] | None = None) -> Any:
    """Resolve a milestone argument, substituting every ``{slot}`` reference.

    A reference standing alone keeps the slot's own type, so an integer argument stays
    an integer; a reference inside a longer string is substituted in place, exactly as
    the surface renders it, because a call and the turn describing it must agree. A
    ``from_result`` marker is left alone: it is resolved later, once the oracle has
    produced the call it reads from. Slot names touched are recorded in ``consumed``.
    """
    if _is_marker(value):
        return copy.deepcopy(value)
    if isinstance(value, str):
        name = sole_placeholder(value)
        if name is not None:
            if name not in slots:
                raise ExpectedTraceError(f"milestone argument references unbound slot {name!r}")
            if consumed is not None:
                consumed.add(name)
            return slots[name]
        try:
            rendered = substitute(value, slots, what="milestone argument")
        except PlaceholderError as exc:
            raise ExpectedTraceError(str(exc)) from exc
        if consumed is not None:
            consumed.update(placeholder_names(value))
        return rendered
    if isinstance(value, dict):
        return {key: _resolve_value(item, slots, consumed) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, slots, consumed) for item in value]
    return value


def bind_arguments(
    pack: LoadedPack,
    tool_name: str,
    declared: dict[str, Any] | None,
    slots: dict[str, Any],
    consumed: set[str] | None = None,
) -> dict[str, Any]:
    """Bind declared arguments and required same-named slots.

    Optional parameters are never injected implicitly: omitting one must preserve the
    backend's declared default rather than letting an unrelated slot change the gold
    call merely because it has the same name.
    """
    arguments = {
        key: _resolve_value(value, slots, consumed) for key, value in (declared or {}).items()
    }
    for param in _required_tool_parameters(pack, tool_name):
        if param not in arguments and param in slots:
            arguments[param] = slots[param]
            if consumed is not None:
                consumed.add(param)
    return arguments


def _markers(value: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Collect ``(argument_path, marker_body)`` pairs anywhere in an argument tree."""
    if _is_marker(value):
        return [(path, value[DEPENDENT_KEY])]
    if isinstance(value, dict):
        found: list[tuple[str, dict[str, Any]]] = []
        for key, item in value.items():
            found.extend(_markers(item, f"{path}.{key}" if path else str(key)))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found.extend(_markers(item, f"{path}[{index}]"))
        return found
    return []


def _extract(result: Any, path: str, reference: str) -> Any:
    """Read a dotted path (list indices included) out of a tool result.

    What a result contains depends on the fixture row this instance bound, so a miss
    here is an instance-level fault, not a template-level one.
    """
    if isinstance(result, dict) and "error" in result:
        raise TaskDataError(
            f"call {reference!r} returned an error, so it cannot supply an argument: {result['error']}"
        )
    current = result
    for token in str(path).split("."):
        if isinstance(current, list):
            if not token.lstrip("-").isdigit():
                raise ExpectedTraceError(f"path {path!r} on {reference!r} indexes a list with {token!r}")
            index = int(token)
            if not -len(current) <= index < len(current):
                raise TaskDataError(
                    f"path {path!r} on {reference!r} is out of range for this instance; narrow the "
                    "template's slot filter if every instance should produce this value"
                )
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise TaskDataError(f"path {path!r} on {reference!r} has no field {token!r}")
            current = current[token]
        else:
            raise TaskDataError(f"path {path!r} on {reference!r} descends into a scalar")
    if isinstance(current, (dict, list)):
        raise TaskDataError(
            f"path {path!r} on {reference!r} resolves to a container; point it at a scalar argument"
        )
    return current


def _substitute_markers(value: Any, resolved: dict[str, Any], path: str = "") -> Any:
    if _is_marker(value):
        return resolved[path]
    if isinstance(value, dict):
        return {
            key: _substitute_markers(item, resolved, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_substitute_markers(item, resolved, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def _marker_reference(
    marker: dict[str, Any],
    *,
    call_index: int,
    call_group: int,
    index_by_id: dict[str, int],
    group_by_id: dict[str, int],
) -> str:
    """Validate one marker against the contract and return the producing call id."""
    if not isinstance(marker, dict):
        raise ExpectedTraceError(f"{DEPENDENT_KEY} must be an object with 'call' and 'path'")
    producer = marker.get("call")
    path = marker.get("path")
    if not isinstance(producer, str) or not isinstance(path, str) or not path:
        raise ExpectedTraceError(f"{DEPENDENT_KEY} needs a 'call' milestone id and a non-empty 'path'")
    if producer not in index_by_id:
        raise ExpectedTraceError(
            f"{DEPENDENT_KEY} references unknown milestone id {producer!r}; the producing tool_call "
            "must declare that id"
        )
    if index_by_id[producer] >= call_index:
        raise ExpectedTraceError(
            f"{DEPENDENT_KEY} references {producer!r}, which is not issued before the consuming call"
        )
    if group_by_id[producer] >= call_group:
        raise ExpectedTraceError(
            f"{DEPENDENT_KEY} references {producer!r} in call_group {group_by_id[producer]}, but the "
            f"consuming call is in call_group {call_group}; a dependency needs a later group because "
            "the producer must return first"
        )
    return producer


def build_expected_calls(
    pack: LoadedPack,
    task: dict[str, Any],
    plan: dict[str, Any],
    *,
    resolve_results: ResultResolver | None = None,
    resolve_trace: TraceResolver | None = None,
) -> list[dict[str, Any]]:
    """Return the locked expected_tool_calls list for one task.

    ``turn_index`` is the 0-based ordinal of the assistant message that issues the
    call, counted across assistant messages in the rendered conversation.
    ``resolve_trace`` drives all calls in one backend episode. ``resolve_results`` is
    retained as a small unit-test seam for callers that only resolve trace prefixes.
    """
    updates = {update["entry_index"]: update for update in task.get("slot_updates") or []}
    slots = dict(task.get("slots_initial") or task.get("slots") or {})
    calls: list[dict[str, Any]] = []
    # What each call read from slots, and the value it read, so a correction landing
    # after the call can be told from a call that legitimately used an older value.
    bound_from_slots: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    group_by_id: dict[str, int] = {}
    assistant_index = -1
    for step in plan["steps"]:
        if step["kind"] == "user":
            update_index = step.get("update_index")
            if update_index is not None:
                update = updates.get(update_index)
                if update is None:
                    raise ExpectedTraceError(
                        f"task {task['task_id']!r} plans a slot correction that expansion did not bind"
                    )
                slots.update(update["values"])
        elif step["kind"] == "text":
            assistant_index += 1
        elif step["kind"] == "calls":
            assistant_index += 1
            group = int(step["call_group"])
            for position, milestone in enumerate(step["milestones"]):
                tool_name = str(milestone.get("tool"))
                identifier = milestone.get("id")
                if identifier is not None:
                    index_by_id[str(identifier)] = len(calls)
                    group_by_id[str(identifier)] = group
                consumed: set[str] = set()
                arguments = bind_arguments(
                    pack, tool_name, milestone.get("args"), slots, consumed
                )
                calls.append(
                    {
                        "turn_index": assistant_index,
                        "call_group": group,
                        "position_in_group": position,
                        "function_name": tool_name,
                        "arguments": arguments,
                    }
                )
                bound_from_slots.append({name: slots[name] for name in consumed if name in slots})

    dependent_indices = [
        index for index, call in enumerate(calls) if _markers(call["arguments"])
    ]
    dependent = set(dependent_indices)
    policy = str(task.get("turn_policy"))
    template_id = task.get("template_id")
    if dependent and policy != "dependent_call":
        raise ExpectedTraceError(
            f"template {template_id!r} reads a prior call's result but declares turn_policy "
            f"{policy!r}; declare dependent_call so scoring keeps the anti-hallucinated-id meaning"
        )
    if policy == "dependent_call" and not dependent:
        raise ExpectedTraceError(
            f"template {template_id!r} is dependent_call but binds every argument from slots; a "
            f"dependent call must read its argument from a prior result via {DEPENDENT_KEY}"
        )
    if dependent and resolve_results is None and resolve_trace is None:
        raise ExpectedTraceError(
            f"template {template_id!r} needs oracle results to bind {DEPENDENT_KEY} arguments"
        )

    def resolve_call(index: int, outcomes: list[Any]) -> None:
        call = calls[index]
        references = {
            argument_path: (
                _marker_reference(
                    marker,
                    call_index=index,
                    call_group=int(call["call_group"]),
                    index_by_id=index_by_id,
                    group_by_id=group_by_id,
                ),
                str(marker["path"]),
            )
            for argument_path, marker in _markers(call["arguments"])
        }
        if len(outcomes) != index:
            raise ExpectedTraceError(
                f"oracle returned {len(outcomes)} results for the {index} calls preceding "
                f"{call['function_name']!r}"
            )
        resolved = {
            argument_path: _extract(outcomes[index_by_id[producer]], path, producer)
            for argument_path, (producer, path) in references.items()
        }
        call["arguments"] = _substitute_markers(call["arguments"], resolved)

    if dependent and resolve_trace is not None:
        outcomes: list[Any] = []

        def incremental_steps() -> EpisodeSteps:
            yield {"op": "reset"}
            for index, call in enumerate(calls):
                if index in dependent:
                    resolve_call(index, outcomes)
                result = yield {
                    "op": "call_tool",
                    "name": call["function_name"],
                    "arguments": call["arguments"],
                    "turn_index": call["turn_index"],
                }
                outcomes.append(result)

        resolved_outcomes = resolve_trace(incremental_steps())
        if len(resolved_outcomes) != len(calls):
            raise ExpectedTraceError(
                f"oracle returned {len(resolved_outcomes)} results while resolving "
                f"{len(calls)} dependent trace calls"
            )
    else:
        # Compatibility path for unit-level resolvers: each dependent call asks for
        # the prefix it needs. Production uses the single-session path above.
        for index in dependent_indices:
            outcomes = resolve_results(calls[:index])  # type: ignore[misc]
            resolve_call(index, outcomes)

    final = corrected_slot_values(task)
    for call, from_slots in zip(calls, bound_from_slots):
        for name in sorted(set(from_slots) & set(final)):
            if from_slots[name] != final[name]:
                raise ExpectedTraceError(
                    f"template {template_id!r} calls {call['function_name']!r} with slot {name!r} "
                    f"still holding the value the user replaced; a correction must land before the "
                    "call that consumes it"
                )
    return calls


def corrected_slot_values(task: dict[str, Any]) -> dict[str, Any]:
    """Map each corrected slot to the value in force once every update has landed.

    Identity is the slot, never the bare value: comparing values would flag an
    unrelated argument that merely happens to equal the replaced one.
    """
    corrected = {
        name
        for update in task.get("slot_updates") or []
        for name in (update.get("values") or {})
    }
    slots = task.get("slots") or {}
    return {name: slots[name] for name in sorted(corrected) if name in slots}


def _oracle_trace_resolver(
    worker: ProcessWorker,
    config: BfclConfig,
    pack: LoadedPack,
    task: dict[str, Any],
) -> TraceResolver:
    """Resolve a dependent trace in one worker episode instead of replaying prefixes."""
    runtime = config.oracle_runtime

    def resolve(steps: EpisodeSteps) -> list[Any]:
        # Resolved before the guarded call: a pack that cannot state its seed state is
        # a pack defect, not a trace the oracle declined to resolve.
        fixtures = oracle_reset_fixtures(pack)
        try:
            outputs = worker.run_episode(
                backend_path=pack.paths.backend_path,
                endpoint_config=getattr(pack, "endpoint_config", None),
                fixtures=fixtures,
                clock_iso=runtime.clock,
                seed=int(task.get("seed") or 0),
                task_id=str(task["task_id"]),
                steps=steps,
                import_root=pack.paths.pack_root,
                import_timeout_s=runtime.import_timeout_s,
                reset_timeout_s=runtime.reset_timeout_s,
                tool_timeout_s=runtime.tool_timeout_s,
                episode_timeout_s=runtime.episode_timeout_s,
                fixture_source_path=oracle_fixture_source_path(pack),
            )
        except Exception as exc:  # noqa: BLE001 — retain worker/backend failures as fatal
            raise ExpectedTraceError(
                f"oracle could not resolve dependent trace for task {task['task_id']!r}: {exc}"
            ) from exc
        return outputs[1:]

    return resolve


def run_expected_trace(
    config: BfclConfig,
    pack: LoadedPack,
    tasks: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Derive and cache expected traces, dropping instances their own data cannot bind.

    Returns ``(traces, drop_reasons)``. ``tasks`` is filtered in place to the kept
    instances so later stages do not look for missing traces, while every expanded
    ``task_id`` — including drops — stays in the stage artifact and can be carried
    forward as a skip row by schema/replay.
    """
    worker = ProcessWorker(
        default_timeout_s=config.oracle_runtime.episode_timeout_s,
        worker=config.oracle_runtime.worker,
    )
    traces: dict[str, list[dict[str, Any]]] = {}
    drop_reasons: dict[str, str] = {}
    dependent_tasks = 0
    rows: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        try:
            calls = build_expected_calls(
                pack,
                task,
                plans[task_id],
                resolve_trace=_oracle_trace_resolver(worker, config, pack, task),
            )
        except TaskDataError as exc:
            logger.warning("BFCL expected_trace dropped task %s: %s", task_id, exc)
            drop_reasons[task_id] = str(exc)
            rows.append(expected_trace_row(task, [], drop_reason=str(exc)))
            continue
        traces[task_id] = calls
        kept.append(task)
        rows.append(expected_trace_row(task, calls))
        if str(task.get("turn_policy")) == "dependent_call":
            dependent_tasks += 1

    dropped = len(tasks) - len(kept)
    write_stage_table(
        stage_cache_dir(config) / EXPECTED_TRACES,
        rows,
        expected_traces_schema(),
    )
    if tasks and not kept:
        raise ExpectedTraceError(
            f"every one of the {dropped} expanded instances failed to bind a trace; the fault is in "
            "the templates rather than in one fixture row — see stage_cache/expected_traces.parquet"
        )
    tasks[:] = kept
    logger.info(
        "BFCL expected_trace derived %d traces (%d bound from oracle results, %d dropped)",
        len(traces),
        dependent_tasks,
        dropped,
    )
    return traces, drop_reasons
