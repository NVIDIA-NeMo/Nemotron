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

"""Replay each expected trace against the oracle twice and run its assertions."""

from __future__ import annotations

import logging
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    oracle_fixture_source_path,
    oracle_reset_fixtures,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    REPLAY_VALIDATED_TASKS,
    replay_validated_row,
    replay_validated_tasks_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)

NONDETERMINISTIC = "nondeterministic_replay"
REPLAY_FAILED = "executable_replay_failed"


def replay_once(
    worker: ProcessWorker,
    config: BfclConfig,
    pack: LoadedPack,
    task: dict[str, Any],
    expected_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one replay episode: reset, the expected calls, final state, then assertions.

    Assertions run inside the same worker, so they observe live state and the
    trace the worker accumulated from this episode's calls.
    """
    runtime = config.oracle_runtime
    steps: list[dict[str, Any]] = [{"op": "reset"}]
    first_call = len(steps)
    steps.extend(
        {
            "op": "call_tool",
            "name": call["function_name"],
            "arguments": call["arguments"],
            "turn_index": call["turn_index"],
        }
        for call in expected_calls
    )
    state_index = len(steps)
    steps.append({"op": "get_state"})
    assertion_names = list(task.get("success_assertions") or [])
    steps.extend({"op": "run_assertion", "name": name, "task": task} for name in assertion_names)

    outputs = worker.run_episode(
        backend_path=pack.paths.backend_path,
        endpoint_config=getattr(pack, "endpoint_config", None),
        fixtures=oracle_reset_fixtures(pack),
        clock_iso=runtime.clock,
        seed=int(task.get("seed") or 0),
        # Determinism attempts must receive byte-identical context. Attempt
        # metadata belongs to the parent verdict, not the backend RunContext.
        task_id=str(task["task_id"]),
        steps=steps,
        assertions_path=pack.paths.assertions_path,
        import_root=pack.paths.pack_root,
        import_timeout_s=runtime.import_timeout_s,
        reset_timeout_s=runtime.reset_timeout_s,
        tool_timeout_s=runtime.tool_timeout_s,
        assertion_timeout_s=runtime.assertion_timeout_s,
        episode_timeout_s=runtime.episode_timeout_s,
        fixture_source_path=oracle_fixture_source_path(pack),
    )
    return {
        "results": outputs[first_call : first_call + len(expected_calls)],
        "state": outputs[state_index],
        "assertions": outputs[state_index + 1 :],
    }


def replay_task(
    worker: ProcessWorker,
    config: BfclConfig,
    pack: LoadedPack,
    task: dict[str, Any],
    expected_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay twice after independent resets and report divergence or assertion failures."""
    try:
        first = replay_once(worker, config, pack, task, expected_calls)
        second = replay_once(worker, config, pack, task, expected_calls)
    except Exception as exc:  # noqa: BLE001 — replay failure is reported, never fatal
        return {"passed": False, "reason": REPLAY_FAILED, "detail": str(exc)}

    try:
        for field in ("results", "state", "assertions"):
            if canonical_json(first[field]) != canonical_json(second[field]):
                return {"passed": False, "reason": NONDETERMINISTIC, "detail": f"{field} diverged"}
    except (TypeError, ValueError) as exc:
        return {
            "passed": False,
            "reason": REPLAY_FAILED,
            "detail": f"replay produced non-JSON data: {exc}",
        }

    failed = [item for item in first["assertions"] if not item.get("passed")]
    if failed:
        return {
            "passed": False,
            "reason": "assertion_failed",
            "detail": "; ".join(f"{item['name']}: {item['detail']}" for item in failed),
        }

    return {
        "passed": True,
        "reason": None,
        "results": first["results"],
        "state": first["state"],
        "assertions": first["assertions"],
    }


def run_executable_replay(
    config: BfclConfig,
    pack: LoadedPack,
    tasks: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
    schema_failures: dict[str, list[dict[str, Any]]],
    *,
    skipped: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Replay every schema-valid task and cache the verdicts.

    ``skipped`` tasks keep a row in the stage table so joins across artifacts still
    cover every expanded ``task_id``, without spending an oracle episode on them.
    """
    worker = ProcessWorker(
        default_timeout_s=config.oracle_runtime.episode_timeout_s,
        worker=config.oracle_runtime.worker,
    )
    skipped = skipped or {}
    verdicts: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in skipped:
            verdicts[task_id] = {
                "passed": False,
                "reason": "trace_not_derived",
                "detail": skipped[task_id],
            }
            continue
        if schema_failures.get(task_id):
            detail = canonical_json(schema_failures[task_id])
            verdicts[task_id] = {
                "passed": False,
                "reason": "expected_trace_schema_mismatch",
                "detail": detail,
            }
            continue
        verdicts[task_id] = replay_task(worker, config, pack, task, traces[task_id])

    write_stage_table(
        stage_cache_dir(config) / REPLAY_VALIDATED_TASKS,
        [replay_validated_row(task, verdicts[str(task["task_id"])]) for task in tasks],
        replay_validated_tasks_schema(),
    )
    passed = sum(1 for verdict in verdicts.values() if verdict["passed"])
    logger.info("BFCL executable_replay validated %d/%d tasks", passed, len(verdicts))
    return verdicts
