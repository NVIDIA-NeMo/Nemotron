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

"""Task-local live oracle sessions backed by the existing process isolator.

The evaluator process never imports pack code. A small command bridge keeps one
``ProcessWorker.run_episode`` invocation alive while candidate calls happen
between oracle operations, preserving task state without weakening isolation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import load_endpoint_config
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    OracleAssertionError,
    OracleCallError,
    OracleResetError,
    OracleSessionError,
    OracleStateError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalLimits
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedEvalSource,
    VerifiedOracleSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    ResolvedPackPaths,
    load_held_out_policy,
    oracle_runtime_fixtures,
    resolve_declared_pack_paths,
)

_FIXTURES_CACHE: dict[tuple[str, str], Any] = {}
_RUNTIME_FIXTURES_CACHE: dict[tuple[str, ...], tuple[Any, bool]] = {}
_FIXTURES_CACHE_LIMIT = 8


def _load_pack_fixtures_revision(path: Path) -> tuple[Any, str]:
    """Parse and validate fixtures once per revision, not once per task.

    A batch run opens one session per task against the same pack, and validating
    a large fixture set on every one of them costs more than the episode. The
    bytes are still read and hashed each time, so pack drift is still caught. The
    parsed value only ever crosses a process boundary, so no session can observe
    another's mutation of it.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    key = (str(path), digest)
    if key in _FIXTURES_CACHE:
        return _FIXTURES_CACHE[key], digest
    fixtures = json.loads(raw.decode("utf-8"))
    if not isinstance(fixtures, dict):
        raise ValueError("fixtures.json is not an object")
    validate_json_value(fixtures, label="oracle fixtures")
    if len(_FIXTURES_CACHE) >= _FIXTURES_CACHE_LIMIT:
        _FIXTURES_CACHE.clear()
    _FIXTURES_CACHE[key] = fixtures
    return fixtures, digest


def _load_pack_fixtures(path: Path) -> Any:
    return _load_pack_fixtures_revision(path)[0]


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_runtime_fixture_projection(paths: ResolvedPackPaths) -> tuple[Any, bool]:
    """Return the fixtures this pack revision lets an oracle session observe.

    A pack that reserves rows outside backend state has them dropped here, before
    any session exists to receive them. The policy is settled once per revision
    for the same reason the fixtures are parsed once, while every input is still
    read and hashed per session so a pack edited mid-run is caught.
    """
    if paths.fixtures_path is None:
        return None, False
    fixtures, fixtures_digest = _load_pack_fixtures_revision(paths.fixtures_path)
    if paths.held_out_path is None:
        return fixtures, False
    manifest_text = paths.manifest_path.read_text(encoding="utf-8")
    templates_text = paths.templates_path.read_text(encoding="utf-8")
    held_out_text = paths.held_out_path.read_text(encoding="utf-8")
    key = (
        str(paths.manifest_path),
        _text_digest(manifest_text),
        _text_digest(templates_text),
        _text_digest(held_out_text),
        fixtures_digest,
    )
    if key in _RUNTIME_FIXTURES_CACHE:
        return _RUNTIME_FIXTURES_CACHE[key]
    manifest = yaml.safe_load(manifest_text) or {}
    templates = yaml.safe_load(templates_text) or []
    if not isinstance(manifest, dict) or not isinstance(templates, list):
        raise ValueError("held-out isolation needs a manifest mapping and a template list")
    held_out = load_held_out_policy(
        paths.held_out_path,
        source=str(manifest.get("held_out")),
        manifest=manifest,
        fixtures=fixtures,
        templates=templates,
        text=held_out_text,
    )
    projected = oracle_runtime_fixtures(
        manifest=manifest,
        fixtures=fixtures,
        held_out=held_out,
    )
    isolates_fixture_file = bool(
        held_out
        and (held_out.get("policy") or {}).get("fixtures_in_backend_state", True) is False
        and paths.backend_path is not None
    )
    if len(_RUNTIME_FIXTURES_CACHE) >= _FIXTURES_CACHE_LIMIT:
        _RUNTIME_FIXTURES_CACHE.clear()
    result = (projected, isolates_fixture_file)
    _RUNTIME_FIXTURES_CACHE[key] = result
    return result


def _load_runtime_fixtures(paths: ResolvedPackPaths) -> Any:
    return _load_runtime_fixture_projection(paths)[0]


@runtime_checkable
class OracleSession(Protocol):
    """The live operations an executable driver may perform on one task."""

    @property
    def oracle_verification_identity(self) -> ContentHash: ...

    async def reset(self) -> None: ...

    async def call_tool(
        self, function_name: str, arguments: dict[str, Any], *, turn_index: int
    ) -> Any: ...

    async def get_state(self) -> dict[str, Any]: ...

    async def run_assertion(
        self, name: str, *, task: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class _ProcessEpisodeBridge:
    """Feed operations to one persistent ProcessWorker episode."""

    def __init__(
        self,
        *,
        worker: ProcessWorker,
        paths: ResolvedPackPaths,
        oracle: VerifiedOracleSource,
        fixtures: dict[str, Any] | None,
        isolate_fixture_file: bool,
        endpoint_config: Any,
        task: ExecutableTaskSpec,
        limits: EvalLimits,
    ) -> None:
        self._worker = worker
        self._paths = paths
        self._oracle = oracle
        self._fixtures = fixtures
        self._isolate_fixture_file = isolate_fixture_file
        self._endpoint_config = endpoint_config
        self._task = task
        self._limits = limits
        self._commands: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._replies: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._exchange_lock = threading.Lock()
        self._closed = False
        self._close_requested = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"bfcl-oracle-session-{task.task_id}",
            daemon=True,
        )
        # Construction resolves and verifies data files but must not execute pack
        # code. The driver performs source-integrity and candidate authorization
        # before its first reset, which is the boundary that starts this worker.
        self._started = False

    def _steps(self) -> Iterator[dict[str, Any]]:
        command = self._commands.get()
        while command is not None:
            result = yield command
            self._replies.put(("value", result))
            command = self._commands.get()

    def _run(self) -> None:
        try:
            self._worker.run_episode(
                backend_path=self._paths.backend_path,
                endpoint_config=self._endpoint_config,
                fixtures=self._fixtures,
                clock_iso=self._task.oracle_clock,
                seed=self._task.seed,
                task_id=self._task.task_id,
                steps=self._steps(),
                assertions_path=self._paths.assertions_path,
                import_root=self._paths.pack_root,
                import_timeout_s=self._limits.tool_timeout_s,
                reset_timeout_s=self._limits.tool_timeout_s,
                tool_timeout_s=self._limits.tool_timeout_s,
                assertion_timeout_s=self._limits.tool_timeout_s,
                episode_timeout_s=self._limits.episode_timeout_s,
                fixture_source_path=(
                    self._paths.fixtures_path if self._isolate_fixture_file else None
                ),
            )
        except BaseException as exc:  # noqa: BLE001 — handed to the waiting driver
            self._replies.put(("error", exc))

    def _exchange_sync(self, operation: dict[str, Any]) -> Any:
        with self._exchange_lock:
            if self._closed:
                raise RuntimeError("oracle session is closed")
            if not self._started:
                self._thread.start()
                self._started = True
            self._commands.put(operation)
            try:
                kind, value = self._replies.get(
                    timeout=float(self._limits.episode_timeout_s) + 2.0
                )
            except queue.Empty as exc:
                self._closed = True
                raise TimeoutError("oracle session bridge exceeded the episode deadline") from exc
            if kind == "error":
                self._closed = True
                raise value
            return value

    async def exchange(self, operation: dict[str, Any]) -> Any:
        return await asyncio.to_thread(self._exchange_sync, operation)

    def _close_sync(self) -> None:
        with self._exchange_lock:
            self._closed = True
            if self._started and not self._close_requested:
                self._close_requested = True
                self._commands.put(None)
            started = self._started
        if not started:
            return
        # A session closed by a bridge timeout may still hold a live worker. Wait
        # for it on the same deadline as an orderly close and report a worker that
        # outlives its task, rather than leaking it behind a shorter join.
        self._thread.join(timeout=float(self._limits.tool_timeout_s) + 2.0)
        if self._thread.is_alive():
            raise TimeoutError("oracle session worker did not close before its deadline")

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)


class _IsolatedOracleSession:
    """Shared adapter behavior; concrete classes pin the verified oracle kind."""

    expected_kind: str

    def __init__(
        self,
        *,
        source: VerifiedEvalSource,
        task: ExecutableTaskSpec,
        limits: EvalLimits,
        worker: ProcessWorker | None = None,
    ) -> None:
        oracle = source.oracle
        if oracle is None or oracle.kind != self.expected_kind:
            raise OracleSessionError(
                "eval.oracle",
                "does not match this session adapter",
                actual=None if oracle is None else oracle.kind,
                expected=self.expected_kind,
                recovery="construct the adapter selected by open_oracle_session",
            )
        if task.oracle_verification_identity != oracle.verification_identity:
            raise OracleSessionError(
                "eval.oracle.identity",
                "does not match the executable task projection",
                actual=oracle.verification_identity,
                expected=task.oracle_verification_identity,
                recovery="rebuild the task from this verified source",
            )
        try:
            paths = resolve_declared_pack_paths(
                OraclePackRef(manifest_path=oracle.pack_manifest_path),
                (oracle.pack_root,),
            )
            fixtures, isolate_fixture_file = _load_runtime_fixture_projection(paths)
            endpoint_config = (
                load_endpoint_config(
                    paths.endpoint_config_path,
                    allowed_roots=(paths.pack_root,),
                )
                if paths.endpoint_config_path is not None
                else None
            )
        except Exception as exc:
            raise OracleSessionError(
                "eval.oracle.pack",
                f"cannot prepare the verified oracle: {type(exc).__name__}",
                expected="the complete verified pack revision",
                recovery="restore the pack and run source verification again",
            ) from exc
        resolved_resource = (
            paths.backend_path if oracle.kind == "python" else paths.endpoint_config_path
        )
        if resolved_resource != oracle.resource_path:
            raise OracleSessionError(
                "eval.oracle.resource",
                "resolved a different execution resource than source verification",
                actual=str(resolved_resource),
                expected=str(oracle.resource_path),
                recovery="stop and verify the source again",
            )
        selected_worker = worker or ProcessWorker(
            default_timeout_s=float(limits.tool_timeout_s),
            worker="process",
        )
        if selected_worker.worker != "process":
            raise OracleSessionError(
                "eval.oracle.isolation",
                "would import or execute pack code outside process isolation",
                actual=selected_worker.worker,
                expected="ProcessWorker(worker='process')",
                recovery="use process isolation for executable evaluation",
            )
        self._oracle = oracle
        self._bridge = _ProcessEpisodeBridge(
            worker=selected_worker,
            paths=paths,
            oracle=oracle,
            fixtures=fixtures,
            isolate_fixture_file=isolate_fixture_file,
            endpoint_config=endpoint_config,
            task=task,
            limits=limits,
        )

    @property
    def oracle_verification_identity(self) -> ContentHash:
        return self._oracle.verification_identity

    async def reset(self) -> None:
        try:
            await self._bridge.exchange({"op": "reset"})
        except Exception as exc:
            raise OracleResetError(
                "eval.oracle.reset",
                f"failed as {type(exc).__name__}",
                expected="a clean task-local oracle state",
                recovery="inspect the verified backend or endpoint and retry the whole task",
            ) from exc

    async def call_tool(
        self, function_name: str, arguments: dict[str, Any], *, turn_index: int
    ) -> Any:
        try:
            return await self._bridge.exchange(
                {
                    "op": "call_tool",
                    "name": function_name,
                    "arguments": arguments,
                    "turn_index": turn_index,
                }
            )
        except TimeoutError:
            raise
        except Exception as exc:
            raise OracleCallError(
                f"eval.oracle.call[{function_name}]",
                f"failed as {type(exc).__name__}",
                expected="one JSON object result or structured business rejection",
                recovery="inspect the oracle; never retry a possibly mutating call in this episode",
            ) from exc

    async def get_state(self) -> dict[str, Any]:
        try:
            state = await self._bridge.exchange({"op": "get_state"})
            if not isinstance(state, dict):
                raise TypeError("oracle state is not a JSON object")
            validate_json_value(state, label="oracle state")
            return state
        except Exception as exc:
            raise OracleStateError(
                "eval.oracle.state",
                f"failed as {type(exc).__name__}",
                expected="one canonical JSON object",
                recovery="inspect the oracle state contract and retry the whole task",
            ) from exc

    async def run_assertion(
        self, name: str, *, task: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            verdict = await self._bridge.exchange(
                {"op": "run_assertion", "name": name, "task": task}
            )
            if (
                not isinstance(verdict, dict)
                or verdict.get("name") != name
                or verdict.get("status")
                not in {
                    "passed",
                    "failed",
                    "not_applicable",
                    "infrastructure_error",
                }
            ):
                raise TypeError("assertion returned an invalid verdict")
            return verdict
        except Exception as exc:
            raise OracleAssertionError(
                f"eval.oracle.assertion[{name}]",
                f"failed as {type(exc).__name__}",
                expected=(
                    "a passed, failed, not_applicable, or infrastructure_error "
                    "assertion verdict"
                ),
                recovery="fix the assertion module and rerun the whole executable task",
            ) from exc

    async def close(self) -> None:
        try:
            await self._bridge.close()
        except Exception as exc:
            raise OracleSessionError(
                "eval.oracle.close",
                f"failed as {type(exc).__name__}",
                expected="the task-local worker and endpoint session to close",
                recovery="discard this task evidence and inspect oracle cleanup",
            ) from exc


class PythonOracleSession(_IsolatedOracleSession):
    """A Python pack imported only inside a process worker."""

    expected_kind = "python"


class EndpointOracleSession(_IsolatedOracleSession):
    """A BFCL Oracle HTTP v1 session owned by a process worker."""

    expected_kind = "endpoint"


def open_oracle_session(
    *,
    source: VerifiedEvalSource,
    task: ExecutableTaskSpec,
    limits: EvalLimits,
    worker: ProcessWorker | None = None,
) -> OracleSession:
    """Construct the adapter selected by the source-verified oracle kind."""

    oracle = source.oracle
    if oracle is None:
        raise OracleSessionError(
            "eval.oracle",
            "is missing from an executable source",
            expected="a VerifiedOracleSource",
            recovery="run source verification with executable mode enabled",
        )
    adapter = PythonOracleSession if oracle.kind == "python" else EndpointOracleSession
    return adapter(source=source, task=task, limits=limits, worker=worker)


__all__ = [
    "EndpointOracleSession",
    "OracleSession",
    "PythonOracleSession",
    "open_oracle_session",
]
