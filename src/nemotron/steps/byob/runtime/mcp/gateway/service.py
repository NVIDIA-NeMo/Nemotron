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

"""Mode A MCP episode lifecycle behind the BFCL Oracle HTTP v1 contract."""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    validate_function_arguments,
)
from nemotron.steps.byob.runtime.mcp.client import (
    ConnectedMcpClient,
    open_mcp_connection,
)
from nemotron.steps.byob.runtime.mcp.config import (
    LoadedMcpOracleConfig,
    McpOracleConfig,
    TrustedExecutablePolicies,
)
from nemotron.steps.byob.runtime.mcp.discovery import (
    DiscoveryReport,
    discover_mcp_oracle,
)
from nemotron.steps.byob.runtime.mcp.gateway.conformance import (
    ConformanceEvidence,
    build_attestation,
    discovery_evidence,
)
from nemotron.steps.byob.runtime.mcp.gateway.errors import (
    GatewayError,
    bad_request,
    unavailable,
    upstream_failure,
)
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    GatewayArtifacts,
    GatewayIdentity,
    build_gateway_identity,
)
from nemotron.steps.byob.runtime.mcp.gateway.result_mapping import (
    control_result_object,
    ensure_gateway_error,
    map_call_result,
)

ConnectionFactory = Callable[
    [McpOracleConfig],
    AbstractAsyncContextManager[ConnectedMcpClient],
]
_CONTEXT_KEYS = frozenset({"clock", "seed", "timeout_s", "task_id"})


@dataclass
class GatewaySession:
    session_id: str
    episode_id: str
    worker: asyncio.Task[None]
    commands: asyncio.Queue[_CallCommand | None]
    created_at: float
    last_used_at: float
    poisoned: bool = False
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _drain_future(future: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned outcome so asyncio does not log it as unhandled."""
    if not future.cancelled():
        future.exception()


def _drain_when_done(future: asyncio.Future[Any]) -> None:
    if future.done():
        _drain_future(future)
    else:
        future.add_done_callback(_drain_future)


@dataclass(frozen=True)
class _CallCommand:
    name: str
    arguments: dict[str, Any]
    meta: dict[str, Any] | None
    timeout_s: float
    timeout_code: str
    timeout_message: str
    response: asyncio.Future[dict[str, Any]]


class GatewayService:
    """Transport-neutral gateway core used by the HTTP adapter and unit tests."""

    def __init__(
        self,
        loaded: LoadedMcpOracleConfig,
        *,
        artifacts: GatewayArtifacts,
        executable_policies: TrustedExecutablePolicies | None = None,
        environ: Mapping[str, str] | None = None,
        connection_factory: ConnectionFactory | None = None,
        conformance_evidence: ConformanceEvidence | None = None,
    ) -> None:
        self.loaded = loaded
        self.config = loaded.value
        self.artifacts = artifacts
        self.executable_policies = executable_policies
        self.environ = environ
        self._injected_factory = connection_factory
        self._injected_conformance_evidence = conformance_evidence
        self._sessions: dict[str, GatewaySession] = {}
        self._registry_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._creating_sessions = 0
        self._starting_workers: set[asyncio.Task[None]] = set()
        self._teardown_tasks: set[asyncio.Task[None]] = set()
        self._reaper_task: asyncio.Task[None] | None = None
        self._shutting_down = False
        self._started = False
        self._report: DiscoveryReport | None = None
        self._identity: GatewayIdentity | None = None
        self._conformance_evidence: ConformanceEvidence | None = None
        self._tool_definitions: dict[str, dict[str, Any]] = {}
        self._output_schemas: dict[str, dict[str, Any] | None] = {}
        self._published_to_source: dict[str, str] = {}

    def _connection_factory(
        self,
        config: McpOracleConfig,
    ) -> AbstractAsyncContextManager[ConnectedMcpClient]:
        if self._injected_factory is not None:
            return self._injected_factory(config)
        return open_mcp_connection(
            config,
            environ=self.environ,
            executable_policies=self.executable_policies,
        )

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            if self.config.mode != "A":
                raise unavailable(
                    "mcp_mode_not_executable",
                    "the gateway executes cooperative mode A only; "
                    "mode B shim and mode C snapshot mechanics remain unimplemented",
                )
            report = await discover_mcp_oracle(
                self.loaded,
                environ=self.environ,
                executable_policies=self.executable_policies,
                connection_factory=self._connection_factory,
            )
            self._load_catalog(report)
            self._identity = build_gateway_identity(
                self.config,
                report,
                self.artifacts,
            )
            self._report = report
            self._conformance_evidence = (
                self._injected_conformance_evidence
                if self._injected_conformance_evidence is not None
                else discovery_evidence(report)
            )
            self._shutting_down = False
            self._started = True
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(),
                name="bfcl-mcp-session-reaper",
            )

    async def _reaper_loop(self) -> None:
        interval = max(
            0.05,
            min(
                float(self.config.limits.session_idle_ttl_s),
                float(self.config.limits.episode_timeout_s),
            )
            / 2.0,
        )
        try:
            while True:
                await asyncio.sleep(interval)
                await self.reap_expired()
        except asyncio.CancelledError:
            raise

    def _load_catalog(self, report: DiscoveryReport) -> None:
        catalog = report.document["catalog"]
        tools = catalog["tools"]
        evidence = catalog["evidence"]
        source_to_published = catalog["selected_source_to_published"]
        self._published_to_source = {str(published): str(source) for source, published in source_to_published.items()}
        self._tool_definitions = {}
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise GatewayError(
                    "mcp_gateway_catalog_invalid",
                    "discovery report contains an invalid normalized tool",
                )
            self._tool_definitions[function["name"]] = function
        self._output_schemas = {
            str(item["published_name"]): (
                dict(item["output_schema"]) if isinstance(item.get("output_schema"), dict) else None
            )
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("published_name"), str)
        }
        if set(self._tool_definitions) != set(self._published_to_source):
            raise GatewayError(
                "mcp_gateway_catalog_invalid",
                "normalized tools and source alias mapping disagree",
            )

    def _require_started(self) -> None:
        if not self._started or self._identity is None or self._report is None or self._conformance_evidence is None:
            raise unavailable(
                "mcp_gateway_not_ready",
                "gateway startup discovery has not completed",
            )

    def metadata(self) -> dict[str, str]:
        self._require_started()
        assert self._identity is not None
        return self._identity.as_dict()

    def conformance(self) -> dict[str, Any]:
        """Serve the attestation for whatever this gateway can currently demonstrate.

        Built from the discovery evidence rather than from a stored constant, so a gateway
        whose startup discovery observed less than a full probe suite reports the lower level
        instead of a level someone typed in.
        """
        self._require_started()
        assert self._identity is not None and self._report is not None and self._conformance_evidence is not None
        return build_attestation(
            self.config,
            self._report,
            self.artifacts,
            self._identity,
            self._conformance_evidence,
        )

    def probe_report(self) -> dict[str, Any]:
        """Return the exact report whose digest the attestation cites."""
        self._require_started()
        assert self._conformance_evidence is not None
        return self._conformance_evidence.probe_report

    def gateway_conformance_report(self) -> dict[str, Any]:
        """Bind the build-time suite to the live gateway and discovered target."""
        self._require_started()
        assert self._identity is not None and self._report is not None and self._conformance_evidence is not None
        return self._conformance_evidence.gateway_report(
            self.artifacts.validated().gateway_artifact_digest,
            effective_content_digest=self._identity.content_digest,
            tool_catalog_digest=self._report.tool_catalog_digest,
        )

    def list_tools(self) -> list[str]:
        self._require_started()
        # Discovery already sorts normalized tools by published name.
        assert self._report is not None
        return [str(tool["function"]["name"]) for tool in self._report.document["catalog"]["tools"]]

    @staticmethod
    def _validate_context(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise bad_request("mcp_context_invalid", "context must be a JSON object")
        missing = sorted(_CONTEXT_KEYS - set(value))
        unknown = sorted(set(value) - _CONTEXT_KEYS)
        if missing or unknown:
            raise bad_request(
                "mcp_context_invalid",
                f"context fields mismatch: missing={missing}, unknown={unknown}",
            )
        clock = value["clock"]
        if not isinstance(clock, str) or not clock:
            raise bad_request("mcp_context_invalid", "context.clock must be an ISO string")
        try:
            parsed_clock = datetime.fromisoformat(clock)
        except ValueError as exc:
            raise bad_request(
                "mcp_context_invalid",
                "context.clock must be a valid ISO datetime",
            ) from exc
        if parsed_clock.tzinfo is None or parsed_clock.utcoffset() is None:
            raise bad_request(
                "mcp_context_invalid",
                "context.clock must include an explicit UTC offset",
            )
        seed = value["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise bad_request("mcp_context_invalid", "context.seed must be an integer")
        timeout = value["timeout_s"]
        if (
            not isinstance(timeout, int | float)
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise bad_request(
                "mcp_context_invalid",
                "context.timeout_s must be a positive finite number",
            )
        task_id = value["task_id"]
        if not isinstance(task_id, str) or not task_id.strip():
            raise bad_request(
                "mcp_context_invalid",
                "context.task_id must be a non-empty string",
            )
        return {
            # Forwarded verbatim: re-serializing a parsed clock would hand the server a
            # different string than BFCL sent, which anything that hashes or echoes the
            # clock would then disagree about.
            "clock": clock,
            "seed": seed,
            "timeout_s": float(timeout),
            "task_id": task_id,
        }

    async def _discover_borrowed(
        self,
        client: ConnectedMcpClient,
    ) -> DiscoveryReport:
        @asynccontextmanager
        async def borrowed(
            config: McpOracleConfig,
        ) -> AsyncIterator[ConnectedMcpClient]:
            yield client

        return await discover_mcp_oracle(
            self.loaded,
            connection_factory=borrowed,
        )

    @staticmethod
    def _session_creation_error(exc: Exception) -> GatewayError:
        if isinstance(exc, GatewayError):
            return exc
        return upstream_failure(
            "mcp_session_create_failed",
            f"MCP session creation failed: {type(exc).__name__}",
        )

    async def _episode_worker(
        self,
        *,
        context: dict[str, Any],
        fixtures: dict[str, Any] | None,
        ready: asyncio.Future[str],
        commands: asyncio.Queue[_CallCommand | None],
    ) -> None:
        """Own one MCP context from enter through exit in a single asyncio task."""
        active: _CallCommand | None = None
        episode_id: str | None = None
        graceful_close = False
        try:
            async with self._connection_factory(self.config) as client:
                try:
                    await self._discover_borrowed(client)
                    reset_tool = self.config.control.reset_episode
                    if reset_tool is None:
                        raise GatewayError(
                            "mcp_reset_unavailable",
                            "mode A requires control.reset_episode",
                        )
                    try:
                        raw_reset = await asyncio.wait_for(
                            client.call_tool(
                                reset_tool,
                                {"fixtures": fixtures, "context": context},
                            ),
                            timeout=float(self.config.limits.reset_timeout_s),
                        )
                    except TimeoutError as exc:
                        raise upstream_failure(
                            "mcp_reset_timeout",
                            "MCP reset_episode exceeded reset_timeout_s",
                            timeout=True,
                        ) from exc
                    reset = control_result_object(raw_reset, operation=reset_tool)
                    observed_episode_id = reset.get("episode_id")
                    if not isinstance(observed_episode_id, str) or not observed_episode_id:
                        raise upstream_failure(
                            "mcp_reset_invalid",
                            "MCP reset_episode must return a non-empty string episode_id",
                        )
                    episode_id = observed_episode_id
                except Exception as exc:
                    ready.set_exception(self._session_creation_error(exc))
                    return

                ready.set_result(episode_id)
                while True:
                    active = await commands.get()
                    if active is None:
                        graceful_close = True
                        break
                    try:
                        raw = await asyncio.wait_for(
                            client.call_tool(
                                active.name,
                                active.arguments,
                                meta=active.meta,
                            ),
                            timeout=active.timeout_s,
                        )
                    except TimeoutError:
                        active.response.set_exception(
                            upstream_failure(
                                active.timeout_code,
                                active.timeout_message,
                                timeout=True,
                            )
                        )
                        break
                    except Exception as exc:
                        active.response.set_exception(ensure_gateway_error(exc, operation=active.name))
                        break
                    else:
                        active.response.set_result(raw)
                    finally:
                        active = None

                end_tool = self.config.control.end_episode
                if graceful_close and end_tool is not None:
                    raw_end = await asyncio.wait_for(
                        client.call_tool(
                            end_tool,
                            {"episode_id": episode_id},
                        ),
                        timeout=float(self.config.limits.tool_timeout_s),
                    )
                    control_result_object(raw_end, operation=end_tool)
        except asyncio.CancelledError:
            if not ready.done():
                ready.set_exception(
                    unavailable(
                        "mcp_gateway_shutting_down",
                        "gateway stopped while creating the MCP session",
                    )
                )
            if active is not None and not active.response.done():
                active.response.set_exception(
                    upstream_failure(
                        "mcp_call_cancelled",
                        "MCP call was cancelled and its episode was poisoned",
                    )
                )
            raise
        except Exception as exc:
            if not ready.done():
                ready.set_exception(self._session_creation_error(exc))
            elif active is not None and not active.response.done():
                active.response.set_exception(ensure_gateway_error(exc, operation=active.name))
            else:
                raise

    async def _discard_starting_worker(
        self,
        worker: asyncio.Task[None],
    ) -> None:
        async with self._registry_lock:
            if worker in self._starting_workers:
                self._starting_workers.remove(worker)
                self._creating_sessions -= 1

    async def create_session(
        self,
        *,
        context: Any,
        fixtures: Any,
    ) -> dict[str, Any]:
        self._require_started()
        normalized_context = self._validate_context(context)
        if fixtures is not None and not isinstance(fixtures, dict):
            raise bad_request(
                "mcp_fixtures_invalid",
                "fixtures must be a JSON object or null",
            )
        await self.reap_expired()
        loop = asyncio.get_running_loop()
        worker_started_at = time.monotonic()
        ready: asyncio.Future[str] = loop.create_future()
        commands: asyncio.Queue[_CallCommand | None] = asyncio.Queue()
        async with self._registry_lock:
            if self._shutting_down:
                raise unavailable(
                    "mcp_gateway_shutting_down",
                    "gateway is shutting down",
                )
            if len(self._sessions) + self._creating_sessions >= self.config.limits.max_concurrent_episodes:
                raise unavailable(
                    "mcp_session_limit",
                    "gateway reached max_concurrent_episodes",
                )
            self._creating_sessions += 1
            worker = asyncio.create_task(
                self._episode_worker(
                    context=normalized_context,
                    fixtures=fixtures,
                    ready=ready,
                    commands=commands,
                ),
                name="bfcl-mcp-episode",
            )
            self._starting_workers.add(worker)
        try:
            episode_id = await asyncio.wait_for(
                asyncio.shield(ready),
                timeout=float(self.config.limits.episode_timeout_s),
            )
        except TimeoutError as exc:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            _drain_when_done(ready)
            await self._discard_starting_worker(worker)
            raise upstream_failure(
                "mcp_session_create_timeout",
                "MCP session readiness exceeded episode_timeout_s",
                timeout=True,
            ) from exc
        except BaseException:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            _drain_when_done(ready)
            await self._discard_starting_worker(worker)
            raise

        reject_for_shutdown = False
        try:
            async with self._registry_lock:
                if worker in self._starting_workers:
                    self._starting_workers.remove(worker)
                    self._creating_sessions -= 1
                if self._shutting_down:
                    reject_for_shutdown = True
                else:
                    now = time.monotonic()
                    session_id = secrets.token_urlsafe(24)
                    while session_id in self._sessions:
                        session_id = secrets.token_urlsafe(24)
                    self._sessions[session_id] = GatewaySession(
                        session_id=session_id,
                        episode_id=episode_id,
                        worker=worker,
                        commands=commands,
                        created_at=worker_started_at,
                        last_used_at=now,
                    )
        except BaseException:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await self._discard_starting_worker(worker)
            raise
        if reject_for_shutdown:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            raise unavailable(
                "mcp_gateway_shutting_down",
                "gateway shut down while creating the MCP session",
            )
        return {"session_id": session_id, "oracle": self.metadata()}

    def _is_expired(self, session: GatewaySession, now: float) -> bool:
        return now - session.last_used_at >= float(
            self.config.limits.session_idle_ttl_s
        ) or now - session.created_at >= float(self.config.limits.episode_timeout_s)

    def _start_teardown(
        self,
        session: GatewaySession,
        *,
        acquire_lock: bool,
        suppress_errors: bool,
    ) -> asyncio.Task[None]:
        async def teardown() -> None:
            if acquire_lock:
                async with session.lock:
                    await self._close_resources(
                        session,
                        suppress_errors=suppress_errors,
                    )
            else:
                await self._close_resources(
                    session,
                    suppress_errors=suppress_errors,
                )

        task = asyncio.create_task(teardown(), name="bfcl-mcp-teardown")
        self._teardown_tasks.add(task)
        task.add_done_callback(self._teardown_tasks.discard)
        return task

    async def _revalidate_locked_session(self, session: GatewaySession) -> None:
        teardown: asyncio.Task[None] | None = None
        async with self._registry_lock:
            current = self._sessions.get(session.session_id)
            if current is session and self._is_expired(session, time.monotonic()):
                self._sessions.pop(session.session_id)
                current = None
                teardown = self._start_teardown(
                    session,
                    acquire_lock=False,
                    suppress_errors=True,
                )
        if teardown is not None:
            await asyncio.shield(teardown)
        if current is not session:
            raise GatewayError(
                "mcp_session_unknown",
                "session is unknown or expired",
                http_status=404,
            )
        self._ensure_usable(session)

    async def _session(self, session_id: str) -> GatewaySession:
        teardown: asyncio.Task[None] | None = None
        async with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is not None and self._is_expired(session, time.monotonic()):
                self._sessions.pop(session_id)
                teardown = self._start_teardown(
                    session,
                    acquire_lock=True,
                    suppress_errors=True,
                )
                session = None
        if teardown is not None:
            await asyncio.shield(teardown)
        if session is None:
            raise GatewayError(
                "mcp_session_unknown",
                "session is unknown or expired",
                http_status=404,
            )
        return session

    async def _poison(
        self,
        session: GatewaySession,
        error: GatewayError,
    ) -> None:
        if not error.poison_session:
            return
        session.poisoned = True
        # The caller already holds ``session.lock``, and teardown has to finish even if
        # the client abandons this request, so it runs as a task shutdown can drain.
        await asyncio.shield(
            self._start_teardown(
                session,
                acquire_lock=False,
                suppress_errors=True,
            )
        )

    @staticmethod
    def _ensure_usable(session: GatewaySession) -> None:
        if session.poisoned or session.closed:
            raise GatewayError(
                "mcp_session_poisoned",
                "session is poisoned after an ambiguous upstream failure",
                http_status=409,
            )
        if session.worker.done():
            session.poisoned = True
            raise GatewayError(
                "mcp_session_poisoned",
                "MCP episode worker stopped unexpectedly",
                http_status=409,
            )

    async def _execute(
        self,
        session: GatewaySession,
        *,
        name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None = None,
        timeout_code: str,
        timeout_message: str,
    ) -> dict[str, Any]:
        self._ensure_usable(session)
        response: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        session.commands.put_nowait(
            _CallCommand(
                name=name,
                arguments=arguments,
                meta=meta,
                timeout_s=float(self.config.limits.tool_timeout_s),
                timeout_code=timeout_code,
                timeout_message=timeout_message,
                response=response,
            )
        )
        try:
            return await asyncio.shield(response)
        except asyncio.CancelledError:
            session.poisoned = True
            # This task is unwinding, so cleanup cannot be awaited here; hand it to a
            # tracked teardown task instead of leaving the episode open upstream.
            self._start_teardown(session, acquire_lock=False, suppress_errors=True)
            # The worker may still settle this future after the caller is gone.
            _drain_when_done(response)
            raise

    async def call_tool(
        self,
        session_id: str,
        *,
        name: Any,
        arguments: Any,
        turn_index: Any,
    ) -> dict[str, Any]:
        session = await self._session(session_id)
        if not isinstance(name, str) or name not in self._tool_definitions:
            raise bad_request("mcp_tool_unknown", "tool is not in the published catalog")
        if not isinstance(arguments, dict):
            raise bad_request("mcp_arguments_invalid", "arguments must be a JSON object")
        if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
            raise bad_request(
                "mcp_turn_index_invalid",
                "turn_index must be a non-negative integer",
            )
        failures = validate_function_arguments(
            self._tool_definitions[name],
            arguments,
        )
        if failures:
            raise bad_request(
                "mcp_arguments_invalid",
                "arguments do not satisfy the published tool schema",
            )

        async with session.lock:
            await self._revalidate_locked_session(session)
            source_name = self._published_to_source[name]
            upstream_arguments = dict(arguments)
            meta: dict[str, Any] | None = None
            if self.config.control.episode_binding == "argument":
                episode_argument = self.config.control.episode_argument
                assert episode_argument is not None
                if episode_argument in upstream_arguments:
                    raise bad_request(
                        "mcp_episode_binding_override",
                        "model arguments may not set the gateway-owned episode id",
                    )
                upstream_arguments[episode_argument] = session.episode_id
            elif self.config.control.episode_binding == "meta":
                meta = {"bfcl": {"episode_id": session.episode_id}}
            try:
                raw = await self._execute(
                    session,
                    name=source_name,
                    arguments=upstream_arguments,
                    meta=meta,
                    timeout_code="mcp_call_timeout",
                    timeout_message="MCP tools/call exceeded tool_timeout_s",
                )
                mapped = map_call_result(
                    raw,
                    config=self.config.results,
                    output_schema=self._output_schemas.get(name),
                )
            except Exception as exc:
                error = ensure_gateway_error(exc, operation=f"tools/call {source_name!r}")
                await self._poison(session, error)
                raise error from exc
            session.last_used_at = time.monotonic()
            return mapped

    async def get_state(self, session_id: str) -> dict[str, Any]:
        session = await self._session(session_id)
        async with session.lock:
            await self._revalidate_locked_session(session)
            state_tool = self.config.control.get_episode_state
            if state_tool is None:
                raise GatewayError(
                    "mcp_state_unavailable",
                    "mode A requires control.get_episode_state",
                )
            try:
                raw = await self._execute(
                    session,
                    name=state_tool,
                    arguments={"episode_id": session.episode_id},
                    timeout_code="mcp_state_timeout",
                    timeout_message="MCP get_episode_state exceeded tool_timeout_s",
                )
                state = control_result_object(raw, operation=state_tool)
            except Exception as exc:
                error = ensure_gateway_error(exc, operation=state_tool)
                await self._poison(session, error)
                raise error from exc
            session.last_used_at = time.monotonic()
            return state

    @property
    def _close_grace_s(self) -> float:
        """Budget a worker gets to stop itself before the gateway cancels it.

        A call that is still in flight burns at most its own ``tool_timeout_s`` before
        the worker abandons it, and a graceful stop then runs ``end_episode`` under the
        same budget.
        """
        return 2.0 * float(self.config.limits.tool_timeout_s)

    async def _close_resources(
        self,
        session: GatewaySession,
        *,
        suppress_errors: bool,
    ) -> None:
        if session.closed:
            return
        close_error: BaseException | None = None
        if not session.worker.done():
            # Ask for a graceful stop even when poisoned. After a failed call the worker
            # has already left the command loop and is closing its own transport, and
            # cancelling that mid-flight is what orphans an upstream episode.
            session.commands.put_nowait(None)
            _, pending = await asyncio.wait(
                {session.worker},
                timeout=self._close_grace_s,
            )
            if pending:
                session.worker.cancel()
        result = await asyncio.gather(session.worker, return_exceptions=True)
        observed = result[0]
        if isinstance(observed, BaseException):
            close_error = observed
        session.closed = True
        if close_error is not None and not suppress_errors:
            raise upstream_failure(
                "mcp_close_failed",
                f"MCP episode cleanup failed: {type(close_error).__name__}",
            ) from close_error

    async def delete_session(self, session_id: str) -> None:
        async with self._registry_lock:
            session = self._sessions.pop(session_id, None)
            teardown = (
                self._start_teardown(
                    session,
                    acquire_lock=True,
                    suppress_errors=False,
                )
                if session is not None
                else None
            )
        if session is None:
            return
        assert teardown is not None
        await asyncio.shield(teardown)

    async def reap_expired(self) -> int:
        now = time.monotonic()
        async with self._registry_lock:
            expired_ids = [
                session_id for session_id, session in self._sessions.items() if self._is_expired(session, now)
            ]
            expired = [self._sessions.pop(session_id) for session_id in expired_ids]
            teardowns = [
                self._start_teardown(
                    session,
                    acquire_lock=True,
                    suppress_errors=True,
                )
                for session in expired
            ]
        if teardowns:
            await asyncio.shield(asyncio.gather(*teardowns))
        return len(teardowns)

    async def shutdown(self) -> None:
        async with self._start_lock:
            async with self._registry_lock:
                self._shutting_down = True
                self._started = False
                starting = list(self._starting_workers)
                sessions = list(self._sessions.values())
                existing_teardowns = list(self._teardown_tasks)
                reaper = self._reaper_task
                self._reaper_task = None
                self._sessions.clear()
                session_teardowns = [
                    self._start_teardown(
                        session,
                        acquire_lock=True,
                        suppress_errors=True,
                    )
                    for session in sessions
                ]
            if reaper is not None:
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
            for worker in starting:
                worker.cancel()
            if starting:
                await asyncio.gather(*starting, return_exceptions=True)
            async with self._registry_lock:
                for worker in starting:
                    if worker in self._starting_workers:
                        self._starting_workers.remove(worker)
                        self._creating_sessions -= 1
            all_teardowns = [*existing_teardowns, *session_teardowns]
            if all_teardowns:
                await asyncio.shield(
                    asyncio.gather(
                        *all_teardowns,
                        return_exceptions=True,
                    )
                )
