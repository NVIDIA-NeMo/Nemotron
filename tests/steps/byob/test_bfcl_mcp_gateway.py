from __future__ import annotations

import asyncio
import hashlib
import json
from base64 import b64encode
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    attestation_digest,
    verify_conformance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.client import (
    McpServerIdentity,
    McpToolPage,
    SdkConnectedMcpClient,
)
from nemotron.steps.byob.runtime.mcp.config import LoadedMcpOracleConfig, McpOracleConfig
from nemotron.steps.byob.runtime.mcp.discovery import catalog_identity_document
from nemotron.steps.byob.runtime.mcp.gateway import (
    GatewayArtifacts,
    GatewayError,
    GatewayService,
    create_gateway_app,
)
from nemotron.steps.byob.runtime.mcp.gateway.conformance import (
    ConformanceEvidence,
    ProbeOutcome,
    load_conformance_evidence,
    run_gateway_timeout_conformance,
)
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    BFCL_ORACLE_PROTOCOL_VERSION,
)
from nemotron.steps.byob.runtime.mcp.gateway.result_mapping import (
    control_result_object,
    map_call_result,
)
from nemotron.steps.byob.runtime.mcp.normalization import normalize_catalog
from nemotron.steps.byob.runtime.release_seal import ReleaseSealAuthority

ZERO_DIGEST = "sha256:" + "0" * 64
SERVER_DIGEST = "sha256:" + "a" * 64
GATEWAY_DIGEST = "sha256:" + "b" * 64
EVIDENCE_AUTHORITY = ReleaseSealAuthority(
    issuer="bfcl-release-authority",
    key_id="release-key-1",
    private_key=Ed25519PrivateKey.generate(),
)

TOOLS = [
    {
        "name": "bfcl.describe",
        "description": "Describe this oracle.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "bfcl.reset",
        "description": "Reset one episode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fixtures": {"type": ["object", "null"]},
                "context": {"type": "object"},
            },
        },
    },
    {
        "name": "bfcl.state",
        "description": "Get state.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
    },
    {
        "name": "bfcl.end",
        "description": "End episode.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
    },
    {
        "name": "inventory.lookup",
        "description": "Look up one item.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "episode_id": {"type": "string"},
            },
            "required": ["item_id", "episode_id"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
    },
]


def _raw_config(
    *,
    tool_timeout_s: float = 1.0,
    episode_timeout_s: float = 30.0,
    session_idle_ttl_s: float = 10.0,
) -> dict[str, Any]:
    return {
        "profile_version": "bfcl-mcp-oracle-v1",
        "mode": "A",
        "mcp_protocol_versions": ["2026-07-28"],
        "transport": {
            "kind": "streamable_http",
            "url": "https://mcp.example.test/mcp",
        },
        "expected": {
            "server_name": "catalog",
            "server_version": "1.0.0",
            "tool_catalog_digest": ZERO_DIGEST,
            "oracle_id": "catalog-oracle",
            "oracle_version": "1.0.0",
            "server_content_digest": SERVER_DIGEST,
        },
        "control": {
            "reset_strategy": "control_tool",
            "state_strategy": "control_tool",
            "describe_oracle": "bfcl.describe",
            "reset_episode": "bfcl.reset",
            "get_episode_state": "bfcl.state",
            "end_episode": "bfcl.end",
            "episode_binding": "argument",
            "episode_argument": "episode_id",
        },
        "fixtures": {"direction": "pushed"},
        "tools": {
            "include": ["inventory.lookup"],
            "aliases": {"inventory.lookup": "inventory_lookup"},
            "mutates": [],
            "requires_confirmation": [],
            "trust_annotations": False,
        },
        "isolation": "namespace_per_episode",
        "limits": {
            "connect_timeout_s": 1,
            "handshake_timeout_s": 1,
            "tool_timeout_s": tool_timeout_s,
            "reset_timeout_s": 1,
            "episode_timeout_s": episode_timeout_s,
            "max_response_bytes": 100_000,
            "max_tools": 16,
            "max_catalog_pages": 4,
            "max_concurrent_episodes": 2,
            "session_idle_ttl_s": session_idle_ttl_s,
        },
    }


def _loaded(
    *,
    tool_timeout_s: float = 1.0,
    episode_timeout_s: float = 30.0,
    session_idle_ttl_s: float = 10.0,
) -> LoadedMcpOracleConfig:
    config = McpOracleConfig.model_validate(
        _raw_config(
            tool_timeout_s=tool_timeout_s,
            episode_timeout_s=episode_timeout_s,
            session_idle_ttl_s=session_idle_ttl_s,
        )
    )
    catalog = normalize_catalog(TOOLS, config)
    document = catalog_identity_document(
        config,
        negotiated_mcp_version="2026-07-28",
        server_name="catalog",
        server_version="1.0.0",
        catalog=catalog,
    )
    digest = "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
    config = config.model_copy(update={"expected": config.expected.model_copy(update={"tool_catalog_digest": digest})})
    return LoadedMcpOracleConfig(
        path=Path("/tmp/mcp_oracle.yaml"),
        value=config,
        raw_document=config.model_dump(mode="json", exclude_none=False),
    )


class _FakeClient:
    sdk_version = "2.1.0-test"
    protocol_version = "2026-07-28"
    server_identity = McpServerIdentity("catalog", "1.0.0")
    capabilities = {"tools": {"listChanged": False}}

    def __init__(
        self,
        index: int,
        *,
        hang_business_call: bool = False,
        factory: _Factory | None = None,
    ):
        self.index = index
        self.hang_business_call = hang_business_call
        self.factory = factory
        self.closed = False
        self.business_calls = 0
        self.episode_id: str | None = None
        self.state: dict[str, Any] = {}
        self.reset_context: dict[str, Any] | None = None

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        assert cursor is None
        if self.factory is not None and self.factory.hang_discovery:
            assert self.factory.discovery_started is not None
            assert self.factory.release_discovery is not None
            self.factory.discovery_started.set()
            await self.factory.release_discovery.wait()
        return McpToolPage(tuple(TOOLS), None)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name == "bfcl.describe":
            return {
                "structuredContent": {
                    "oracle_id": "catalog-oracle",
                    "oracle_version": "1.0.0",
                    "content_digest": SERVER_DIGEST,
                }
            }
        if name == "bfcl.reset":
            if self.factory is not None and self.factory.hang_reset:
                assert self.factory.reset_started is not None
                self.factory.reset_started.set()
                await asyncio.sleep(60)
            self.episode_id = f"episode-{self.index}"
            self.reset_context = arguments["context"]
            self.state = {"calls": 0, "fixtures": arguments["fixtures"]}
            if self.factory is not None and self.factory.pause_reset_return:
                assert self.factory.reset_ready is not None
                assert self.factory.release_reset is not None
                self.factory.reset_ready.set()
                await self.factory.release_reset.wait()
            return {"structuredContent": {"episode_id": self.episode_id}}
        if name == "bfcl.state":
            assert arguments["episode_id"] == self.episode_id
            return {"structuredContent": dict(self.state)}
        if name == "bfcl.end":
            assert arguments["episode_id"] == self.episode_id
            return {"structuredContent": {"closed": True}}
        if name == "inventory.lookup":
            self.business_calls += 1
            assert arguments["episode_id"] == self.episode_id
            assert meta is None
            if self.factory is not None and self.factory.block_business_call:
                assert self.factory.business_started is not None
                assert self.factory.release_business is not None
                self.factory.business_started.set()
                await self.factory.release_business.wait()
            if self.hang_business_call:
                await asyncio.sleep(60)
            self.state["calls"] += 1
            return {"structuredContent": {"item": f"{self.episode_id}:{arguments['item_id']}"}}
        raise AssertionError(f"unexpected tool {name}")


class _Factory:
    def __init__(
        self,
        *,
        hang_business_call: bool = False,
        hang_reset: bool = False,
        block_business_call: bool = False,
        hang_discovery: bool = False,
        pause_reset_return: bool = False,
        close_delay_s: float = 0.0,
        close_exception: BaseException | None = None,
    ):
        self.hang_business_call = hang_business_call
        self.hang_reset = hang_reset
        self.block_business_call = block_business_call
        self.hang_discovery = hang_discovery
        self.pause_reset_return = pause_reset_return
        self.close_delay_s = close_delay_s
        self.close_exception = close_exception
        self.reset_started: asyncio.Event | None = None
        self.business_started: asyncio.Event | None = None
        self.release_business: asyncio.Event | None = None
        self.discovery_started: asyncio.Event | None = None
        self.release_discovery: asyncio.Event | None = None
        self.reset_ready: asyncio.Event | None = None
        self.release_reset: asyncio.Event | None = None
        self.clients: list[_FakeClient] = []
        self.context_owners: list[tuple[asyncio.Task[Any] | None, asyncio.Task[Any] | None]] = []

    def __call__(self, config: McpOracleConfig):
        return self._open(config)

    @asynccontextmanager
    async def _open(self, config: McpOracleConfig):
        entered_by = asyncio.current_task()
        client = _FakeClient(
            len(self.clients),
            hang_business_call=self.hang_business_call,
            factory=self,
        )
        self.clients.append(client)
        try:
            yield client
        finally:
            # Stands in for an episode transport whose shutdown is itself an await, such
            # as the DELETE a Streamable HTTP session sends under terminate_on_close.
            # Client 0 is the startup discovery connection and closes immediately.
            if self.close_delay_s and client.index > 0:
                await asyncio.sleep(self.close_delay_s)
            client.closed = True
            self.context_owners.append((entered_by, asyncio.current_task()))
            if self.close_exception is not None and client.index > 0:
                raise self.close_exception


def _service(
    *,
    factory: _Factory | None = None,
    tool_timeout_s: float = 1.0,
    episode_timeout_s: float = 30.0,
    session_idle_ttl_s: float = 10.0,
    conformance_evidence: ConformanceEvidence | None = None,
) -> tuple[GatewayService, _Factory]:
    factory = factory or _Factory()
    return (
        GatewayService(
            _loaded(
                tool_timeout_s=tool_timeout_s,
                episode_timeout_s=episode_timeout_s,
                session_idle_ttl_s=session_idle_ttl_s,
            ),
            artifacts=GatewayArtifacts(GATEWAY_DIGEST),
            connection_factory=factory,
            conformance_evidence=conformance_evidence,
        ),
        factory,
    )


def _context(task_id: str = "task-1") -> dict[str, Any]:
    return {
        "clock": "2026-08-25T12:00:00+00:00",
        "seed": 7,
        "timeout_s": 5,
        "task_id": task_id,
    }


def _signed_gateway_suite() -> dict[str, Any]:
    suite: dict[str, Any] = {
        "kind": "gateway",
        "profile_version": "bfcl-mcp-gateway-conformance-v1",
        "p9": {
            "timeout_observed": True,
            "business_call_attempts": 1,
            "episode_poisoned": True,
            "transport_cleanup_completed": True,
            "unknown_commit_state_preserved": True,
        },
        "producer": {
            "issuer": EVIDENCE_AUTHORITY.issuer,
            "signing_key_id": EVIDENCE_AUTHORITY.key_id,
            "gateway_artifact_digest": GATEWAY_DIGEST,
        },
    }
    suite["evidence_digest"] = attestation_digest(suite)
    suite["signature"] = b64encode(
        EVIDENCE_AUTHORITY.private_key.sign(suite["evidence_digest"].encode("ascii"))
    ).decode("ascii")
    return suite


def test_result_mapping_covers_success_error_and_confirmation() -> None:
    config = _loaded().value.results
    assert map_call_result(
        {"structuredContent": {"item": "A"}},
        config=config,
        output_schema=None,
    ) == {"item": "A"}
    assert map_call_result(
        {
            "isError": True,
            "structuredContent": {"error": {"code": "not_found", "id": "A"}},
        },
        config=config,
        output_schema=None,
    ) == {"error": {"code": "not_found", "id": "A"}}
    assert map_call_result(
        {"structuredContent": {"status": "awaiting_confirmation"}},
        config=config,
        output_schema=None,
    ) == {"status": "awaiting_confirmation"}
    # The SDK always dumps the default discriminator, so it must stay mappable.
    assert map_call_result(
        {"resultType": "complete", "isError": False, "structuredContent": {"item": "A"}},
        config=config,
        output_schema=None,
    ) == {"item": "A"}


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"resultType": "input_required"}, "mcp_input_required_unsupported"),
        ({"task": {"taskId": "t-1"}}, "mcp_async_task_unsupported"),
        (
            {"resultType": "vendorExtension", "structuredContent": {"episode_id": "e"}},
            "mcp_unsupported_result_type",
        ),
    ],
)
def test_control_results_refuse_unsupported_shapes(
    payload: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(GatewayError) as caught:
        control_result_object(payload, operation="bfcl.reset")
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"content": [{"type": "text", "text": "{}"}]}, "mcp_result_not_object"),
        (
            {"structuredContent": {"error": {"code": "bad"}}},
            "mcp_error_flag_inconsistent",
        ),
        (
            {"isError": True, "structuredContent": {"error": {"message": "bad"}}},
            "mcp_unstructured_error",
        ),
        (
            {"resultType": "input_required", "structuredContent": {}},
            "mcp_input_required_unsupported",
        ),
        (
            {"task": {"taskId": "t-1", "status": "working"}},
            "mcp_async_task_unsupported",
        ),
        (
            {"resultType": "streamingChunk", "structuredContent": {"item": "A"}},
            "mcp_unsupported_result_type",
        ),
    ],
)
def test_result_mapping_fails_closed(
    payload: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(GatewayError) as caught:
        map_call_result(
            payload,
            config=_loaded().value.results,
            output_schema=None,
        )
    assert caught.value.code == code
    assert caught.value.http_status >= 500


def test_result_mapping_enforces_output_schema() -> None:
    with pytest.raises(GatewayError) as caught:
        map_call_result(
            {"structuredContent": {"wrong": "shape"}},
            config=_loaded().value.results,
            output_schema=TOOLS[-1]["outputSchema"],
        )
    assert caught.value.code == "mcp_output_schema_mismatch"


def test_gateway_identity_is_deterministic_and_protocol_compatible() -> None:
    async def run() -> None:
        service, factory = _service()
        await service.start()
        try:
            first = service.metadata()
            second = service.metadata()
            assert first == second
            assert first["protocol_version"] == BFCL_ORACLE_PROTOCOL_VERSION
            assert first["oracle_id"] == "catalog-oracle"
            assert first["content_digest"].startswith("sha256:")
        finally:
            await service.shutdown()
        assert factory.context_owners
        assert all(entered is exited for entered, exited in factory.context_owners)

    asyncio.run(run())


def test_catalog_digest_covers_output_schema_and_runtime_annotations() -> None:
    config = _loaded().value

    def digest(tools: list[dict[str, Any]]) -> str:
        catalog = normalize_catalog(tools, config)
        document = catalog_identity_document(
            config,
            negotiated_mcp_version="2026-07-28",
            server_name="catalog",
            server_version="1.0.0",
            catalog=catalog,
        )
        return "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()

    baseline = digest(TOOLS)
    assert baseline == "sha256:6ca7d204b21701f38da881dbf187a0adfe24d95becc0bfe4571774477b56f846"
    schema_changed = json.loads(json.dumps(TOOLS))
    schema_changed[-1]["outputSchema"]["properties"]["item"]["type"] = "integer"
    annotation_changed = json.loads(json.dumps(TOOLS))
    annotation_changed[-1]["annotations"] = {"readOnlyHint": False}

    assert digest(schema_changed) != baseline
    assert digest(annotation_changed) != baseline


def test_two_sessions_are_isolated_and_cleanup_is_idempotent() -> None:
    async def run() -> None:
        service, factory = _service()
        await service.start()
        try:
            first = await service.create_session(
                context=_context("first"),
                fixtures={"items": [{"id": "A"}]},
            )
            second = await service.create_session(
                context=_context("second"),
                fixtures={"items": [{"id": "B"}]},
            )
            first_result, second_result = await asyncio.gather(
                service.call_tool(
                    first["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                ),
                service.call_tool(
                    second["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "B"},
                    turn_index=0,
                ),
            )
            assert first_result["item"] != second_result["item"]
            assert (await service.get_state(first["session_id"]))["calls"] == 1
            assert (await service.get_state(second["session_id"]))["calls"] == 1
            await service.delete_session(first["session_id"])
            await service.delete_session(first["session_id"])
            assert factory.clients[1].closed is True
            assert factory.clients[2].closed is False
        finally:
            await service.shutdown()
        assert factory.context_owners
        assert all(entered is exited for entered, exited in factory.context_owners)

    asyncio.run(run())


def test_delete_reports_cancelled_transport_cleanup_as_failure() -> None:
    async def run() -> None:
        service, _ = _service(
            factory=_Factory(close_exception=asyncio.CancelledError()),
        )
        await service.start()
        try:
            created = await service.create_session(context=_context(), fixtures=None)
            with pytest.raises(GatewayError) as caught:
                await service.delete_session(created["session_id"])
            assert caught.value.code == "mcp_close_failed"
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_background_reaper_reclaims_abandoned_sessions() -> None:
    async def run() -> None:
        service, factory = _service(
            session_idle_ttl_s=0.02,
            episode_timeout_s=1.0,
        )
        await service.start()
        try:
            created = await service.create_session(context=_context(), fixtures=None)
            await asyncio.sleep(0.12)
            assert created["session_id"] not in service._sessions
            assert factory.clients[1].closed is True
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_session_readiness_is_bounded_by_episode_deadline() -> None:
    async def run() -> None:
        factory = _Factory(pause_reset_return=True)
        service, _ = _service(
            factory=factory,
            episode_timeout_s=0.02,
        )
        await service.start()
        factory.reset_ready = asyncio.Event()
        factory.release_reset = asyncio.Event()
        try:
            with pytest.raises(GatewayError) as caught:
                await service.create_session(context=_context(), fixtures=None)
            assert caught.value.code == "mcp_session_create_timeout"
            assert service._creating_sessions == 0
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_timeout_poisoning_never_retries_a_business_call() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True),
            tool_timeout_s=0.01,
        )
        await service.start()
        try:
            created = await service.create_session(
                context=_context(),
                fixtures=None,
            )
            with pytest.raises(GatewayError) as caught:
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                )
            assert caught.value.code == "mcp_call_timeout"
            session_client = factory.clients[1]
            assert session_client.business_calls == 1
            assert session_client.closed is True
            with pytest.raises(GatewayError) as poisoned:
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=1,
                )
            assert poisoned.value.code == "mcp_session_poisoned"
            assert session_client.business_calls == 1
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_bfcl_owned_gateway_timeout_suite_records_p9_evidence() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True),
            tool_timeout_s=0.01,
        )
        suite = await run_gateway_timeout_conformance(
            service,
            context=_context(),
            business_tool="inventory_lookup",
            arguments={"item_id": "A"},
            business_call_attempts=lambda: sum(client.business_calls for client in factory.clients),
            transport_cleanup_completed=lambda: all(client.closed for client in factory.clients),
        )

        assert suite == {
            "kind": "gateway",
            "profile_version": "bfcl-mcp-gateway-conformance-v1",
            "p9": {
                "timeout_observed": True,
                "business_call_attempts": 1,
                "episode_poisoned": True,
                "transport_cleanup_completed": True,
                "unknown_commit_state_preserved": True,
            },
        }

    asyncio.run(run())


def test_poisoning_lets_the_transport_finish_its_own_shutdown() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True, close_delay_s=0.01),
            tool_timeout_s=0.05,
        )
        await service.start()
        try:
            created = await service.create_session(context=_context(), fixtures=None)
            with pytest.raises(GatewayError):
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                )
            assert factory.clients[1].closed is True
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_a_transport_that_will_not_close_is_cancelled_within_the_grace_period() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True, close_delay_s=60.0),
            tool_timeout_s=0.01,
        )
        await service.start()
        try:
            created = await service.create_session(context=_context(), fixtures=None)
            session = service._sessions[created["session_id"]]
            with pytest.raises(GatewayError):
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                )
            assert session.closed is True
            assert session.worker.cancelled() is True
            assert factory.clients[1].closed is False
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_reset_receives_the_bfcl_context_verbatim() -> None:
    async def run() -> None:
        service, factory = _service()
        await service.start()
        try:
            context = _context()
            await service.create_session(context=context, fixtures=None)
            assert factory.clients[1].reset_context == context
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_the_conformance_route_serves_the_exact_bytes_its_digest_covers() -> None:
    service, _ = _service()
    app = create_gateway_app(service)
    with TestClient(app) as client:
        response = client.get("/v1/conformance")
        assert response.status_code == 200
        document = response.json()

        # The pack pins a digest over the served bytes, so re-encoding on either side would
        # break verification even though the JSON is semantically identical.
        assert response.content == canonical_json(document).encode("utf-8")
        assert attestation_digest(document) == attestation_digest(json.loads(response.content))

        # Both routes must describe the same build.
        assert document["effective_content_digest"] == client.get("/v1/metadata").json()["content_digest"]
        assert document["gateway_artifact_digest"] == GATEWAY_DIGEST

        # The gateway declares only what it ran. P4 is not implemented, so discovery alone
        # remains L0 rather than claiming the executable level.
        assert document["level"] == "L0"
        assert document["gateway_evidence_kind"] == "locally_verified"
        assert {check["id"] for check in document["checks"]} == {"P1", "P2", "P3"}


def test_verified_probe_evidence_can_move_the_live_route_to_l2() -> None:
    evidence = ConformanceEvidence(
        probes=tuple(
            ProbeOutcome(
                id=f"P{index}",
                requirement="conditional" if index in {7, 8} else "required",
                status="pass",
            )
            for index in range(1, 12)
        ),
        suite=_signed_gateway_suite(),
        trusted_producer=True,
    )
    service, _ = _service(conformance_evidence=evidence)
    with TestClient(create_gateway_app(service)) as client:
        document = client.get("/v1/conformance").json()
        metadata = client.get("/v1/metadata").json()
        probe_report = client.get("/v1/conformance/probe-report").json()
        gateway_report = client.get("/v1/conformance/gateway-report").json()

    assert document["level"] == "L2"
    assert {check["id"] for check in document["checks"]} == {f"P{index}" for index in range(1, 12)}
    assert attestation_digest(probe_report) == document["probe_report_digest"]
    assert attestation_digest(gateway_report) == document["gateway_conformance_report_digest"]
    verdict = verify_conformance(
        document,
        expected_digest=attestation_digest(document),
        metadata_content_digest=metadata["content_digest"],
        probe_report=probe_report,
        gateway_conformance_report=gateway_report,
        trusted_evidence_keys={EVIDENCE_AUTHORITY.key_id: EVIDENCE_AUTHORITY.public_key},
        expected_evidence_issuer=EVIDENCE_AUTHORITY.issuer,
    )
    assert verdict.publishable is True
    assert verdict.effective_level == "L2"


def test_gateway_loads_only_complete_ordered_probes_and_passing_p9_suite() -> None:
    probes = {
        "probes": [
            {
                "id": f"P{index}",
                "requirement": ("conditional" if index in {7, 8} else "required"),
                "status": "pass",
                "reason": None,
            }
            for index in range(1, 12)
        ]
    }
    suite = {
        "kind": "gateway",
        "profile_version": "bfcl-mcp-gateway-conformance-v1",
        "p9": {
            "timeout_observed": True,
            "business_call_attempts": 1,
            "episode_poisoned": True,
            "transport_cleanup_completed": True,
            "unknown_commit_state_preserved": True,
        },
    }

    evidence = load_conformance_evidence(probes, suite)
    assert evidence.probe_report == probes
    assert evidence.trusted_producer is False

    suite["p9"]["business_call_attempts"] = 2
    with pytest.raises(ValueError, match="passing P9"):
        load_conformance_evidence(probes, suite)

    signed = _signed_gateway_suite()
    trusted = load_conformance_evidence(
        probes,
        signed,
        trusted_public_keys={EVIDENCE_AUTHORITY.key_id: EVIDENCE_AUTHORITY.public_key},
        expected_issuer=EVIDENCE_AUTHORITY.issuer,
        gateway_artifact_digest=GATEWAY_DIGEST,
    )
    assert trusted.trusted_producer is True
    signed["p9"]["business_call_attempts"] = 2
    with pytest.raises(ValueError):
        load_conformance_evidence(
            probes,
            signed,
            trusted_public_keys={EVIDENCE_AUTHORITY.key_id: EVIDENCE_AUTHORITY.public_key},
            expected_issuer=EVIDENCE_AUTHORITY.issuer,
            gateway_artifact_digest=GATEWAY_DIGEST,
        )


def test_a_gateway_attestation_cannot_publish_on_its_own_word() -> None:
    service, _ = _service()
    app = create_gateway_app(service)
    with TestClient(app) as client:
        document = client.get("/v1/conformance").json()
        metadata = client.get("/v1/metadata").json()

    verdict = verify_conformance(
        document,
        expected_digest=attestation_digest(document),
        metadata_content_digest=metadata["content_digest"],
    )
    # Digests all agree, and it still cannot publish: the reports referenced by the gateway
    # were not independently supplied to the verifier.
    assert verdict.findings == (
        "probe_report_missing",
        "gateway_conformance_report_missing",
    )
    assert verdict.attested_level == "L0"
    assert verdict.publishable is False


def test_http_adapter_matches_bfcl_oracle_v1_routes() -> None:
    service, _ = _service()
    app = create_gateway_app(service)
    with TestClient(app) as client:
        metadata = client.get("/v1/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["protocol_version"] == BFCL_ORACLE_PROTOCOL_VERSION
        assert client.get("/v1/tools").json() == {"tools": ["inventory_lookup"]}

        created = client.post(
            "/v1/sessions",
            json={"context": _context(), "fixtures": None},
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        result = client.post(
            f"/v1/sessions/{session_id}/calls",
            json={
                "name": "inventory_lookup",
                "arguments": {"item_id": "A"},
                "turn_index": 0,
            },
        )
        assert result.status_code == 200
        assert result.json()["item"].endswith(":A")
        assert client.get(f"/v1/sessions/{session_id}/state").json()["calls"] == 1
        assert client.delete(f"/v1/sessions/{session_id}").status_code == 204
        assert client.delete(f"/v1/sessions/{session_id}").status_code == 204


def test_http_adapter_rejects_unknown_sessions_and_duplicate_json_keys() -> None:
    service, _ = _service()
    app = create_gateway_app(service)
    with TestClient(app) as client:
        unknown = client.get("/v1/sessions/missing/state")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "mcp_session_unknown"
        duplicate = client.post(
            "/v1/sessions",
            content='{"context":{},"context":{},"fixtures":null}',
            headers={"Content-Type": "application/json"},
        )
        assert duplicate.status_code == 400
        assert duplicate.json()["error"]["code"] == "mcp_request_invalid"


def test_http_adapter_enforces_optional_client_authentication() -> None:
    service, _ = _service()
    app = create_gateway_app(service, client_bearer_token="gateway-secret")
    with TestClient(app) as client:
        assert client.get("/v1/metadata").status_code == 401
        authorized = client.get(
            "/v1/metadata",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        assert authorized.status_code == 200
        # A non-ASCII credential must be rejected, not raise inside the comparison.
        non_ascii = client.get(
            "/v1/metadata",
            headers={b"Authorization": "Bearer gateway-sécret".encode()},
        )
        assert non_ascii.status_code == 401


class _DumpedResult:
    def model_dump(
        self,
        *,
        mode: str,
        by_alias: bool,
        exclude_none: bool,
    ) -> dict[str, Any]:
        assert mode == "json"
        assert by_alias is True
        assert exclude_none is True
        return {
            "resultType": "input_required",
            "inputRequests": {},
            "requestState": "opaque",
        }


class _LowLevelSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None,
        allow_input_required: bool,
        allow_claimed: bool,
    ) -> _DumpedResult:
        self.calls += 1
        assert allow_input_required is True
        assert allow_claimed is True
        return _DumpedResult()


class _RawSdkClient:
    def __init__(self) -> None:
        self.session = _LowLevelSession()


def test_sdk_facade_exposes_input_required_without_automatic_retry() -> None:
    raw = _RawSdkClient()
    facade = SdkConnectedMcpClient(
        raw,
        max_response_bytes=100_000,
        sdk_version="2.1.0-test",
    )
    result = asyncio.run(facade.call_tool("lookup", {"id": "A"}))
    assert raw.session.calls == 1
    with pytest.raises(GatewayError) as caught:
        map_call_result(
            result,
            config=_loaded().value.results,
            output_schema=None,
        )
    assert caught.value.code == "mcp_input_required_unsupported"


def test_cancelled_session_creation_releases_capacity_and_transport() -> None:
    async def run() -> None:
        factory = _Factory(hang_reset=True)
        factory.reset_started = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        creation = asyncio.create_task(service.create_session(context=_context(), fixtures=None))
        await asyncio.wait_for(factory.reset_started.wait(), timeout=1)
        creation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creation
        assert service._creating_sessions == 0
        assert not service._starting_workers
        assert factory.clients[1].closed is True

        factory.hang_reset = False
        created = await service.create_session(context=_context(), fixtures=None)
        await service.delete_session(created["session_id"])
        await service.shutdown()

    asyncio.run(run())


def test_shutdown_cancels_and_awaits_in_progress_session_creation() -> None:
    async def run() -> None:
        factory = _Factory(hang_reset=True)
        factory.reset_started = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        creation = asyncio.create_task(service.create_session(context=_context(), fixtures=None))
        await asyncio.wait_for(factory.reset_started.wait(), timeout=1)
        await service.shutdown()
        with pytest.raises(GatewayError) as caught:
            await creation
        assert caught.value.code == "mcp_gateway_shutting_down"
        assert service._creating_sessions == 0
        assert not service._starting_workers
        assert all(client.closed for client in factory.clients)

    asyncio.run(run())


def test_cancellation_while_registering_a_ready_session_cleans_worker() -> None:
    async def run() -> None:
        factory = _Factory(pause_reset_return=True)
        factory.reset_ready = asyncio.Event()
        factory.release_reset = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        creation = asyncio.create_task(service.create_session(context=_context(), fixtures=None))
        await asyncio.wait_for(factory.reset_ready.wait(), timeout=1)
        await service._registry_lock.acquire()
        factory.release_reset.set()
        await asyncio.sleep(0)
        creation.cancel()
        service._registry_lock.release()
        with pytest.raises(asyncio.CancelledError):
            await creation
        assert service._creating_sessions == 0
        assert not service._starting_workers
        assert factory.clients[1].closed is True
        await service.shutdown()

    asyncio.run(run())


def test_shutdown_serializes_with_in_progress_startup() -> None:
    async def run() -> None:
        factory = _Factory(hang_discovery=True)
        factory.discovery_started = asyncio.Event()
        factory.release_discovery = asyncio.Event()
        service, _ = _service(factory=factory)
        startup = asyncio.create_task(service.start())
        await asyncio.wait_for(factory.discovery_started.wait(), timeout=1)
        shutdown = asyncio.create_task(service.shutdown())
        factory.release_discovery.set()
        await asyncio.gather(startup, shutdown)
        with pytest.raises(GatewayError) as caught:
            service.metadata()
        assert caught.value.code == "mcp_gateway_not_ready"
        assert all(client.closed for client in factory.clients)

    asyncio.run(run())


def test_expiry_waits_for_an_inflight_call_before_cleanup() -> None:
    async def run() -> None:
        factory = _Factory(block_business_call=True)
        factory.business_started = asyncio.Event()
        factory.release_business = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        created = await service.create_session(context=_context(), fixtures=None)
        session_id = created["session_id"]
        call = asyncio.create_task(
            service.call_tool(
                session_id,
                name="inventory_lookup",
                arguments={"item_id": "A"},
                turn_index=0,
            )
        )
        await asyncio.wait_for(factory.business_started.wait(), timeout=1)
        session = service._sessions[session_id]
        expired_lookup = asyncio.create_task(service.get_state(session_id))
        await asyncio.sleep(0)
        session.created_at -= float(service.config.limits.episode_timeout_s) + 1
        assert factory.clients[1].closed is False

        factory.release_business.set()
        assert (await call)["item"].endswith(":A")
        with pytest.raises(GatewayError) as caught:
            await expired_lookup
        assert caught.value.code == "mcp_session_unknown"
        assert factory.clients[1].closed is True
        await service.shutdown()

    asyncio.run(run())


def test_cancelled_delete_keeps_teardown_tracked_until_completion() -> None:
    async def run() -> None:
        factory = _Factory(block_business_call=True)
        factory.business_started = asyncio.Event()
        factory.release_business = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        created = await service.create_session(context=_context(), fixtures=None)
        session_id = created["session_id"]
        call = asyncio.create_task(
            service.call_tool(
                session_id,
                name="inventory_lookup",
                arguments={"item_id": "A"},
                turn_index=0,
            )
        )
        await asyncio.wait_for(factory.business_started.wait(), timeout=1)
        deletion = asyncio.create_task(service.delete_session(session_id))
        await asyncio.sleep(0)
        deletion.cancel()
        with pytest.raises(asyncio.CancelledError):
            await deletion
        assert service._teardown_tasks

        factory.release_business.set()
        await call
        await asyncio.gather(*list(service._teardown_tasks))
        assert factory.clients[1].closed is True
        await service.shutdown()

    asyncio.run(run())


def test_http_adapter_streams_request_limit_and_rejects_nonfinite_json() -> None:
    service, _ = _service()
    app = create_gateway_app(service, max_request_bytes=32)
    with TestClient(app) as client:
        oversized = client.post(
            "/v1/sessions",
            content=(chunk for chunk in (b'{"context":"', b"x" * 64, b'"}')),
            headers={"Content-Type": "application/json"},
        )
        assert oversized.status_code == 413
        nonfinite = client.post(
            "/v1/sessions",
            content=b'{"context":NaN,"fixtures":null}',
            headers={"Content-Type": "application/json"},
        )
        assert nonfinite.status_code == 400
        assert nonfinite.json()["error"]["code"] == "mcp_request_invalid"


@pytest.mark.parametrize(
    "artifacts",
    [
        GatewayArtifacts(GATEWAY_DIGEST, shim_artifact_digest=ZERO_DIGEST),
        GatewayArtifacts(GATEWAY_DIGEST, snapshot_digest=ZERO_DIGEST),
    ],
)
def test_mode_a_rejects_irrelevant_artifact_digests(
    artifacts: GatewayArtifacts,
) -> None:
    async def run() -> None:
        factory = _Factory()
        service = GatewayService(
            _loaded(),
            artifacts=artifacts,
            connection_factory=factory,
        )
        with pytest.raises(GatewayError) as caught:
            await service.start()
        assert caught.value.code == "mcp_gateway_identity_invalid"

    asyncio.run(run())
