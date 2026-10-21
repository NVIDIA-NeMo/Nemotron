from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)
from nemotron.steps.byob.runtime.mcp.client import open_mcp_connection
from nemotron.steps.byob.runtime.mcp.config import (
    LoadedMcpOracleConfig,
    McpOracleConfig,
    TrustedExecutablePolicies,
)
from nemotron.steps.byob.runtime.mcp.discovery import catalog_identity_document
from nemotron.steps.byob.runtime.mcp.gateway import GatewayArtifacts, GatewayService
from nemotron.steps.byob.runtime.mcp.normalization import normalize_catalog
from tests.steps.byob.mcp_fixture_server import BUSINESS_TOOLS, CONTROL_TOOLS

_MCP_VERSION = importlib.metadata.version("mcp")
pytestmark = pytest.mark.skipif(
    _MCP_VERSION.split(".", 1)[0] != "2",
    reason="real transport integration requires the isolated bfcl-mcp SDK v2 runtime",
)

_DIGEST = "sha256:" + "0" * 64
_CONTENT_DIGEST = "sha256:" + "a" * 64


def _config(server: Path) -> McpOracleConfig:
    executable = Path(sys.executable).resolve()
    return McpOracleConfig.model_validate(
        {
            "profile_version": "bfcl-mcp-oracle-v1",
            "mode": "A",
            "mcp_protocol_versions": ["2026-07-28"],
            "transport": {
                "kind": "stdio",
                "command": [executable.name, str(server)],
                "cwd": str(server.parent),
                "env_passthrough": [],
                "executable_policy": "python-fixture",
            },
            "expected": {
                "server_name": "bfcl-fixture-oracle",
                "server_version": "1.0.0",
                "tool_catalog_digest": _DIGEST,
                "oracle_id": "bfcl-fixture-oracle",
                "oracle_version": "1.0.0",
                "server_content_digest": _CONTENT_DIGEST,
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
                "include": ["inventory.lookup", "inventory.reserve"],
                "aliases": {
                    "inventory.lookup": "inventory_lookup",
                    "inventory.reserve": "inventory_reserve",
                },
                "mutates": ["inventory_reserve"],
                "requires_confirmation": ["inventory_reserve"],
                "trust_annotations": False,
            },
            "results": {
                "error_path": "error",
                "status_field": "status",
                "pending_status": "pending_confirmation",
                "confirmation_parameter": "confirmed",
            },
            "isolation": "namespace_per_episode",
            "limits": {
                "connect_timeout_s": 2,
                "handshake_timeout_s": 2,
                "tool_timeout_s": 2,
                "reset_timeout_s": 2,
                "episode_timeout_s": 10,
                "max_response_bytes": 100_000,
                "max_tools": 16,
                "max_catalog_pages": 4,
                "max_concurrent_episodes": 2,
                "session_idle_ttl_s": 10,
            },
        }
    )


def _policies(server: Path) -> TrustedExecutablePolicies:
    executable = Path(sys.executable).resolve()
    digest = "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest()
    return TrustedExecutablePolicies.model_validate(
        {
            "schema_version": "bfcl-trusted-executables-v1",
            "policies": {
                "python-fixture": {
                    "executable": str(executable),
                    "sha256": digest,
                    "allowed_argv": [[str(server)]],
                    "allowed_cwd_roots": [str(server.parent)],
                }
            },
        }
    )


def _http_config(server: Path, url: str) -> McpOracleConfig:
    raw = _config(server).model_dump(mode="json", exclude_none=False)
    raw["transport"] = {
        "kind": "streamable_http",
        "url": url,
        "auth": {"bearer_token_env": "MCP_FIXTURE_TOKEN", "headers": {}},
        "tls": {"ca_bundle_path": None},
    }
    return McpOracleConfig.model_validate(raw)


def _loaded_http_config(server: Path, url: str) -> LoadedMcpOracleConfig:
    return _loaded_config(_http_config(server, url), server)


def _loaded_config(
    config: McpOracleConfig,
    source: Path,
) -> LoadedMcpOracleConfig:
    catalog = normalize_catalog([*CONTROL_TOOLS, *BUSINESS_TOOLS], config)
    identity = catalog_identity_document(
        config,
        negotiated_mcp_version="2026-07-28",
        server_name="bfcl-fixture-oracle",
        server_version="1.0.0",
        catalog=catalog,
    )
    digest = "sha256:" + hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()
    config = config.model_copy(
        update={
            "expected": config.expected.model_copy(
                update={"tool_catalog_digest": digest}
            )
        }
    )
    return LoadedMcpOracleConfig(
        path=source.with_name("mcp_oracle.yaml"),
        value=config,
        raw_document=config.model_dump(mode="json", exclude_none=False),
    )


def _unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_listener(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"HTTP fixture exited before listening: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("HTTP fixture did not listen within ten seconds")


def _structured(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("structuredContent")
    assert isinstance(value, dict)
    return value


def test_real_sdk_v2_stdio_transport_runs_the_complete_oracle_lifecycle() -> None:
    server = Path(__file__).with_name("mcp_fixture_server.py").resolve()

    async def run() -> None:
        async with open_mcp_connection(
            _config(server),
            environ={},
            executable_policies=_policies(server),
        ) as client:
            assert client.protocol_version == "2026-07-28"
            assert client.server_identity.as_dict() == {
                "name": "bfcl-fixture-oracle",
                "version": "1.0.0",
            }
            first = await client.list_tools()
            second = await client.list_tools(first.next_cursor)
            assert [tool.name for tool in first.tools] == [
                "bfcl.describe",
                "bfcl.reset",
                "bfcl.state",
                "bfcl.end",
            ]
            assert [tool.name for tool in second.tools] == [
                "inventory.lookup",
                "inventory.reserve",
            ]

            reset = _structured(
                await client.call_tool(
                    "bfcl.reset",
                    {
                        "fixtures": {"inventory": {"gpu": {"stock": 3}}},
                        "context": {"seed": 7},
                    },
                )
            )
            episode_id = reset["episode_id"]
            pending = _structured(
                await client.call_tool(
                    "inventory.reserve",
                    {
                        "episode_id": episode_id,
                        "item_id": "gpu",
                        "quantity": 2,
                        "confirmed": False,
                    },
                )
            )
            state = _structured(
                await client.call_tool("bfcl.state", {"episode_id": episode_id})
            )
            assert pending == {"status": "pending_confirmation", "remaining": 3}
            assert state["inventory"]["gpu"]["stock"] == 3
            assert _structured(
                await client.call_tool("bfcl.end", {"episode_id": episode_id})
            ) == {"closed": True}

    asyncio.run(run())


def test_real_sdk_v2_streamable_http_transport_enforces_auth_and_identity() -> None:
    server = Path(__file__).with_name("mcp_fixture_server.py").resolve()
    port = _unused_loopback_port()
    token = "fixture-token-long-enough"
    process = subprocess.Popen(
        [
            sys.executable,
            str(server),
            "--http",
            "--port",
            str(port),
            "--bearer-token",
            token,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_listener(process, port)

        async def run() -> None:
            async with open_mcp_connection(
                _http_config(server, f"http://127.0.0.1:{port}/mcp"),
                environ={"MCP_FIXTURE_TOKEN": token},
            ) as client:
                assert client.server_identity.as_dict() == {
                    "name": "bfcl-fixture-oracle",
                    "version": "1.0.0",
                }
                first = await client.list_tools()
                assert first.next_cursor == "business-tools"
                identity = _structured(
                    await client.call_tool("bfcl.describe", {})
                )
                assert identity["content_digest"] == _CONTENT_DIGEST

        asyncio.run(run())
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_gateway_lifecycle_and_isolation_run_through_real_http_transport() -> None:
    server = Path(__file__).with_name("mcp_fixture_server.py").resolve()
    port = _unused_loopback_port()
    token = "fixture-token-long-enough"
    process = subprocess.Popen(
        [
            sys.executable,
            str(server),
            "--http",
            "--port",
            str(port),
            "--bearer-token",
            token,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_listener(process, port)

        async def run() -> None:
            service = GatewayService(
                _loaded_http_config(
                    server,
                    f"http://127.0.0.1:{port}/mcp",
                ),
                artifacts=GatewayArtifacts(
                    gateway_artifact_digest="sha256:" + "b" * 64,
                ),
                environ={"MCP_FIXTURE_TOKEN": token},
            )
            await service.start()
            try:
                first, second = await asyncio.gather(
                    service.create_session(
                        context={
                            "task_id": "first",
                            "seed": 7,
                            "clock": "2026-08-27T09:00:00+07:00",
                            "timeout_s": 10,
                        },
                        fixtures={"inventory": {"gpu": {"stock": 3}}},
                    ),
                    service.create_session(
                        context={
                            "task_id": "second",
                            "seed": 7,
                            "clock": "2026-08-27T09:00:00+07:00",
                            "timeout_s": 10,
                        },
                        fixtures={"inventory": {"gpu": {"stock": 9}}},
                    ),
                )
                first_result, second_result = await asyncio.gather(
                    service.call_tool(
                        first["session_id"],
                        name="inventory_lookup",
                        arguments={"item_id": "gpu"},
                        turn_index=0,
                    ),
                    service.call_tool(
                        second["session_id"],
                        name="inventory_lookup",
                        arguments={"item_id": "gpu"},
                        turn_index=0,
                    ),
                )
                assert first_result["item"]["stock"] == 3
                assert second_result["item"]["stock"] == 9
                assert (
                    await service.get_state(first["session_id"])
                )["inventory"]["gpu"]["stock"] == 3
                assert (
                    await service.get_state(second["session_id"])
                )["inventory"]["gpu"]["stock"] == 9
                await asyncio.gather(
                    service.delete_session(first["session_id"]),
                    service.delete_session(second["session_id"]),
                )
            finally:
                await service.shutdown()

        asyncio.run(run())
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_gateway_lifecycle_runs_through_real_stdio_transport() -> None:
    server = Path(__file__).with_name("mcp_fixture_server.py").resolve()

    async def run() -> None:
        service = GatewayService(
            _loaded_config(_config(server), server),
            artifacts=GatewayArtifacts(
                gateway_artifact_digest="sha256:" + "b" * 64,
            ),
            executable_policies=_policies(server),
            environ={},
        )
        await service.start()
        try:
            session = await service.create_session(
                context={
                    "task_id": "stdio",
                    "seed": 7,
                    "clock": "2026-08-27T09:00:00+07:00",
                    "timeout_s": 10,
                },
                fixtures={"inventory": {"gpu": {"stock": 4}}},
            )
            result = await service.call_tool(
                session["session_id"],
                name="inventory_lookup",
                arguments={"item_id": "gpu"},
                turn_index=0,
            )
            assert result["item"]["stock"] == 4
            assert (
                await service.get_state(session["session_id"])
            )["inventory"]["gpu"]["stock"] == 4
            await service.delete_session(session["session_id"])
        finally:
            await service.shutdown()

    asyncio.run(run())
