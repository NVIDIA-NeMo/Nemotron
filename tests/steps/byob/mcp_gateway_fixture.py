"""A real BFCL gateway over TLS, backed by an MCP oracle written only for these tests.

Certifying MCP Mode A above A0 means calling it, so the probes have to reach a gateway
that is actually listening rather than an application object a test client drives in
memory. This module runs the production gateway under uvicorn on a loopback port and puts
an independent oracle behind it: the tools, the episode lifecycle, and the business rules
here are written from the contract rather than imported from it, which is what lets a test
distinguish a regression from a shared mistake.

One MCP connection is one episode. The gateway opens a connection per session, so a fresh
connection means a fresh world built from the fixtures that session was opened with, which
is the isolation the ladder asks about rather than one this fixture assumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.client import McpServerIdentity, McpToolPage
from nemotron.steps.byob.runtime.mcp.config import (
    McpOracleConfig,
    load_mcp_oracle_config,
)
from nemotron.steps.byob.runtime.mcp.discovery import catalog_identity_document
from nemotron.steps.byob.runtime.mcp.gateway import (
    GatewayArtifacts,
    GatewayService,
    create_gateway_app,
)
from nemotron.steps.byob.runtime.mcp.normalization import normalize_catalog
from tests.steps.byob.http_oracle_fixture_server import write_localhost_certificate

PROTOCOL_VERSION = "2026-07-28"
SERVER_NAME = "bfcl-library-oracle"
SERVER_VERSION = "1.0.0"
SERVER_CONTENT_DIGEST = "sha256:" + "a" * 64
# One book that lives in cold storage: reading it is legitimate but slow, which is how the
# timeout probe gets a call that misses its deadline without inventing a broken tool.
COLD_STORAGE_BOOK = "BK-COLD"

CONTROL_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "bfcl.describe",
        "description": "Return immutable oracle identity.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "bfcl.reset",
        "description": "Create one isolated episode from supplied fixtures.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fixtures": {"type": ["object", "null"]},
                "context": {"type": "object"},
            },
            "required": ["context"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bfcl.state",
        "description": "Return the complete state of one episode.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bfcl.end",
        "description": "Delete one episode.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
            "additionalProperties": False,
        },
    },
)

BUSINESS_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "library.get_book_status",
        "description": "Return circulation status for one book.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "book_id": {"type": "string"},
            },
            "required": ["episode_id", "book_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "library.checkout_book",
        "description": "Check out an available book after explicit confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "book_id": {"type": "string"},
                "patron_id": {"type": "string"},
                "confirm": {"type": "boolean"},
            },
            "required": ["episode_id", "book_id", "patron_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False},
    },
)

ALL_TOOLS: tuple[dict[str, Any], ...] = CONTROL_TOOLS + BUSINESS_TOOLS


@dataclass
class LibraryEpisode:
    books: dict[str, dict[str, Any]]
    patrons: set[str]
    loans: list[dict[str, Any]]
    clock: str | None

    def snapshot(self) -> dict[str, Any]:
        # Deliberately excludes the seed and task id the gateway passes at session open:
        # echoing those back would make two identical resets look different and turn a
        # determinism finding into an artefact of the fixture.
        return {
            "books": {
                book_id: dict(book) for book_id, book in sorted(self.books.items())
            },
            "patrons": sorted(self.patrons),
            "loans": [dict(loan) for loan in self.loans],
            "clock": self.clock,
        }


class LibraryOracleClient:
    """One MCP connection: control tools, two business tools, and its own episode."""

    sdk_version = "2.1.0-test"
    protocol_version = PROTOCOL_VERSION
    server_identity = McpServerIdentity(SERVER_NAME, SERVER_VERSION)
    capabilities = {"tools": {"listChanged": False}}

    def __init__(self, *, index: int, slow_call_s: float) -> None:
        self.index = index
        self.slow_call_s = slow_call_s
        self.episode_id: str | None = None
        self.episode: LibraryEpisode | None = None
        self.business_calls = 0

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        if cursor is None:
            return McpToolPage(tuple(CONTROL_TOOLS), "business")
        if cursor == "business":
            return McpToolPage(tuple(BUSINESS_TOOLS), None)
        raise AssertionError(f"unknown catalog cursor {cursor!r}")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name == "bfcl.describe":
            return _ok(
                {
                    "oracle_id": SERVER_NAME,
                    "oracle_version": SERVER_VERSION,
                    "content_digest": SERVER_CONTENT_DIGEST,
                }
            )
        if name == "bfcl.reset":
            return _ok({"episode_id": self._reset(arguments)})
        episode = self._require_episode(arguments)
        if episode is None:
            return _error("episode_not_found", "episode does not exist")
        if name == "bfcl.state":
            return _ok(episode.snapshot())
        if name == "bfcl.end":
            self.episode = None
            self.episode_id = None
            return _ok({"closed": True})
        self.business_calls += 1
        if name == "library.get_book_status":
            return await self._get_book_status(episode, arguments)
        if name == "library.checkout_book":
            return self._checkout_book(episode, arguments)
        return _error("tool_not_found", f"unknown tool {name}")

    def _reset(self, arguments: dict[str, Any]) -> str:
        fixtures = arguments.get("fixtures") or {}
        context = arguments.get("context") or {}
        self.episode_id = f"episode-{self.index:04d}"
        self.episode = LibraryEpisode(
            books={
                str(entry["book_id"]): {
                    "status": str(entry["status"]),
                    "copies": int(entry["copies"]),
                }
                for entry in fixtures.get("books", [])
            },
            patrons={str(entry["patron_id"]) for entry in fixtures.get("patrons", [])},
            loans=[],
            clock=context.get("clock"),
        )
        return self.episode_id

    def _require_episode(self, arguments: dict[str, Any]) -> LibraryEpisode | None:
        if (
            self.episode is None
            or arguments.get("episode_id") != self.episode_id
        ):
            return None
        return self.episode

    async def _get_book_status(
        self,
        episode: LibraryEpisode,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        book_id = str(arguments.get("book_id", ""))
        if book_id == COLD_STORAGE_BOOK:
            await asyncio.sleep(self.slow_call_s)
        book = episode.books.get(book_id)
        if book is None:
            return _error("not_found", "book does not exist")
        return _ok({"book_id": book_id, "status": book["status"]})

    def _checkout_book(
        self,
        episode: LibraryEpisode,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        book_id = str(arguments.get("book_id", ""))
        patron_id = str(arguments.get("patron_id", ""))
        book = episode.books.get(book_id)
        if book is None or patron_id not in episode.patrons:
            return _error("not_found", "book or patron does not exist")
        if arguments.get("confirm") is not True:
            return _ok({"status": "awaiting_confirmation", "book_id": book_id})
        if book["status"] != "available" or book["copies"] <= 0:
            return _error("unavailable", "book is unavailable")
        book["copies"] -= 1
        if book["copies"] == 0:
            book["status"] = "on_loan"
        loan = {
            "loan_id": f"LN-{len(episode.loans) + 1:06d}",
            "book_id": book_id,
            "patron_id": patron_id,
            "created_at": episode.clock,
        }
        episode.loans.append(loan)
        return _ok({"status": "checked_out", "loan": dict(loan)})


def _ok(value: dict[str, Any]) -> dict[str, Any]:
    return {"isError": False, "structuredContent": value}


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "isError": True,
        "structuredContent": {"error": {"code": code, "message": message}},
    }


@dataclass
class LibraryConnectionFactory:
    """Hand every gateway session its own connection, and therefore its own world."""

    slow_call_s: float = 5.0
    clients: list[LibraryOracleClient] = field(default_factory=list)

    def __call__(self, config: McpOracleConfig) -> Any:
        return self._open(config)

    @asynccontextmanager
    async def _open(self, config: McpOracleConfig) -> Any:
        client = LibraryOracleClient(
            index=len(self.clients) + 1,
            slow_call_s=self.slow_call_s,
        )
        self.clients.append(client)
        try:
            yield client
        finally:
            client.episode = None


def raw_oracle_config(*, tool_timeout_s: float = 2.0) -> dict[str, Any]:
    """The reviewed Mode A profile these tests probe, before its catalog is pinned."""
    return {
        "profile_version": "bfcl-mcp-oracle-v1",
        "mode": "A",
        "mcp_protocol_versions": [PROTOCOL_VERSION],
        "transport": {
            "kind": "streamable_http",
            "url": "https://library-mcp.example.test/mcp",
        },
        "expected": {
            "server_name": SERVER_NAME,
            "server_version": SERVER_VERSION,
            "tool_catalog_digest": "sha256:" + "0" * 64,
            "oracle_id": SERVER_NAME,
            "oracle_version": SERVER_VERSION,
            "server_content_digest": SERVER_CONTENT_DIGEST,
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
            "include": ["library.get_book_status", "library.checkout_book"],
            "aliases": {
                "library.get_book_status": "get_book_status",
                "library.checkout_book": "checkout_book",
            },
            "mutates": ["checkout_book"],
            "requires_confirmation": ["checkout_book"],
            "trust_annotations": False,
        },
        "results": {
            "error_path": "error",
            "status_field": "status",
            "pending_status": "awaiting_confirmation",
            "confirmation_parameter": "confirm",
        },
        "isolation": "namespace_per_episode",
        "limits": {
            "connect_timeout_s": 2,
            "handshake_timeout_s": 2,
            "tool_timeout_s": tool_timeout_s,
            "reset_timeout_s": 2,
            "episode_timeout_s": 20,
            "max_response_bytes": 100_000,
            "max_tools": 16,
            "max_catalog_pages": 4,
            "max_concurrent_episodes": 4,
            "session_idle_ttl_s": 30,
        },
    }


def pin_catalog_digest(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill in the tool_catalog_digest the reviewed profile has to pin for discovery."""
    config = McpOracleConfig.model_validate(raw)
    catalog = normalize_catalog(list(ALL_TOOLS), config)
    document = catalog_identity_document(
        config,
        negotiated_mcp_version=PROTOCOL_VERSION,
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        catalog=catalog,
    )
    digest = hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
    pinned = {**raw, "expected": {**raw["expected"], "tool_catalog_digest": f"sha256:{digest}"}}
    return pinned


@dataclass(frozen=True)
class RunningGateway:
    base_url: str
    certificate_path: Path
    factory: LibraryConnectionFactory


@contextlib.contextmanager
def serve_mcp_gateway(
    oracle_config_path: Path,
    *,
    gateway_artifact_digest: str,
    root: Path,
    certificate_root: Path | None = None,
    slow_call_s: float = 5.0,
    host: str = "127.0.0.1",
) -> Iterator[RunningGateway]:
    """Run the production gateway over TLS for the duration of the block."""
    certificate_path, key_path = write_localhost_certificate(
        root,
        certificate_root=certificate_root,
        host=host,
    )
    factory = LibraryConnectionFactory(slow_call_s=slow_call_s)
    service = GatewayService(
        load_mcp_oracle_config(oracle_config_path),
        artifacts=GatewayArtifacts(gateway_artifact_digest),
        connection_factory=factory,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_gateway_app(service),
            host=host,
            port=0,
            ssl_certfile=str(certificate_path),
            ssl_keyfile=str(key_path),
            log_level="warning",
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30.0
        while not server.started:
            if time.monotonic() > deadline:
                raise AssertionError("the gateway did not start within thirty seconds")
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield RunningGateway(
            base_url=f"https://{host}:{port}",
            certificate_path=certificate_path,
            factory=factory,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)
