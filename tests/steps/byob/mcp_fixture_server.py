"""Deterministic stdio MCP oracle used by integration and failure-mode tests.

This fixture deliberately implements the wire contract without importing the MCP SDK. It can
therefore detect SDK/client framing regressions instead of sharing the same implementation as the
code under test. One process owns one isolated set of episodes.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from copy import deepcopy
from typing import Any

PROTOCOL_VERSION = "2026-07-28"
SERVER_CONTENT_DIGEST = "sha256:" + "a" * 64
PAGE_TWO_CURSOR = "business-tools"

CONTROL_TOOLS = (
    {
        "name": "bfcl.describe",
        "description": "Return immutable oracle identity.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
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

BUSINESS_TOOLS = (
    {
        "name": "inventory.lookup",
        "description": "Look up one item in the episode inventory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "item_id": {"type": "string"},
            },
            "required": ["episode_id", "item_id"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {"item": {"type": ["object", "null"]}},
            "required": ["item"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "inventory.reserve",
        "description": "Reserve units only after explicit confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "item_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
                "confirmed": {"type": "boolean"},
            },
            "required": ["episode_id", "item_id", "quantity", "confirmed"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending_confirmation", "reserved"]},
                "remaining": {"type": "integer"},
            },
            "required": ["status", "remaining"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False},
    },
)

TINY_LIBRARY_TOOLS = (
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
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["episode_id", "book_id", "patron_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False},
    },
)


class FixtureOracle:
    """Stateful JSON-RPC handler with deterministic episode identifiers."""

    def __init__(
        self,
        *,
        server_version: str = "1.0.0",
        content_digest: str = SERVER_CONTENT_DIGEST,
        domain: str = "inventory",
    ) -> None:
        if domain not in {"inventory", "tiny_library"}:
            raise ValueError(f"unsupported fixture domain {domain!r}")
        self._next_episode = 1
        self._episodes: dict[str, dict[str, Any]] = {}
        self._server_version = server_version
        self._content_digest = content_digest
        self._domain = domain
        self._business_tools = (
            BUSINESS_TOOLS if domain == "inventory" else TINY_LIBRARY_TOOLS
        )

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            result = dict(result)
            result.setdefault("ttlMs", 0)
            result.setdefault("cacheScope", "private")
            result.setdefault("resultType", "complete")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def _episode(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        episode_id = arguments.get("episode_id")
        if not isinstance(episode_id, str) or episode_id not in self._episodes:
            return None
        return episode_id, self._episodes[episode_id]

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one request; notifications intentionally produce no response."""
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            return None
        if method == "server/discover":
            return self._result(
                request_id,
                {
                    "_meta": {
                        "io.modelcontextprotocol/serverInfo": {
                            "name": "bfcl-fixture-oracle",
                            "version": self._server_version,
                        }
                    },
                    "supportedVersions": [PROTOCOL_VERSION],
                    "capabilities": {"tools": {"listChanged": False}},
                    "ttlMs": 0,
                    "cacheScope": "private",
                },
            )
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "bfcl-fixture-oracle",
                        "version": self._server_version,
                    },
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            cursor = (request.get("params") or {}).get("cursor")
            if cursor is None:
                return self._result(
                    request_id,
                    {"tools": list(CONTROL_TOOLS), "nextCursor": PAGE_TWO_CURSOR},
                )
            if cursor == PAGE_TWO_CURSOR:
                return self._result(request_id, {"tools": list(self._business_tools)})
            return self._error(request_id, -32602, "unknown tools/list cursor")
        if method != "tools/call":
            return self._error(request_id, -32601, "method not found")

        params = request.get("params")
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return self._error(request_id, -32602, "tool arguments must be an object")
        if arguments.get("_fixture_failure") == "hang":
            time.sleep(60)
        if arguments.get("_fixture_failure") == "crash":
            raise RuntimeError("requested fixture crash")

        if name == "bfcl.describe":
            return self._tool_result(
                request_id,
                {
                    "oracle_id": "bfcl-fixture-oracle",
                    "oracle_version": self._server_version,
                    "content_digest": self._content_digest,
                    "build_id": "fixture-build-1",
                },
            )
        if name == "bfcl.reset":
            episode_id = f"episode-{self._next_episode:04d}"
            self._next_episode += 1
            fixtures = deepcopy(arguments.get("fixtures") or {})
            if self._domain == "inventory":
                state = {
                    "inventory": deepcopy(fixtures.get("inventory") or {}),
                    "context": deepcopy(arguments.get("context") or {}),
                    "reservations": [],
                }
            else:
                state = {
                    "books": deepcopy(fixtures.get("books") or []),
                    "patrons": deepcopy(fixtures.get("patrons") or []),
                    "loans": deepcopy(fixtures.get("loans") or []),
                    "context": deepcopy(arguments.get("context") or {}),
                }
            self._episodes[episode_id] = state
            return self._tool_result(request_id, {"episode_id": episode_id})

        episode = self._episode(arguments)
        if episode is None:
            return self._tool_error(request_id, "episode_not_found", "episode does not exist")
        episode_id, state = episode
        if name == "bfcl.state":
            return self._tool_result(request_id, deepcopy(state))
        if name == "bfcl.end":
            del self._episodes[episode_id]
            return self._tool_result(request_id, {"closed": True})
        if name == "inventory.lookup":
            item = deepcopy(state["inventory"].get(str(arguments.get("item_id"))))
            return self._tool_result(request_id, {"item": item})
        if name == "inventory.reserve":
            item_id = str(arguments.get("item_id"))
            quantity = arguments.get("quantity")
            if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
                return self._tool_error(request_id, "invalid_quantity", "quantity must be positive")
            item = state["inventory"].get(item_id)
            if not isinstance(item, dict) or not isinstance(item.get("stock"), int):
                return self._tool_error(request_id, "item_not_found", "inventory item does not exist")
            if arguments.get("confirmed") is not True:
                return self._tool_result(
                    request_id,
                    {"status": "pending_confirmation", "remaining": item["stock"]},
                )
            if item["stock"] < quantity:
                return self._tool_error(request_id, "insufficient_stock", "not enough stock")
            item["stock"] -= quantity
            state["reservations"].append({"item_id": item_id, "quantity": quantity})
            return self._tool_result(
                request_id,
                {"status": "reserved", "remaining": item["stock"]},
            )
        if name == "library.get_book_status":
            book_id = arguments.get("book_id")
            if not isinstance(book_id, str):
                return self._tool_error(
                    request_id,
                    "invalid_argument",
                    "book_id must be a string",
                )
            book = next(
                (
                    item
                    for item in state["books"]
                    if item.get("book_id") == book_id
                ),
                None,
            )
            if book is None:
                return self._tool_error(
                    request_id,
                    "not_found",
                    "book does not exist",
                )
            return self._tool_result(request_id, deepcopy(book))
        if name == "library.checkout_book":
            return self._checkout_book(request_id, state, arguments)
        return self._tool_error(request_id, "tool_not_found", "unknown tool")

    def _checkout_book(
        self,
        request_id: Any,
        state: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        book_id = arguments.get("book_id")
        patron_id = arguments.get("patron_id")
        confirm = arguments.get("confirm", False)
        if (
            not isinstance(book_id, str)
            or not isinstance(patron_id, str)
            or not isinstance(confirm, bool)
        ):
            return self._tool_error(
                request_id,
                "invalid_argument",
                "book_id, patron_id, or confirm has an invalid type",
            )
        book = next(
            (item for item in state["books"] if item.get("book_id") == book_id),
            None,
        )
        if book is None:
            return self._tool_error(request_id, "not_found", "book does not exist")
        if not any(
            item.get("patron_id") == patron_id for item in state["patrons"]
        ):
            return self._tool_error(
                request_id,
                "not_found",
                "patron does not exist",
            )
        if confirm is not True:
            return self._tool_result(
                request_id,
                {"status": "awaiting_confirmation"},
            )
        copies = book.get("copies")
        if (
            not isinstance(copies, int)
            or isinstance(copies, bool)
            or copies <= 0
            or book.get("status") != "available"
        ):
            return self._tool_error(
                request_id,
                "unavailable",
                "book is unavailable",
            )
        book["copies"] = copies - 1
        if book["copies"] == 0:
            book["status"] = "on_loan"
        loan = {
            "loan_id": f"LN-{len(state['loans']) + 1:06d}",
            "book_id": book_id,
            "patron_id": patron_id,
            "status": "active",
            "created_at": state["context"].get("clock"),
        }
        state["loans"].append(loan)
        return self._tool_result(
            request_id,
            {"status": "checked_out", "loan": deepcopy(loan)},
        )

    def _tool_result(self, request_id: Any, value: dict[str, Any]) -> dict[str, Any]:
        return self._result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(value, sort_keys=True)}],
                "structuredContent": value,
                "isError": False,
            },
        )

    def _tool_error(self, request_id: Any, code: str, message: str) -> dict[str, Any]:
        value = {"error": {"code": code, "message": message}}
        result = self._tool_result(request_id, value)
        result["result"]["isError"] = True
        return result


def create_streamable_http_app(
    *,
    bearer_token: str | None = None,
    server_version: str = "1.0.0",
    content_digest: str = SERVER_CONTENT_DIGEST,
    max_request_bytes: int = 1024 * 1024,
    domain: str = "inventory",
) -> Any:
    """Create a bounded HTTP fixture with auth and identity-drift controls."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    oracle = FixtureOracle(
        server_version=server_version,
        content_digest=content_digest,
        domain=domain,
    )

    async def mcp(request: Request) -> Response:
        if bearer_token is not None:
            authorization = request.headers.get("authorization", "")
            supplied = (
                authorization.removeprefix("Bearer ")
                if authorization.startswith("Bearer ")
                else ""
            )
            if not secrets.compare_digest(supplied, bearer_token):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        if request.method == "DELETE":
            return Response(status_code=200)
        body = await request.body()
        if len(body) > max_request_bytes:
            return JSONResponse({"error": "request_too_large"}, status_code=413)
        try:
            document = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        if not isinstance(document, dict):
            return JSONResponse({"error": "request_must_be_object"}, status_code=400)
        response = oracle.handle(document)
        if response is None:
            return Response(status_code=202)
        return JSONResponse(
            response,
            headers={
                "MCP-Protocol-Version": PROTOCOL_VERSION,
                "MCP-Session-Id": "fixture-session",
            },
        )

    return Starlette(routes=[Route("/mcp", mcp, methods=["POST", "DELETE"])])


def _stdio_main(*, domain: str) -> None:
    oracle = FixtureOracle(domain=domain)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = oracle.handle(request)
        except Exception as exc:  # noqa: BLE001 - fixture crash must terminate the process
            print(f"fixture server failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(70) from exc
        if response is not None:
            print(json.dumps(response, separators=(",", ":"), sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--bearer-token")
    parser.add_argument("--server-version", default="1.0.0")
    parser.add_argument("--content-digest", default=SERVER_CONTENT_DIGEST)
    parser.add_argument(
        "--domain",
        choices=("inventory", "tiny_library"),
        default="inventory",
    )
    args = parser.parse_args()
    if not args.http:
        _stdio_main(domain=args.domain)
        return
    import uvicorn

    uvicorn.run(
        create_streamable_http_app(
            bearer_token=args.bearer_token,
            server_version=args.server_version,
            content_digest=args.content_digest,
            domain=args.domain,
        ),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
