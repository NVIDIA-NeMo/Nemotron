from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from tests.steps.byob.mcp_fixture_server import (
    PAGE_TWO_CURSOR,
    FixtureOracle,
    create_streamable_http_app,
)


def _request(
    oracle: FixtureOracle,
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = oracle.handle(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            **({"params": params} if params is not None else {}),
        }
    )
    assert response is not None
    return response


def _call(
    oracle: FixtureOracle,
    request_id: int,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    response = _request(
        oracle,
        request_id,
        "tools/call",
        {"name": name, "arguments": arguments},
    )
    return response["result"]


def test_fixture_catalog_is_paginated_and_separates_control_tools() -> None:
    oracle = FixtureOracle()

    first = _request(oracle, 1, "tools/list", {})
    second = _request(oracle, 2, "tools/list", {"cursor": PAGE_TWO_CURSOR})

    assert first["result"]["nextCursor"] == PAGE_TWO_CURSOR
    assert [tool["name"] for tool in first["result"]["tools"]] == [
        "bfcl.describe",
        "bfcl.reset",
        "bfcl.state",
        "bfcl.end",
    ]
    assert [tool["name"] for tool in second["result"]["tools"]] == [
        "inventory.lookup",
        "inventory.reserve",
    ]


def test_tiny_library_profile_is_deterministic_and_confirmation_safe() -> None:
    oracle = FixtureOracle(domain="tiny_library")
    catalog = _request(
        oracle,
        1,
        "tools/list",
        {"cursor": PAGE_TWO_CURSOR},
    )
    assert [tool["name"] for tool in catalog["result"]["tools"]] == [
        "library.get_book_status",
        "library.checkout_book",
    ]
    reset = _call(
        oracle,
        2,
        "bfcl.reset",
        {
            "fixtures": {
                "books": [
                    {
                        "book_id": "BK-100",
                        "title": "Algorithms",
                        "status": "available",
                        "copies": 1,
                    }
                ],
                "patrons": [{"patron_id": "P-1", "name": "Ada"}],
                "loans": [],
            },
            "context": {"clock": "2026-03-02T09:00:00+07:00"},
        },
    )
    episode_id = reset["structuredContent"]["episode_id"]
    before = _call(
        oracle,
        3,
        "bfcl.state",
        {"episode_id": episode_id},
    )["structuredContent"]
    pending = _call(
        oracle,
        4,
        "library.checkout_book",
        {
            "episode_id": episode_id,
            "book_id": "BK-100",
            "patron_id": "P-1",
            "confirm": False,
        },
    )
    assert pending["structuredContent"]["status"] == "awaiting_confirmation"
    assert (
        _call(
            oracle,
            5,
            "bfcl.state",
            {"episode_id": episode_id},
        )["structuredContent"]
        == before
    )
    committed = _call(
        oracle,
        6,
        "library.checkout_book",
        {
            "episode_id": episode_id,
            "book_id": "BK-100",
            "patron_id": "P-1",
            "confirm": True,
        },
    )
    assert committed["structuredContent"]["loan"]["loan_id"] == "LN-000001"
    assert committed["structuredContent"]["loan"]["created_at"] == (
        "2026-03-02T09:00:00+07:00"
    )


def test_fixture_reset_is_deterministic_but_episode_state_is_isolated() -> None:
    oracle = FixtureOracle()
    fixtures = {"inventory": {"gpu": {"stock": 3}}}
    first = _call(oracle, 1, "bfcl.reset", {"fixtures": fixtures, "context": {"seed": 7}})
    second = _call(oracle, 2, "bfcl.reset", {"fixtures": fixtures, "context": {"seed": 7}})
    first_id = first["structuredContent"]["episode_id"]
    second_id = second["structuredContent"]["episode_id"]

    reserved = _call(
        oracle,
        3,
        "inventory.reserve",
        {
            "episode_id": first_id,
            "item_id": "gpu",
            "quantity": 2,
            "confirmed": True,
        },
    )
    first_state = _call(oracle, 4, "bfcl.state", {"episode_id": first_id})
    second_state = _call(oracle, 5, "bfcl.state", {"episode_id": second_id})

    assert (first_id, second_id) == ("episode-0001", "episode-0002")
    assert reserved["structuredContent"]["remaining"] == 1
    assert first_state["structuredContent"]["inventory"]["gpu"]["stock"] == 1
    assert second_state["structuredContent"]["inventory"]["gpu"]["stock"] == 3


def test_unconfirmed_mutation_is_pending_and_leaves_state_unchanged() -> None:
    oracle = FixtureOracle()
    reset = _call(
        oracle,
        1,
        "bfcl.reset",
        {
            "fixtures": {"inventory": {"gpu": {"stock": 3}}},
            "context": {},
        },
    )
    episode_id = reset["structuredContent"]["episode_id"]

    pending = _call(
        oracle,
        2,
        "inventory.reserve",
        {
            "episode_id": episode_id,
            "item_id": "gpu",
            "quantity": 2,
            "confirmed": False,
        },
    )
    state = _call(oracle, 3, "bfcl.state", {"episode_id": episode_id})

    assert pending["structuredContent"] == {
        "status": "pending_confirmation",
        "remaining": 3,
    }
    assert state["structuredContent"]["inventory"]["gpu"]["stock"] == 3
    assert state["structuredContent"]["reservations"] == []


def test_fixture_errors_are_stable_structured_tool_results() -> None:
    oracle = FixtureOracle()

    missing = _call(
        oracle,
        1,
        "inventory.lookup",
        {"episode_id": "missing", "item_id": "gpu"},
    )

    assert missing["isError"] is True
    assert missing["structuredContent"]["error"] == {
        "code": "episode_not_found",
        "message": "episode does not exist",
    }


def test_fixture_runs_as_a_newline_delimited_stdio_server() -> None:
    server = Path(__file__).with_name("mcp_fixture_server.py")
    process = subprocess.Popen(
        [sys.executable, str(server)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        requests = (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        for request in requests:
            process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        initialized = json.loads(process.stdout.readline())
        catalog = json.loads(process.stdout.readline())
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert initialized["result"]["protocolVersion"] == "2026-07-28"
    assert initialized["result"]["serverInfo"]["name"] == "bfcl-fixture-oracle"
    assert catalog["result"]["nextCursor"] == PAGE_TWO_CURSOR


def test_streamable_http_fixture_enforces_auth_and_protocol_headers() -> None:
    with TestClient(create_streamable_http_app(bearer_token="fixture-secret")) as client:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
        assert client.post("/mcp", json=request).status_code == 401
        response = client.post(
            "/mcp",
            json=request,
            headers={"Authorization": "Bearer fixture-secret"},
        )

    assert response.status_code == 200
    assert response.headers["mcp-protocol-version"] == "2026-07-28"
    assert response.headers["mcp-session-id"] == "fixture-session"
    assert response.json()["result"]["serverInfo"]["version"] == "1.0.0"


def test_streamable_http_fixture_exposes_identity_drift_and_request_limits() -> None:
    drift_digest = "sha256:" + "d" * 64
    with TestClient(
        create_streamable_http_app(
            server_version="2.0.0-drifted",
            content_digest=drift_digest,
            max_request_bytes=128,
        )
    ) as client:
        initialize = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        too_large = client.post(
            "/mcp",
            content=b"{" + b" " * 256 + b"}",
            headers={"content-type": "application/json"},
        )
        describe = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "bfcl.describe", "arguments": {}},
            },
        )

    assert initialize.json()["result"]["serverInfo"]["version"] == "2.0.0-drifted"
    assert too_large.status_code == 413
    assert describe.json()["result"]["structuredContent"]["content_digest"] == drift_digest


def test_stdio_fixture_exposes_bounded_crash_and_hang_failure_modes() -> None:
    server = Path(__file__).with_name("mcp_fixture_server.py")
    crash = subprocess.run(
        [sys.executable, str(server)],
        input=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "bfcl.describe",
                    "arguments": {"_fixture_failure": "crash"},
                },
            }
        )
        + "\n",
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert crash.returncode == 70
    assert "requested fixture crash" in crash.stderr

    hang = subprocess.Popen(
        [sys.executable, str(server)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "bfcl.describe",
                    "arguments": {"_fixture_failure": "hang"},
                },
            }
        )
        + "\n"
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            hang.communicate(request, timeout=0.1)
    finally:
        hang.kill()
        hang.communicate(timeout=5)
