# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool and policy contracts: read-only data, the quality bar, one proposal, and report checks."""

import io

import pytest

import world
from investigator import contract, endpoint, policy, sandbox, tools

SCENARIO = "model-switch"
BEST = {"Account & password": "Vesk 4 Lite", "Billing": "Vesk 4 Lite",
        "Technical troubleshooting": "Vesk 4", "Integrations & API": "Vesk 4"}


def _conn():
    world.ensure(SCENARIO)
    return world.connect(SCENARIO)


def test_sql_is_read_only():
    conn = _conn()
    assert "rows" in tools.run_sql(conn, "SELECT COUNT(*) FROM usage_daily")
    for statement in ("DELETE FROM usage_daily", "INSERT INTO change_log VALUES ('x', 'y', 'z', 'w')", "DROP TABLE billing_daily"):
        assert "error" in tools.run_sql(conn, statement), statement


def test_quality_bar_rejects_a_route_below_the_floor():
    state = {"proposals": []}
    cheap = {**BEST, "Technical troubleshooting": "Dunlin-3-8B-Instruct"}
    result = tools.propose_change({"title": "t", "routes": cheap, "expected_monthly_savings_usd": 1, "rationale": "r", "why": "w"},
                                  state, enforce=True)
    assert "quality-bar" in result["error"] and state["proposals"] == []


def test_one_proposal_per_investigation():
    state = {"proposals": []}
    arguments = {"title": "t", "routes": BEST, "expected_monthly_savings_usd": 1, "rationale": "r", "why": "w"}
    assert tools.propose_change(arguments, state, enforce=True)["proposal_id"] == "PRP-0001"
    assert "only one proposal" in tools.propose_change(arguments, state, enforce=True)["error"]


def test_report_checks():
    assert contract.validate_report({})[0].startswith("missing required fields")
    report = {field: "x" for field in contract.report_schema()} | {"drivers": [{"metric": "m"}], "proposal_id": "none",
                                                                  "manager_summary": "Tokens rose."}
    assert contract.validate_report(report) == []
    problems = policy.check_report(report, {"proposals": []})
    assert any("terms a product manager" in p for p in problems)


def test_python_refuses_to_run_without_a_sandbox(monkeypatch):
    monkeypatch.setattr(sandbox, "available", lambda: None)
    assert "error" in sandbox.run_python(SCENARIO, "print(1)")


def test_python_sandbox_blocks_network_and_writes():
    if sandbox.available() is None:
        return
    world.ensure(SCENARIO)
    ok = sandbox.run_python(SCENARIO, "print(conn.execute('SELECT COUNT(*) FROM change_log').fetchone()[0])")
    assert ok["exit_code"] == 0 and ok["stdout"].strip().isdigit()
    net = sandbox.run_python(SCENARIO, "import socket; socket.create_connection(('example.com', 80), timeout=3)")
    assert net["exit_code"] != 0
    write = sandbox.run_python(SCENARIO, f"open({str(world.WORLD / 'escape.txt')!r}, 'w').write('x')")
    assert write["exit_code"] != 0 and not (world.WORLD / "escape.txt").exists()


def test_python_sandbox_cannot_read_the_project_files():
    from pathlib import Path

    backend, readme = sandbox.available(), Path(__file__).resolve().parents[1] / "README.md"
    if backend is None or (backend == "sandbox-exec" and Path.home() not in readme.parents):
        return  # sandbox-exec hides the home folder; bwrap mounts only the data and the interpreter
    world.ensure(SCENARIO)
    leak = sandbox.run_python(SCENARIO, f"print(open({str(readme)!r}).read()[:20])")
    assert leak["exit_code"] != 0 and not leak["stdout"].strip()


def test_endpoint_accepts_nvidia_or_generic_api_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    assert endpoint.settings()["api_key"] == "nvidia-key"
    monkeypatch.setenv("MODEL_API_KEY", "generic-key")
    assert endpoint.settings()["api_key"] == "generic-key"


def test_endpoint_missing_settings_points_to_env_template(monkeypatch):
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(RuntimeError, match=r"Copy \.env\.example to \.env and set NVIDIA_API_KEY"):
        endpoint.chat([])


@pytest.mark.parametrize(("status", "message"), [
    (401, "NVIDIA Build rejected the API key"),
    (403, "NVIDIA Build rejected the API key"),
    (404, "endpoint or model was not found"),
])
def test_endpoint_http_errors_are_actionable(monkeypatch, status, message):
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "nvidia/example")
    monkeypatch.setattr(
        endpoint.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            endpoint.urllib.error.HTTPError("https://example.invalid", status, "error", None, io.BytesIO(b"{}"))
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        endpoint.chat([])


def test_endpoint_connection_error_is_actionable(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "nvidia/example")
    monkeypatch.setattr(
        endpoint.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(endpoint.urllib.error.URLError("connection refused")),
    )
    with pytest.raises(RuntimeError, match="Could not reach MODEL_BASE_URL"):
        endpoint.chat([])
