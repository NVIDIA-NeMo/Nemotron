"""MCP Mode A on the same observation ladder as local Python and the HTTP endpoint.

Intake used to stop at A0 for MCP because it only ever discovered a catalog. These tests
run the reviewed probe plan against a gateway that is genuinely listening, over TLS, in
worker processes of their own, and then check the tier the certification report derives
from what was observed. A0, A1, and A2 are separated by which observation is missing, so
each refusal here names the evidence that was not produced rather than a policy switch.
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointIdentity,
)
from nemotron.steps.byob.runtime.mcp.authoring.intake import LoadedMcpIntake
from nemotron.steps.byob.runtime.mcp.authoring.runner import run_intake
from nemotron.steps.byob.runtime.mcp.gateway.identity import GatewayIdentity
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    CertificationAuthority,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)
from nemotron.steps.byob.runtime.source_adapters.intake import SourceIntakeError
from nemotron.steps.byob.runtime.source_adapters.probe_engine import AdapterProbePlan
from tests.steps.byob.mcp_gateway_fixture import (
    COLD_STORAGE_BOOK,
    RunningGateway,
    pin_catalog_digest,
    raw_oracle_config,
    serve_mcp_gateway,
)

GATEWAY_DIGEST = "sha256:" + "b" * 64
BASE_URL = "https://library-gateway.example.test"
CLOCK = "2026-03-02T09:00:00+00:00"
FIXTURES: dict[str, Any] = {
    "books": [
        {"book_id": "BK-1", "status": "available", "copies": 2},
        {"book_id": COLD_STORAGE_BOOK, "status": "available", "copies": 1},
    ],
    "patrons": [{"patron_id": "P-1"}],
}


def _probe_plan(**overrides: Any) -> AdapterProbePlan:
    document: dict[str, Any] = {
        "schema_version": "bfcl-adapter-probe-plan-v1",
        "clock": CLOCK,
        "seed": 7,
        "fixtures": copy.deepcopy(FIXTURES),
        "cases": [
            {
                "case_id": "checkout.success",
                "tool": "checkout_book",
                "arguments": {
                    "book_id": "BK-1",
                    "patron_id": "P-1",
                    "confirm": True,
                },
                "expectation": "success",
                "expected_state_change": True,
            },
            {
                "case_id": "status.cold",
                "tool": "get_book_status",
                "arguments": {"book_id": COLD_STORAGE_BOOK},
                "expectation": "timeout",
            },
            {
                "case_id": "status.missing",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-404"},
                "expectation": "structured_error",
                "expected_error_code": "not_found",
            },
            {
                "case_id": "status.success",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-1"},
                "expectation": "success",
                "expected_state_change": False,
            },
        ],
        "confirmation_parameter": "confirm",
        "status_field": "status",
        "pending_status": "awaiting_confirmation",
        "error_path": ["error", "code"],
    }
    document.update(overrides)
    return AdapterProbePlan.model_validate(document)


def _write_intake(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    oracle = pin_catalog_digest(raw_oracle_config())
    (root / "mcp_oracle.yaml").write_text(yaml.safe_dump(oracle), encoding="utf-8")
    path = root / "mcp_intake.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "intake_version": "bfcl-mcp-intake-v1",
                "kind": "mcp",
                "mcp_oracle_config": "mcp_oracle.yaml",
                "pack": {"pack_id": "library-mcp", "version": "1.0.0"},
                "gateway": {
                    "base_url": BASE_URL,
                    "gateway_artifact_digest": GATEWAY_DIGEST,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def gateway(tmp_path: Path) -> Any:
    intake_path = _write_intake(tmp_path / "intake")
    with serve_mcp_gateway(
        intake_path.parent / "mcp_oracle.yaml",
        gateway_artifact_digest=GATEWAY_DIGEST,
        root=tmp_path,
        slow_call_s=5.0,
    ) as running:
        yield intake_path, running


def _endpoint_factory(running: RunningGateway) -> Any:
    """Send both the attestation fetch and the probes to the gateway that is listening.

    A published pack has to name an oracle other hosts can reach, so the reviewed intake
    declares a real hostname. One factory serves both uses, which is what stops a test
    from attesting to one endpoint while probing another.
    """

    def factory(intake: LoadedMcpIntake, identity: GatewayIdentity) -> EndpointConfig:
        return EndpointConfig(
            path=Path("<mcp-probe-test>"),
            base_url=running.base_url,
            expected=EndpointIdentity.from_mapping(
                identity.as_dict(),
                source="gateway identity",
            ),
            ca_bundle_path=running.certificate_path,
        )

    return factory


def _connection_factory(running: RunningGateway) -> Any:
    """Let discovery see the same catalog the running gateway serves."""

    @asynccontextmanager
    async def factory(config: Any) -> Any:
        async with running.factory(config) as client:
            yield client

    return factory


def _run(
    intake_path: Path,
    running: RunningGateway,
    output: Path,
    *,
    probe_plan: AdapterProbePlan | None,
    required_tier: AdapterTier = AdapterTier.A0,
) -> Any:
    brief = output.parent / f"{output.name}-brief.txt"
    brief.write_text(
        "Evaluate deterministic library circulation over MCP.",
        encoding="utf-8",
    )
    return asyncio.run(
        run_intake(
            intake_path,
            output,
            connection_factory=_connection_factory(running),
            endpoint_config_factory=_endpoint_factory(running),
            domain_brief_path=brief,
            certification_authority=CertificationAuthority(
                key_id="test-root",
                private_key=Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
            ),
            held_out_decision=build_not_applicable_decision(
                "Synthetic MCP fixture has no held-out evaluation.",
                reviewed_by="mcp-probe-tests",
            ),
            probe_plan=probe_plan,
            required_tier=required_tier,
        )
    )


def test_mcp_intake_without_a_probe_plan_still_stops_at_a0(
    gateway: tuple[Path, RunningGateway],
    tmp_path: Path,
) -> None:
    intake_path, running = gateway

    result = _run(intake_path, running, tmp_path / "a0", probe_plan=None)

    assert result.source_evidence is not None
    assert result.source_evidence.certification.attained_tier == "A0"


def test_mcp_probes_reach_a2_over_live_gateway_sessions(
    gateway: tuple[Path, RunningGateway],
    tmp_path: Path,
) -> None:
    intake_path, running = gateway

    result = _run(
        intake_path,
        running,
        tmp_path / "a2",
        probe_plan=_probe_plan(),
        required_tier=AdapterTier.A2,
    )

    assert result.source_evidence is not None
    assert result.source_evidence.certification.attained_tier == "A2"
    # The probes have to have been the reason: a gateway that was never called cannot
    # have opened the sessions that answered reset, isolation, and confirmation.
    assert running.factory.clients
    assert any(client.business_calls for client in running.factory.clients)


def test_mcp_a2_needs_a_deadline_the_gateway_was_actually_held_to(
    gateway: tuple[Path, RunningGateway],
    tmp_path: Path,
) -> None:
    intake_path, running = gateway
    plan = _probe_plan()
    without_timeout = _probe_plan(
        cases=[
            case.model_dump(mode="json", exclude_none=True)
            for case in plan.cases
            if case.expectation != "timeout"
        ]
    )

    with pytest.raises(SourceIntakeError) as refusal:
        _run(
            intake_path,
            running,
            tmp_path / "no-deadline",
            probe_plan=without_timeout,
            required_tier=AdapterTier.A2,
        )

    assert refusal.value.code == "adapter_under_certified"
    assert not (tmp_path / "no-deadline").exists()

    attained = _run(
        intake_path,
        running,
        tmp_path / "a1",
        probe_plan=without_timeout,
        required_tier=AdapterTier.A1,
    )
    assert attained.source_evidence is not None
    assert attained.source_evidence.certification.attained_tier == "A1"


def test_mcp_probes_refuse_a_plan_that_leaves_a_published_tool_unobserved(
    gateway: tuple[Path, RunningGateway],
    tmp_path: Path,
) -> None:
    intake_path, running = gateway
    plan = _probe_plan()
    partial = _probe_plan(
        cases=[
            case.model_dump(mode="json", exclude_none=True)
            for case in plan.cases
            if case.tool != "get_book_status"
        ]
    )

    with pytest.raises(ValueError, match="result_shape_incomplete"):
        _run(
            intake_path,
            running,
            tmp_path / "partial",
            probe_plan=partial,
            required_tier=AdapterTier.A2,
        )
    assert not (tmp_path / "partial").exists()


def test_mcp_probes_refuse_a_plan_that_names_no_session_fixtures(
    gateway: tuple[Path, RunningGateway],
    tmp_path: Path,
) -> None:
    intake_path, running = gateway

    with pytest.raises(ValueError, match="fixtures each session is opened with"):
        _run(
            intake_path,
            running,
            tmp_path / "no-fixtures",
            probe_plan=_probe_plan(fixtures=None),
            required_tier=AdapterTier.A2,
        )
    assert not (tmp_path / "no-fixtures").exists()


def test_mode_c_cannot_be_probed_because_it_cannot_be_reset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "intake"
    root.mkdir(parents=True)
    raw = raw_oracle_config()
    raw["mode"] = "C"
    raw["control"] = {
        "reset_strategy": "no_op_verified",
        "state_strategy": "read_only_projection",
        "describe_oracle": "bfcl.describe",
        "episode_binding": "argument",
        "episode_argument": "episode_id",
        "state_projection": [
            {"tool": "library.get_book_status", "arguments": {"book_id": "BK-1"}}
        ],
    }
    raw["fixtures"] = {
        "direction": "snapshot",
        "snapshot_calls": [
            {
                "tool": "library.get_book_status",
                "arguments": {"book_id": "BK-1"},
                "collection": "books",
            },
        ],
    }
    raw["tools"]["mutates"] = []
    raw["tools"]["requires_confirmation"] = []
    raw["isolation"] = "read_only"
    (root / "mcp_oracle.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    intake_path = root / "mcp_intake.yaml"
    intake_path.write_text(
        yaml.safe_dump(
            {
                "intake_version": "bfcl-mcp-intake-v1",
                "kind": "mcp",
                "mcp_oracle_config": "mcp_oracle.yaml",
                "pack": {"pack_id": "library-mcp", "version": "1.0.0"},
                "gateway": {
                    "base_url": BASE_URL,
                    "gateway_artifact_digest": GATEWAY_DIGEST,
                    "snapshot_digest": "sha256:" + "c" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    brief = tmp_path / "brief.txt"
    brief.write_text("Read-only circulation snapshot.", encoding="utf-8")

    with pytest.raises(SourceIntakeError) as refusal:
        asyncio.run(
            run_intake(
                intake_path,
                tmp_path / "mode-c",
                domain_brief_path=brief,
                certification_authority=CertificationAuthority(
                    key_id="test-root",
                    private_key=Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
                ),
                held_out_decision=build_not_applicable_decision(
                    "Synthetic MCP fixture has no held-out evaluation.",
                    reviewed_by="mcp-probe-tests",
                ),
                probe_plan=_probe_plan(),
            )
        )

    assert refusal.value.code == "adapter_not_supported"
