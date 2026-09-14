from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    HTTP_PROFILE_VERSION,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    PROTOCOL_VERSION,
    EndpointConfig,
    EndpointIdentity,
    ExpectedAttestation,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
    endpoint_conformance,
)

CONTENT_DIGEST = "sha256:" + "1" * 64


def _config(document: dict[str, object]) -> EndpointConfig:
    return EndpointConfig(
        path=Path("/tmp/endpoint.yaml"),
        base_url="https://oracle.example",
        expected=EndpointIdentity(
            protocol_version=PROTOCOL_VERSION,
            oracle_id="core-http-oracle",
            oracle_version="1.0.0",
            content_digest=CONTENT_DIGEST,
        ),
        attestation=ExpectedAttestation(
            kind=ATTESTATION_KIND,
            expected_digest=attestation_digest(document),
        ),
    )


def test_endpoint_gate_forwards_operator_evidence_trust(monkeypatch) -> None:
    captured: dict[str, object] = {}
    public_key = Ed25519PrivateKey.generate().public_key()
    document: dict[str, object] = {"schema_version": ATTESTATION_KIND}

    def fake_verify_conformance(value, **kwargs):
        captured["document"] = value
        captured.update(kwargs)
        return SimpleNamespace(
            publishable=False,
            findings=(),
            caps=(),
            effective_level="L0",
            as_dict=lambda: {},
        )

    monkeypatch.setattr(
        endpoint_conformance,
        "verify_conformance",
        fake_verify_conformance,
    )

    endpoint_conformance.run_endpoint_conformance_check(
        _config(document),
        {"content_digest": CONTENT_DIGEST},
        fetch=lambda _config: document,
        trusted_evidence_keys={"operator-key": public_key},
        expected_evidence_issuer="operator-conformance",
    )

    assert captured["trusted_evidence_keys"] == {"operator-key": public_key}
    assert captured["expected_evidence_issuer"] == "operator-conformance"


def test_unsigned_http_evidence_fails_closed_through_endpoint_gate() -> None:
    checks = [
        {
            "id": f"H{index}",
            "requirement": "conditional" if index in {7, 8} else "required",
            "status": "pass",
            "reason": None,
        }
        for index in range(1, 12)
    ]
    probe_report = {"probes": checks}
    gateway_report = {
        "issuer": "operator-conformance",
        "gateway_artifact_digest": "sha256:" + "a" * 64,
        "effective_content_digest": CONTENT_DIGEST,
        "tool_catalog_digest": "sha256:" + "b" * 64,
        "suite": {
            "kind": "endpoint",
            "profile_version": "bfcl-http-conformance-v1",
            "p9": {
                "timeout_observed": True,
                "business_call_attempts": 1,
                "episode_poisoned": True,
                "transport_cleanup_completed": True,
                "unknown_commit_state_preserved": True,
            },
        },
    }
    document: dict[str, object] = {
        "schema_version": ATTESTATION_KIND,
        "provider_kind": "http",
        "profile_version": HTTP_PROFILE_VERSION,
        "level": "L2",
        "effective_content_digest": CONTENT_DIGEST,
        "gateway_artifact_digest": "sha256:" + "a" * 64,
        "shim_artifact_digest": None,
        "tool_catalog_digest": "sha256:" + "b" * 64,
        "server_content_digest": "sha256:" + "c" * 64,
        "snapshot_digest": None,
        "probe_report_digest": attestation_digest(probe_report),
        "gateway_conformance_report_digest": attestation_digest(gateway_report),
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": "operator-conformance",
        "state_observability": "complete",
        "read_only_boundary": None,
        "checks": checks,
    }

    entry = endpoint_conformance.run_endpoint_conformance_check(
        _config(document),
        {"content_digest": CONTENT_DIGEST},
        fetch=lambda _config: document,
        probe_report=probe_report,
        gateway_conformance_report=gateway_report,
        trusted_evidence_keys={
            "operator-key": Ed25519PrivateKey.generate().public_key(),
        },
        expected_evidence_issuer="operator-conformance",
    )

    assert entry is not None
    assert entry["status"] == "fail"
    assert {
        failure["reason"] for failure in entry["failures"]
    } >= {"gateway_evidence_signature_missing"}
