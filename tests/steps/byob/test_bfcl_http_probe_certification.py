"""HTTP sources climb the same ladder as local ones, over a real connection.

Every observation here comes from a call that crossed a TLS socket to a server this test
does not share code with. That is the point: an endpoint certified at A2 is claiming that
calls, errors, resets, isolation, confirmations, and deadlines were all seen to behave, and
none of those claims survives being asserted against a stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    HTTP_PROFILE_VERSION,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    PROTOCOL_VERSION,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    CertificationAuthority,
    CertificationProbe,
    certification_input_digest,
    derive_attained_tier,
    http_package_reference_profile,
    project_probe_executions,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import PackIdentity
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)
from nemotron.steps.byob.runtime.source_adapters.http_package import (
    inspect_http_package,
)
from nemotron.steps.byob.runtime.source_adapters.http_package_probes import (
    run_http_package_probes,
)
from nemotron.steps.byob.runtime.source_adapters.intake import (
    CERTIFICATION_FILE_NAME,
    DOMAIN_BRIEF_REPORT_FILE_NAME,
    DOMAIN_BRIEF_SOURCE_FILE_NAME,
    HELD_OUT_REDACTION_FILE_NAME,
    OBSERVATIONS_FILE_NAME,
    SourceIntakeError,
    run_conventional_intake,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import (
    AdapterProbePlan,
    ProbeError,
)
from tests.steps.byob.http_oracle_fixture_server import (
    ORACLE_ID,
    ORACLE_VERSION,
    LibraryOracle,
    RunningOracle,
    serve_library_oracle,
)

CONTENT_DIGEST = "sha256:" + "a" * 64
GATEWAY_DIGEST = "sha256:" + "b" * 64
REPORT_DIGEST = "sha256:" + "c" * 64
FIXTURES = {"books": [{"book_id": "bk-1", "status": "available"}]}
PACK = PackIdentity(pack_id="library", version="1.0.0")
AUTHORITY = CertificationAuthority(
    key_id="http-probe-test-root",
    private_key=Ed25519PrivateKey.from_private_bytes(b"\x09" * 32),
)
DECISION = build_not_applicable_decision(
    "Synthetic endpoint fixture has no held-out evaluation.",
    reviewed_by="http-probe-tests",
)


def _brief(root: Path) -> Path:
    path = root / "domain-brief.txt"
    path.write_text(
        "Create a benchmark for deterministic library loans.",
        encoding="utf-8",
    )
    return path


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "x-mutates": True,
            "x-requires-confirmation": True,
            "function": {
                "name": "borrow_book",
                "description": "Borrow one book once the borrower confirms.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "book_id": {"type": "string"},
                        "confirm": {"type": "boolean"},
                    },
                    "required": ["book_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_book_status",
                "description": "Return the loan status of one book.",
                "parameters": {
                    "type": "object",
                    "properties": {"book_id": {"type": "string"}},
                    "required": ["book_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rebuild_catalog_index",
                "description": "Rebuild the search index, fully or incrementally.",
                "parameters": {
                    "type": "object",
                    "properties": {"full": {"type": "boolean"}},
                    "required": ["full"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _attestation(catalog_digest: str) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_KIND,
        "provider_kind": "http",
        "profile_version": HTTP_PROFILE_VERSION,
        "level": "L0",
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": "bfcl-http-verifier",
        "state_observability": "diagnostic",
        "read_only_boundary": None,
        "effective_content_digest": CONTENT_DIGEST,
        "gateway_artifact_digest": GATEWAY_DIGEST,
        "tool_catalog_digest": catalog_digest,
        "probe_report_digest": GATEWAY_DIGEST,
        "gateway_conformance_report_digest": REPORT_DIGEST,
        "shim_artifact_digest": None,
        "server_content_digest": CONTENT_DIGEST,
        "snapshot_digest": None,
        "checks": [
            {"id": "H1", "requirement": "required", "status": "pass", "reason": None}
        ],
    }


def _probe_plan(
    *,
    include_timeout: bool = True,
    include_error: bool = True,
    tools: tuple[str, ...] = (
        "borrow_book",
        "get_book_status",
        "rebuild_catalog_index",
    ),
) -> AdapterProbePlan:
    success = {
        "borrow_book": {
            "case_id": "borrow",
            "tool": "borrow_book",
            "arguments": {"book_id": "bk-1", "confirm": True},
            "expectation": "success",
            "expected_state_change": True,
        },
        "get_book_status": {
            "case_id": "status",
            "tool": "get_book_status",
            "arguments": {"book_id": "bk-1"},
            "expectation": "success",
            "expected_state_change": False,
        },
        "rebuild_catalog_index": {
            "case_id": "index",
            "tool": "rebuild_catalog_index",
            "arguments": {"full": False},
            "expectation": "success",
            "expected_state_change": False,
        },
    }
    cases: list[dict[str, Any]] = [success[name] for name in tools]
    if include_error:
        cases.append(
            {
                "case_id": "missing",
                "tool": "get_book_status",
                "arguments": {"book_id": "bk-404"},
                "expectation": "structured_error",
                "expected_error_code": "not_found",
            }
        )
    if include_timeout:
        cases.append(
            {
                "case_id": "stall",
                "tool": "rebuild_catalog_index",
                "arguments": {"full": True},
                "expectation": "timeout",
            }
        )
    return AdapterProbePlan.model_validate(
        {
            "schema_version": "bfcl-adapter-probe-plan-v1",
            "clock": "2026-03-02T09:00:00+07:00",
            "seed": 7,
            "fixtures": FIXTURES,
            "cases": sorted(cases, key=lambda case: case["case_id"]),
        }
    )


def _package(root: Path, running: RunningOracle) -> Path:
    package = root / "http_package"
    tools = _tools()
    (package / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (package / "endpoint_config.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol_version": PROTOCOL_VERSION,
                "base_url": running.base_url,
                "expected": {
                    "oracle_id": ORACLE_ID,
                    "oracle_version": ORACLE_VERSION,
                    "content_digest": CONTENT_DIGEST,
                },
                "attestation": {
                    "kind": ATTESTATION_KIND,
                    "expected_digest": attestation_digest(
                        running.oracle.attestation
                    ),
                },
                "tls": {"ca_bundle_path": str(running.certificate_path)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return package


@pytest.fixture
def oracle(tmp_path: Path) -> Any:
    catalog_digest = sha256_json(
        sorted(_tools(), key=lambda item: item["function"]["name"])
    )
    library = LibraryOracle(
        content_digest=CONTENT_DIGEST,
        attestation=_attestation(catalog_digest),
        slow_call_s=5.0,
    )
    # The reviewed package pins the bundle it trusts, so the certificate has to live
    # inside it; the private key stays outside, where no reviewer would sign it off.
    package = tmp_path / "http_package"
    package.mkdir()
    with serve_library_oracle(
        library,
        root=tmp_path,
        certificate_root=package,
    ) as running:
        yield running


def _tier(
    package: Path,
    root: Path,
    plan: AdapterProbePlan,
) -> tuple[AdapterTier, dict[CertificationProbe, Any]]:
    inspection = inspect_http_package(package, allowed_roots=(root,), environ={})
    run = run_http_package_probes(inspection, plan, environ={})
    profile = http_package_reference_profile()
    outcomes = project_probe_executions(
        profile,
        run.records,
        input_digest=certification_input_digest(
            run.descriptor,
            source_identity_digest=inspection.source_identity_digest,
            profile=profile,
            execution_inputs_digest=run.plan_digest,
        ),
    )
    return derive_attained_tier(profile, outcomes), {
        outcome.probe: outcome for outcome in outcomes
    }


def test_http_endpoint_probes_reach_a2_over_live_sessions(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)

    tier, outcomes = _tier(package, tmp_path, _probe_plan())

    assert tier is AdapterTier.A2
    assert outcomes[CertificationProbe.EXECUTABLE_OBSERVATION].status == "pass"
    assert outcomes[CertificationProbe.CONFIRMATION_SAFETY].status == "pass"
    timeout_evidence = outcomes[CertificationProbe.TIMEOUT_CLEANUP].evidence
    assert timeout_evidence["observation"]["evidence"]["timeout_observed"] is True
    assert timeout_evidence["observation"]["evidence"]["fresh_episode_succeeded"] is True
    # A deadline that leaves a session open on the endpoint is a leak, not a cleanup.
    assert oracle.oracle.sessions == {}


def test_http_a2_needs_a_deadline_the_endpoint_was_actually_held_to(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)

    tier, outcomes = _tier(package, tmp_path, _probe_plan(include_timeout=False))

    assert tier is AdapterTier.A1
    assert outcomes[CertificationProbe.TIMEOUT_CLEANUP].reason == "probe_missing"


def test_http_probes_refuse_a_plan_that_leaves_a_published_tool_unobserved(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)
    inspection = inspect_http_package(package, allowed_roots=(tmp_path,), environ={})
    plan = _probe_plan(tools=("borrow_book", "get_book_status"), include_timeout=False)

    with pytest.raises(ProbeError) as raised:
        run_http_package_probes(inspection, plan, environ={})

    assert raised.value.code == "result_shape_incomplete"
    assert "rebuild_catalog_index" in raised.value.detail


def test_http_probes_refuse_a_plan_that_names_no_session_fixtures(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)
    inspection = inspect_http_package(package, allowed_roots=(tmp_path,), environ={})
    plan = AdapterProbePlan.model_validate(
        {
            **_probe_plan().model_dump(mode="json"),
            "fixtures": None,
        }
    )

    with pytest.raises(ProbeError) as raised:
        run_http_package_probes(inspection, plan, environ={})

    assert raised.value.code == "probe_evidence_invalid"


def test_http_intake_publishes_an_a2_bundle_that_loads_at_a2(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)
    output = tmp_path / "out-a2"

    result = run_conventional_intake(
        {
            "declaration_version": "bfcl-source-declaration-v1",
            "http_package": {"path": package.name},
        },
        output,
        source_base_dir=tmp_path,
        allowed_roots=(tmp_path,),
        pack=PACK,
        domain_brief_path=_brief(tmp_path),
        certification_authority=AUTHORITY,
        held_out_decision=DECISION,
        required_tier=AdapterTier.A2,
        probe_plan=_probe_plan(),
        http_environ={},
    )

    assert result.finalized.certification.attained_tier is AdapterTier.A2
    assert result.finalized.evidence.schema_version == "bfcl-source-evidence-v2"
    assert not result.finalized.evidence.unresolved_gaps
    view = load_evidence_bundle(
        result.evidence_path,
        certification_report_path=result.output_root / CERTIFICATION_FILE_NAME,
        trusted_certification_keys={AUTHORITY.key_id: AUTHORITY.public_key},
        domain_brief_source_path=result.output_root / DOMAIN_BRIEF_SOURCE_FILE_NAME,
        domain_brief_report_path=result.output_root / DOMAIN_BRIEF_REPORT_FILE_NAME,
        held_out_redaction_report_path=(
            result.output_root / HELD_OUT_REDACTION_FILE_NAME
        ),
        source_observations_path=result.output_root / OBSERVATIONS_FILE_NAME,
        required_certification_tier=AdapterTier.A2,
    )
    assert view.certification_verified


def test_http_intake_refuses_a2_and_publishes_nothing_without_a_deadline_case(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)
    output = tmp_path / "under-certified"

    with pytest.raises(SourceIntakeError) as refused:
        run_conventional_intake(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                "http_package": {"path": package.name},
            },
            output,
            source_base_dir=tmp_path,
            allowed_roots=(tmp_path,),
            pack=PACK,
            domain_brief_path=_brief(tmp_path),
            certification_authority=AUTHORITY,
            held_out_decision=DECISION,
            required_tier=AdapterTier.A2,
            probe_plan=_probe_plan(include_timeout=False),
            http_environ={},
        )

    assert refused.value.code == "adapter_under_certified"
    assert not output.exists()


def test_http_probes_report_a_catalog_that_drifted_after_review(
    tmp_path: Path,
    oracle: RunningOracle,
) -> None:
    package = _package(tmp_path, oracle)
    inspection = inspect_http_package(package, allowed_roots=(tmp_path,), environ={})
    oracle.oracle.content_digest = "sha256:" + "d" * 64

    run = run_http_package_probes(inspection, _probe_plan(), environ={})
    by_probe = {record.observation.probe: record for record in run.records}

    assert by_probe[CertificationProbe.CATALOG_INTEGRITY].observation.status == "fail"
    assert by_probe[CertificationProbe.IDENTITY_INTEGRITY].observation.status == "fail"
    assert (
        by_probe[CertificationProbe.IDENTITY_INTEGRITY].observation.reason
        == "identity_drift"
    )
