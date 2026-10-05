from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import TypeAdapter

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    HTTP_PROFILE_VERSION,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    PROTOCOL_VERSION,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    AuthorizationError,
    ExposureSubject,
    authorize_model_exposure_by_human,
    build_exposure_subject,
    verify_exposure_authorization,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import (
    BundleError,
    load_evidence_bundle,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    PUBLISHED_CERTIFICATION_PROFILES,
    AdapterTier,
    CertificationAuthority,
    CertificationError,
    CertificationRefusalCode,
    certification_profile_for,
    load_certification_report,
    verify_certification_report,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import (
    load_domain_brief_redaction_report,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    PackIdentity,
    SourceEvidenceDocument,
    load_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
    load_held_out_redaction_report,
    verify_held_out_redaction_report,
)
from nemotron.steps.byob.runtime.source_adapters.intake import (
    CERTIFICATION_FILE_NAME,
    DOMAIN_BRIEF_REPORT_FILE_NAME,
    DOMAIN_BRIEF_SOURCE_FILE_NAME,
    EVIDENCE_FILE_NAME,
    EXPOSURE_SUBJECT_FILE_NAME,
    HELD_OUT_REDACTION_FILE_NAME,
    OBSERVATIONS_FILE_NAME,
    PROVENANCE_FILE_NAME,
    SourceIntakeError,
    run_conventional_intake,
)
from tests.steps.byob.test_bfcl_local_authoring_adapter import (
    _probe_plan,
    _runtime_package,
)
from tests.steps.byob.test_bfcl_mcp_authoring import _intake as _mcp_intake

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
AUTHORITY = CertificationAuthority(
    key_id="intake-test-root",
    private_key=Ed25519PrivateKey.from_private_bytes(b"\x08" * 32),
)
PACK = PackIdentity(pack_id="inventory", version="1.0.0")
DECISION = build_not_applicable_decision(
    "Synthetic intake fixture has no held-out evaluation.",
    reviewed_by="intake-tests",
)
EXPOSURE_SUBJECT_ADAPTER = TypeAdapter(ExposureSubject)


@dataclass(frozen=True)
class _ContractCase:
    adapter: str
    output_root: Path
    evidence_path: Path
    certification_path: Path
    domain_brief_source_path: Path
    domain_brief_report_path: Path
    held_out_redaction_path: Path
    exposure_subject_path: Path
    observations_path: Path
    authority: CertificationAuthority


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one public item.",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "string"}},
                    "required": ["item_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _brief(tmp_path: Path) -> Path:
    path = tmp_path / "domain-brief.txt"
    path.write_text(
        "Create a benchmark for deterministic inventory lookup.",
        encoding="utf-8",
    )
    return path


def _local_package(tmp_path: Path) -> Path:
    package = tmp_path / "local-source"
    package.mkdir()
    (package / "backend.py").write_text("def lookup(item_id):\n    return None\n")
    (package / "tools.json").write_text(json.dumps(_tools()), encoding="utf-8")
    (package / "dependency-lock.json").write_text(
        json.dumps(
            {
                "schema_version": "bfcl-python-dependency-lock-v1",
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )
    return package


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
        "effective_content_digest": DIGEST_A,
        "gateway_artifact_digest": DIGEST_B,
        "tool_catalog_digest": catalog_digest,
        "probe_report_digest": DIGEST_B,
        "gateway_conformance_report_digest": DIGEST_C,
        "shim_artifact_digest": None,
        "server_content_digest": DIGEST_A,
        "snapshot_digest": None,
        "checks": [
            {
                "id": "H1",
                "requirement": "required",
                "status": "pass",
                "reason": None,
            }
        ],
    }


class _HttpClient:
    def __init__(self, attestation: dict[str, Any]) -> None:
        self.attestation = attestation
        self.closed = False

    def metadata(self) -> dict[str, str]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "oracle_id": "reviewed-oracle",
            "oracle_version": "1.0.0",
            "content_digest": DIGEST_A,
        }

    def list_tools(self) -> list[str]:
        return ["lookup"]

    def conformance(self) -> dict[str, Any]:
        return self.attestation

    def close(self, *, suppress_errors: bool = False) -> None:
        self.closed = True


def _http_package(tmp_path: Path) -> tuple[Path, _HttpClient]:
    package = tmp_path / "http-source"
    package.mkdir()
    tools = _tools()
    (package / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    attestation = _attestation(sha256_json(tools))
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "base_url": "https://oracle.example",
        "expected": {
            "oracle_id": "reviewed-oracle",
            "oracle_version": "1.0.0",
            "content_digest": DIGEST_A,
        },
        "attestation": {
            "kind": ATTESTATION_KIND,
            "expected_digest": attestation_digest(attestation),
        },
        "max_request_bytes": 4096,
        "max_response_bytes": 4096,
    }
    (package / "endpoint_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    return package, _HttpClient(attestation)


def _contract_case(tmp_path: Path, adapter: str) -> _ContractCase:
    root = tmp_path / adapter
    root.mkdir()
    if adapter == "mcp_mode_a":
        result = _mcp_intake(root, output="output", v2=True)
        assert result.certification_path is not None
        assert result.domain_brief_source_path is not None
        assert result.domain_brief_report_path is not None
        assert result.held_out_redaction_path is not None
        assert result.exposure_subject_path is not None
        assert result.observations_path is not None
        return _ContractCase(
            adapter=adapter,
            output_root=result.output_root,
            evidence_path=result.output_root / EVIDENCE_FILE_NAME,
            certification_path=result.certification_path,
            domain_brief_source_path=result.domain_brief_source_path,
            domain_brief_report_path=result.domain_brief_report_path,
            held_out_redaction_path=result.held_out_redaction_path,
            exposure_subject_path=result.exposure_subject_path,
            observations_path=result.observations_path,
            authority=CertificationAuthority(
                key_id="test-root",
                private_key=Ed25519PrivateKey.from_private_bytes(b"\x03" * 32),
            ),
        )

    if adapter == "local_python":
        package = _local_package(root)
        factory = None
    elif adapter == "http_package":
        package, client = _http_package(root)

        def factory(_config: Any, _headers: Any, _timeout: float) -> _HttpClient:
            return client

    else:
        raise AssertionError(f"unknown contract adapter {adapter}")
    result = run_conventional_intake(
        {
            "declaration_version": "bfcl-source-declaration-v1",
            adapter: {"path": package.name},
        },
        root / "output",
        source_base_dir=root,
        allowed_roots=(root,),
        pack=PACK,
        domain_brief_path=_brief(root),
        certification_authority=AUTHORITY,
        held_out_decision=DECISION,
        http_client_factory=factory,
    )
    return _ContractCase(
        adapter=adapter,
        output_root=result.output_root,
        evidence_path=result.evidence_path,
        certification_path=result.output_root / CERTIFICATION_FILE_NAME,
        domain_brief_source_path=(
            result.output_root / DOMAIN_BRIEF_SOURCE_FILE_NAME
        ),
        domain_brief_report_path=(
            result.output_root / DOMAIN_BRIEF_REPORT_FILE_NAME
        ),
        held_out_redaction_path=(
            result.output_root / HELD_OUT_REDACTION_FILE_NAME
        ),
        exposure_subject_path=result.output_root / EXPOSURE_SUBJECT_FILE_NAME,
        observations_path=result.output_root / OBSERVATIONS_FILE_NAME,
        authority=AUTHORITY,
    )


@pytest.mark.parametrize("adapter", ["local_python", "http_package"])
def test_conventional_transports_publish_the_same_v2_envelope(
    tmp_path: Path,
    adapter: str,
) -> None:
    if adapter == "local_python":
        package = _local_package(tmp_path)
        client = None
    else:
        package, client = _http_package(tmp_path)
    declaration = {
        "declaration_version": "bfcl-source-declaration-v1",
        adapter: {"path": package.name},
    }
    factory = (
        None
        if client is None
        else lambda _config, _headers, _timeout: client
    )

    result = run_conventional_intake(
        declaration,
        tmp_path / f"out-{adapter}",
        source_base_dir=tmp_path,
        allowed_roots=(tmp_path,),
        pack=PACK,
        domain_brief_path=_brief(tmp_path),
        certification_authority=AUTHORITY,
        held_out_decision=DECISION,
        http_client_factory=factory,
    )

    assert result.finalized.certification.attained_tier is AdapterTier.A0
    assert result.finalized.evidence.schema_version == "bfcl-source-evidence-v2"
    assert result.finalized.evidence.source_adapter.kind == adapter
    assert result.finalized.exposure_subject.evidence_digest == (
        result.finalized.evidence.bundle_digest
    )
    assert {
        EVIDENCE_FILE_NAME,
        CERTIFICATION_FILE_NAME,
        EXPOSURE_SUBJECT_FILE_NAME,
        PROVENANCE_FILE_NAME,
    } <= {path.name for path in result.output_root.iterdir()}
    view = load_evidence_bundle(
        result.evidence_path,
        certification_report_path=result.output_root / CERTIFICATION_FILE_NAME,
        trusted_certification_keys={AUTHORITY.key_id: AUTHORITY.public_key},
        domain_brief_source_path=(
            result.output_root / DOMAIN_BRIEF_SOURCE_FILE_NAME
        ),
        domain_brief_report_path=(
            result.output_root / DOMAIN_BRIEF_REPORT_FILE_NAME
        ),
        held_out_redaction_report_path=(
            result.output_root / HELD_OUT_REDACTION_FILE_NAME
        ),
        source_observations_path=result.output_root / OBSERVATIONS_FILE_NAME,
    )
    assert view.is_v2
    assert view.certification_verified
    if client is not None:
        assert client.closed is True


def test_refusal_leaves_no_partial_intake_directory(tmp_path: Path) -> None:
    package = _local_package(tmp_path)
    output = tmp_path / "refused"
    required_decision = DECISION.model_copy(
        update={
            "status": "required",
            "policy_digest": DIGEST_A,
            "reviewed_reason": None,
        }
    )

    with pytest.raises(ValueError, match="held-out decision"):
        run_conventional_intake(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                "local_python": {"path": package.name},
            },
            output,
            source_base_dir=tmp_path,
            allowed_roots=(tmp_path,),
            pack=PACK,
            domain_brief_path=_brief(tmp_path),
            certification_authority=AUTHORITY,
            held_out_decision=required_decision,
        )

    assert not output.exists()


def test_existing_output_refuses_before_adapter_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("adapter execution must not start for an existing output")

    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.source_adapters.intake.inspect_local_python_package",
        forbidden,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        run_conventional_intake(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                "local_python": {"path": "never-inspected"},
            },
            output,
            source_base_dir=tmp_path,
            allowed_roots=(tmp_path,),
            pack=PACK,
            domain_brief_path=_brief(tmp_path),
            certification_authority=AUTHORITY,
            held_out_decision=DECISION,
        )


@pytest.mark.parametrize(
    "adapter",
    ["local_python", "http_package", "mcp_mode_a"],
)
def test_all_adapters_obey_the_same_v2_trust_contract(
    tmp_path: Path,
    adapter: str,
) -> None:
    case = _contract_case(tmp_path, adapter)
    evidence = load_source_evidence(case.evidence_path)
    report = load_certification_report(case.certification_path)
    profile = certification_profile_for(adapter)
    observations = json.loads(case.observations_path.read_text(encoding="utf-8"))

    assert evidence.schema_version == "bfcl-source-evidence-v2"
    assert evidence.source_adapter.kind == adapter
    assert set(evidence.model_dump(mode="json")) == set(
        SourceEvidenceDocument.model_fields
    )
    assert observations["profile_id"] == profile.profile_id
    assert observations["profile_digest"] == profile.digest
    assert observations["outcomes"] == [
        outcome.model_dump(mode="json") for outcome in report.outcomes
    ]
    verify_certification_report(
        report,
        descriptor=evidence.source_adapter,
        source_identity_digest=sha256_json(evidence.identity.model_dump(mode="json")),
        profile=profile,
        required_tier=AdapterTier.A0,
        trusted_public_keys={case.authority.key_id: case.authority.public_key},
        execution_inputs_digest=observations["execution_inputs_digest"],
    )
    with pytest.raises(CertificationError) as under_certified:
        verify_certification_report(
            report,
            descriptor=evidence.source_adapter,
            source_identity_digest=sha256_json(
                evidence.identity.model_dump(mode="json")
            ),
            profile=profile,
            required_tier=AdapterTier.A2,
            trusted_public_keys={case.authority.key_id: case.authority.public_key},
            execution_inputs_digest=observations["execution_inputs_digest"],
        )
    assert (
        under_certified.value.code
        == CertificationRefusalCode.ADAPTER_UNDER_CERTIFIED.value
    )

    brief_report = load_domain_brief_redaction_report(
        case.domain_brief_report_path,
        brief=evidence.domain_brief,
        source_path=case.domain_brief_source_path,
    )
    held_out_report = load_held_out_redaction_report(
        case.held_out_redaction_path
    )
    verify_held_out_redaction_report(
        held_out_report,
        decision=evidence.fixtures.held_out,
        evidence_digest=evidence.bundle_digest,
        trusted_public_keys={case.authority.key_id: case.authority.public_key},
    )
    persisted_subject = EXPOSURE_SUBJECT_ADAPTER.validate_python(
        json.loads(case.exposure_subject_path.read_text(encoding="utf-8"))
    )
    expected_subject = build_exposure_subject(
        evidence,
        domain_brief_report=brief_report,
        held_out_redaction_report=held_out_report,
    )
    assert persisted_subject == expected_subject

    authorization = authorize_model_exposure_by_human(
        persisted_subject,
        authorized_by="adapter-contract-tests",
    )
    verify_exposure_authorization(
        authorization,
        expected_subject=expected_subject,
    )
    stale_subject = expected_subject.model_copy(
        update={"evidence_digest": DIGEST_C}
    )
    with pytest.raises(AuthorizationError, match="does not cover"):
        verify_exposure_authorization(
            authorization,
            expected_subject=stale_subject,
        )

    changed_identity = evidence.identity.model_copy(
        update={
            "source_config_digest": (
                DIGEST_B
                if evidence.identity.source_config_digest != DIGEST_B
                else DIGEST_C
            )
        }
    )
    with pytest.raises(CertificationError, match="trusted inputs"):
        verify_certification_report(
            report,
            descriptor=evidence.source_adapter,
            source_identity_digest=sha256_json(
                changed_identity.model_dump(mode="json")
            ),
            profile=profile,
            required_tier=AdapterTier.A0,
            trusted_public_keys={case.authority.key_id: case.authority.public_key},
            execution_inputs_digest=observations["execution_inputs_digest"],
        )

    a0_outcomes = report.outcomes[:2]
    if adapter == "mcp_mode_a":
        assert profile.profile_id == "mcp-mode-a-v1"
        assert all(outcome.evidence is not None for outcome in a0_outcomes)
    else:
        for outcome in a0_outcomes:
            assert outcome.evidence is not None
            execution = outcome.evidence["execution"]
            assert execution["cleanup_status"] in {"passed", "not_required"}


@pytest.mark.parametrize(
    ("include_timeout", "required_tier"),
    [(False, AdapterTier.A1), (True, AdapterTier.A2)],
)
def test_local_observation_intake_enforces_a1_and_a2_boundaries(
    tmp_path: Path,
    include_timeout: bool,
    required_tier: AdapterTier,
) -> None:
    package = tmp_path / "runtime-source"
    package.mkdir()
    _runtime_package(package)
    output = tmp_path / f"out-{required_tier.value}"

    result = run_conventional_intake(
        {
            "declaration_version": "bfcl-source-declaration-v1",
            "local_python": {"path": package.name},
        },
        output,
        source_base_dir=tmp_path,
        allowed_roots=(tmp_path,),
        pack=PACK,
        domain_brief_path=_brief(tmp_path),
        certification_authority=AUTHORITY,
        held_out_decision=DECISION,
        required_tier=required_tier,
        probe_plan=_probe_plan(include_timeout=include_timeout),
    )

    assert result.finalized.certification.attained_tier is required_tier
    assert not result.finalized.evidence.unresolved_gaps or (
        required_tier is AdapterTier.A1
    )
    if required_tier is AdapterTier.A1:
        assert {
            gap.code for gap in result.finalized.evidence.unresolved_gaps
        } == {"confirmation_behavior", "reset_isolation"}
        with pytest.raises(BundleError, match="below required A2"):
            load_evidence_bundle(
                result.evidence_path,
                certification_report_path=result.output_root
                / CERTIFICATION_FILE_NAME,
                trusted_certification_keys={
                    AUTHORITY.key_id: AUTHORITY.public_key
                },
                domain_brief_source_path=result.output_root
                / DOMAIN_BRIEF_SOURCE_FILE_NAME,
                domain_brief_report_path=result.output_root
                / DOMAIN_BRIEF_REPORT_FILE_NAME,
                held_out_redaction_report_path=result.output_root
                / HELD_OUT_REDACTION_FILE_NAME,
                source_observations_path=result.output_root
                / OBSERVATIONS_FILE_NAME,
                required_certification_tier=AdapterTier.A2,
            )
    view = load_evidence_bundle(
        result.evidence_path,
        certification_report_path=result.output_root / CERTIFICATION_FILE_NAME,
        trusted_certification_keys={AUTHORITY.key_id: AUTHORITY.public_key},
        domain_brief_source_path=(
            result.output_root / DOMAIN_BRIEF_SOURCE_FILE_NAME
        ),
        domain_brief_report_path=(
            result.output_root / DOMAIN_BRIEF_REPORT_FILE_NAME
        ),
        held_out_redaction_report_path=(
            result.output_root / HELD_OUT_REDACTION_FILE_NAME
        ),
        source_observations_path=result.output_root / OBSERVATIONS_FILE_NAME,
        required_certification_tier=required_tier,
    )
    assert view.certification_verified


def test_a2_refusal_from_an_a1_run_publishes_nothing(tmp_path: Path) -> None:
    package = tmp_path / "runtime-source"
    package.mkdir()
    _runtime_package(package)
    output = tmp_path / "under-certified"

    with pytest.raises(SourceIntakeError) as refused:
        run_conventional_intake(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                "local_python": {"path": package.name},
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
        )

    assert refused.value.code == "adapter_under_certified"
    assert not output.exists()


def test_every_stable_refusal_is_documented_and_profile_reasons_are_versioned() -> None:
    reference = (
        Path(__file__).parents[3]
        / "src/nemotron/steps/byob/references/"
        "bfcl-source-adapter-certification-profiles.md"
    ).read_text(encoding="utf-8")
    documented = {
        line.removeprefix("- `").removesuffix("`")
        for line in reference.splitlines()
        if line.startswith("- `") and line.endswith("`")
    }
    stable = {code.value for code in CertificationRefusalCode}

    assert stable <= documented
    for profile in PUBLISHED_CERTIFICATION_PROFILES.values():
        for requirement in profile.probes:
            assert set(requirement.allowed_failure_reasons) <= stable
