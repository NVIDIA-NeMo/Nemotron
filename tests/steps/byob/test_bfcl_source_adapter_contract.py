from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json, sha256_text
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    AdapterDescriptor,
    AdapterRequest,
    CleanupKind,
    CleanupSemantics,
    FixtureAccessKind,
    FixtureAccessPolicy,
    OracleSourceAdapter,
    ProbeSafetyKind,
    ProbeSafetyPolicy,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import DomainBriefEvidence
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    CapabilityEvidence,
    CertificationReference,
    ConfirmationVocabulary,
    FixtureEvidence,
    PackIdentity,
    SourceEvidenceError,
    SourceIdentity,
    ToolEvidence,
    UnresolvedGap,
    UnsignedSourceEvidence,
    UntrustedText,
    build_source_evidence,
    load_source_evidence,
    write_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _descriptor() -> AdapterDescriptor:
    return AdapterDescriptor(
        contract_version="bfcl-source-adapter-v1",
        kind="fixture",
        implementation_name="bfcl.fixture",
        implementation_version="1.0.0",
        capabilities=(
            AdapterCapability.DESCRIBE_TOOLS,
            AdapterCapability.PIN_IDENTITY,
        ),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.READ_ONLY,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.IDENTITY_ONLY,
            max_calls=1,
            timeout_s=1.0,
        ),
        cleanup=CleanupSemantics(kind=CleanupKind.NONE, timeout_s=1.0),
    )


def _unsigned(
    *,
    descriptor: AdapterDescriptor | None = None,
) -> UnsignedSourceEvidence:
    selected = _descriptor() if descriptor is None else descriptor
    descriptor_digest = sha256_json(selected.model_dump(mode="json"))
    return UnsignedSourceEvidence(
        schema_version="bfcl-source-evidence-v2",
        source_adapter=selected,
        certification=CertificationReference(
            reference_version="bfcl-adapter-certification-reference-v1",
            report_schema_version="bfcl-adapter-certification-report-v1",
            report_digest=SHA_A,
            descriptor_digest=descriptor_digest,
            issuer="bfcl-verifier",
            profile_id="fixture-a0",
            attained_tier="A0",
        ),
        pack=PackIdentity(pack_id="tiny-library", version="1.0.0"),
        domain_brief=DomainBriefEvidence(
            schema_version="bfcl-domain-brief-v1",
            language="en",
            untrusted_text="Evaluate safe library operations.",
            source_digest=SHA_A,
            content_digest=sha256_text("Evaluate safe library operations."),
            redaction_report_digest=SHA_D,
        ),
        identity=SourceIdentity(
            subject="fixture source",
            effective_content_digest=SHA_B,
            source_config_digest=SHA_C,
        ),
        capabilities=(
            CapabilityEvidence(
                capability=AdapterCapability.DESCRIBE_TOOLS,
                status="observed",
                evidence_digests=(SHA_A,),
            ),
            CapabilityEvidence(
                capability=AdapterCapability.PIN_IDENTITY,
                status="observed",
                evidence_digests=(SHA_B,),
            ),
        ),
        vocabulary=ConfirmationVocabulary(),
        fixtures=FixtureEvidence(
            direction="read_only",
            content_digest=SHA_D,
            held_out=build_not_applicable_decision(
                "Synthetic contract fixture has no held-out data.",
                reviewed_by="contract-tests",
            ),
        ),
        tools=(
            ToolEvidence(
                published_name="library.lookup",
                source_name="library.lookup",
                description=UntrustedText(untrusted_text="Look up a book."),
                parameter_schema={
                    "type": "object",
                    "properties": {"book_id": {"type": "string"}},
                    "required": ["book_id"],
                },
                output_schema=None,
                annotations=None,
                mutates=False,
                requires_confirmation=False,
                raw_digest=SHA_D,
            ),
        ),
        unresolved_gaps=(
            UnresolvedGap(
                code="fixture_samples",
                field="fixtures",
                reason="No model-visible fixture samples were approved.",
            ),
        ),
    )


def test_descriptor_declares_capabilities_but_cannot_claim_certification() -> None:
    descriptor = _descriptor()

    assert "attained_tier" not in AdapterDescriptor.model_fields
    assert "certification" not in AdapterDescriptor.model_fields
    assert descriptor.capabilities == (
        AdapterCapability.DESCRIBE_TOOLS,
        AdapterCapability.PIN_IDENTITY,
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AdapterDescriptor.model_validate(
            {**descriptor.model_dump(mode="json"), "attained_tier": "A2"}
        )


def test_descriptor_is_strict_canonical_and_bounded() -> None:
    raw = _descriptor().model_dump(mode="json")
    raw["capabilities"] = ["pin_identity", "describe_tools"]
    with pytest.raises(ValidationError, match="sorted"):
        AdapterDescriptor.model_validate(raw)

    raw = _descriptor().model_dump(mode="json")
    raw["probe_safety"]["max_calls"] = 0
    with pytest.raises(ValidationError, match="positive"):
        AdapterDescriptor.model_validate(raw)

    raw = _descriptor().model_dump(mode="json")
    raw["implementation_name"] = "Unsafe Adapter"
    with pytest.raises(ValidationError, match="lowercase"):
        AdapterDescriptor.model_validate(raw)


def test_fake_adapter_can_only_return_unsigned_evidence() -> None:
    evidence = _unsigned()

    class _FakeAdapter:
        descriptor = evidence.source_adapter

        def collect_evidence(self, request: AdapterRequest) -> UnsignedSourceEvidence:
            assert request.workspace_id == "run-1"
            return evidence

    adapter = _FakeAdapter()
    request = AdapterRequest(
        request_version="bfcl-source-adapter-request-v1",
        source_declaration_digest=SHA_A,
        workspace_id="run-1",
    )

    assert isinstance(adapter, OracleSourceAdapter)
    assert adapter.collect_evidence(request) is evidence


def test_v2_evidence_round_trip_is_byte_stable_and_digest_bound(
    tmp_path: Path,
) -> None:
    evidence = build_source_evidence(_unsigned())
    first = write_source_evidence(evidence, tmp_path / "first.json")
    loaded = load_source_evidence(first)
    second = write_source_evidence(loaded, tmp_path / "second.json")

    assert loaded == evidence
    assert first.read_bytes() == second.read_bytes()
    assert evidence.bundle_digest == sha256_json(
        evidence.model_dump(mode="json", exclude={"bundle_digest"})
    )


def test_v2_evidence_rejects_unknown_fields_and_tampering(tmp_path: Path) -> None:
    evidence = build_source_evidence(_unsigned())
    document = evidence.model_dump(mode="json")

    unknown = {**document, "gold_eligible": True}
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(SourceEvidenceError, match="Extra inputs are not permitted"):
        load_source_evidence(path)

    tampered = evidence.model_dump(mode="json")
    tampered["pack"]["version"] = "2.0.0"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SourceEvidenceError, match="modified"):
        load_source_evidence(path)


def test_v2_evidence_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"bfcl-source-evidence-v2",'
        '"schema_version":"bfcl-source-evidence-v2"}',
        encoding="utf-8",
    )

    with pytest.raises(SourceEvidenceError, match="repeats JSON key 'schema_version'"):
        load_source_evidence(path)


def test_certification_reference_must_cover_exact_descriptor() -> None:
    document = _unsigned().model_dump(mode="json")
    document["certification"]["descriptor_digest"] = SHA_D

    with pytest.raises(ValidationError, match="does not cover"):
        UnsignedSourceEvidence.model_validate(document)


def test_capability_evidence_must_match_descriptor_exactly() -> None:
    document = _unsigned().model_dump(mode="json")
    document["capabilities"] = document["capabilities"][:-1]

    with pytest.raises(ValidationError, match="exactly match"):
        UnsignedSourceEvidence.model_validate(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tools", [], "at least one tool"),
        (
            "capabilities",
            [
                {
                    "capability": "describe_tools",
                    "status": "observed",
                    "evidence_digests": [],
                    "reason": None,
                },
                {
                    "capability": "pin_identity",
                    "status": "observed",
                    "evidence_digests": [SHA_B],
                    "reason": None,
                },
            ],
            "requires at least one digest",
        ),
    ],
)
def test_v2_evidence_rejects_incomplete_contract_fields(
    field: str,
    value: Any,
    message: str,
) -> None:
    document = _unsigned().model_dump(mode="json")
    document[field] = value

    with pytest.raises(ValidationError, match=message):
        UnsignedSourceEvidence.model_validate(document)
