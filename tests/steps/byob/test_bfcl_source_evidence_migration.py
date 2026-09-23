from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json, sha256_text
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    AdapterDescriptor,
    CleanupKind,
    CleanupSemantics,
    FixtureAccessKind,
    FixtureAccessPolicy,
    ProbeSafetyKind,
    ProbeSafetyPolicy,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import (
    DomainBriefEvidence,
    DomainBriefRedactionReport,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    CertificationReference,
    write_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)
from nemotron.steps.byob.runtime.source_adapters.migration import (
    MIGRATION_APPROVAL_VERSION,
    EvidenceMigrationError,
    MigrationContext,
    load_normalized_approval,
    migrate_legacy_mcp_evidence,
    normalize_source_evidence,
    write_migration_record,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
SHA_E = "sha256:" + "e" * 64


def _descriptor() -> AdapterDescriptor:
    return AdapterDescriptor(
        contract_version="bfcl-source-adapter-v1",
        kind="mcp_mode_a",
        implementation_name="bfcl.mcp_mode_a",
        implementation_version="1.0.0",
        capabilities=(
            AdapterCapability.DESCRIBE_TOOLS,
            AdapterCapability.PIN_IDENTITY,
        ),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.PUSHED,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.IDENTITY_ONLY,
            max_calls=4,
            timeout_s=5.0,
        ),
        cleanup=CleanupSemantics(kind=CleanupKind.EPISODE, timeout_s=5.0),
    )


def _context(*, held_out_redacted: bool = True) -> MigrationContext:
    descriptor = _descriptor()
    text = "Evaluate deterministic inventory operations."
    report_document = {
        "schema_version": "bfcl-domain-brief-redaction-v1",
        "source_digest": SHA_B,
        "sanitized_digest": sha256_text(text),
        "redactions": [],
        "advisory": [],
    }
    report = DomainBriefRedactionReport.model_validate(
        {
            **report_document,
            "record_digest": sha256_json(report_document),
        }
    )
    return MigrationContext(
        source_adapter=descriptor,
        certification=CertificationReference(
            reference_version="bfcl-adapter-certification-reference-v1",
            report_schema_version="bfcl-adapter-certification-report-v1",
            report_digest=SHA_A,
            descriptor_digest=sha256_json(descriptor.model_dump(mode="json")),
            issuer="bfcl-source-adapter-verifier-v1",
            profile_id="mcp-mode-a-v1",
            attained_tier="A0",
        ),
        domain_brief=DomainBriefEvidence(
            schema_version="bfcl-domain-brief-v1",
            language="en",
            untrusted_text=text,
            source_digest=SHA_B,
            content_digest=sha256_text(text),
            redaction_report_digest=report.record_digest,
        ),
        domain_brief_report=report,
        held_out=build_not_applicable_decision(
            (
                "Legacy fixture was reviewed as fully redacted."
                if held_out_redacted
                else "Legacy fixture declares no held-out evaluation."
            ),
            reviewed_by="migration-tests",
        ),
    )


def _legacy_document() -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "bfcl-mcp-evidence-v1",
        "profile_version": "bfcl-mcp-oracle-v1",
        "status": "requires_review",
        "attained_level": "L0",
        "mode": "A",
        "pack": {"pack_id": "acme-inventory", "version": "1.0.0"},
        "oracle": {
            "protocol_version": "bfcl-oracle-http-v1",
            "oracle_id": "inventory-oracle",
            "oracle_version": "1.0.0",
            "content_digest": SHA_D,
        },
        "identity": {
            "tool_catalog_digest": SHA_E,
            "server_content_digest": None,
            "gateway_artifact_digest": SHA_A,
            "shim_artifact_digest": None,
            "snapshot_digest": None,
            "effective_content_digest": SHA_D,
            "intake_config_digest": SHA_B,
            "source_config_digest": SHA_C,
            "discovery_report_digest": SHA_E,
        },
        "vocabulary": {
            "confirmation_parameter": "confirm",
            "status_field": "status",
            "pending_status": "awaiting_confirmation",
            "error_path": "error",
        },
        "fixtures": {"direction": "pushed", "snapshot_calls": []},
        "tools": [
            {
                "published_name": "inventory.lookup",
                "source_name": "inventory.lookup",
                "description": {"untrusted_text": "Look up one item."},
                "declared": {
                    "mutates": False,
                    "mutation_source": "config",
                    "requires_confirmation": False,
                },
                "untrusted_schemas": {
                    "parameters": {
                        "type": "object",
                        "properties": {"item_id": {"type": "string"}},
                        "required": ["item_id"],
                    },
                    "output_schema": {"type": "object"},
                    "annotations": None,
                },
                "raw_digest": SHA_A,
                "trust_annotations": False,
            }
        ],
        "catalog": {"exclusions": [], "warnings": []},
        "review": {"advisory": []},
        "unknowns": [
            {
                "field": "observed_error_codes",
                "blocks": "negative validation cases",
                "resolved_by": "executable probes",
            }
        ],
        "assumptions": ["Server text is untrusted."],
    }
    document["bundle_digest"] = sha256_json(document)
    return document


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _approval_document(result) -> dict[str, Any]:
    assert result.migration is not None
    warnings = sorted(
        f"{item.location}:{item.code}" for item in result.migration.warnings
    )
    return {
        "approval_version": MIGRATION_APPROVAL_VERSION,
        "approved_by": "reviewer@example.test",
        "source_bundle_digest": result.source_digest,
        "normalized_bundle_digest": result.evidence.bundle_digest,
        "migration_record_digest": result.migration.record_digest,
        "acknowledged_warnings": warnings,
        "acknowledged_findings": [],
        "note": None,
    }


def test_v1_migration_is_deterministic_digest_bound_and_preserves_unknowns(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "legacy.json", _legacy_document())

    first = migrate_legacy_mcp_evidence(source, context=_context())
    second = migrate_legacy_mcp_evidence(source, context=_context())

    assert first == second
    assert first.evidence.schema_version == "bfcl-source-evidence-v2"
    assert first.migration is not None
    assert first.migration.source_digest == _legacy_document()["bundle_digest"]
    assert first.migration.normalized_digest == first.evidence.bundle_digest
    assert first.evidence.unresolved_gaps[0].field == "observed_error_codes"
    assert first.evidence.tools[0].description.untrusted_text == "Look up one item."
    assert first.evidence.identity.effective_content_digest == SHA_D

    first_path = write_migration_record(
        first.migration,
        tmp_path / "first-record.json",
    )
    second_path = write_migration_record(
        second.migration,
        tmp_path / "second-record.json",
    )
    assert first_path.read_bytes() == second_path.read_bytes()


def test_version_negotiation_passes_native_v2_without_a_migration_record(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "legacy.json", _legacy_document())
    migrated = migrate_legacy_mcp_evidence(source, context=_context())
    v2_path = write_source_evidence(migrated.evidence, tmp_path / "v2.json")

    normalized = normalize_source_evidence(v2_path)

    assert normalized.evidence == migrated.evidence
    assert normalized.source_digest == migrated.evidence.bundle_digest
    assert normalized.migration is None


def test_legacy_and_unknown_versions_require_explicit_supported_migration(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "legacy.json", _legacy_document())
    with pytest.raises(EvidenceMigrationError, match="explicit migration context"):
        normalize_source_evidence(source)

    unknown = _legacy_document()
    unknown["schema_version"] = "bfcl-source-evidence-v99"
    unknown["bundle_digest"] = sha256_json(
        {key: value for key, value in unknown.items() if key != "bundle_digest"}
    )
    path = _write(tmp_path / "unknown.json", unknown)
    with pytest.raises(EvidenceMigrationError, match="unsupported"):
        normalize_source_evidence(path, legacy_context=_context())


def test_legacy_tampering_and_schema_drift_fail_closed(tmp_path: Path) -> None:
    tampered = _legacy_document()
    tampered["pack"]["version"] = "2.0.0"
    path = _write(tmp_path / "tampered.json", tampered)
    with pytest.raises(EvidenceMigrationError, match="digest mismatch"):
        migrate_legacy_mcp_evidence(path, context=_context())

    extended = _legacy_document()
    extended["adapter_claimed_gold"] = True
    extended["bundle_digest"] = sha256_json(
        {key: value for key, value in extended.items() if key != "bundle_digest"}
    )
    path = _write(tmp_path / "extended.json", extended)
    with pytest.raises(EvidenceMigrationError, match="field mismatch"):
        migrate_legacy_mcp_evidence(path, context=_context())


def test_v2_only_reviewed_context_changes_normalized_identity(tmp_path: Path) -> None:
    source = _write(tmp_path / "legacy.json", _legacy_document())

    redacted = migrate_legacy_mcp_evidence(source, context=_context())
    visible = migrate_legacy_mcp_evidence(
        source,
        context=_context(held_out_redacted=False),
    )

    assert redacted.source_digest == visible.source_digest
    assert redacted.evidence.bundle_digest != visible.evidence.bundle_digest
    assert redacted.migration is not None
    assert visible.migration is not None
    assert redacted.migration.record_digest != visible.migration.record_digest


def test_migration_preserves_each_legacy_advisory_for_exact_acknowledgement(
    tmp_path: Path,
) -> None:
    document = _legacy_document()
    document["review"]["advisory"] = [
        {
            "location": "tools.inventory.description",
            "code": "instruction_like_text",
            "detail": "Description contains imperative language.",
            "severity": "review",
        },
        {
            "location": "tools.inventory.annotations.note",
            "code": "suspicious_phrase",
            "detail": "Annotation asks the model to ignore policy.",
            "severity": "review",
        },
    ]
    document["bundle_digest"] = sha256_json(
        {key: value for key, value in document.items() if key != "bundle_digest"}
    )
    source = _write(tmp_path / "legacy-findings.json", document)

    normalized = migrate_legacy_mcp_evidence(source, context=_context())

    assert normalized.migration is not None
    finding_warnings = [
        warning
        for warning in normalized.migration.warnings
        if warning.code.startswith("legacy_finding_")
    ]
    assert len(finding_warnings) == 2
    assert {warning.location for warning in finding_warnings} == {
        "tools.inventory.description",
        "tools.inventory.annotations.note",
    }


def test_legacy_mcp_migration_rejects_non_mcp_descriptor() -> None:
    context = _context()
    descriptor = context.source_adapter.model_copy(update={"kind": "local_python"})
    certification = context.certification.model_copy(
        update={
            "descriptor_digest": sha256_json(descriptor.model_dump(mode="json")),
            "profile_id": "local-python-v1",
        }
    )

    with pytest.raises(ValueError, match="mcp_mode_a descriptor"):
        MigrationContext(
            source_adapter=descriptor,
            certification=certification,
            domain_brief=context.domain_brief,
            domain_brief_report=context.domain_brief_report,
            held_out=context.held_out,
        )


def test_new_approval_binds_source_normalized_and_migration_digests(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "legacy.json", _legacy_document())
    normalized = migrate_legacy_mcp_evidence(source, context=_context())
    approval = _approval_document(normalized)
    path = _write(tmp_path / "approval.json", approval)

    loaded = load_normalized_approval(path, normalized)

    assert loaded.normalized_bundle_digest == normalized.evidence.bundle_digest

    for field, replacement in (
        ("source_bundle_digest", SHA_A),
        ("normalized_bundle_digest", SHA_B),
        ("migration_record_digest", SHA_C),
    ):
        swapped = {**approval, field: replacement}
        path = _write(tmp_path / f"{field}.json", swapped)
        with pytest.raises(EvidenceMigrationError, match=field):
            load_normalized_approval(path, normalized)


def test_legacy_approval_and_warning_omission_cannot_authorize_migration(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "legacy.json", _legacy_document())
    normalized = migrate_legacy_mcp_evidence(source, context=_context())
    legacy_approval = {
        "approval_version": "bfcl-authoring-approval-v1",
        "approved_by": "reviewer@example.test",
        "bundle_digest": normalized.source_digest,
        "acknowledged_findings": [],
        "note": None,
    }
    path = _write(tmp_path / "legacy-approval.json", legacy_approval)
    with pytest.raises(EvidenceMigrationError, match="legacy approval is not sufficient"):
        load_normalized_approval(path, normalized)

    incomplete = _approval_document(normalized)
    incomplete["acknowledged_warnings"] = []
    path = _write(tmp_path / "incomplete.json", incomplete)
    with pytest.raises(EvidenceMigrationError, match="acknowledged_warnings"):
        load_normalized_approval(path, normalized)
