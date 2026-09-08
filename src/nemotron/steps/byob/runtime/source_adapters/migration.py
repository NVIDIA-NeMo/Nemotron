# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Version negotiation and deterministic v1-to-v2 source-evidence migration.

The legacy MCP bundle did not contain a domain brief, a generic adapter
descriptor, or a BFCL-owned certification reference.  Migration therefore
requires those reviewed inputs explicitly; it never fabricates authority or
model-visible prose to make an old document fit the new schema.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    AdapterDescriptor,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import (
    DomainBriefEvidence,
    DomainBriefRedactionReport,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    SOURCE_EVIDENCE_VERSION,
    CapabilityEvidence,
    CertificationReference,
    ConfirmationVocabulary,
    FixtureEvidence,
    IdentityArtifact,
    PackIdentity,
    SourceEvidenceDocument,
    SourceIdentity,
    ToolEvidence,
    UnresolvedGap,
    UnsignedSourceEvidence,
    UntrustedText,
    build_source_evidence,
    load_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import HeldOutDecision

LEGACY_MCP_EVIDENCE_VERSION: Literal["bfcl-mcp-evidence-v1"] = "bfcl-mcp-evidence-v1"
MIGRATION_RECORD_VERSION: Literal["bfcl-source-evidence-migration-v1"] = "bfcl-source-evidence-migration-v1"
MIGRATION_APPROVAL_VERSION: Literal["bfcl-source-evidence-approval-v2"] = "bfcl-source-evidence-approval-v2"
TRANSFORMER_ID: Literal["bfcl.mcp-evidence-v1-to-source-evidence-v2"] = "bfcl.mcp-evidence-v1-to-source-evidence-v2"
TRANSFORMER_VERSION: Literal["1.0.0"] = "1.0.0"
_TRANSFORMER_SPEC = {
    "id": TRANSFORMER_ID,
    "version": TRANSFORMER_VERSION,
    "source": LEGACY_MCP_EVIDENCE_VERSION,
    "target": SOURCE_EVIDENCE_VERSION,
    "authority": "explicit-context-only",
    "unknowns": "preserve-as-unresolved-gaps",
    "review": "record-warning",
}
TRANSFORMER_DIGEST = sha256_json(_TRANSFORMER_SPEC)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_LEGACY_FIELDS = {
    "schema_version",
    "profile_version",
    "status",
    "attained_level",
    "mode",
    "pack",
    "oracle",
    "identity",
    "vocabulary",
    "fixtures",
    "tools",
    "catalog",
    "review",
    "unknowns",
    "assumptions",
    "bundle_digest",
}


class EvidenceMigrationError(ValueError):
    """Raised when evidence cannot be normalized without inventing information."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_digest(value: str, field: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


class MigrationContext(_StrictModel):
    """Reviewed v2-only fields that cannot be recovered from a legacy bundle."""

    source_adapter: AdapterDescriptor
    certification: CertificationReference
    domain_brief: DomainBriefEvidence
    domain_brief_report: DomainBriefRedactionReport
    held_out: HeldOutDecision
    # A legacy bundle names a fixture *direction* but never the fixtures themselves,
    # because discovery has none. Once probes run, the reviewed snapshot they ran against
    # is a fact about this evidence, so the caller that supplied the plan states its
    # digest here rather than letting a later phase re-derive it from an unsigned file.
    fixture_content_digest: StrictStr | None = None

    @field_validator("fixture_content_digest")
    @classmethod
    def _fixture_digest(cls, value: str | None) -> str | None:
        if value is not None and _DIGEST.fullmatch(value) is None:
            raise ValueError("fixture_content_digest must be sha256:<64 lowercase hex>")
        return value

    @model_validator(mode="after")
    def _descriptor_binding(self) -> MigrationContext:
        descriptor_digest = sha256_json(self.source_adapter.model_dump(mode="json"))
        if self.certification.descriptor_digest != descriptor_digest:
            raise ValueError("migration certification does not cover the supplied adapter descriptor")
        if self.source_adapter.kind != "mcp_mode_a":
            raise ValueError("legacy MCP evidence requires an mcp_mode_a descriptor")
        if self.certification.profile_id != "mcp-mode-a-v1":
            raise ValueError("legacy MCP evidence requires the MCP certification profile")
        if (
            self.domain_brief.redaction_report_digest != self.domain_brief_report.record_digest
            or self.domain_brief.source_digest != self.domain_brief_report.source_digest
            or self.domain_brief.content_digest != self.domain_brief_report.sanitized_digest
        ):
            raise ValueError("migration domain brief does not match its redaction report")
        return self


class MigrationWarning(_StrictModel):
    code: StrictStr
    location: StrictStr
    message: StrictStr

    @field_validator("code")
    @classmethod
    def _code(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("migration warning code must be a safe identifier")
        return value

    @field_validator("location", "message")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("migration warning text must be non-empty")
        return value


class EvidenceMigrationRecord(_StrictModel):
    schema_version: Literal["bfcl-source-evidence-migration-v1"]
    transformer_id: Literal["bfcl.mcp-evidence-v1-to-source-evidence-v2"]
    transformer_version: Literal["1.0.0"]
    transformer_digest: StrictStr
    source_schema_version: Literal["bfcl-mcp-evidence-v1"]
    source_digest: StrictStr
    normalized_schema_version: Literal["bfcl-source-evidence-v2"]
    normalized_digest: StrictStr
    warnings: tuple[MigrationWarning, ...]
    record_digest: StrictStr

    @field_validator(
        "transformer_digest",
        "source_digest",
        "normalized_digest",
        "record_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "migration digest")

    @field_validator("warnings")
    @classmethod
    def _canonical_warnings(
        cls,
        value: tuple[MigrationWarning, ...],
    ) -> tuple[MigrationWarning, ...]:
        keys = [(item.code, item.location) for item in value]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("migration warnings must be unique and sorted")
        return value

    @model_validator(mode="after")
    def _bindings(self) -> EvidenceMigrationRecord:
        if self.transformer_digest != TRANSFORMER_DIGEST:
            raise ValueError("migration transformer digest mismatch")
        unsigned = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("migration record digest mismatch")
        return self


class NormalizedSourceEvidence(_StrictModel):
    """The v2 document and, only when transformed, its immutable migration record."""

    source_digest: StrictStr
    evidence: SourceEvidenceDocument
    migration: EvidenceMigrationRecord | None

    @field_validator("source_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "source evidence digest")

    @model_validator(mode="after")
    def _migration_binding(self) -> NormalizedSourceEvidence:
        normalized_root = self.evidence.bundle_digest
        if self.migration is None:
            expected_native_root = (
                self.evidence.revision.root_bundle_digest
                if self.evidence.revision is not None
                else self.evidence.bundle_digest
            )
            if self.source_digest != expected_native_root:
                raise ValueError("native v2 source digest must equal normalized digest")
        elif self.evidence.revision is None:
            if (
                self.migration.source_digest != self.source_digest
                or self.migration.normalized_digest != normalized_root
            ):
                raise ValueError("migration record does not bind source and normalized evidence")
        elif (
            self.migration.source_digest != self.source_digest
            or self.evidence.revision.root_bundle_digest != self.source_digest
        ):
            raise ValueError("migrated revision root must equal the legacy source digest")
        return self


class NormalizedEvidenceApproval(_StrictModel):
    approval_version: Literal["bfcl-source-evidence-approval-v2"]
    approved_by: StrictStr
    source_bundle_digest: StrictStr
    normalized_bundle_digest: StrictStr
    migration_record_digest: StrictStr | None
    acknowledged_warnings: tuple[StrictStr, ...] = ()
    acknowledged_findings: tuple[StrictStr, ...] = ()
    note: StrictStr | None = None

    @field_validator("approved_by")
    @classmethod
    def _approved_by(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("approval must name its reviewer")
        return value.strip()

    @field_validator(
        "source_bundle_digest",
        "normalized_bundle_digest",
        "migration_record_digest",
    )
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is not None:
            _require_digest(value, "approval digest")
        return value

    @field_validator("acknowledged_warnings", "acknowledged_findings")
    @classmethod
    def _canonical_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("acknowledged warnings must be unique and sorted")
        return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceMigrationError(f"evidence repeats JSON key {key!r}")
        result[key] = value
    return result


def _read_document(path: Path) -> dict[str, Any]:
    source = path.resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except EvidenceMigrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceMigrationError(f"cannot read evidence {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceMigrationError(f"evidence must be a JSON object: {source}")
    return value


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceMigrationError(f"legacy evidence {field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceMigrationError(f"legacy evidence {field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceMigrationError(f"legacy evidence {field} must be a non-empty string")
    return value


def _legacy_document(path: Path) -> dict[str, Any]:
    document = _read_document(path)
    if document.get("schema_version") != LEGACY_MCP_EVIDENCE_VERSION:
        raise EvidenceMigrationError(f"unsupported source evidence schema_version {document.get('schema_version')!r}")
    if set(document) != _LEGACY_FIELDS:
        missing = sorted(_LEGACY_FIELDS - set(document))
        unknown = sorted(set(document) - _LEGACY_FIELDS)
        raise EvidenceMigrationError(f"legacy evidence field mismatch; missing={missing!r}, unknown={unknown!r}")
    claimed = document.get("bundle_digest")
    observed = sha256_json({key: value for key, value in document.items() if key != "bundle_digest"})
    if claimed != observed:
        raise EvidenceMigrationError(f"legacy evidence digest mismatch: claimed {claimed!r}, observed {observed!r}")
    return document


def _identity(document: dict[str, Any]) -> SourceIdentity:
    identity = _require_mapping(document["identity"], "identity")
    oracle = _require_mapping(document["oracle"], "oracle")
    effective = identity.get("effective_content_digest", oracle.get("content_digest"))
    source_config = identity.get("source_config_digest")
    if not isinstance(effective, str) or not isinstance(source_config, str):
        raise EvidenceMigrationError("legacy identity lacks effective_content_digest or source_config_digest")
    subject = oracle.get("oracle_id")
    if not isinstance(subject, str) or not subject.strip():
        raise EvidenceMigrationError("legacy oracle lacks a non-empty oracle_id")
    artifacts = []
    for legacy_name, role in (
        ("discovery_report_digest", "discovery_report"),
        ("gateway_artifact_digest", "gateway_artifact"),
        ("shim_artifact_digest", "shim_artifact"),
        ("snapshot_digest", "snapshot"),
        ("tool_catalog_digest", "tool_catalog"),
        ("authorization_context_digest", "authorization_context"),
    ):
        digest = identity.get(legacy_name)
        if digest is not None:
            artifacts.append(IdentityArtifact(role=role, digest=digest))
    return SourceIdentity(
        subject=subject,
        effective_content_digest=effective,
        source_config_digest=source_config,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.role)),
    )


def legacy_mcp_source_identity(document: Mapping[str, Any]) -> SourceIdentity:
    """Derive the exact v2 source identity used by the legacy MCP transformer."""

    return _identity(dict(document))


def _capabilities(
    descriptor: AdapterDescriptor,
    *,
    source_digest: str,
) -> tuple[CapabilityEvidence, ...]:
    observed = {
        AdapterCapability.DESCRIBE_TOOLS,
        AdapterCapability.PIN_IDENTITY,
    }
    return tuple(
        CapabilityEvidence(
            capability=capability,
            status="observed" if capability in observed else "declared",
            evidence_digests=(source_digest,) if capability in observed else (),
        )
        for capability in descriptor.capabilities
    )


def _tools(document: dict[str, Any]) -> tuple[ToolEvidence, ...]:
    converted = []
    for index, raw in enumerate(_require_list(document["tools"], "tools")):
        tool = _require_mapping(raw, f"tools[{index}]")
        if set(tool) != {
            "published_name",
            "source_name",
            "description",
            "declared",
            "untrusted_schemas",
            "raw_digest",
            "trust_annotations",
        }:
            raise EvidenceMigrationError(f"legacy evidence tools[{index}] has an invalid field set")
        description = _require_mapping(tool["description"], f"tools[{index}].description")
        declared = _require_mapping(tool["declared"], f"tools[{index}].declared")
        schemas = _require_mapping(
            tool["untrusted_schemas"],
            f"tools[{index}].untrusted_schemas",
        )
        if set(description) != {"untrusted_text"}:
            raise EvidenceMigrationError(f"legacy evidence tools[{index}].description is not tagged untrusted")
        if not {"mutates", "requires_confirmation"} <= set(declared):
            raise EvidenceMigrationError(f"legacy evidence tools[{index}].declared is incomplete")
        if set(schemas) != {"parameters", "output_schema", "annotations"}:
            raise EvidenceMigrationError(f"legacy evidence tools[{index}].untrusted_schemas is incomplete")
        converted.append(
            ToolEvidence(
                published_name=tool["published_name"],
                source_name=tool["source_name"],
                description=UntrustedText(untrusted_text=description["untrusted_text"]),
                parameter_schema=schemas["parameters"],
                output_schema=schemas["output_schema"],
                annotations=schemas["annotations"],
                mutates=declared["mutates"],
                requires_confirmation=declared["requires_confirmation"],
                raw_digest=tool["raw_digest"],
            )
        )
    return tuple(sorted(converted, key=lambda item: item.published_name))


def _gaps(document: dict[str, Any]) -> tuple[UnresolvedGap, ...]:
    converted = []
    for index, raw in enumerate(_require_list(document["unknowns"], "unknowns")):
        unknown = _require_mapping(raw, f"unknowns[{index}]")
        if set(unknown) != {"field", "blocks", "resolved_by"}:
            raise EvidenceMigrationError(f"legacy evidence unknowns[{index}] has an invalid field set")
        reason = f"Blocks: {unknown['blocks']} Resolved by: {unknown['resolved_by']}"
        converted.append(
            UnresolvedGap(
                code="legacy_unresolved",
                field=unknown["field"],
                reason=reason,
            )
        )
    return tuple(sorted(converted, key=lambda item: (item.code, item.field)))


def _warnings(document: dict[str, Any]) -> tuple[MigrationWarning, ...]:
    warnings = [
        MigrationWarning(
            code="legacy_transport_metadata",
            location="legacy",
            message=(
                "Legacy profile, mode, oracle, catalog, review, and assumptions remain "
                "available only through the source bundle digest."
            ),
        )
    ]
    fixtures = _require_mapping(document["fixtures"], "fixtures")
    if fixtures.get("direction") != "none":
        warnings.append(
            MigrationWarning(
                code="fixture_content_digest_unavailable",
                location="fixtures.content_digest",
                message="Legacy evidence did not bind fixture content independently.",
            )
        )
    review = _require_mapping(document["review"], "review")
    advisory = _require_list(review.get("advisory"), "review.advisory")
    for index, raw in enumerate(advisory):
        finding = _require_mapping(raw, f"review.advisory[{index}]")
        location = _require_string(
            finding.get("location"),
            f"review.advisory[{index}].location",
        )
        code = _require_string(
            finding.get("code"),
            f"review.advisory[{index}].code",
        )
        detail = _require_string(
            finding.get("detail"),
            f"review.advisory[{index}].detail",
        )
        identity = sha256_json({"location": location, "code": code, "detail": detail}).removeprefix("sha256:")[:16]
        warnings.append(
            MigrationWarning(
                code=f"legacy_finding_{identity}",
                location=location,
                message=f"{code}: {detail}",
            )
        )
    return tuple(sorted(warnings, key=lambda item: (item.code, item.location)))


def migrate_legacy_mcp_evidence(
    path: Path,
    *,
    context: MigrationContext,
) -> NormalizedSourceEvidence:
    """Normalize one digest-valid legacy MCP bundle to strict evidence v2."""

    document = _legacy_document(path)
    source_digest = str(document["bundle_digest"])
    pack = _require_mapping(document["pack"], "pack")
    vocabulary = _require_mapping(document["vocabulary"], "vocabulary")
    fixtures = _require_mapping(document["fixtures"], "fixtures")
    direction = fixtures.get("direction")
    if direction not in {"none", "read_only", "pushed", "snapshot"}:
        raise EvidenceMigrationError(f"legacy fixture direction {direction!r} is unsupported")
    unsigned = UnsignedSourceEvidence(
        schema_version=SOURCE_EVIDENCE_VERSION,
        source_adapter=context.source_adapter,
        certification=context.certification,
        pack=PackIdentity(
            pack_id=_require_string(pack.get("pack_id"), "pack.pack_id"),
            version=_require_string(pack.get("version"), "pack.version"),
        ),
        domain_brief=context.domain_brief,
        identity=_identity(document),
        capabilities=_capabilities(
            context.source_adapter,
            source_digest=source_digest,
        ),
        vocabulary=ConfirmationVocabulary(
            parameter=vocabulary.get("confirmation_parameter"),
            status_field=vocabulary.get("status_field"),
            pending_status=vocabulary.get("pending_status"),
            error_path=vocabulary.get("error_path"),
        ),
        fixtures=FixtureEvidence(
            direction=direction,
            content_digest=context.fixture_content_digest,
            held_out=context.held_out,
        ),
        tools=_tools(document),
        unresolved_gaps=_gaps(document),
    )
    normalized = build_source_evidence(unsigned)
    record_document: dict[str, Any] = {
        "schema_version": MIGRATION_RECORD_VERSION,
        "transformer_id": TRANSFORMER_ID,
        "transformer_version": TRANSFORMER_VERSION,
        "transformer_digest": TRANSFORMER_DIGEST,
        "source_schema_version": LEGACY_MCP_EVIDENCE_VERSION,
        "source_digest": source_digest,
        "normalized_schema_version": SOURCE_EVIDENCE_VERSION,
        "normalized_digest": normalized.bundle_digest,
        "warnings": [warning.model_dump(mode="json") for warning in _warnings(document)],
    }
    record_document["record_digest"] = sha256_json(record_document)
    return NormalizedSourceEvidence(
        source_digest=source_digest,
        evidence=normalized,
        migration=EvidenceMigrationRecord.model_validate(record_document),
    )


def normalize_source_evidence(
    path: Path,
    *,
    legacy_context: MigrationContext | None = None,
) -> NormalizedSourceEvidence:
    """Negotiate v1/v2 explicitly; unknown and newer versions fail closed."""

    document = _read_document(path)
    version = document.get("schema_version")
    if version == SOURCE_EVIDENCE_VERSION:
        evidence = load_source_evidence(path)
        source_digest = (
            evidence.revision.root_bundle_digest if evidence.revision is not None else evidence.bundle_digest
        )
        return NormalizedSourceEvidence(
            source_digest=source_digest,
            evidence=evidence,
            migration=None,
        )
    if version == LEGACY_MCP_EVIDENCE_VERSION:
        if legacy_context is None:
            raise EvidenceMigrationError("legacy evidence requires explicit migration context")
        return migrate_legacy_mcp_evidence(path, context=legacy_context)
    raise EvidenceMigrationError(f"unsupported source evidence schema_version {version!r}")


def verified_source_digest(path: Path) -> str:
    """Return a verified v1/v2 bundle digest without transforming its contents."""

    document = _read_document(path)
    version = document.get("schema_version")
    if version == SOURCE_EVIDENCE_VERSION:
        evidence = load_source_evidence(path)
        return evidence.revision.root_bundle_digest if evidence.revision is not None else evidence.bundle_digest
    if version == LEGACY_MCP_EVIDENCE_VERSION:
        return str(_legacy_document(path)["bundle_digest"])
    raise EvidenceMigrationError(f"unsupported source evidence schema_version {version!r}")


def write_migration_record(
    record: EvidenceMigrationRecord,
    path: Path,
) -> Path:
    EvidenceMigrationRecord.model_validate(record.model_dump(mode="json"))
    return write_canonical_json(record.model_dump(mode="json"), path)


def load_migration_record(path: Path) -> EvidenceMigrationRecord:
    """Load and verify one immutable migration record."""

    document = _read_document(path)
    try:
        return EvidenceMigrationRecord.model_validate(document)
    except ValueError as exc:
        raise EvidenceMigrationError(f"invalid migration record: {exc}") from exc


def load_normalized_approval(
    path: Path,
    normalized: NormalizedSourceEvidence,
    *,
    required_findings: frozenset[str] = frozenset(),
) -> NormalizedEvidenceApproval:
    """Require a new approval that binds both sides of any transformation."""

    document = _read_document(path)
    if document.get("approval_version") != MIGRATION_APPROVAL_VERSION:
        raise EvidenceMigrationError(
            "normalized evidence requires approval_version "
            f"{MIGRATION_APPROVAL_VERSION!r}; legacy approval is not sufficient"
        )
    try:
        approval = NormalizedEvidenceApproval.model_validate(document)
    except ValueError as exc:
        raise EvidenceMigrationError(f"invalid normalized approval: {exc}") from exc
    expected_record_digest = normalized.migration.record_digest if normalized.migration is not None else None
    mismatches = []
    if approval.source_bundle_digest != normalized.source_digest:
        mismatches.append("source_bundle_digest")
    if approval.normalized_bundle_digest != normalized.evidence.bundle_digest:
        mismatches.append("normalized_bundle_digest")
    if approval.migration_record_digest != expected_record_digest:
        mismatches.append("migration_record_digest")
    expected_warnings = (
        {f"{warning.location}:{warning.code}" for warning in normalized.migration.warnings}
        if normalized.migration is not None
        else set()
    )
    acknowledged = set(approval.acknowledged_warnings)
    if acknowledged != expected_warnings:
        mismatches.append("acknowledged_warnings")
    if set(approval.acknowledged_findings) != required_findings:
        mismatches.append("acknowledged_findings")
    if mismatches:
        raise EvidenceMigrationError("normalized approval does not match evidence: " + ", ".join(sorted(mismatches)))
    return approval
