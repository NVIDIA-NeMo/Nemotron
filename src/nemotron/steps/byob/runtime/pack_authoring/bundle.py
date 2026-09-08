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

"""Reading an evidence bundle, and the approval gate in front of it.

The drafting phase sees a file, never a server. That is what makes this side auditable: the
exact bytes a human approved are the exact bytes the model reads, and the digest proves it.

Two gates live here. The digest gate catches a bundle edited after review. The approval gate
refuses to draft from a bundle nobody signed, because the whole point of flagging suspicious
tool text for a human is lost if the next phase runs anyway.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.schemas import UNKNOWN_FIELDS
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterCertificationReport,
    AdapterTier,
    CertificationError,
    CertificationProfile,
    certification_profile_by_id,
    load_certification_report,
    verify_certification_report,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import (
    DomainBriefError,
    DomainBriefRedactionReport,
    load_domain_brief_redaction_report,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    SOURCE_EVIDENCE_VERSION,
    SourceEvidenceDocument,
    load_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutError,
    HeldOutRedactionReport,
    load_held_out_redaction_report,
    load_held_out_sensitive_terms,
    load_required_held_out_policy,
    verify_held_out_redaction_report,
)
from nemotron.steps.byob.runtime.source_adapters.migration import (
    MIGRATION_APPROVAL_VERSION,
    EvidenceMigrationError,
    EvidenceMigrationRecord,
    NormalizedSourceEvidence,
    load_migration_record,
    load_normalized_approval,
    verified_source_digest,
)

EVIDENCE_BUNDLE_VERSION = "bfcl-mcp-evidence-v1"
APPROVAL_VERSION = "bfcl-authoring-approval-v1"


class BundleError(Exception):
    """Raised when a bundle cannot be trusted as authoring input."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise BundleError(f"authoring input repeats JSON key {key!r}")
        document[key] = value
    return document


@dataclass(frozen=True)
class ToolEvidence:
    """One tool as the drafting model is allowed to see it."""

    published_name: str
    description: str
    parameters: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any] | None
    mutates: bool
    requires_confirmation: bool

    @property
    def parameter_names(self) -> frozenset[str]:
        properties = self.parameters.get("properties")
        if not isinstance(properties, Mapping):
            return frozenset()
        return frozenset(str(name) for name in properties)

    @property
    def required_parameters(self) -> tuple[str, ...]:
        required = self.parameters.get("required")
        if not isinstance(required, list):
            return ()
        return tuple(sorted(str(name) for name in required))


@dataclass(frozen=True)
class EvidenceView:
    """A verified bundle, plus the accessors the generators actually need."""

    document: dict[str, Any]
    path: Path
    source_evidence: SourceEvidenceDocument | None = None
    certification_report: AdapterCertificationReport | None = None
    migration: EvidenceMigrationRecord | None = None
    source_digest: str | None = None
    source_document: dict[str, Any] | None = None
    domain_brief_report: DomainBriefRedactionReport | None = None
    held_out_redaction_report: HeldOutRedactionReport | None = None

    @property
    def is_v2(self) -> bool:
        return self.document.get("schema_version") == SOURCE_EVIDENCE_VERSION

    @property
    def digest(self) -> str:
        return str(self.document["bundle_digest"])

    @property
    def pack_id(self) -> str:
        return str(self.document["pack"]["pack_id"])

    @property
    def attained_level(self) -> str:
        """Legacy compatibility accessor; new drafting uses certification_tier."""

        if self.is_v2:
            tier = self.certification_tier
            if tier is None:  # Defensive: v2 branch above otherwise raises first.
                raise BundleError("v2 evidence has no certification tier")
            return tier
        return str(self.document["attained_level"])

    @property
    def vocabulary(self) -> dict[str, str]:
        if self.is_v2:
            raw = self.document["vocabulary"]
            return {
                "confirmation_parameter": str(raw.get("parameter") or ""),
                "status_field": str(raw.get("status_field") or ""),
                "pending_status": str(raw.get("pending_status") or ""),
                "error_path": str(raw.get("error_path") or ""),
            }
        return {str(k): str(v) for k, v in self.document["vocabulary"].items()}

    @property
    def certification_tier(self) -> str | None:
        """Return a BFCL-verified v2 tier; legacy labels never become certification."""

        if self.is_v2:
            if self.certification_report is None:
                raise BundleError(
                    "v2 evidence has no independently verified certification report"
                )
            return self.certification_report.attained_tier.value
        return None

    @property
    def legacy_level(self) -> str | None:
        if self.is_v2:
            return None
        return str(self.document["attained_level"])

    @property
    def certification_verified(self) -> bool:
        return self.is_v2 and self.certification_report is not None

    @property
    def domain_brief(self) -> str | None:
        if not self.is_v2:
            return None
        value = self.document["domain_brief"]["untrusted_text"]
        return str(value)

    @property
    def unresolved_unknowns(self) -> frozenset[str]:
        """The still-open unknowns, named in the vocabulary a draft may declare.

        A v2 gap record carries its vocabulary term in a different key depending on which
        producer wrote it: source intake puts the term in `code` and the document location
        in `field`, while legacy migration puts `legacy_unresolved` in `code` and the term
        in `field`. Reading `field` alone therefore reported locations -- `tools`,
        `vocabulary` -- as if they were unknowns. Nothing downstream can use those: a draft
        cannot declare them, so every gate that asks whether an unknown is still open was
        answered no, and a draft naming the real unknown was refused for naming one the
        bundle supposedly did not list.

        Both keys are read, and only terms in the drafting vocabulary survive. A gap
        outside it, such as reset isolation, blocks no drafting decision and is dropped
        rather than passed on as a blocker no draft could ever resolve.
        """
        entries = (
            self.document.get("unresolved_gaps", [])
            if self.is_v2
            else self.document.get("unknowns", [])
        )
        named = {
            str(entry[key])
            for entry in entries
            for key in ("code", "field")
            if entry.get(key) is not None
        }
        return frozenset(named & UNKNOWN_FIELDS)

    @property
    def tools(self) -> tuple[ToolEvidence, ...]:
        entries: list[ToolEvidence] = []
        for entry in self.document["tools"]:
            if self.is_v2:
                parameters = entry["parameter_schema"]
                output_schema = entry["output_schema"]
                annotations = entry["annotations"]
                mutates = entry["mutates"]
                requires_confirmation = entry["requires_confirmation"]
            else:
                schemas = entry["untrusted_schemas"]
                parameters = schemas["parameters"]
                output_schema = schemas["output_schema"]
                annotations = schemas["annotations"]
                mutates = entry["declared"]["mutates"]
                requires_confirmation = entry["declared"]["requires_confirmation"]
            entries.append(
                ToolEvidence(
                    published_name=str(entry["published_name"]),
                    description=str(entry["description"]["untrusted_text"]),
                    parameters=parameters,
                    output_schema=output_schema,
                    annotations=annotations,
                    mutates=bool(mutates),
                    requires_confirmation=bool(requires_confirmation),
                )
            )
        return tuple(entries)

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(tool.published_name for tool in self.tools)

    def tool(self, published_name: str) -> ToolEvidence:
        for tool in self.tools:
            if tool.published_name == published_name:
                return tool
        raise BundleError(f"no tool named {published_name!r} in the evidence bundle")


def _verify_digest(document: Mapping[str, Any], source: Path) -> None:
    claimed = document.get("bundle_digest")
    unsigned = {key: value for key, value in document.items() if key != "bundle_digest"}
    observed = sha256_json(unsigned)
    if claimed != observed:
        raise BundleError(
            f"evidence bundle {source} was modified after its digest was computed: "
            f"claimed {claimed!r}, observed {observed!r}"
        )


def _profile_for(
    profile_id: str,
) -> CertificationProfile:
    try:
        return certification_profile_by_id(profile_id)
    except CertificationError as exc:
        raise BundleError(
            f"no trusted certification profile was supplied for {profile_id!r}"
        ) from exc


def _verify_certification_reference(
    evidence: SourceEvidenceDocument,
    report: AdapterCertificationReport,
) -> None:
    reference = evidence.certification
    expected = {
        "report_schema_version": report.schema_version,
        "report_digest": report.report_digest,
        "descriptor_digest": report.descriptor_digest,
        "issuer": report.issuer,
        "profile_id": report.profile_id,
        "attained_tier": report.attained_tier.value,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if getattr(reference, field) != value
    ]
    if mismatches:
        raise BundleError(
            "source evidence certification reference does not match report: "
            + ", ".join(sorted(mismatches))
        )


def _load_source_observations(
    path: Path,
    *,
    report: AdapterCertificationReport,
    profile: CertificationProfile,
) -> str | None:
    source = path.resolve()
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read source observations {source}: {exc}") from exc
    required = {
        "schema_version",
        "profile_id",
        "profile_digest",
        "execution_inputs_digest",
        "outcomes",
        "document_digest",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise BundleError("source observations have an invalid field set")
    if document["schema_version"] != "bfcl-source-observations-v1":
        raise BundleError("source observations have an unsupported schema version")
    unsigned = {
        key: value for key, value in document.items() if key != "document_digest"
    }
    if document["document_digest"] != sha256_json(unsigned):
        raise BundleError("source observations digest mismatch")
    if (
        document["profile_id"] != profile.profile_id
        or document["profile_digest"] != profile.digest
        or document["outcomes"]
        != [outcome.model_dump(mode="json") for outcome in report.outcomes]
    ):
        raise BundleError(
            "source observations do not match the certification report and profile"
        )
    execution_inputs_digest = document["execution_inputs_digest"]
    if execution_inputs_digest is not None and not isinstance(
        execution_inputs_digest, str
    ):
        raise BundleError("source observations execution digest must be a string or null")
    return execution_inputs_digest


def load_evidence_bundle(
    path: Path,
    *,
    certification_report_path: Path | None = None,
    trusted_certification_keys: Mapping[str, Ed25519PublicKey] | None = None,
    domain_brief_source_path: Path | None = None,
    domain_brief_report_path: Path | None = None,
    held_out_redaction_report_path: Path | None = None,
    held_out_policy_path: Path | None = None,
    held_out_content_path: Path | None = None,
    source_bundle_path: Path | None = None,
    migration_record_path: Path | None = None,
    source_observations_path: Path | None = None,
    required_certification_tier: AdapterTier = AdapterTier.A0,
) -> EvidenceView:
    """Load legacy evidence or strict v2 evidence with independent certification."""
    source = path.resolve()
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read evidence bundle {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleError(f"evidence bundle must be a JSON object: {source}")
    version = raw.get("schema_version")
    if version == SOURCE_EVIDENCE_VERSION:
        if certification_report_path is None:
            raise BundleError(
                "v2 evidence requires an independent certification report"
            )
        if not trusted_certification_keys:
            raise BundleError(
                "v2 evidence requires trusted certification public keys"
            )
        if domain_brief_source_path is None or domain_brief_report_path is None:
            raise BundleError(
                "v2 evidence requires the domain brief source and redaction report"
            )
        if held_out_redaction_report_path is None:
            raise BundleError("v2 evidence requires a held-out redaction report")
        try:
            evidence = load_source_evidence(source)
            report = load_certification_report(certification_report_path)
            brief_report = load_domain_brief_redaction_report(
                domain_brief_report_path,
                brief=evidence.domain_brief,
                source_path=domain_brief_source_path,
            )
            held_out_report = load_held_out_redaction_report(
                held_out_redaction_report_path
            )
            sensitive_terms: tuple[str, ...] = ()
            decision = evidence.fixtures.held_out
            if decision.status == "required":
                if held_out_policy_path is None:
                    raise BundleError(
                        "required held-out evidence needs its reviewed policy path"
                    )
                observed_decision = load_required_held_out_policy(
                    held_out_policy_path,
                    reviewed_by=decision.reviewed_by,
                )
                if observed_decision != decision:
                    raise BundleError(
                        "held-out policy path does not match the evidence decision"
                    )
                sensitive_terms = load_held_out_sensitive_terms(
                    held_out_policy_path,
                    content_path=held_out_content_path,
                )
            elif held_out_policy_path is not None or held_out_content_path is not None:
                raise BundleError(
                    "not_applicable held-out evidence cannot carry policy or content"
                )
            profile = _profile_for(evidence.certification.profile_id)
            _verify_certification_reference(evidence, report)
            execution_inputs_digest = (
                _load_source_observations(
                    source_observations_path,
                    report=report,
                    profile=profile,
                )
                if source_observations_path is not None
                else None
            )
            verify_certification_report(
                report,
                descriptor=evidence.source_adapter,
                source_identity_digest=sha256_json(
                    evidence.identity.model_dump(mode="json")
                ),
                profile=profile,
                required_tier=required_certification_tier,
                trusted_public_keys=trusted_certification_keys,
                execution_inputs_digest=execution_inputs_digest,
            )
            verify_held_out_redaction_report(
                held_out_report,
                decision=evidence.fixtures.held_out,
                evidence_digest=evidence.bundle_digest,
                trusted_public_keys=trusted_certification_keys,
                sensitive_terms=sensitive_terms,
                evidence=evidence.model_dump(mode="json"),
            )
            migration: EvidenceMigrationRecord | None = None
            original_digest = (
                evidence.revision.root_bundle_digest
                if evidence.revision is not None
                else evidence.bundle_digest
            )
            if migration_record_path is not None:
                if source_bundle_path is None:
                    raise BundleError(
                        "a migration record requires the original source bundle"
                    )
                migration = load_migration_record(migration_record_path)
                original_digest = verified_source_digest(source_bundle_path)
                source_document = json.loads(
                    source_bundle_path.resolve().read_text(encoding="utf-8"),
                    object_pairs_hook=_unique_object,
                )
                if not isinstance(source_document, dict):
                    raise BundleError("original source bundle must be a JSON object")
                NormalizedSourceEvidence(
                    source_digest=original_digest,
                    evidence=evidence,
                    migration=migration,
                )
            elif source_bundle_path is not None:
                raise BundleError(
                    "source_bundle_path is invalid without migration_record_path"
                )
            else:
                source_document = None
            return EvidenceView(
                document=evidence.model_dump(mode="json"),
                path=source,
                source_evidence=evidence,
                certification_report=report,
                migration=migration,
                source_digest=original_digest,
                source_document=source_document,
                domain_brief_report=brief_report,
                held_out_redaction_report=held_out_report,
            )
        except (
            CertificationError,
            DomainBriefError,
            EvidenceMigrationError,
            HeldOutError,
            ValueError,
        ) as exc:
            raise BundleError(f"cannot verify v2 evidence {source}: {exc}") from exc
    if version != EVIDENCE_BUNDLE_VERSION:
        raise BundleError(
            f"evidence bundle {source} declares schema_version {version!r}; "
            f"this drafting phase reads {EVIDENCE_BUNDLE_VERSION!r} or "
            f"{SOURCE_EVIDENCE_VERSION!r}"
        )
    for required in ("bundle_digest", "tools", "pack", "vocabulary", "identity"):
        if required not in raw:
            raise BundleError(f"evidence bundle {source} has no {required!r}")
    _verify_digest(raw, source)
    if not raw["tools"]:
        raise BundleError(f"evidence bundle {source} selected no tools to draft against")
    return EvidenceView(document=raw, path=source, source_digest=str(raw["bundle_digest"]))


@dataclass(frozen=True)
class Approval:
    """A human's recorded decision about one exact bundle."""

    approved_by: str
    bundle_digest: str
    acknowledged_findings: tuple[str, ...]
    note: str | None
    approval_version: str = APPROVAL_VERSION
    source_bundle_digest: str | None = None
    migration_record_digest: str | None = None
    acknowledged_warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        if self.approval_version == MIGRATION_APPROVAL_VERSION:
            return {
                "approval_version": self.approval_version,
                "approved_by": self.approved_by,
                "source_bundle_digest": self.source_bundle_digest,
                "normalized_bundle_digest": self.bundle_digest,
                "migration_record_digest": self.migration_record_digest,
                "acknowledged_warnings": list(self.acknowledged_warnings),
                "acknowledged_findings": list(self.acknowledged_findings),
                "note": self.note,
            }
        return {
            "approval_version": self.approval_version,
            "approved_by": self.approved_by,
            "bundle_digest": self.bundle_digest,
            "acknowledged_findings": list(self.acknowledged_findings),
            "note": self.note,
        }


def load_approval(path: Path, bundle: EvidenceView) -> Approval:
    """Load an approval and prove it refers to this bundle and its open flags."""
    if bundle.is_v2:
        if bundle.source_evidence is None or bundle.source_digest is None:
            raise BundleError("v2 evidence view is missing its verified source binding")
        normalized = NormalizedSourceEvidence(
            source_digest=bundle.source_digest,
            evidence=bundle.source_evidence,
            migration=bundle.migration,
        )
        try:
            if bundle.domain_brief_report is None:
                raise BundleError(
                    "v2 evidence view is missing its domain brief report"
                )
            required_findings = frozenset(
                f"{finding.location}:{finding.code}"
                for finding in bundle.domain_brief_report.advisory
            )
            approval = load_normalized_approval(
                path,
                normalized,
                required_findings=required_findings,
            )
        except EvidenceMigrationError as exc:
            raise BundleError(f"cannot verify v2 approval: {exc}") from exc
        return Approval(
            approved_by=approval.approved_by,
            bundle_digest=approval.normalized_bundle_digest,
            note=approval.note,
            approval_version=approval.approval_version,
            source_bundle_digest=approval.source_bundle_digest,
            migration_record_digest=approval.migration_record_digest,
            acknowledged_warnings=approval.acknowledged_warnings,
            acknowledged_findings=approval.acknowledged_findings,
        )
    source = path.resolve()
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read approval {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleError(f"approval must be a JSON object: {source}")
    if raw.get("approval_version") != APPROVAL_VERSION:
        raise BundleError(
            f"approval {source} must declare approval_version {APPROVAL_VERSION!r}"
        )
    approved_by = raw.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise BundleError(f"approval {source} must name who approved the bundle")
    digest = raw.get("bundle_digest")
    if digest != bundle.digest:
        # An approval of a different bundle is the failure this gate exists to catch: it
        # is how reviewed text gets swapped for unreviewed text between the two phases.
        raise BundleError(
            f"approval {source} covers bundle {digest!r}, not {bundle.digest!r}"
        )
    acknowledged = raw.get("acknowledged_findings", [])
    if not isinstance(acknowledged, list) or not all(
        isinstance(item, str) for item in acknowledged
    ):
        raise BundleError(f"approval {source} acknowledged_findings must be strings")
    advisory = {
        f"{finding['location']}:{finding['code']}"
        for finding in bundle.document.get("review", {}).get("advisory", [])
    }
    unacknowledged = sorted(advisory - set(acknowledged))
    if unacknowledged:
        # Every flag a human was asked about has to be answered by name. A blanket
        # approval would let a newly appearing flag ride along on an old decision.
        raise BundleError(
            "approval does not acknowledge every flagged finding: "
            + ", ".join(unacknowledged)
        )
    unknown = sorted(set(acknowledged) - advisory)
    if unknown:
        raise BundleError(
            "approval acknowledges findings this bundle does not contain: "
            + ", ".join(unknown)
        )
    note = raw.get("note")
    if note is not None and not isinstance(note, str):
        raise BundleError(f"approval {source} note must be a string when present")
    return Approval(
        approved_by=approved_by.strip(),
        bundle_digest=bundle.digest,
        acknowledged_findings=tuple(sorted(acknowledged)),
        note=note,
    )
