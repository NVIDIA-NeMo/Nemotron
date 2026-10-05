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

"""What the intake phase did, recorded so a later reviewer can re-derive it.

The drafting phase will add model identity, prompt hashes, and approvals to a record of its
own. This one covers the half that ran before any model existed, and says so explicitly:
`model` is null because nothing was inferred here, not because the field was forgotten. A
provenance record that leaves the distinction to the reader is the one that lets an
unattributed inference through later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.mcp.authoring.evidence import EvidenceBundle
from nemotron.steps.byob.runtime.mcp.authoring.intake import LoadedMcpIntake
from nemotron.steps.byob.runtime.mcp.authoring.pack_artifacts import (
    PENDING_PACK_ARTIFACTS,
    EmittedArtifact,
)
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.errors import McpProtocolError
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import ExposureSubject
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterCertificationReport,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import SourceEvidenceDocument
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutRedactionReport,
)
from nemotron.steps.byob.runtime.source_adapters.migration import (
    EvidenceMigrationRecord,
)

INTAKE_PROVENANCE_VERSION = "bfcl-mcp-intake-provenance-v1"
MCP_INTAKE_ADAPTER_VERSION = "1.0.0"


@dataclass(frozen=True)
class IntakeProvenance:
    document: dict[str, Any]

    def verify_digest(self) -> None:
        claimed = self.document.get("record_digest")
        unsigned = {
            key: value for key, value in self.document.items() if key != "record_digest"
        }
        observed = sha256_json(unsigned)
        if claimed != observed:
            raise McpProtocolError(
                "intake provenance was modified after record_digest was computed: "
                f"claimed {claimed!r}, observed {observed!r}"
            )


def build_intake_provenance(
    intake: LoadedMcpIntake,
    report: DiscoveryReport,
    bundle: EvidenceBundle,
    artifacts: list[EmittedArtifact],
    *,
    output_root: Path,
    evidence_path: Path,
    attestation_path: Path,
    attestation_document: dict[str, Any],
    source_evidence: SourceEvidenceDocument | None = None,
    legacy_evidence_path: Path | None = None,
    certification: AdapterCertificationReport | None = None,
    certification_path: Path | None = None,
    migration: EvidenceMigrationRecord | None = None,
    migration_path: Path | None = None,
    domain_brief_source_path: Path | None = None,
    domain_brief_report_path: Path | None = None,
    held_out_redaction: HeldOutRedactionReport | None = None,
    held_out_redaction_path: Path | None = None,
    exposure_subject: ExposureSubject | None = None,
    exposure_subject_path: Path | None = None,
    observations_document: dict[str, Any] | None = None,
    observations_path: Path | None = None,
    resolved_authoring_config_digest: str | None = None,
) -> IntakeProvenance:
    """Record the inputs, the outputs, and what remains unauthored."""
    bundle.verify_digest()
    report.verify_digest()
    root = output_root.resolve()
    identity = bundle.document["identity"]
    document: dict[str, Any] = {
        "schema_version": INTAKE_PROVENANCE_VERSION,
        "phase": "intake",
        "attained_level": "L0",
        "adapter": {
            "name": "nemotron-bfcl-mcp-intake",
            "version": MCP_INTAKE_ADAPTER_VERSION,
        },
        "pack": dict(bundle.document["pack"]),
        "mode": bundle.document["mode"],
        # Digests rather than absolute paths: the same reviewed inputs must produce the
        # same record on any host, and a path is not evidence of content.
        "inputs": {
            "intake_config_digest": identity["intake_config_digest"],
            "mcp_oracle_config_digest": identity["source_config_digest"],
            "discovery_report_digest": identity["discovery_report_digest"],
        },
        "identity": dict(identity),
        "oracle": dict(bundle.document["oracle"]),
        "evidence_bundle": {
            "path": evidence_path.resolve().relative_to(root).as_posix(),
            "digest": bundle.bundle_digest,
        },
        "gateway_attestation": {
            "path": attestation_path.resolve().relative_to(root).as_posix(),
            "digest": sha256_json(attestation_document),
        },
        "artifacts": [artifact.as_dict(root=root) for artifact in artifacts],
        "excluded_tools": list(bundle.document["catalog"]["exclusions"]),
        "review": {
            "status": bundle.document["status"],
            "advisory_findings": list(bundle.document["review"]["advisory"]),
            # Filled by the human act, not by this run.
            "approvals": [],
        },
        # No model read anything in this phase. The drafting phase records its own.
        "model": None,
        "pending_artifacts": list(PENDING_PACK_ARTIFACTS),
    }
    if resolved_authoring_config_digest is not None:
        document["inputs"]["resolved_authoring_config_digest"] = (
            resolved_authoring_config_digest
        )
    if source_evidence is not None:
        if (
            legacy_evidence_path is None
            or certification is None
            or certification_path is None
            or migration is None
            or migration_path is None
            or domain_brief_source_path is None
            or domain_brief_report_path is None
            or held_out_redaction is None
            or held_out_redaction_path is None
            or exposure_subject is None
            or exposure_subject_path is None
            or observations_document is None
            or observations_path is None
        ):
            raise ValueError("v2 intake provenance requires every trust sidecar")
        document["attained_level"] = source_evidence.certification.attained_tier
        document["evidence_bundle"] = {
            "path": evidence_path.resolve().relative_to(root).as_posix(),
            "digest": source_evidence.bundle_digest,
            "schema_version": source_evidence.schema_version,
        }
        document["source_evidence_bundle"] = {
            "path": legacy_evidence_path.resolve().relative_to(root).as_posix(),
            "digest": bundle.bundle_digest,
        }
        document["certification"] = {
            "path": certification_path.resolve().relative_to(root).as_posix(),
            "report_digest": certification.report_digest,
            "signing_key_id": certification.signing_key_id,
        }
        document["migration"] = {
            "path": migration_path.resolve().relative_to(root).as_posix(),
            "record_digest": migration.record_digest,
        }
        document["domain_brief"] = {
            "source_path": domain_brief_source_path.resolve()
            .relative_to(root)
            .as_posix(),
            "report_path": domain_brief_report_path.resolve()
            .relative_to(root)
            .as_posix(),
            "content_digest": source_evidence.domain_brief.content_digest,
            "redaction_report_digest": (
                source_evidence.domain_brief.redaction_report_digest
            ),
        }
        document["held_out_redaction"] = {
            "path": held_out_redaction_path.resolve()
            .relative_to(root)
            .as_posix(),
            "report_digest": held_out_redaction.report_digest,
            "decision_digest": held_out_redaction.decision_digest,
            "terms_commitment_digest": (
                held_out_redaction.terms_commitment_digest
            ),
        }
        document["model_exposure_subject"] = {
            "path": exposure_subject_path.resolve().relative_to(root).as_posix(),
            "digest": sha256_json(exposure_subject.model_dump(mode="json")),
            "evidence_digest": exposure_subject.evidence_digest,
        }
        document["source_observations"] = {
            "path": observations_path.resolve().relative_to(root).as_posix(),
            "digest": sha256_json(observations_document),
            "profile_id": certification.profile_id,
        }
    document["record_digest"] = sha256_json(document)
    return IntakeProvenance(document=document)


def write_intake_provenance(provenance: IntakeProvenance, path: Path) -> Path:
    provenance.verify_digest()
    return Path(write_canonical_json(provenance.document, path))
