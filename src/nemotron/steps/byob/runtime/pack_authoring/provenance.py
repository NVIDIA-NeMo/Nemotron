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

"""What the drafting phase inferred, from which evidence, under whose approval.

The intake record says no model was involved. This one says exactly which model was, which
prompt version it ran under, which request hash each answer is filed against, and which human
approved the evidence it read. Together they cover the whole path from a server's catalog to
a pack draft, which is what makes a later reviewer able to disagree with a specific step
instead of distrusting the artifact as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.authoring_workflow.quota import RunQuotaSnapshot
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    ExposureAuthorization,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import Approval, EvidenceView
from nemotron.steps.byob.runtime.pack_authoring.drafts import DraftBundle
from nemotron.steps.byob.runtime.pack_authoring.model_client import AuthoringModel

DRAFT_PROVENANCE_VERSION = "bfcl-authoring-draft-provenance-v1"
DRAFTING_ADAPTER_VERSION = "1.0.0"


class ProvenanceError(Exception):
    """Raised when a provenance record does not describe the artifacts beside it."""


@dataclass(frozen=True)
class DraftProvenance:
    document: dict[str, Any]

    def verify_digest(self) -> None:
        claimed = self.document.get("record_digest")
        unsigned = {
            key: value for key, value in self.document.items() if key != "record_digest"
        }
        observed = sha256_json(unsigned)
        if claimed != observed:
            raise ProvenanceError(
                "draft provenance was modified after record_digest was computed: "
                f"claimed {claimed!r}, observed {observed!r}"
            )


def build_draft_provenance(
    evidence: EvidenceView,
    approval: Approval,
    model: AuthoringModel,
    drafts: DraftBundle,
    *,
    assertions_compiled: bool,
    compilation_refusals: tuple[str, ...] = (),
    exposure_authorization: ExposureAuthorization | None = None,
    quota_snapshot: RunQuotaSnapshot | None = None,
    resolved_authoring_config_digest: str | None = None,
) -> DraftProvenance:
    """Record the drafting phase and what it still could not produce."""
    documents = drafts.as_documents()
    document: dict[str, Any] = {
        "schema_version": DRAFT_PROVENANCE_VERSION,
        "phase": "drafting",
        "adapter": {
            "name": "nemotron-bfcl-pack-authoring",
            "version": DRAFTING_ADAPTER_VERSION,
        },
        "pack": dict(evidence.document["pack"]),
        "evidence": {
            "schema_version": evidence.document["schema_version"],
            "bundle_digest": evidence.digest,
            "source_bundle_digest": evidence.source_digest,
            "migration_record_digest": (
                evidence.migration.record_digest
                if evidence.migration is not None
                else None
            ),
            "certification": {
                "tier": evidence.certification_tier,
                "bfcl_verified": evidence.certification_verified,
                "legacy_level": evidence.legacy_level,
                "report_digest": (
                    evidence.certification_report.report_digest
                    if evidence.certification_report is not None
                    else None
                ),
                "profile_id": (
                    evidence.certification_report.profile_id
                    if evidence.certification_report is not None
                    else None
                ),
            },
            "domain_brief_digest": (
                evidence.document["domain_brief"]["content_digest"]
                if evidence.is_v2
                else None
            ),
            "held_out_redaction_report_digest": (
                evidence.held_out_redaction_report.report_digest
                if evidence.held_out_redaction_report is not None
                else None
            ),
            "unresolved_unknowns": sorted(evidence.unresolved_unknowns),
        },
        "approval": approval.as_dict(),
        "model_exposure_authorization": (
            exposure_authorization.model_dump(mode="json")
            if exposure_authorization is not None
            else None
        ),
        "model": model.as_provenance(),
        "calls": [record.as_dict() for record in drafts.calls],
        "run_quota": (
            quota_snapshot.model_dump(mode="json")
            if quota_snapshot is not None
            else None
        ),
        # Each artifact is digested so a later edit to a draft is visible rather than
        # inheriting the trust of the run that generated it.
        "artifact_digests": {
            name: sha256_json(value) for name, value in sorted(documents.items())
        },
        "assertions_compiled": assertions_compiled,
        "compilation_refusals": list(compilation_refusals),
        "blocked_on": sorted(
            {
                field
                for case in drafts.validation_cases.cases
                for field in case.blocked_on
            }
            | {
                field
                for template in drafts.task_templates.templates
                for field in template.blocked_on
            }
            | {
                field
                for spec in drafts.assertions.assertions
                for field in spec.blocked_on
            }
        ),
    }
    if resolved_authoring_config_digest is not None:
        document["resolved_authoring_config_digest"] = (
            resolved_authoring_config_digest
        )
    document["record_digest"] = sha256_json(document)
    return DraftProvenance(document=document)


def write_draft_provenance(provenance: DraftProvenance, path: Path) -> Path:
    provenance.verify_digest()
    return write_canonical_json(provenance.document, path)
