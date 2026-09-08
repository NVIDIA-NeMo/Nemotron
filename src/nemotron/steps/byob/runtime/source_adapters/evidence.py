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

"""Strict transport-neutral evidence for BFCL assisted authoring.

The bundle stores source observations and a reference to independent
certification.  It does not make an adapter authoritative: the referenced report
must later be loaded and verified by BFCL before a release can use its tier.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
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
from nemotron.steps.byob.runtime.source_adapters.domain_brief import DomainBriefEvidence
from nemotron.steps.byob.runtime.source_adapters.held_out import HeldOutDecision

SOURCE_EVIDENCE_VERSION: Literal[
    "bfcl-source-evidence-v2"
] = "bfcl-source-evidence-v2"
CERTIFICATION_REFERENCE_VERSION: Literal[
    "bfcl-adapter-certification-reference-v1"
] = "bfcl-adapter-certification-reference-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class SourceEvidenceError(ValueError):
    """Raised when source evidence is incomplete, ambiguous, or modified."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_digest(value: str, field: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_safe_name(value: str, field: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} must be a safe lowercase identifier")
    return value


class PackIdentity(_StrictModel):
    pack_id: StrictStr
    version: StrictStr

    @field_validator("pack_id")
    @classmethod
    def _pack_id(cls, value: str) -> str:
        return _require_safe_name(value, "pack_id")

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pack version must be non-empty")
        return value


class IdentityArtifact(_StrictModel):
    role: StrictStr
    digest: StrictStr

    @field_validator("role")
    @classmethod
    def _role(cls, value: str) -> str:
        return _require_safe_name(value, "identity artifact role")

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "identity artifact digest")


class SourceIdentity(_StrictModel):
    subject: StrictStr
    effective_content_digest: StrictStr
    source_config_digest: StrictStr
    artifacts: tuple[IdentityArtifact, ...] = ()

    @field_validator("subject")
    @classmethod
    def _subject(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity subject must be non-empty")
        return value

    @field_validator("effective_content_digest", "source_config_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "identity digest")

    @field_validator("artifacts")
    @classmethod
    def _canonical_artifacts(
        cls,
        value: tuple[IdentityArtifact, ...],
    ) -> tuple[IdentityArtifact, ...]:
        roles = [item.role for item in value]
        if len(roles) != len(set(roles)):
            raise ValueError("identity artifact roles must be unique")
        if roles != sorted(roles):
            raise ValueError("identity artifacts must be sorted by role")
        return value


class CertificationReference(_StrictModel):
    """Pointer to a BFCL-issued report; not a certification report itself."""

    reference_version: Literal["bfcl-adapter-certification-reference-v1"]
    report_schema_version: StrictStr
    report_digest: StrictStr
    descriptor_digest: StrictStr
    issuer: StrictStr
    profile_id: StrictStr
    attained_tier: Literal["A0", "A1", "A2"]

    @field_validator("report_schema_version", "issuer", "profile_id")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("certification reference strings must be non-empty")
        return value

    @field_validator("report_digest", "descriptor_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "certification digest")


class CapabilityEvidence(_StrictModel):
    capability: AdapterCapability
    status: Literal["declared", "observed", "unavailable"]
    evidence_digests: tuple[StrictStr, ...] = ()
    reason: StrictStr | None = None

    @field_validator("evidence_digests")
    @classmethod
    def _digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for digest in value:
            _require_digest(digest, "capability evidence digest")
        if len(value) != len(set(value)):
            raise ValueError("capability evidence digests must be unique")
        if tuple(sorted(value)) != value:
            raise ValueError("capability evidence digests must be sorted")
        return value

    @model_validator(mode="after")
    def _status_contract(self) -> CapabilityEvidence:
        if self.status == "observed" and not self.evidence_digests:
            raise ValueError("observed capability evidence requires at least one digest")
        if self.status == "unavailable" and not (self.reason and self.reason.strip()):
            raise ValueError("unavailable capability evidence requires a reason")
        if self.status != "unavailable" and self.reason is not None:
            raise ValueError("only unavailable capability evidence may carry a reason")
        return self


class UntrustedText(_StrictModel):
    untrusted_text: StrictStr


class ToolEvidence(_StrictModel):
    published_name: StrictStr
    source_name: StrictStr
    description: UntrustedText
    parameter_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    mutates: StrictBool
    requires_confirmation: StrictBool
    raw_digest: StrictStr

    @field_validator("published_name", "source_name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _require_safe_name(value, "tool name")

    @field_validator("raw_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "tool raw digest")


class ConfirmationVocabulary(_StrictModel):
    parameter: StrictStr | None = None
    status_field: StrictStr | None = None
    pending_status: StrictStr | None = None
    error_path: StrictStr | None = None


class FixtureEvidence(_StrictModel):
    direction: Literal["none", "read_only", "pushed", "snapshot"]
    content_digest: StrictStr | None = None
    held_out: HeldOutDecision

    @field_validator("content_digest")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is not None:
            _require_digest(value, "fixture content digest")
        return value

    @model_validator(mode="after")
    def _content_contract(self) -> FixtureEvidence:
        if self.direction == "none" and self.content_digest is not None:
            raise ValueError("fixture direction none cannot carry a content digest")
        return self


class UnresolvedGap(_StrictModel):
    code: StrictStr
    field: StrictStr
    reason: StrictStr
    evidence_refs: tuple[StrictStr, ...] = ()

    @field_validator("code", "field")
    @classmethod
    def _name(cls, value: str) -> str:
        return _require_safe_name(value, "unresolved gap identifier")

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("unresolved gap reason must be non-empty")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def _canonical_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("unresolved gap evidence references must be unique")
        if tuple(sorted(value)) != value:
            raise ValueError("unresolved gap evidence references must be sorted")
        return value


SemanticAnswerValue = StrictBool | StrictInt | StrictFloat | StrictStr
_FORBIDDEN_SEMANTIC_SEGMENTS = frozenset(
    {
        "adapter",
        "attained_tier",
        "bundle_digest",
        "certification",
        "identity",
        "signature",
        "source_adapter",
    }
)


def validate_semantic_target(value: str) -> str:
    if (
        not value.startswith("/semantic/")
        or len(value) > 512
        or any(character.isspace() for character in value)
    ):
        raise ValueError("semantic target must be a bounded /semantic/ path")
    segments = {segment.casefold() for segment in value.split("/")}
    if segments & _FORBIDDEN_SEMANTIC_SEGMENTS:
        raise ValueError("semantic target cannot name an authority field")
    return value


class SemanticAnswer(_StrictModel):
    question_id: StrictStr
    question_digest: StrictStr
    target_path: StrictStr
    value: SemanticAnswerValue
    evidence_refs: tuple[StrictStr, ...]

    @field_validator("question_id")
    @classmethod
    def _question_id(cls, value: str) -> str:
        if not re.fullmatch(r"q_[0-9a-f]{24}", value):
            raise ValueError("semantic answer has an invalid question identity")
        return value

    @field_validator("question_digest")
    @classmethod
    def _question_digest(cls, value: str) -> str:
        return _require_digest(value, "semantic answer question digest")

    @field_validator("target_path")
    @classmethod
    def _target(cls, value: str) -> str:
        return validate_semantic_target(value)


class EvidenceRevisionLink(_StrictModel):
    root_bundle_digest: StrictStr
    parent_bundle_digest: StrictStr
    question_artifact_digest: StrictStr
    answer_set_digest: StrictStr

    @field_validator(
        "parent_bundle_digest",
        "root_bundle_digest",
        "question_artifact_digest",
        "answer_set_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "evidence revision digest")


class UnsignedSourceEvidence(_StrictModel):
    """All model-visible source evidence except the bundle's own digest."""

    schema_version: Literal["bfcl-source-evidence-v2"]
    source_adapter: AdapterDescriptor
    certification: CertificationReference
    pack: PackIdentity
    domain_brief: DomainBriefEvidence
    identity: SourceIdentity
    capabilities: tuple[CapabilityEvidence, ...]
    vocabulary: ConfirmationVocabulary
    fixtures: FixtureEvidence
    tools: tuple[ToolEvidence, ...]
    unresolved_gaps: tuple[UnresolvedGap, ...] = ()
    semantic_answers: tuple[SemanticAnswer, ...] = ()
    revision: EvidenceRevisionLink | None = None

    @field_validator("capabilities")
    @classmethod
    def _canonical_capabilities(
        cls,
        value: tuple[CapabilityEvidence, ...],
    ) -> tuple[CapabilityEvidence, ...]:
        names = [item.capability.value for item in value]
        if len(names) != len(set(names)):
            raise ValueError("capability evidence must contain unique capabilities")
        if names != sorted(names):
            raise ValueError("capability evidence must be sorted by capability name")
        return value

    @field_validator("tools")
    @classmethod
    def _canonical_tools(cls, value: tuple[ToolEvidence, ...]) -> tuple[ToolEvidence, ...]:
        if not value:
            raise ValueError("source evidence must select at least one tool")
        names = [item.published_name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("published tool names must be unique")
        if names != sorted(names):
            raise ValueError("tool evidence must be sorted by published name")
        return value

    @field_validator("unresolved_gaps")
    @classmethod
    def _canonical_gaps(
        cls,
        value: tuple[UnresolvedGap, ...],
    ) -> tuple[UnresolvedGap, ...]:
        keys = [(item.code, item.field) for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("unresolved gaps must have unique code and field pairs")
        if keys != sorted(keys):
            raise ValueError("unresolved gaps must be sorted by code and field")
        return value

    @field_validator("semantic_answers")
    @classmethod
    def _canonical_answers(
        cls,
        value: tuple[SemanticAnswer, ...],
    ) -> tuple[SemanticAnswer, ...]:
        identities = [item.question_id for item in value]
        targets = [item.target_path for item in value]
        if identities != sorted(set(identities)):
            raise ValueError("semantic answers must have sorted unique question identities")
        if len(targets) != len(set(targets)):
            raise ValueError("semantic answers cannot write one target more than once")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> UnsignedSourceEvidence:
        descriptor_digest = sha256_json(self.source_adapter.model_dump(mode="json"))
        if self.certification.descriptor_digest != descriptor_digest:
            raise ValueError(
                "certification reference does not cover this adapter descriptor"
            )
        declared = tuple(item.value for item in self.source_adapter.capabilities)
        evidenced = tuple(item.capability.value for item in self.capabilities)
        if declared != evidenced:
            raise ValueError(
                "capability evidence must exactly match the adapter descriptor"
            )
        if bool(self.semantic_answers) != (self.revision is not None):
            raise ValueError(
                "semantic answers and evidence revision lineage must appear together"
            )
        return self


class SourceEvidenceDocument(UnsignedSourceEvidence):
    """A complete evidence document whose digest covers every other field."""

    bundle_digest: StrictStr

    @field_validator("bundle_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value, "bundle digest")

    @model_validator(mode="after")
    def _verify_bundle_digest(self) -> SourceEvidenceDocument:
        unsigned = self.model_dump(mode="json", exclude={"bundle_digest"})
        observed = sha256_json(unsigned)
        if self.bundle_digest != observed:
            raise ValueError(
                "source evidence was modified after bundle_digest was computed"
            )
        return self


def build_source_evidence(evidence: UnsignedSourceEvidence) -> SourceEvidenceDocument:
    """Bind a canonical digest to validated unsigned evidence."""

    document = evidence.model_dump(mode="json")
    document["bundle_digest"] = sha256_json(document)
    return SourceEvidenceDocument.model_validate(document)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise SourceEvidenceError(f"source evidence repeats JSON key {key!r}")
        document[key] = value
    return document


def load_source_evidence(path: Path) -> SourceEvidenceDocument:
    """Read strict v2 evidence without accepting duplicate JSON keys."""

    source = path.resolve()
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        return SourceEvidenceDocument.model_validate(document)
    except SourceEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SourceEvidenceError(f"cannot load source evidence {source}: {exc}") from exc


def write_source_evidence(
    evidence: SourceEvidenceDocument,
    path: Path,
) -> Path:
    """Write the exact canonical representation covered by ``bundle_digest``."""

    SourceEvidenceDocument.model_validate(evidence.model_dump(mode="json"))
    return write_canonical_json(evidence.model_dump(mode="json"), path)
