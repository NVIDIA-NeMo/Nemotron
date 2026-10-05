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

"""Transport-neutral certification, evidence, and atomic intake publication."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_validator

from nemotron.steps.byob.runtime.authoring_workflow.credentials import CredentialResolver
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    ExposureSubject,
    build_exposure_subject,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterCertificationReport,
    AdapterTier,
    CertificationAuthority,
    CertificationProfile,
    ProbeOutcome,
    build_certification_report,
    certification_input_digest,
    certification_reference,
    http_package_reference_profile,
    local_python_reference_profile,
    project_probe_executions,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    AdapterDescriptor,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import (
    DomainBriefEvidence,
    DomainBriefRedactionReport,
    load_domain_brief,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    SOURCE_EVIDENCE_VERSION,
    CapabilityEvidence,
    ConfirmationVocabulary,
    FixtureEvidence,
    PackIdentity,
    SourceEvidenceDocument,
    SourceIdentity,
    ToolEvidence,
    UnresolvedGap,
    UnsignedSourceEvidence,
    build_source_evidence,
    write_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutDecision,
    HeldOutRedactionReport,
    build_held_out_redaction_report,
    load_held_out_sensitive_terms,
    load_required_held_out_policy,
)
from nemotron.steps.byob.runtime.source_adapters.http_package import (
    HttpClientFactory,
    inspect_http_package,
)
from nemotron.steps.byob.runtime.source_adapters.http_package_probes import (
    run_http_package_probes,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    inspect_local_python_package,
)
from nemotron.steps.byob.runtime.source_adapters.local_python_probes import (
    run_local_python_probes,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import (
    AdapterProbePlan,
)
from nemotron.steps.byob.runtime.source_adapters.registry import (
    ResolvedAdapter,
    SourceDeclaration,
    resolve_source_adapter,
)

INTAKE_RECORD_VERSION: Literal[
    "bfcl-source-intake-record-v1"
] = "bfcl-source-intake-record-v1"
EVIDENCE_FILE_NAME = "evidence_bundle.json"
CERTIFICATION_FILE_NAME = "adapter_certification.json"
DOMAIN_BRIEF_SOURCE_FILE_NAME = "domain_brief.source.txt"
DOMAIN_BRIEF_REPORT_FILE_NAME = "domain_brief_redaction.json"
HELD_OUT_REDACTION_FILE_NAME = "held_out_redaction.json"
EXPOSURE_SUBJECT_FILE_NAME = "model_exposure_subject.json"
OBSERVATIONS_FILE_NAME = "source_observations.json"
PROVENANCE_FILE_NAME = "intake_provenance.json"
_DIGEST_PREFIX = "sha256:"


class SourceIntakeError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceIntakeRecord(_StrictModel):
    schema_version: Literal["bfcl-source-intake-record-v1"]
    declaration_digest: StrictStr
    adapter_kind: StrictStr
    profile_id: StrictStr
    profile_digest: StrictStr
    source_identity_digest: StrictStr
    certification_report_digest: StrictStr
    evidence_digest: StrictStr
    domain_brief_report_digest: StrictStr
    held_out_redaction_report_digest: StrictStr
    exposure_subject_digest: StrictStr
    observations_digest: StrictStr
    resolved_authoring_config_digest: StrictStr | None = None
    record_digest: StrictStr

    @field_validator(
        "declaration_digest",
        "profile_digest",
        "source_identity_digest",
        "certification_report_digest",
        "evidence_digest",
        "domain_brief_report_digest",
        "held_out_redaction_report_digest",
        "exposure_subject_digest",
        "observations_digest",
        "record_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        if not value.startswith(_DIGEST_PREFIX) or len(value) != 71:
            raise ValueError("intake record digests must be lowercase SHA-256")
        try:
            int(value.removeprefix(_DIGEST_PREFIX), 16)
        except ValueError as exc:
            raise ValueError("intake record digest is not hexadecimal") from exc
        return value

    @model_validator(mode="after")
    def _record_digest(self) -> SourceIntakeRecord:
        unsigned = self.model_dump(
            mode="json",
            exclude={"record_digest"},
            exclude_none=True,
        )
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("source intake record digest mismatch")
        return self

    @field_validator("resolved_authoring_config_digest")
    @classmethod
    def _optional_digest(cls, value: str | None) -> str | None:
        if value is not None:
            cls._digest(value)
        return value


OutcomeProjector = Callable[[str], Sequence[ProbeOutcome]]
EvidenceFactory = Callable[
    [
        AdapterCertificationReport,
        DomainBriefEvidence,
        DomainBriefRedactionReport,
        HeldOutDecision,
    ],
    SourceEvidenceDocument,
]


@dataclass(frozen=True)
class FinalizedSourceIntake:
    certification: AdapterCertificationReport
    evidence: SourceEvidenceDocument
    domain_brief: DomainBriefEvidence
    domain_brief_report: DomainBriefRedactionReport
    held_out_redaction: HeldOutRedactionReport
    exposure_subject: ExposureSubject
    observations_document: dict[str, Any]


@dataclass(frozen=True)
class SourceCollection:
    resolved: ResolvedAdapter
    descriptor: AdapterDescriptor
    profile: CertificationProfile
    identity: SourceIdentity
    pack: PackIdentity
    tools: tuple[ToolEvidence, ...]
    fixture_direction: Literal["none", "read_only", "pushed", "snapshot"]
    fixture_content_digest: str | None
    vocabulary: ConfirmationVocabulary
    project_outcomes: OutcomeProjector
    execution_inputs_digest: str | None = None

    @property
    def source_identity_digest(self) -> str:
        return sha256_json(self.identity.model_dump(mode="json"))


@dataclass(frozen=True)
class SourceIntakeResult:
    output_root: Path
    collection: SourceCollection
    finalized: FinalizedSourceIntake
    provenance: SourceIntakeRecord

    @property
    def evidence_path(self) -> Path:
        return self.output_root / EVIDENCE_FILE_NAME


def validate_held_out_inputs(
    decision: HeldOutDecision,
    *,
    policy_path: Path | None,
    content_path: Path | None,
) -> tuple[str, ...]:
    if decision.status == "required":
        if policy_path is None:
            raise SourceIntakeError(
                "held_out_policy_missing",
                "required held-out decision needs its reviewed policy",
            )
        observed = load_required_held_out_policy(
            policy_path,
            reviewed_by=decision.reviewed_by,
        )
        if observed != decision:
            raise SourceIntakeError(
                "held_out_policy_mismatch",
                "held-out policy does not match the reviewed decision",
            )
        return load_held_out_sensitive_terms(
            policy_path,
            content_path=content_path,
        )
    if policy_path is not None or content_path is not None:
        raise SourceIntakeError(
            "held_out_policy_unexpected",
            "not-applicable held-out decision cannot carry policy content",
        )
    return ()


def _capability_evidence(
    descriptor: AdapterDescriptor,
    report: AdapterCertificationReport,
) -> tuple[CapabilityEvidence, ...]:
    observed = {
        AdapterCapability.DESCRIBE_TOOLS,
        AdapterCapability.PIN_IDENTITY,
    }
    if report.attained_tier in {AdapterTier.A1, AdapterTier.A2}:
        observed.add(AdapterCapability.OBSERVE)
    if report.attained_tier is AdapterTier.A2:
        observed.update(
            {
                AdapterCapability.DESCRIBE_STATE,
                AdapterCapability.GET_STATE,
                AdapterCapability.RESET_STATE,
            }
        )
    return tuple(
        CapabilityEvidence(
            capability=capability,
            status="observed" if capability in observed else "declared",
            evidence_digests=(
                (report.report_digest,) if capability in observed else ()
            ),
        )
        for capability in descriptor.capabilities
    )


def _unresolved_gaps(tier: AdapterTier) -> tuple[UnresolvedGap, ...]:
    gaps: list[UnresolvedGap] = []
    if tier is AdapterTier.A0:
        gaps.extend(
            [
                UnresolvedGap(
                    code="observed_error_codes",
                    field="tools",
                    reason="No A1 structured-error observation was certified.",
                ),
                UnresolvedGap(
                    code="observed_result_shapes",
                    field="tools",
                    reason="No A1 successful result shape was certified.",
                ),
            ]
        )
    if tier in {AdapterTier.A0, AdapterTier.A1}:
        gaps.extend(
            [
                UnresolvedGap(
                    code="confirmation_behavior",
                    field="vocabulary",
                    reason="No A2 confirmation-safety behavior was certified.",
                ),
                UnresolvedGap(
                    code="reset_isolation",
                    field="capabilities",
                    reason="No A2 reset and episode-isolation behavior was certified.",
                ),
            ]
        )
    return tuple(sorted(gaps, key=lambda item: (item.code, item.field)))


def finalize_source_intake(
    *,
    descriptor: AdapterDescriptor,
    identity: SourceIdentity,
    profile: CertificationProfile,
    project_outcomes: OutcomeProjector,
    execution_inputs_digest: str | None,
    certification_authority: CertificationAuthority,
    required_tier: AdapterTier,
    domain_brief_path: Path,
    domain_brief_language: str,
    domain_brief_redactions: Mapping[str, str] | None,
    held_out_decision: HeldOutDecision,
    held_out_sensitive_terms: Sequence[str],
    evidence_factory: EvidenceFactory,
    resolved_authoring_config_digest: str | None = None,
) -> FinalizedSourceIntake:
    """Run the shared trust spine used by every transport."""
    brief, brief_report = load_domain_brief(
        domain_brief_path,
        language=domain_brief_language,
        redactions=domain_brief_redactions,
    )
    source_identity_digest = sha256_json(identity.model_dump(mode="json"))
    probe_input = certification_input_digest(
        descriptor,
        source_identity_digest=source_identity_digest,
        profile=profile,
        execution_inputs_digest=execution_inputs_digest,
    )
    outcomes = tuple(project_outcomes(probe_input))
    certification = build_certification_report(
        descriptor,
        source_identity_digest=source_identity_digest,
        profile=profile,
        outcomes=outcomes,
        authority=certification_authority,
        execution_inputs_digest=execution_inputs_digest,
    )
    tier_order = {
        AdapterTier.NONE: 0,
        AdapterTier.A0: 1,
        AdapterTier.A1: 2,
        AdapterTier.A2: 3,
    }
    if tier_order[certification.attained_tier] < tier_order[required_tier]:
        raise SourceIntakeError(
            "adapter_under_certified",
            f"adapter attained {certification.attained_tier.value}, "
            f"below required {required_tier.value}",
        )
    evidence = evidence_factory(
        certification,
        brief,
        brief_report,
        held_out_decision,
    )
    expected_reference = certification_reference(certification)
    if (
        evidence.source_adapter != descriptor
        or evidence.certification != expected_reference
        or evidence.identity != identity
        or evidence.domain_brief != brief
        or evidence.fixtures.held_out != held_out_decision
    ):
        raise SourceIntakeError(
            "evidence_binding_mismatch",
            "evidence factory changed a verifier-owned intake input",
        )
    if evidence.revision is not None or evidence.semantic_answers:
        raise SourceIntakeError(
            "evidence_revision_unexpected",
            "initial intake evidence cannot contain answer revision lineage",
        )
    held_out_redaction = build_held_out_redaction_report(
        evidence.model_dump(mode="json"),
        decision=held_out_decision,
        sensitive_terms=held_out_sensitive_terms,
        authority=certification_authority,
    )
    exposure_subject = build_exposure_subject(
        evidence,
        domain_brief_report=brief_report,
        held_out_redaction_report=held_out_redaction,
        resolved_authoring_config_digest=resolved_authoring_config_digest,
    )
    observations_document = {
        "schema_version": "bfcl-source-observations-v1",
        "profile_id": profile.profile_id,
        "profile_digest": profile.digest,
        "execution_inputs_digest": execution_inputs_digest,
        "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
    }
    observations_document["document_digest"] = sha256_json(observations_document)
    return FinalizedSourceIntake(
        certification=certification,
        evidence=evidence,
        domain_brief=brief,
        domain_brief_report=brief_report,
        held_out_redaction=held_out_redaction,
        exposure_subject=exposure_subject,
        observations_document=observations_document,
    )


def _conventional_evidence_factory(
    collection: SourceCollection,
) -> EvidenceFactory:
    def build(
        report: AdapterCertificationReport,
        brief: DomainBriefEvidence,
        _brief_report: DomainBriefRedactionReport,
        held_out: HeldOutDecision,
    ) -> SourceEvidenceDocument:
        # The report is already signed and digest-bound before its reference enters
        # model-visible evidence.
        reference = certification_reference(report)
        attained = report.attained_tier
        unsigned = UnsignedSourceEvidence(
            schema_version=SOURCE_EVIDENCE_VERSION,
            source_adapter=collection.descriptor,
            certification=reference,
            pack=collection.pack,
            domain_brief=brief,
            identity=collection.identity,
            capabilities=_capability_evidence(
                collection.descriptor,
                report,
            ),
            vocabulary=collection.vocabulary,
            fixtures=FixtureEvidence(
                direction=collection.fixture_direction,
                content_digest=collection.fixture_content_digest,
                held_out=held_out,
            ),
            tools=collection.tools,
            unresolved_gaps=_unresolved_gaps(attained),
        )
        return build_source_evidence(unsigned)

    return build


def _write_atomic_intake(
    final_root: Path,
    *,
    declaration: SourceDeclaration,
    collection: SourceCollection,
    finalized: FinalizedSourceIntake,
    domain_brief_path: Path,
    resolved_authoring_config_digest: str | None = None,
) -> SourceIntakeRecord:
    if final_root.exists():
        raise FileExistsError(f"intake output already exists: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(
        dir=final_root.parent,
        prefix=f".{final_root.name}.staging-",
    )
    root = Path(staging.name)
    try:
        write_source_evidence(finalized.evidence, root / EVIDENCE_FILE_NAME)
        write_canonical_json(
            finalized.certification.model_dump(mode="json"),
            root / CERTIFICATION_FILE_NAME,
        )
        (root / DOMAIN_BRIEF_SOURCE_FILE_NAME).write_bytes(
            domain_brief_path.resolve().read_bytes()
        )
        write_canonical_json(
            finalized.domain_brief_report.model_dump(mode="json"),
            root / DOMAIN_BRIEF_REPORT_FILE_NAME,
        )
        write_canonical_json(
            finalized.held_out_redaction.model_dump(mode="json"),
            root / HELD_OUT_REDACTION_FILE_NAME,
        )
        write_canonical_json(
            finalized.exposure_subject.model_dump(mode="json"),
            root / EXPOSURE_SUBJECT_FILE_NAME,
        )
        write_canonical_json(
            finalized.observations_document,
            root / OBSERVATIONS_FILE_NAME,
        )
        record_document: dict[str, Any] = {
            "schema_version": INTAKE_RECORD_VERSION,
            "declaration_digest": declaration.digest,
            "adapter_kind": collection.descriptor.kind,
            "profile_id": collection.profile.profile_id,
            "profile_digest": collection.profile.digest,
            "source_identity_digest": collection.source_identity_digest,
            "certification_report_digest": finalized.certification.report_digest,
            "evidence_digest": finalized.evidence.bundle_digest,
            "domain_brief_report_digest": finalized.domain_brief_report.record_digest,
            "held_out_redaction_report_digest": (
                finalized.held_out_redaction.report_digest
            ),
            "exposure_subject_digest": sha256_json(
                finalized.exposure_subject.model_dump(mode="json")
            ),
            "observations_digest": finalized.observations_document["document_digest"],
        }
        if resolved_authoring_config_digest is not None:
            record_document["resolved_authoring_config_digest"] = (
                resolved_authoring_config_digest
            )
        record_document["record_digest"] = sha256_json(record_document)
        record = SourceIntakeRecord.model_validate(record_document)
        write_canonical_json(
            record.model_dump(mode="json", exclude_none=True),
            root / PROVENANCE_FILE_NAME,
        )
        root.replace(final_root)
    except Exception:
        staging.cleanup()
        raise
    staging.cleanup()
    return record


def run_conventional_intake(
    declaration: SourceDeclaration | Mapping[str, Any],
    output_root: Path,
    *,
    source_base_dir: Path,
    allowed_roots: tuple[Path, ...],
    pack: PackIdentity,
    domain_brief_path: Path,
    certification_authority: CertificationAuthority,
    held_out_decision: HeldOutDecision,
    held_out_policy_path: Path | None = None,
    held_out_content_path: Path | None = None,
    domain_brief_language: str = "en",
    domain_brief_redactions: Mapping[str, str] | None = None,
    required_tier: AdapterTier = AdapterTier.A0,
    probe_plan: AdapterProbePlan | None = None,
    http_environ: Mapping[str, str] | None = None,
    credential_resolver: CredentialResolver | None = None,
    http_client_factory: HttpClientFactory | None = None,
    resolved_authoring_config_digest: str | None = None,
) -> SourceIntakeResult:
    """Collect and atomically publish one local or HTTP v2 evidence bundle."""
    final_root = output_root.resolve()
    if final_root.exists():
        raise FileExistsError(f"intake output already exists: {final_root}")
    validated_declaration = (
        declaration
        if isinstance(declaration, SourceDeclaration)
        else SourceDeclaration.model_validate(declaration)
    )
    resolved = resolve_source_adapter(validated_declaration)
    sensitive_terms = validate_held_out_inputs(
        held_out_decision,
        policy_path=held_out_policy_path,
        content_path=held_out_content_path,
    )
    source_path = Path(resolved.source.path)
    if not source_path.is_absolute():
        source_path = source_base_dir / source_path

    if resolved.adapter_id == "local_python":
        local_inspection = inspect_local_python_package(
            source_path,
            allowed_roots=allowed_roots,
        )
        if probe_plan is None:
            descriptor = local_inspection.descriptor
            records = local_inspection.execution_records
            execution_inputs_digest = None
        else:
            probe_run = run_local_python_probes(
                local_inspection,
                probe_plan,
                allowed_roots=allowed_roots,
                held_out_sensitive_terms=sensitive_terms,
            )
            descriptor = probe_run.descriptor
            records = probe_run.records
            execution_inputs_digest = probe_run.plan_digest
        fixture_digest = next(
            (
                artifact.digest
                for artifact in local_inspection.identity.artifacts
                if artifact.role == "fixtures"
            ),
            None,
        )
        fixture_direction: Literal["none", "read_only", "pushed", "snapshot"]
        if probe_plan is not None and probe_plan.fixtures is not None:
            fixture_digest = sha256_json(probe_plan.fixtures)
            fixture_direction = "snapshot"
            vocabulary = ConfirmationVocabulary(
                parameter=probe_plan.confirmation_parameter,
                status_field=probe_plan.status_field,
                pending_status=probe_plan.pending_status,
                error_path=".".join(probe_plan.error_path),
            )
        else:
            fixture_direction = "read_only" if fixture_digest else "none"
            vocabulary = ConfirmationVocabulary()
        collection = SourceCollection(
            resolved=resolved,
            descriptor=descriptor,
            profile=local_python_reference_profile(),
            identity=local_inspection.identity,
            pack=pack,
            tools=local_inspection.tools,
            fixture_direction=fixture_direction,
            fixture_content_digest=fixture_digest,
            vocabulary=vocabulary,
            project_outcomes=lambda input_digest: project_probe_executions(
                local_python_reference_profile(),
                records,
                input_digest=input_digest,
            ),
            execution_inputs_digest=execution_inputs_digest,
        )
    elif resolved.adapter_id == "http_package":
        http_inspection = inspect_http_package(
            source_path,
            allowed_roots=allowed_roots,
            environ=http_environ,
            credential_resolver=credential_resolver,
            client_factory=http_client_factory,
        )
        http_fixture_direction: Literal["none", "read_only", "pushed", "snapshot"]
        if probe_plan is None:
            records = http_inspection.execution_records
            execution_inputs_digest = None
            http_fixture_direction = "none"
            http_fixture_digest = None
            vocabulary = ConfirmationVocabulary(
                parameter="confirm",
                status_field="status",
                pending_status="awaiting_confirmation",
                error_path="error.code",
            )
        else:
            http_probe_run = run_http_package_probes(
                http_inspection,
                probe_plan,
                environ=http_environ,
                credential_resolver=credential_resolver,
                held_out_sensitive_terms=sensitive_terms,
            )
            records = http_probe_run.records
            execution_inputs_digest = http_probe_run.plan_digest
            # The endpoint is handed these fixtures at every session open, so the
            # reviewed snapshot is what the observations were made against.
            http_fixture_direction = "snapshot"
            http_fixture_digest = sha256_json(probe_plan.fixtures)
            vocabulary = ConfirmationVocabulary(
                parameter=probe_plan.confirmation_parameter,
                status_field=probe_plan.status_field,
                pending_status=probe_plan.pending_status,
                error_path=".".join(probe_plan.error_path),
            )
        collection = SourceCollection(
            resolved=resolved,
            descriptor=http_inspection.descriptor,
            profile=http_package_reference_profile(),
            identity=http_inspection.identity,
            pack=pack,
            tools=http_inspection.tools,
            fixture_direction=http_fixture_direction,
            fixture_content_digest=http_fixture_digest,
            vocabulary=vocabulary,
            project_outcomes=lambda input_digest: project_probe_executions(
                http_package_reference_profile(),
                records,
                input_digest=input_digest,
            ),
            execution_inputs_digest=execution_inputs_digest,
        )
    else:
        raise SourceIntakeError(
            "adapter_not_supported",
            "MCP must enter through its compatibility collector",
        )

    if collection.descriptor.kind != resolved.descriptor_kind:
        raise SourceIntakeError(
            "adapter_source_mismatch",
            "resolved source kind does not match the collector descriptor",
        )
    finalized = finalize_source_intake(
        descriptor=collection.descriptor,
        identity=collection.identity,
        profile=collection.profile,
        project_outcomes=collection.project_outcomes,
        execution_inputs_digest=collection.execution_inputs_digest,
        certification_authority=certification_authority,
        required_tier=required_tier,
        domain_brief_path=domain_brief_path,
        domain_brief_language=domain_brief_language,
        domain_brief_redactions=domain_brief_redactions,
        held_out_decision=held_out_decision,
        held_out_sensitive_terms=sensitive_terms,
        evidence_factory=_conventional_evidence_factory(collection),
        resolved_authoring_config_digest=resolved_authoring_config_digest,
    )
    provenance = _write_atomic_intake(
        final_root,
        declaration=validated_declaration,
        collection=collection,
        finalized=finalized,
        domain_brief_path=domain_brief_path,
        resolved_authoring_config_digest=resolved_authoring_config_digest,
    )
    return SourceIntakeResult(
        output_root=final_root,
        collection=collection,
        finalized=finalized,
        provenance=provenance,
    )
