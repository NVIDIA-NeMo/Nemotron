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

"""One intake run: discover, sanitize, derive, and record.

The order is the argument. Identity is pinned before any evidence is written, so a bundle
can never describe one catalog while the pack points at another. The pack files are written
only after the bundle survives hygiene, so a blocked run leaves no artifact that looks
authored. Provenance is written last, because it is the only file that claims the others
exist.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.authoring_workflow.credentials import CredentialResolver
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import EndpointConfig
from nemotron.steps.byob.runtime.mcp.authoring.attestation import (
    AttestationFetcher,
    fetch_gateway_attestation,
    temporary_endpoint_config,
    validate_gateway_attestation,
)
from nemotron.steps.byob.runtime.mcp.authoring.evidence import (
    EvidenceBundle,
    build_evidence_bundle,
    write_evidence_bundle,
)
from nemotron.steps.byob.runtime.mcp.authoring.intake import (
    LoadedMcpIntake,
    load_mcp_intake,
)
from nemotron.steps.byob.runtime.mcp.authoring.pack_artifacts import (
    EmittedArtifact,
    emit_pack_artifacts,
)
from nemotron.steps.byob.runtime.mcp.authoring.probes import (
    discovery_identity_record,
    reviewed_probe_tools,
    run_mcp_gateway_probes,
)
from nemotron.steps.byob.runtime.mcp.authoring.provenance import (
    IntakeProvenance,
    build_intake_provenance,
    write_intake_provenance,
)
from nemotron.steps.byob.runtime.mcp.config import (
    StreamableHttpTransportConfig,
    TrustedExecutablePolicies,
)
from nemotron.steps.byob.runtime.mcp.discovery import (
    ConnectionFactory,
    DiscoveryReport,
    discover_mcp_oracle,
    write_discovery_report,
)
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    GatewayArtifacts,
    GatewayIdentity,
    build_gateway_identity,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import ExposureSubject
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterCertificationReport,
    AdapterTier,
    CertificationAuthority,
    ProbeOutcome,
    certification_reference,
    mcp_reference_profile,
    project_mcp_probe_report,
    project_probe_executions,
)
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
from nemotron.steps.byob.runtime.source_adapters.evidence import SourceEvidenceDocument
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutDecision,
    HeldOutRedactionReport,
)
from nemotron.steps.byob.runtime.source_adapters.intake import (
    SourceIntakeError,
    finalize_source_intake,
    validate_held_out_inputs,
)
from nemotron.steps.byob.runtime.source_adapters.migration import (
    MigrationContext,
    NormalizedSourceEvidence,
    legacy_mcp_source_identity,
    migrate_legacy_mcp_evidence,
    write_migration_record,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import AdapterProbePlan

PACK_DIRECTORY_NAME = "pack"
EVIDENCE_FILE_NAME = "evidence_bundle.json"
LEGACY_EVIDENCE_FILE_NAME = "evidence_bundle.v1.json"
CERTIFICATION_FILE_NAME = "adapter_certification.json"
MIGRATION_FILE_NAME = "evidence_migration.json"
DOMAIN_BRIEF_SOURCE_FILE_NAME = "domain_brief.source.txt"
DOMAIN_BRIEF_REPORT_FILE_NAME = "domain_brief_redaction.json"
HELD_OUT_REDACTION_FILE_NAME = "held_out_redaction.json"
EXPOSURE_SUBJECT_FILE_NAME = "model_exposure_subject.json"
OBSERVATIONS_FILE_NAME = "source_observations.json"
DISCOVERY_FILE_NAME = "discovery_report.json"
PROVENANCE_FILE_NAME = "intake_provenance.json"
ATTESTATION_FILE_NAME = "gateway_attestation.json"


@dataclass(frozen=True)
class IntakeResult:
    intake: LoadedMcpIntake
    report: DiscoveryReport
    identity: GatewayIdentity
    attestation: dict[str, Any]
    bundle: EvidenceBundle
    provenance: IntakeProvenance
    artifacts: list[EmittedArtifact]
    output_root: Path
    source_evidence: SourceEvidenceDocument | None = None
    certification_path: Path | None = None
    migration_path: Path | None = None
    domain_brief_source_path: Path | None = None
    domain_brief_report_path: Path | None = None
    held_out_redaction_path: Path | None = None
    held_out_redaction: HeldOutRedactionReport | None = None
    exposure_subject_path: Path | None = None
    exposure_subject: ExposureSubject | None = None
    observations_path: Path | None = None

    @property
    def pack_root(self) -> Path:
        return self.output_root / PACK_DIRECTORY_NAME


async def run_intake(
    intake_path: Path,
    output_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    executable_policies: TrustedExecutablePolicies | None = None,
    connection_factory: ConnectionFactory | None = None,
    attestation_fetcher: AttestationFetcher | None = None,
    endpoint_config_factory: Callable[
        [LoadedMcpIntake, GatewayIdentity], EndpointConfig
    ] = temporary_endpoint_config,
    credential_resolver: CredentialResolver | None = None,
    allow_insecure_localhost: bool = False,
    domain_brief_path: Path | None = None,
    domain_brief_language: str = "en",
    certification_authority: CertificationAuthority | None = None,
    held_out_decision: HeldOutDecision | None = None,
    held_out_policy_path: Path | None = None,
    held_out_content_path: Path | None = None,
    probe_plan: AdapterProbePlan | None = None,
    resolved_authoring_config_digest: str | None = None,
    required_tier: AdapterTier = AdapterTier.A0,
) -> IntakeResult:
    """Turn a reviewed MCP intake declaration into a reviewable pack draft."""
    final_root = output_root.resolve()
    if final_root.exists():
        raise FileExistsError(f"intake output already exists: {final_root}")
    v2_requested = any(
        value is not None
        for value in (
            domain_brief_path,
            certification_authority,
            held_out_decision,
            held_out_policy_path,
            held_out_content_path,
        )
    )
    if v2_requested and (
        domain_brief_path is None
        or certification_authority is None
        or held_out_decision is None
    ):
        raise SourceIntakeError(
            "intake_inputs_incomplete",
            "MCP v2 intake requires domain brief, certification authority, and "
            "an explicit held-out decision",
        )
    held_out_sensitive_terms: tuple[str, ...] = ()
    if held_out_decision is not None:
        held_out_sensitive_terms = validate_held_out_inputs(
            held_out_decision,
            policy_path=held_out_policy_path,
            content_path=held_out_content_path,
        )
    intake = load_mcp_intake(
        intake_path,
        allow_insecure_localhost=allow_insecure_localhost,
    )
    source_transport = intake.oracle.value.transport
    if (
        isinstance(source_transport, StreamableHttpTransportConfig)
        and source_transport.auth.credential_references()
        and intake.oracle.value.expected.authorization_context_digest is None
    ):
        raise SourceIntakeError(
            "unsupported_auth",
            "authenticated MCP authoring requires expected principal_digest, "
            "permission_digest, and authorization_context_digest",
        )
    if probe_plan is not None:
        if not v2_requested:
            raise SourceIntakeError(
                "intake_inputs_incomplete",
                "a probe plan only has meaning for v2 intake, which issues the "
                "certification report the observations feed",
            )
        if intake.oracle.value.mode != "A":
            raise SourceIntakeError(
                "adapter_not_supported",
                "only mode A can be probed above A0: reset and state are control tools "
                f"there, and this profile declares mode {intake.oracle.value.mode}",
            )
    identity_started = time.monotonic()
    report = await discover_mcp_oracle(
        intake.oracle,
        environ=environ,
        executable_policies=executable_policies,
        connection_factory=connection_factory,
    )
    identity = build_gateway_identity(
        intake.oracle.value,
        report,
        GatewayArtifacts(
            gateway_artifact_digest=intake.value.gateway.gateway_artifact_digest,
            shim_artifact_digest=intake.value.gateway.shim_artifact_digest,
            snapshot_digest=intake.value.gateway.snapshot_digest,
        ),
    )
    if intake.value.gateway.authorization_context_digest is not None:
        identity = replace(
            identity,
            principal_digest=intake.value.gateway.principal_digest,
            permission_digest=intake.value.gateway.permission_digest,
            authorization_context_digest=(
                intake.value.gateway.authorization_context_digest
            ),
        )
    temporary_endpoint = endpoint_config_factory(intake, identity)
    if attestation_fetcher is None:
        raw_attestation = await asyncio.to_thread(
            fetch_gateway_attestation,
            temporary_endpoint,
            environ=environ,
            timeout_s=float(intake.oracle.value.limits.handshake_timeout_s),
        )
    else:
        raw_attestation = await asyncio.to_thread(
            attestation_fetcher,
            temporary_endpoint,
        )
    attestation = validate_gateway_attestation(
        raw_attestation,
        intake=intake,
        report=report,
        identity=identity,
    )
    bundle = build_evidence_bundle(intake, report, identity)

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(
        dir=final_root.parent,
        prefix=f".{final_root.name}.staging-",
    )
    root = Path(staging.name)
    task = asyncio.current_task()

    def cleanup_staging(_task: asyncio.Task[Any]) -> None:
        staging.cleanup()

    if task is not None:
        task.add_done_callback(cleanup_staging)
    attestation_path = write_canonical_json(
        attestation,
        root / ATTESTATION_FILE_NAME,
    )
    artifacts = emit_pack_artifacts(
        intake,
        report,
        identity,
        attestation,
        root / PACK_DIRECTORY_NAME,
    )
    write_discovery_report(report, root / DISCOVERY_FILE_NAME)
    normalized: NormalizedSourceEvidence | None = None
    certification_path: Path | None = None
    migration_path: Path | None = None
    persisted_brief_path: Path | None = None
    brief_report_path: Path | None = None
    held_out_redaction_path: Path | None = None
    exposure_subject_path: Path | None = None
    observations_path: Path | None = None
    held_out_redaction: HeldOutRedactionReport | None = None
    exposure_subject: ExposureSubject | None = None
    legacy_path: Path | None = None
    certification: AdapterCertificationReport | None = None
    if v2_requested:
        assert domain_brief_path is not None
        assert certification_authority is not None
        assert held_out_decision is not None
        config = intake.oracle.value
        legacy_path = write_evidence_bundle(
            bundle,
            root / LEGACY_EVIDENCE_FILE_NAME,
        )
        # One probe operation is either a reset or a tool call, and the reviewed profile
        # already states what each is allowed to take.
        probe_timeout_s = float(
            max(config.limits.reset_timeout_s, config.limits.tool_timeout_s)
        )
        descriptor = AdapterDescriptor(
            contract_version="bfcl-source-adapter-v1",
            kind="mcp_mode_a",
            implementation_name="bfcl.mcp_mode_a",
            implementation_version="1.0.0",
            # Without a probe plan this run only reads a catalog and pins an identity, so
            # the descriptor claims only that. Mode A can do more, but a descriptor that
            # said so would be describing the gateway rather than this intake.
            capabilities=(
                tuple(sorted(AdapterCapability, key=lambda item: item.value))
                if probe_plan is not None
                else (
                    AdapterCapability.DESCRIBE_TOOLS,
                    AdapterCapability.PIN_IDENTITY,
                )
            ),
            fixture_access=FixtureAccessPolicy(
                kind=FixtureAccessKind(config.fixtures.direction),
                supports_redaction=True,
            ),
            probe_safety=(
                ProbeSafetyPolicy(
                    kind=ProbeSafetyKind.RESET_ISOLATED,
                    max_calls=32,
                    timeout_s=probe_timeout_s,
                )
                if probe_plan is not None
                else ProbeSafetyPolicy(
                    kind=ProbeSafetyKind.IDENTITY_ONLY,
                    max_calls=1,
                    timeout_s=float(config.limits.handshake_timeout_s),
                )
            ),
            cleanup=CleanupSemantics(
                kind=CleanupKind.SESSION,
                timeout_s=float(config.limits.connect_timeout_s),
            ),
        )
        profile = mcp_reference_profile()
        source_identity = legacy_mcp_source_identity(bundle.document)
        normalized_holder: dict[str, NormalizedSourceEvidence] = {}

        def build_mcp_evidence(
            signed_report: AdapterCertificationReport,
            brief: DomainBriefEvidence,
            brief_report: DomainBriefRedactionReport,
            decision: HeldOutDecision,
        ) -> SourceEvidenceDocument:
            migrated = migrate_legacy_mcp_evidence(
                legacy_path,
                context=MigrationContext(
                    source_adapter=descriptor,
                    certification=certification_reference(signed_report),
                    domain_brief=brief,
                    domain_brief_report=brief_report,
                    held_out=decision,
                    # The gateway is handed these fixtures at every session open, so once
                    # probes have run the reviewed snapshot is what the observations were
                    # made against. Discovery-only intake has no snapshot to name.
                    fixture_content_digest=(
                        sha256_json(probe_plan.fixtures)
                        if probe_plan is not None
                        else None
                    ),
                ),
            )
            normalized_holder["value"] = migrated
            return migrated.evidence

        project_outcomes: Callable[[str], tuple[ProbeOutcome, ...]]
        execution_inputs_digest: str | None
        if probe_plan is None:
            execution_inputs_digest = None

            def project_discovery_checks(input_digest: str) -> tuple[ProbeOutcome, ...]:
                return project_mcp_probe_report(
                    {"probes": list(report.document["checks"])},
                    input_digest=input_digest,
                    structured_error_applicable=False,
                    confirmation_applicable=any(
                        bool(
                            entry.get("function", {}).get(
                                "x-requires-confirmation",
                                False,
                            )
                        )
                        for entry in report.document["catalog"]["tools"]
                    ),
                )

            project_outcomes = project_discovery_checks
        else:
            try:
                probe_run = await asyncio.to_thread(
                    run_mcp_gateway_probes,
                    endpoint_config=temporary_endpoint,
                    tools=reviewed_probe_tools(report.document["catalog"]["tools"]),
                    identity_record=discovery_identity_record(
                        identity_document=identity.as_dict(),
                        attestation_digest=sha256_json(attestation),
                        started=identity_started,
                    ),
                    catalog_pinned=all(
                        check["status"] == "pass"
                        for check in report.document["checks"]
                    ),
                    plan=probe_plan,
                    environ=environ,
                    credential_resolver=credential_resolver,
                    held_out_sensitive_terms=held_out_sensitive_terms,
                    timeout_s=probe_timeout_s,
                )
            except Exception:
                staging.cleanup()
                raise
            execution_inputs_digest = probe_run.plan_digest

            def project_probe_run(input_digest: str) -> tuple[ProbeOutcome, ...]:
                return project_probe_executions(
                    profile,
                    probe_run.records,
                    input_digest=input_digest,
                )

            project_outcomes = project_probe_run

        try:
            finalized = finalize_source_intake(
                descriptor=descriptor,
                identity=source_identity,
                profile=profile,
                project_outcomes=project_outcomes,
                execution_inputs_digest=execution_inputs_digest,
                certification_authority=certification_authority,
                required_tier=required_tier,
                domain_brief_path=domain_brief_path,
                domain_brief_language=domain_brief_language,
                domain_brief_redactions=None,
                held_out_decision=held_out_decision,
                held_out_sensitive_terms=held_out_sensitive_terms,
                evidence_factory=build_mcp_evidence,
                resolved_authoring_config_digest=resolved_authoring_config_digest,
            )
        except Exception:
            staging.cleanup()
            raise
        certification = finalized.certification
        exposure_subject = finalized.exposure_subject
        normalized = normalized_holder["value"]
        persisted_brief_path = root / DOMAIN_BRIEF_SOURCE_FILE_NAME
        persisted_brief_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_brief_path.write_bytes(domain_brief_path.resolve().read_bytes())
        brief_report_path = write_canonical_json(
            finalized.domain_brief_report.model_dump(mode="json"),
            root / DOMAIN_BRIEF_REPORT_FILE_NAME,
        )
        certification_path = write_canonical_json(
            certification.model_dump(mode="json"),
            root / CERTIFICATION_FILE_NAME,
        )
        assert normalized.migration is not None
        migration_path = write_migration_record(
            normalized.migration,
            root / MIGRATION_FILE_NAME,
        )
        normalized_document = finalized.evidence.model_dump(mode="json")
        held_out_redaction = finalized.held_out_redaction
        evidence_path = write_canonical_json(
            normalized_document,
            root / EVIDENCE_FILE_NAME,
        )
        held_out_redaction_path = write_canonical_json(
            held_out_redaction.model_dump(mode="json"),
            root / HELD_OUT_REDACTION_FILE_NAME,
        )
        exposure_subject_path = write_canonical_json(
            finalized.exposure_subject.model_dump(mode="json"),
            root / EXPOSURE_SUBJECT_FILE_NAME,
        )
        observations_path = write_canonical_json(
            finalized.observations_document,
            root / OBSERVATIONS_FILE_NAME,
        )
    else:
        evidence_path = write_evidence_bundle(bundle, root / EVIDENCE_FILE_NAME)
    provenance = build_intake_provenance(
        intake,
        report,
        bundle,
        artifacts,
        output_root=root,
        evidence_path=evidence_path,
        attestation_path=attestation_path,
        attestation_document=attestation,
        source_evidence=normalized.evidence if normalized is not None else None,
        legacy_evidence_path=legacy_path,
        certification=certification,
        certification_path=certification_path,
        migration=normalized.migration if normalized is not None else None,
        migration_path=migration_path,
        domain_brief_source_path=persisted_brief_path,
        domain_brief_report_path=brief_report_path,
        held_out_redaction=held_out_redaction,
        held_out_redaction_path=held_out_redaction_path,
        exposure_subject=exposure_subject,
        exposure_subject_path=exposure_subject_path,
        observations_document=(
            finalized.observations_document if normalized is not None else None
        ),
        observations_path=observations_path,
        resolved_authoring_config_digest=resolved_authoring_config_digest,
    )
    write_intake_provenance(provenance, root / PROVENANCE_FILE_NAME)
    final_paths = {
        path: final_root / path.relative_to(root)
        for path in (
            certification_path,
            migration_path,
            persisted_brief_path,
            brief_report_path,
            held_out_redaction_path,
            exposure_subject_path,
            observations_path,
        )
        if path is not None
    }
    final_artifacts = [
        EmittedArtifact(
            path=final_root / artifact.path.relative_to(root),
            digest=artifact.digest,
        )
        for artifact in artifacts
    ]
    root.replace(final_root)
    if task is not None:
        task.remove_done_callback(cleanup_staging)
    staging.cleanup()
    return IntakeResult(
        intake=intake,
        report=report,
        identity=identity,
        attestation=attestation,
        bundle=bundle,
        provenance=provenance,
        artifacts=final_artifacts,
        output_root=final_root,
        source_evidence=normalized.evidence if normalized is not None else None,
        certification_path=(
            final_paths.get(certification_path)
            if certification_path is not None
            else None
        ),
        migration_path=(
            final_paths.get(migration_path) if migration_path is not None else None
        ),
        domain_brief_source_path=(
            final_paths.get(persisted_brief_path)
            if persisted_brief_path is not None
            else None
        ),
        domain_brief_report_path=(
            final_paths.get(brief_report_path)
            if brief_report_path is not None
            else None
        ),
        held_out_redaction=held_out_redaction,
        held_out_redaction_path=(
            final_paths.get(held_out_redaction_path)
            if held_out_redaction_path is not None
            else None
        ),
        exposure_subject_path=(
            final_paths.get(exposure_subject_path)
            if exposure_subject_path is not None
            else None
        ),
        exposure_subject=exposure_subject,
        observations_path=(
            final_paths.get(observations_path)
            if observations_path is not None
            else None
        ),
    )
