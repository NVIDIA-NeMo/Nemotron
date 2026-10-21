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

"""Verified, transport-neutral assembly of v2 review inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.authoring_release.contracts import (
    AdapterReviewContribution,
    FreezeHookContext,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    ReviewPacketV2,
    build_review_packet,
    load_json_mapping,
)
from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    load_resolved_authoring_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_fingerprint,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
    derive_pack_tier,
)
from nemotron.steps.byob.runtime.mcp.authoring.provenance import IntakeProvenance
from nemotron.steps.byob.runtime.mcp.config import load_unique_yaml_mapping
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    build_exposure_subject,
    load_exposure_authorization,
    verify_exposure_authorization,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.pack_authoring.provenance import DraftProvenance
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    verify_answered_revision,
)
from nemotron.steps.byob.runtime.source_adapters.certification import AdapterTier
from nemotron.steps.byob.runtime.source_adapters.intake import SourceIntakeRecord

AdapterKind = Literal["local_python", "http_package", "mcp_mode_a"]


class ReviewAssemblyError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


@dataclass(frozen=True)
class ReviewContext:
    evidence_path: Path
    certification_report_path: Path
    trusted_certification_keys: Mapping[str, Ed25519PublicKey]
    domain_brief_source_path: Path
    domain_brief_report_path: Path
    held_out_redaction_report_path: Path
    source_observations_path: Path
    intake_provenance_path: Path
    draft_provenance_path: Path
    validation_report_path: Path
    resolved_authoring_config_path: Path
    validation_config_path: Path | None = None
    exposure_authorization_path: Path | None = None
    evidence_approval_path: Path | None = None
    held_out_policy_path: Path | None = None
    held_out_content_path: Path | None = None
    source_bundle_path: Path | None = None
    migration_record_path: Path | None = None
    parent_evidence_path: Path | None = None
    open_questions_path: Path | None = None
    answer_set_path: Path | None = None
    adapter_records: Mapping[str, Path] | None = None
    freeze_sidecars: Mapping[str, Path] | None = None
    organizational_policy_digest: str | None = None

    def record_paths(self) -> dict[str, Path]:
        paths: dict[str, Path | None] = {
            "evidence_bundle": self.evidence_path,
            "certification_report": self.certification_report_path,
            "domain_brief_source": self.domain_brief_source_path,
            "domain_brief_report": self.domain_brief_report_path,
            "held_out_redaction_report": self.held_out_redaction_report_path,
            "source_observations": self.source_observations_path,
            "intake_provenance": self.intake_provenance_path,
            "draft_provenance": self.draft_provenance_path,
            "validation_report": self.validation_report_path,
            "validation_config": self.validation_config_path,
            "resolved_authoring_config": self.resolved_authoring_config_path,
            "model_exposure_authorization": self.exposure_authorization_path,
            "evidence_approval": self.evidence_approval_path,
            "held_out_policy": self.held_out_policy_path,
            "source_bundle": self.source_bundle_path,
            "migration_record": self.migration_record_path,
            "parent_evidence": self.parent_evidence_path,
            "open_questions": self.open_questions_path,
            "answer_set": self.answer_set_path,
        }
        result = {name: path for name, path in paths.items() if path is not None}
        for name, path in sorted((self.adapter_records or {}).items()):
            if name in result:
                raise ReviewAssemblyError(
                    "review_record_name_collision",
                    f"adapter record name collides with common record {name!r}",
                    recovery="use an adapter-prefixed source record name",
                )
            result[name] = path
        return result


def _file_digest(path: Path) -> str:
    try:
        return f"sha256:{hashlib.sha256(path.resolve().read_bytes()).hexdigest()}"
    except OSError as exc:
        raise ReviewAssemblyError(
            "review_record_missing",
            f"cannot read reviewed record {path}: {exc}",
            recovery="restore the exact authoring artifact and retry",
        ) from exc


def _verify_intake(path: Path, evidence_digest: str) -> str:
    document = load_json_mapping(path, "intake provenance")
    version = document.get("schema_version")
    if version == "bfcl-source-intake-record-v1":
        source_record = SourceIntakeRecord.model_validate(document)
        if source_record.evidence_digest != evidence_digest:
            raise ReviewAssemblyError(
                "intake_evidence_mismatch",
                "intake provenance names different evidence",
                recovery="review one coherent intake revision",
            )
        return source_record.record_digest
    if version == "bfcl-mcp-intake-provenance-v1":
        mcp_record = IntakeProvenance(document)
        mcp_record.verify_digest()
        evidence = document.get("evidence_bundle")
        if not isinstance(evidence, Mapping) or evidence.get("digest") != evidence_digest:
            raise ReviewAssemblyError(
                "intake_evidence_mismatch",
                "MCP intake provenance names different evidence",
                recovery="review one coherent intake revision",
            )
        return str(document["record_digest"])
    raise ReviewAssemblyError(
        "intake_version_unsupported",
        f"unsupported intake provenance {version!r}",
        recovery="use a verified transport-neutral or MCP intake record",
    )


class VerifiedReleaseAdapter:
    def __init__(
        self,
        *,
        kind: AdapterKind,
        oracle_variant: Literal["backend", "endpoint"],
        contribution: AdapterReviewContribution,
        sidecars: Mapping[str, bytes],
    ) -> None:
        self._kind = kind
        self._oracle_variant = oracle_variant
        self._contribution = contribution
        self._sidecars = dict(sidecars)

    @property
    def kind(self) -> str:
        return self._kind

    def validate_pack(self, pack_root: Path) -> str:
        root = pack_root.resolve()
        paths = resolve_declared_pack_paths(
            OraclePackRef(manifest_path=root / "manifest.yaml"),
            (root,),
        )
        if self._oracle_variant == "backend":
            valid = paths.backend_path is not None and paths.endpoint_config_path is None
        else:
            valid = paths.endpoint_config_path is not None and paths.backend_path is None
        if not valid:
            raise ReviewAssemblyError(
                "candidate_pack_variant_mismatch",
                f"{self._kind} requires a canonical {self._oracle_variant} pack",
                recovery="build the candidate pack for the selected source adapter",
            )
        return f"sha256:{pack_fingerprint(paths)}"

    def review(
        self,
        pack_root: Path,
        source_digests: Mapping[str, str],
    ) -> AdapterReviewContribution:
        del pack_root, source_digests
        return self._contribution

    def freeze_sidecars(
        self,
        context: FreezeHookContext,
    ) -> Mapping[str, bytes]:
        del context
        return self._sidecars


@dataclass(frozen=True)
class AssembledReview:
    packet: ReviewPacketV2
    adapter: VerifiedReleaseAdapter
    source_records: Mapping[str, Path]


def release_adapter_for_packet(
    packet: ReviewPacketV2,
    *,
    freeze_sidecars: Mapping[str, Path] | None = None,
) -> VerifiedReleaseAdapter:
    packet.verify()
    kind = packet.document["adapter_kind"]
    if kind not in {"local_python", "http_package", "mcp_mode_a"}:
        raise ReviewAssemblyError(
            "review_adapter_unsupported",
            f"unsupported release adapter {kind!r}",
            recovery="use a built-in reviewed source adapter",
        )
    sidecars = {
        name: path.resolve().read_bytes()
        for name, path in sorted((freeze_sidecars or {}).items())
    }
    contribution = AdapterReviewContribution(
        identity_digest=packet.document["source_identity_digest"],
        certification_tier=packet.document["certification_tier"],
        review_data=packet.document["adapter_review"],
        blockers=packet.document["blockers"],
        risks=packet.document["risks"],
    )
    return VerifiedReleaseAdapter(
        kind=kind,
        oracle_variant="backend" if kind == "local_python" else "endpoint",
        contribution=contribution,
        sidecars=sidecars,
    )


def assemble_review(
    *,
    adapter_kind: AdapterKind,
    pack_root: Path,
    context: ReviewContext,
) -> AssembledReview:
    evidence = load_evidence_bundle(
        context.evidence_path,
        certification_report_path=context.certification_report_path,
        trusted_certification_keys=context.trusted_certification_keys,
        domain_brief_source_path=context.domain_brief_source_path,
        domain_brief_report_path=context.domain_brief_report_path,
        held_out_redaction_report_path=context.held_out_redaction_report_path,
        held_out_policy_path=context.held_out_policy_path,
        held_out_content_path=context.held_out_content_path,
        source_bundle_path=context.source_bundle_path,
        migration_record_path=context.migration_record_path,
        source_observations_path=context.source_observations_path,
        required_certification_tier=AdapterTier.A0,
    )
    if evidence.source_evidence is None or evidence.certification_report is None:
        raise ReviewAssemblyError(
            "v2_evidence_required",
            "generalized review requires transport-neutral v2 evidence",
            recovery="migrate and independently certify the source evidence",
        )
    if (
        evidence.domain_brief_report is None
        or evidence.held_out_redaction_report is None
    ):
        raise ReviewAssemblyError(
            "v2_evidence_required",
            "generalized review requires verified brief and held-out reports",
            recovery="reload the complete transport-neutral intake sidecars",
        )
    if evidence.source_evidence.source_adapter.kind != adapter_kind:
        raise ReviewAssemblyError(
            "review_adapter_mismatch",
            "evidence source adapter differs from requested review adapter",
            recovery="select the adapter recorded in verified evidence",
        )
    resolved = load_resolved_authoring_config(
        context.resolved_authoring_config_path
    )
    if resolved.semantic_payload.adapter_kind.value != adapter_kind:
        raise ReviewAssemblyError(
            "resolved_config_adapter_mismatch",
            "resolved configuration names a different source adapter",
            recovery="resume from the configuration-bound authoring revision",
        )
    if (
        resolved.semantic_payload.pack_id.value != evidence.pack_id
        or resolved.semantic_payload.pack_version.value
        != evidence.document["pack"]["version"]
    ):
        raise ReviewAssemblyError(
            "resolved_config_pack_mismatch",
            "resolved configuration names a different pack identity",
            recovery="resume from the configuration-bound authoring revision",
        )
    intake_digest = _verify_intake(context.intake_provenance_path, evidence.digest)
    draft_document = load_json_mapping(context.draft_provenance_path, "draft provenance")
    DraftProvenance(draft_document).verify_digest()
    if draft_document.get("evidence", {}).get("bundle_digest") != evidence.digest:
        raise ReviewAssemblyError(
            "draft_evidence_mismatch",
            "draft provenance names different evidence",
            recovery="redraft from the exact reviewed evidence",
        )
    if (
        draft_document.get("resolved_authoring_config_digest")
        != resolved.resolved_authoring_config_digest
    ):
        raise ReviewAssemblyError(
            "draft_config_mismatch",
            "draft provenance does not bind the resolved authoring configuration",
            recovery="redraft from the exact configuration-bound revision",
        )
    blockers: list[dict[str, Any]] = []
    authorization_digest: str | None = None
    if context.exposure_authorization_path is None:
        blockers.append(
            {
                "code": "model_exposure_authorization_missing",
                "recovery": "run authorize before final review",
            }
        )
    else:
        try:
            authorization = load_exposure_authorization(
                context.exposure_authorization_path
            )
            subject = build_exposure_subject(
                evidence.source_evidence,
                domain_brief_report=evidence.domain_brief_report,
                held_out_redaction_report=evidence.held_out_redaction_report,
                resolved_authoring_config_digest=(
                    resolved.resolved_authoring_config_digest
                ),
            )
            verify_exposure_authorization(
                authorization,
                expected_subject=subject,
                expected_organizational_policy_digest=(
                    context.organizational_policy_digest
                ),
            )
            authorization_digest = authorization.authorization_digest
            if (
                draft_document.get("model_exposure_authorization")
                != authorization.model_dump(mode="json")
            ):
                raise ValueError("draft provenance used different authorization")
        except ValueError as exc:
            blockers.append(
                {
                    "code": "model_exposure_authorization_stale",
                    "detail": str(exc),
                    "recovery": "authorize and redraft the exact current evidence",
                }
            )
    revision = evidence.source_evidence.revision
    try:
        verify_answered_revision(
            evidence.source_evidence,
            parent_evidence_path=context.parent_evidence_path,
            open_questions_path=context.open_questions_path,
            answer_set_path=context.answer_set_path,
        )
    except ValueError as exc:
        blockers.append(
            {
                "code": "answered_revision_incomplete",
                "detail": str(exc),
                "recovery": "supply and replay the bound question and answer artifacts",
            }
        )
    if context.validation_config_path is None:
        blockers.append(
            {
                "code": "validation_authority_missing",
                "recovery": "rerun review with a BFCL validation config",
            }
        )
    else:
        generated_validation = prepare_bfcl(
            context.validation_config_path,
            force_validation=True,
        ).resolve()
        if generated_validation != context.validation_report_path.resolve():
            raise ReviewAssemblyError(
                "validation_report_path_mismatch",
                "fresh BFCL prepare wrote a different validation report path",
                recovery="bind the report produced by the reviewed BFCL config",
            )
    validation = load_json_mapping(context.validation_report_path, "validation report")
    candidate_fingerprint = (
        VerifiedReleaseAdapter(
            kind=adapter_kind,
            oracle_variant="backend" if adapter_kind == "local_python" else "endpoint",
            contribution=AdapterReviewContribution(
                identity_digest="sha256:" + "0" * 64,
                certification_tier="A0",
                review_data={},
            ),
            sidecars={},
        ).validate_pack(pack_root)
    )
    candidate_manifest = load_unique_yaml_mapping(
        pack_root.resolve() / "manifest.yaml",
        "candidate pack manifest",
    )
    if (
        candidate_manifest.get("pack_id") != evidence.document["pack"]["pack_id"]
        or candidate_manifest.get("version") != evidence.document["pack"]["version"]
    ):
        raise ReviewAssemblyError(
            "candidate_pack_identity_mismatch",
            "candidate pack identity differs from verified evidence",
            recovery="compile the candidate pack from the exact evidence revision",
        )
    if (
        validation.get("pack_fingerprint")
        not in {
            candidate_fingerprint,
            candidate_fingerprint.removeprefix("sha256:"),
        }
    ):
        blockers.append(
            {
                "code": "validation_pack_mismatch",
                "recovery": "validate the exact candidate pack",
            }
        )
    gold, tier = derive_pack_tier(validation)
    if not gold or tier != "gold":
        blockers.append(
            {
                "code": "validation_not_gold",
                "tier": tier,
                "recovery": "resolve validation failures before final approval",
            }
        )
    if evidence.certification_tier != "A2":
        blockers.append(
            {
                "code": "adapter_under_certified",
                "tier": evidence.certification_tier,
                "recovery": "complete independently verified A2 probes",
            }
        )
    for code in evidence.unresolved_unknowns:
        blockers.append(
            {
                "code": "evidence_unresolved",
                "field": code,
                "recovery": "answer or observe every blocking evidence gap",
            }
        )
    if draft_document.get("blocked_on"):
        blockers.append(
            {
                "code": "draft_incomplete",
                "fields": sorted(set(draft_document["blocked_on"])),
                "recovery": "resolve drafting blockers and redraft",
            }
        )
    if draft_document.get("assertions_compiled") is not True:
        blockers.append(
            {
                "code": "assertions_not_compiled",
                "recovery": "compile assertions before final review",
            }
        )
    source_records = context.record_paths()
    source_digests = {
        name: _file_digest(path) for name, path in sorted(source_records.items())
    }
    sidecars = {
        name: path.resolve().read_bytes()
        for name, path in sorted((context.freeze_sidecars or {}).items())
    }
    sidecar_digests = {
        name: f"sha256:{hashlib.sha256(data).hexdigest()}"
        for name, data in sorted(sidecars.items())
    }
    identity_digest = sha256_json(
        evidence.source_evidence.identity.model_dump(mode="json")
    )
    contribution = AdapterReviewContribution(
        identity_digest=identity_digest,
        certification_tier=str(evidence.certification_tier),
        review_data={
            "pack": dict(evidence.document["pack"]),
            "certification": {
                "report_digest": evidence.certification_report.report_digest,
                "profile_id": evidence.certification_report.profile_id,
                "tier": evidence.certification_tier,
            },
            "authoring": {
                "model_exposure_authorization_digest": authorization_digest,
                "draft_provenance_digest": draft_document["record_digest"],
                "resolved_authoring_config_digest": (
                    resolved.resolved_authoring_config_digest
                ),
                "evidence_revision": (
                    {
                        "question_artifact_digest": revision.question_artifact_digest,
                        "answer_set_digest": revision.answer_set_digest,
                    }
                    if revision is not None
                    else None
                ),
                "questions_status": "answered" if revision is not None else "not_required",
            },
            "intake_record_digest": intake_digest,
            "validation": {
                "tier": tier,
                "gold": gold,
                "pack_fingerprint": validation.get("pack_fingerprint"),
            },
            "freeze_sidecars": sidecar_digests,
        },
        blockers=blockers,
        risks=(),
    )
    adapter = VerifiedReleaseAdapter(
        kind=adapter_kind,
        oracle_variant="backend" if adapter_kind == "local_python" else "endpoint",
        contribution=contribution,
        sidecars=sidecars,
    )
    packet = build_review_packet(
        adapter=adapter,
        pack_root=pack_root,
        source_digests=source_digests,
    )
    return AssembledReview(
        packet=packet,
        adapter=adapter,
        source_records=source_records,
    )
