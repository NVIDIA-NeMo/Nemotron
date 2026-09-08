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

"""Sanitized Stage F refusals and explicit next-revision authorization."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.authoring_workflow.events import (
    AuthoringEventSink,
    RefusalPayload,
    RevisionAuthorizationPayload,
    emit_authoring_event,
)
from nemotron.steps.byob.runtime.authoring_workflow.revision_store import (
    RevisionStore,
    RevisionStoreError,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import WorkspaceLease
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

REFUSAL_RECORD_VERSION: Literal["bfcl-authoring-refusal-v1"] = (
    "bfcl-authoring-refusal-v1"
)
REVISION_AUTHORIZATION_VERSION: Literal[
    "bfcl-refusal-revision-authorization-v1"
] = "bfcl-refusal-revision-authorization-v1"
REFUSAL_FILE_NAME = "refusal.json"
AUTHORIZATION_FILE_NAME = "revision_authorization.json"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")
_FORBIDDEN_CODES = frozenset(
    {
        "accuracy",
        "metric",
        "metrics",
        "model_output",
        "model_response",
        "output",
        "outputs",
        "pass_rate",
        "response",
        "reward",
        "score",
        "scores",
        "target_model_output",
    }
)


class RefusalClassification(str, Enum):
    DETERMINISTIC_MATERIALIZATION = "deterministic_materialization"
    MODEL_OWNED_PROPOSAL = "model_owned_proposal"
    USER_OWNED_SOURCE_CONTRACT = "user_owned_source_contract"
    ORACLE_OWNED_BEHAVIOR = "oracle_owned_behavior"
    OPERATIONAL_INFRASTRUCTURE = "operational_infrastructure"


class RevisionAction(str, Enum):
    RERUN_DETERMINISTIC = "rerun_deterministic"
    REVISE_PROPOSAL = "revise_proposal"
    AMEND_SOURCE_CONTRACT = "amend_source_contract"
    REPAIR_ORACLE_BEHAVIOR = "repair_oracle_behavior"
    RETRY_INFRASTRUCTURE = "retry_infrastructure"


REQUIRED_ACTION: dict[RefusalClassification, RevisionAction] = {
    RefusalClassification.DETERMINISTIC_MATERIALIZATION: (
        RevisionAction.RERUN_DETERMINISTIC
    ),
    RefusalClassification.MODEL_OWNED_PROPOSAL: RevisionAction.REVISE_PROPOSAL,
    RefusalClassification.USER_OWNED_SOURCE_CONTRACT: (
        RevisionAction.AMEND_SOURCE_CONTRACT
    ),
    RefusalClassification.ORACLE_OWNED_BEHAVIOR: (
        RevisionAction.REPAIR_ORACLE_BEHAVIOR
    ),
    RefusalClassification.OPERATIONAL_INFRASTRUCTURE: (
        RevisionAction.RETRY_INFRASTRUCTURE
    ),
}

_RECOVERY_BY_CODE = {
    "revision_authorization_required": "obtain operator authorization for this refusal",
    "revision_authorization_stale": "authorize the exact current refusal and parent session",
    "revision_action_mismatch": "use the remediation action owned by the classification",
    "workspace_lease_required": "acquire the matching tenant/run workspace lease",
}


class RefusalRecordError(ValueError):
    """A stable refusal-record or authorization failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        recovery: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery or _RECOVERY_BY_CODE.get(
            code,
            "preserve the refusal artifacts and obtain operator review",
        )
        super().__init__(f"{code}: {detail}; recovery: {self.recovery}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SanitizedFinding(_StrictModel):
    finding_code: StrictStr
    classification: RefusalClassification
    reason_code: StrictStr
    artifact_role: StrictStr | None = None
    evidence_digests: tuple[StrictStr, ...] = ()
    finding_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> SanitizedFinding:
        _validate_code(self.finding_code, "finding_code")
        _validate_code(self.reason_code, "reason_code")
        if self.artifact_role is not None:
            _validate_code(self.artifact_role, "artifact_role")
        for digest in self.evidence_digests:
            _validate_digest(digest, "finding evidence digest")
        if self.evidence_digests != tuple(sorted(set(self.evidence_digests))):
            raise ValueError("finding evidence digests must be sorted and unique")
        unsigned = self.model_dump(mode="json", exclude={"finding_digest"})
        if self.finding_digest != sha256_json(unsigned):
            raise ValueError("finding digest mismatch")
        return self


class RefusalRecord(_StrictModel):
    schema_version: Literal["bfcl-authoring-refusal-v1"]
    tenant_id: StrictStr
    run_id: StrictStr
    session_digest: StrictStr
    primary_classification: RefusalClassification
    findings: tuple[SanitizedFinding, ...]
    refused_at: datetime
    refusal_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RefusalRecord:
        _validate_identifier(self.tenant_id, "tenant_id")
        _validate_identifier(self.run_id, "run_id")
        _validate_digest(self.session_digest, "session digest")
        if not self.findings:
            raise ValueError("refusal record requires at least one finding")
        finding_digests = tuple(item.finding_digest for item in self.findings)
        if finding_digests != tuple(sorted(set(finding_digests))):
            raise ValueError("refusal findings must be sorted and unique")
        if self.primary_classification not in {
            item.classification for item in self.findings
        }:
            raise ValueError("primary classification must be represented by a finding")
        if self.refused_at.tzinfo is None:
            raise ValueError("refused_at must be timezone-aware")
        unsigned = self.model_dump(mode="json", exclude={"refusal_digest"})
        if self.refusal_digest != sha256_json(unsigned):
            raise ValueError("refusal record digest mismatch")
        return self


class RevisionAuthorization(_StrictModel):
    schema_version: Literal["bfcl-refusal-revision-authorization-v1"]
    refusal_digest: StrictStr
    parent_session_digest: StrictStr
    action: RevisionAction
    authorized_by: StrictStr
    authorization_code: StrictStr
    authorized_at: datetime
    authorization_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RevisionAuthorization:
        _validate_digest(self.refusal_digest, "refusal digest")
        _validate_digest(self.parent_session_digest, "parent session digest")
        _validate_identifier(self.authorized_by, "authorized_by")
        _validate_code(self.authorization_code, "authorization_code")
        if self.authorized_at.tzinfo is None:
            raise ValueError("authorized_at must be timezone-aware")
        unsigned = self.model_dump(mode="json", exclude={"authorization_digest"})
        if self.authorization_digest != sha256_json(unsigned):
            raise ValueError("revision authorization digest mismatch")
        return self


def _json_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise RefusalRecordError(
            "timestamp_invalid",
            "refusal timestamps must be timezone-aware",
        )
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_digest(value: str, field: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")


def _validate_identifier(value: str, field: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded safe identifier")


def _validate_code(value: str, field: str) -> None:
    if _CODE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded stable code")
    segments = set(value.replace("-", ".").replace("_", ".").split("."))
    if segments & _FORBIDDEN_CODES or value in _FORBIDDEN_CODES:
        raise ValueError(f"{field} cannot describe model output, metrics, or scores")


def build_sanitized_finding(
    *,
    finding_code: str,
    classification: RefusalClassification,
    reason_code: str,
    artifact_role: str | None = None,
    evidence_digests: tuple[str, ...] = (),
) -> SanitizedFinding:
    unsigned = {
        "finding_code": finding_code,
        "classification": classification.value,
        "reason_code": reason_code,
        "artifact_role": artifact_role,
        "evidence_digests": sorted(set(evidence_digests)),
    }
    return SanitizedFinding.model_validate(
        {**unsigned, "finding_digest": sha256_json(unsigned)}
    )


def build_refusal_record(
    *,
    tenant_id: str,
    run_id: str,
    session_digest: str,
    primary_classification: RefusalClassification,
    findings: tuple[SanitizedFinding, ...],
    refused_at: datetime,
) -> RefusalRecord:
    canonical_findings = tuple(sorted(findings, key=lambda item: item.finding_digest))
    unsigned = {
        "schema_version": REFUSAL_RECORD_VERSION,
        "tenant_id": tenant_id,
        "run_id": run_id,
        "session_digest": session_digest,
        "primary_classification": primary_classification.value,
        "findings": [
            finding.model_dump(mode="json") for finding in canonical_findings
        ],
        "refused_at": _json_datetime(refused_at),
    }
    return RefusalRecord.model_validate(
        {**unsigned, "refusal_digest": sha256_json(unsigned)}
    )


def authorize_next_revision(
    record: RefusalRecord,
    *,
    action: RevisionAction,
    authorized_by: str,
    authorization_code: str,
    authorized_at: datetime,
) -> RevisionAuthorization:
    required = REQUIRED_ACTION[record.primary_classification]
    if action != required:
        raise RefusalRecordError(
            "revision_action_mismatch",
            f"{record.primary_classification.value} requires action {required.value}",
        )
    unsigned = {
        "schema_version": REVISION_AUTHORIZATION_VERSION,
        "refusal_digest": record.refusal_digest,
        "parent_session_digest": record.session_digest,
        "action": action.value,
        "authorized_by": authorized_by,
        "authorization_code": authorization_code,
        "authorized_at": _json_datetime(authorized_at),
    }
    return RevisionAuthorization.model_validate(
        {**unsigned, "authorization_digest": sha256_json(unsigned)}
    )


def verify_next_revision_authorization(
    record: RefusalRecord,
    authorization: RevisionAuthorization | None,
    *,
    parent_session_digest: str,
) -> RevisionAction:
    if authorization is None:
        raise RefusalRecordError(
            "revision_authorization_required",
            "a refusal cannot create a new revision without operator authorization",
        )
    if (
        authorization.refusal_digest != record.refusal_digest
        or authorization.parent_session_digest != record.session_digest
        or parent_session_digest != record.session_digest
    ):
        raise RefusalRecordError(
            "revision_authorization_stale",
            "revision authorization does not bind this refusal and parent session",
        )
    required = REQUIRED_ACTION[record.primary_classification]
    if authorization.action != required:
        raise RefusalRecordError(
            "revision_action_mismatch",
            f"authorization action must be {required.value}",
        )
    return authorization.action


def _require_active_lease(
    lease: WorkspaceLease,
    *,
    tenant_id: str,
    run_id: str,
) -> None:
    if (
        not lease.active
        or lease.metadata.tenant_id != tenant_id
        or lease.metadata.run_id != run_id
    ):
        raise RefusalRecordError(
            "workspace_lease_required",
            "persisting refusal state requires the active matching workspace lease",
        )


def persist_refusal_record(
    record: RefusalRecord,
    root: Path,
    *,
    lease: WorkspaceLease,
    event_sink: AuthoringEventSink | None = None,
) -> Path:
    _require_active_lease(
        lease,
        tenant_id=record.tenant_id,
        run_id=record.run_id,
    )
    try:
        path = RevisionStore(root).put_json(
            record.refusal_digest,
            {REFUSAL_FILE_NAME: record.model_dump(mode="json")},
        )
    except RevisionStoreError as exc:
        raise RefusalRecordError(exc.code, exc.detail) from exc
    if event_sink is not None:
        emit_authoring_event(
            event_sink,
            "refusal_recorded",
            RefusalPayload(
                refusal_digest=record.refusal_digest,
                primary_classification=record.primary_classification.value,
                finding_codes=tuple(
                    sorted(finding.finding_code for finding in record.findings)
                ),
                reason_codes=tuple(
                    sorted({finding.reason_code for finding in record.findings})
                ),
            ),
            tenant_id=record.tenant_id,
            run_id=record.run_id,
            session_digest=record.session_digest,
        )
    return path


def persist_revision_authorization(
    record: RefusalRecord,
    authorization: RevisionAuthorization,
    root: Path,
    *,
    lease: WorkspaceLease,
    event_sink: AuthoringEventSink | None = None,
) -> Path:
    _require_active_lease(
        lease,
        tenant_id=record.tenant_id,
        run_id=record.run_id,
    )
    verify_next_revision_authorization(
        record,
        authorization,
        parent_session_digest=record.session_digest,
    )
    try:
        path = RevisionStore(root).put_json(
            authorization.authorization_digest,
            {AUTHORIZATION_FILE_NAME: authorization.model_dump(mode="json")},
        )
    except RevisionStoreError as exc:
        raise RefusalRecordError(exc.code, exc.detail) from exc
    if event_sink is not None:
        emit_authoring_event(
            event_sink,
            "revision_authorized",
            RevisionAuthorizationPayload(
                authorization_digest=authorization.authorization_digest,
                refusal_digest=authorization.refusal_digest,
                parent_session_digest=authorization.parent_session_digest,
                action=authorization.action.value,
                authorization_code=authorization.authorization_code,
            ),
            tenant_id=record.tenant_id,
            run_id=record.run_id,
            session_digest=record.session_digest,
        )
    return path


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RefusalRecordError(
                "refusal_artifact_invalid",
                f"duplicate JSON key {key!r}",
            )
        result[key] = value
    return result


def _load_artifact(
    root: Path,
    content_address: str,
    file_name: str,
) -> dict[str, Any]:
    store = RevisionStore(root)
    try:
        manifest = store.verify(content_address)
    except RevisionStoreError as exc:
        raise RefusalRecordError(exc.code, exc.detail) from exc
    if tuple(item.path for item in manifest.artifacts) != (file_name,):
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            f"content-addressed artifact must contain only {file_name}",
        )
    path = store.root / content_address.removeprefix("sha256:") / file_name
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_mapping,
        )
    except RefusalRecordError:
        raise
    except Exception as exc:
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            f"cannot parse {file_name}: {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(document, dict):
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            f"{file_name} must contain a JSON object",
        )
    return document


def load_refusal_record(root: Path, refusal_digest: str) -> RefusalRecord:
    try:
        record = RefusalRecord.model_validate(
            _load_artifact(root, refusal_digest, REFUSAL_FILE_NAME)
        )
    except RefusalRecordError:
        raise
    except Exception as exc:
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            f"cannot validate refusal record: {type(exc).__name__}: {exc}",
        ) from exc
    if record.refusal_digest != refusal_digest:
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            "refusal record does not match its content address",
        )
    return record


def load_revision_authorization(
    root: Path,
    authorization_digest: str,
) -> RevisionAuthorization:
    try:
        authorization = RevisionAuthorization.model_validate(
            _load_artifact(
                root,
                authorization_digest,
                AUTHORIZATION_FILE_NAME,
            )
        )
    except RefusalRecordError:
        raise
    except Exception as exc:
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            f"cannot validate revision authorization: {type(exc).__name__}: {exc}",
        ) from exc
    if authorization.authorization_digest != authorization_digest:
        raise RefusalRecordError(
            "refusal_artifact_invalid",
            "revision authorization does not match its content address",
        )
    return authorization
