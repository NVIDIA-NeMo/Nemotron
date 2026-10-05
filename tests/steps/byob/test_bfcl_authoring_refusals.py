from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.authoring_workflow.refusal import (
    REQUIRED_ACTION,
    RefusalClassification,
    RefusalRecord,
    RefusalRecordError,
    RevisionAction,
    SanitizedFinding,
    authorize_next_revision,
    build_refusal_record,
    build_sanitized_finding,
    load_refusal_record,
    load_revision_authorization,
    persist_refusal_record,
    persist_revision_authorization,
    verify_next_revision_authorization,
)
from nemotron.steps.byob.runtime.authoring_workflow.resume import AuthoringResumeError
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import WorkspaceLock
from tests.steps.byob.test_bfcl_authoring_revisions import _committed_session

SESSION_DIGEST = "sha256:" + "a" * 64
EVIDENCE_DIGEST = "sha256:" + "b" * 64
NOW = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)


def _record(
    classification: RefusalClassification = RefusalClassification.MODEL_OWNED_PROPOSAL,
    *,
    session_digest: str = SESSION_DIGEST,
) -> RefusalRecord:
    finding = build_sanitized_finding(
        finding_code="assertion_compilation_blocked",
        classification=classification,
        reason_code="unresolved_behavior",
        artifact_role="assertions",
        evidence_digests=(EVIDENCE_DIGEST,),
    )
    return build_refusal_record(
        tenant_id="tenant-a",
        run_id="run-a",
        session_digest=session_digest,
        primary_classification=classification,
        findings=(finding,),
        refused_at=NOW,
    )


def test_refusal_schema_cannot_carry_model_output_or_scores() -> None:
    record = _record()
    document = record.model_dump(mode="json")

    assert set(document["findings"][0]) == {
        "finding_code",
        "classification",
        "reason_code",
        "artifact_role",
        "evidence_digests",
        "finding_digest",
    }
    assert "score" not in json.dumps(document)
    with pytest.raises(ValidationError, match="Extra inputs"):
        SanitizedFinding.model_validate(
            {
                **document["findings"][0],
                "target_model_output": {"answer": "must not persist"},
            }
        )
    with pytest.raises(ValidationError, match="scores"):
        build_sanitized_finding(
            finding_code="quality_score",
            classification=RefusalClassification.MODEL_OWNED_PROPOSAL,
            reason_code="low_score",
        )


@pytest.mark.parametrize(
    "classification",
    list(RefusalClassification),
)
def test_each_stage_f_classification_requires_its_owned_action(
    classification: RefusalClassification,
) -> None:
    record = _record(classification)
    required = REQUIRED_ACTION[classification]
    authorization = authorize_next_revision(
        record,
        action=required,
        authorized_by="operator@example.test",
        authorization_code="reviewed_remediation",
        authorized_at=NOW,
    )

    assert (
        verify_next_revision_authorization(
            record,
            authorization,
            parent_session_digest=SESSION_DIGEST,
        )
        is required
    )
    wrong = next(action for action in RevisionAction if action is not required)
    with pytest.raises(RefusalRecordError) as refused:
        authorize_next_revision(
            record,
            action=wrong,
            authorized_by="operator@example.test",
            authorization_code="wrong_owner",
            authorized_at=NOW,
        )
    assert refused.value.code == "revision_action_mismatch"


def test_next_revision_requires_fresh_operator_authorization() -> None:
    record = _record()
    with pytest.raises(RefusalRecordError) as missing:
        verify_next_revision_authorization(
            record,
            None,
            parent_session_digest=SESSION_DIGEST,
        )
    assert missing.value.code == "revision_authorization_required"

    authorization = authorize_next_revision(
        record,
        action=RevisionAction.REVISE_PROPOSAL,
        authorized_by="operator@example.test",
        authorization_code="reviewed_remediation",
        authorized_at=NOW,
    )
    with pytest.raises(RefusalRecordError) as stale:
        verify_next_revision_authorization(
            record,
            authorization,
            parent_session_digest="sha256:" + "c" * 64,
        )
    assert stale.value.code == "revision_authorization_stale"


def test_refusal_and_authorization_persist_as_separate_immutable_records(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock = WorkspaceLock(
        workspace / ".locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    record = _record()
    authorization = authorize_next_revision(
        record,
        action=RevisionAction.REVISE_PROPOSAL,
        authorized_by="operator@example.test",
        authorization_code="reviewed_remediation",
        authorized_at=NOW,
    )

    with lock.acquire() as lease:
        refusal_path = persist_refusal_record(
            record,
            workspace / "refusals",
            lease=lease,
        )
        authorization_path = persist_revision_authorization(
            record,
            authorization,
            workspace / "refusal_authorizations",
            lease=lease,
        )

    assert refusal_path != authorization_path
    assert load_refusal_record(
        workspace / "refusals",
        record.refusal_digest,
    ) == record
    assert load_revision_authorization(
        workspace / "refusal_authorizations",
        authorization.authorization_digest,
    ) == authorization
    assert "revision_authorization" not in (
        refusal_path / "refusal.json"
    ).read_text(encoding="utf-8")

    with lock.acquire() as lease:
        with pytest.raises(RefusalRecordError) as duplicate:
            persist_refusal_record(
                record,
                workspace / "refusals",
                lease=lease,
            )
    assert duplicate.value.code == "revision_already_exists"


def test_persistence_requires_active_matching_workspace_lease(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock = WorkspaceLock(
        workspace / ".locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    lease = lock.acquire()
    lease.release()

    with pytest.raises(RefusalRecordError) as refused:
        persist_refusal_record(
            _record(),
            workspace / "refusals",
            lease=lease,
        )
    assert refused.value.code == "workspace_lease_required"


def test_refused_session_can_create_revision_only_with_bound_authorization(
    tmp_path: Path,
) -> None:
    workspace, gate, session_digest, _paths = _committed_session(
        tmp_path,
        phase="refused",
    )
    record = _record(session_digest=session_digest)
    authorization = authorize_next_revision(
        record,
        action=RevisionAction.REVISE_PROPOSAL,
        authorized_by="operator@example.test",
        authorization_code="reviewed_remediation",
        authorized_at=NOW,
    )
    with gate.workspace_lock.acquire() as lease:
        persist_refusal_record(
            record,
            workspace / "refusals",
            lease=lease,
        )
        persist_revision_authorization(
            record,
            authorization,
            workspace / "refusal_authorizations",
            lease=lease,
        )

    with pytest.raises(AuthoringResumeError) as missing:
        gate.open_authorized_revision(
            session_digest,
            refusal_digest=record.refusal_digest,
            authorization_digest="sha256:" + "c" * 64,
        )
    assert missing.value.code == "revision_authorization_required"

    with gate.open_authorized_revision(
        session_digest,
        refusal_digest=record.refusal_digest,
        authorization_digest=authorization.authorization_digest,
    ) as resumed:
        assert resumed.verdict.command == "revise"
        assert resumed.verdict.authorized_action == "revise_proposal"
