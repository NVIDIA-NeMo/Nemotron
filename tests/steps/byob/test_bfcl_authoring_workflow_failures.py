from __future__ import annotations

from pathlib import Path

from nemotron.steps.byob.runtime.authoring_workflow.quota import (
    RunQuota,
    RunQuotaError,
    RunQuotaLimits,
)
from nemotron.steps.byob.runtime.authoring_workflow.refusal import (
    RefusalRecordError,
    verify_next_revision_authorization,
)
from nemotron.steps.byob.runtime.authoring_workflow.resume import AuthoringResumeError
from nemotron.steps.byob.runtime.authoring_workflow.revision_store import (
    RevisionStore,
    RevisionStoreError,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import (
    WorkspaceLock,
    WorkspaceLockError,
)
from tests.steps.byob.test_bfcl_authoring_refusals import _record
from tests.steps.byob.test_bfcl_authoring_revisions import _committed_session

ADDRESS = "sha256:" + "d" * 64


def test_epic_ten_rejections_have_stable_codes_and_recovery(
    tmp_path: Path,
) -> None:
    refusals: list[
        RevisionStoreError
        | WorkspaceLockError
        | AuthoringResumeError
        | RefusalRecordError
        | RunQuotaError
    ] = []

    store = RevisionStore(tmp_path / "revisions")
    revision = store.put(ADDRESS, {"evidence.json": b"original"})
    (revision / "evidence.json").write_bytes(b"tampered")
    try:
        store.verify(ADDRESS)
    except RevisionStoreError as exc:
        refusals.append(exc)

    workspace_lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    with workspace_lock.acquire():
        try:
            workspace_lock.acquire()
        except WorkspaceLockError as exc:
            refusals.append(exc)

    workspace, gate, session_digest, _paths = _committed_session(
        tmp_path,
    )
    (workspace / ".interrupted.staging-run").mkdir()
    try:
        gate.open(session_digest, command="draft")
    except AuthoringResumeError as exc:
        refusals.append(exc)

    try:
        verify_next_revision_authorization(
            _record(),
            None,
            parent_session_digest="sha256:" + "a" * 64,
        )
    except RefusalRecordError as exc:
        refusals.append(exc)

    quota = RunQuota(
        RunQuotaLimits(
            max_provider_calls=0,
            max_token_units=0,
            max_batch_size=1,
            max_wall_time_ms=1000,
        )
    )
    try:
        quota.reserve_provider_call(token_units=1, batch_size=1)
    except RunQuotaError as exc:
        refusals.append(exc)

    assert [refusal.code for refusal in refusals] == [
        "artifact_digest_mismatch",
        "workspace_locked",
        "partial_workspace_write",
        "revision_authorization_required",
        "provider_call_quota_exhausted",
    ]
    assert all(refusal.recovery.strip() for refusal in refusals)
