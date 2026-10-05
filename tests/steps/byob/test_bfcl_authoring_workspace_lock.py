from __future__ import annotations

import json
import multiprocessing
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty

import pytest

from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import (
    LOCK_METADATA_VERSION,
    RecoveryAuditRecord,
    WorkspaceLock,
    WorkspaceLockError,
    _build_metadata,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)
from nemotron.steps.byob.scripts import bfcl_author

START = datetime(2026, 8, 28, 10, 0, tzinfo=UTC)


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _contend_for_lock(root: str, result: multiprocessing.Queue[str]) -> None:
    try:
        WorkspaceLock(
            Path(root),
            tenant_id="tenant-a",
            run_id="run-a",
        ).acquire(
            recover_stale=True,
            recovered_by="contender",
            recovery_reason="attempted takeover",
        )
    except WorkspaceLockError as exc:
        result.put(f"{exc.code}|{exc.recovery}")
    else:
        result.put("acquired")


def test_lease_records_namespace_owner_and_renews(tmp_path: Path) -> None:
    clock = _MutableClock(START)
    lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
        lease_seconds=30,
        clock=clock,
        host="worker-a",
        pid=1234,
    )
    lease = lock.acquire()

    assert lease.metadata.schema_version == LOCK_METADATA_VERSION
    assert lease.metadata.tenant_id == "tenant-a"
    assert lease.metadata.run_id == "run-a"
    assert lease.metadata.host == "worker-a"
    assert lease.metadata.pid == 1234
    original_expiry = lease.metadata.lease_expires_at

    clock.value += timedelta(seconds=10)
    renewed = lease.renew()
    assert renewed.lease_id == lease.metadata.lease_id
    assert renewed.lease_expires_at > original_expiry
    lease.release()

    with lock.acquire() as replacement:
        assert replacement.metadata.lease_id != renewed.lease_id


def test_live_owner_cannot_be_stolen_by_another_process(tmp_path: Path) -> None:
    root = tmp_path / "locks"
    lease = WorkspaceLock(
        root,
        tenant_id="tenant-a",
        run_id="run-a",
        lease_seconds=1,
    ).acquire()
    context = multiprocessing.get_context("spawn")
    result: multiprocessing.Queue[str] = context.Queue()
    contender = context.Process(target=_contend_for_lock, args=(str(root), result))
    contender.start()
    contender.join(timeout=15)
    try:
        assert contender.exitcode == 0
        code, recovery = result.get(timeout=2).split("|", 1)
        assert code == "workspace_locked"
        assert recovery
    except Empty as exc:
        raise AssertionError("contender returned no lock verdict") from exc
    finally:
        if contender.is_alive():
            contender.terminate()
            contender.join(timeout=5)
        lease.release()


def test_namespaces_allow_independent_writers(tmp_path: Path) -> None:
    root = tmp_path / "locks"
    first = WorkspaceLock(root, tenant_id="tenant-a", run_id="run-a").acquire()
    second = WorkspaceLock(root, tenant_id="tenant-a", run_id="run-b").acquire()
    third = WorkspaceLock(root, tenant_id="tenant-b", run_id="run-a").acquire()
    try:
        assert len({first.path, second.path, third.path}) == 3
    finally:
        third.release()
        second.release()
        first.release()


def test_stale_recovery_is_explicit_and_audited(tmp_path: Path) -> None:
    clock = _MutableClock(START + timedelta(minutes=2))
    lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
        lease_seconds=30,
        clock=clock,
    )
    lock.lock_path.parent.mkdir(parents=True)
    stale = _build_metadata(
        tenant_id="tenant-a",
        run_id="run-a",
        lease_id="00000000-0000-4000-8000-000000000001",
        host="crashed-worker",
        pid=4321,
        created_at=START,
        renewed_at=START,
        lease_seconds=30,
    )
    lock.lock_path.write_text(
        canonical_json(stale.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceLockError) as implicit:
        lock.acquire()
    assert implicit.value.code == "stale_lock_recovery_required"

    lease = lock.acquire(
        recover_stale=True,
        recovered_by="operator@example.test",
        recovery_reason="worker host was retired",
    )
    try:
        lines = lock.recovery_audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = RecoveryAuditRecord.model_validate(json.loads(lines[0]))
        assert record.previous_lease_id == stale.lease_id
        assert record.previous_metadata_digest == stale.metadata_digest
        assert record.recovered_by == "operator@example.test"
    finally:
        lease.release()


def test_unexpired_orphan_is_not_recovered(tmp_path: Path) -> None:
    clock = _MutableClock(START)
    lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
        clock=clock,
    )
    lock.lock_path.parent.mkdir(parents=True)
    metadata = _build_metadata(
        tenant_id="tenant-a",
        run_id="run-a",
        lease_id="00000000-0000-4000-8000-000000000002",
        host="worker",
        pid=100,
        created_at=START,
        renewed_at=START,
        lease_seconds=60,
    )
    lock.lock_path.write_text(
        canonical_json(metadata.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceLockError) as refused:
        lock.acquire(
            recover_stale=True,
            recovered_by="operator",
            recovery_reason="too early",
        )
    assert refused.value.code == "orphan_lease_not_expired"
    assert not lock.recovery_audit_path.exists()


def test_malformed_lock_metadata_fails_closed(tmp_path: Path) -> None:
    lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    lock.lock_path.parent.mkdir(parents=True)
    lock.lock_path.write_text('{"schema_version":"unknown"}\n', encoding="utf-8")

    with pytest.raises(WorkspaceLockError) as refused:
        lock.acquire()
    assert refused.value.code == "lock_metadata_invalid"


def test_lock_file_symlink_cannot_redirect_owner_metadata(tmp_path: Path) -> None:
    lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    lock.lock_path.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")
    lock.lock_path.symlink_to(victim)

    with pytest.raises(WorkspaceLockError) as refused:
        lock.acquire()
    assert refused.value.code == "lock_path_unsafe"
    assert victim.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize(
    ("tenant_id", "run_id"),
    [
        ("../tenant", "run"),
        ("tenant", "../run"),
        ("tenant/name", "run"),
        ("tenant", ""),
    ],
)
def test_workspace_namespace_rejects_path_injection(
    tmp_path: Path,
    tenant_id: str,
    run_id: str,
) -> None:
    with pytest.raises(WorkspaceLockError) as refused:
        WorkspaceLock(
            tmp_path / "locks",
            tenant_id=tenant_id,
            run_id=run_id,
        )
    assert refused.value.code == "workspace_namespace_invalid"


def test_orphan_staging_cleanup_requires_lease_and_age(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    old = workspace / ".intake.staging-old"
    recent = workspace / ".drafts.staging-recent"
    old.mkdir(parents=True)
    recent.mkdir()
    (old / "partial.json").write_text("{}", encoding="utf-8")
    old_timestamp = START.timestamp() - 600
    os.utime(old / "partial.json", (old_timestamp, old_timestamp))
    os.utime(old, (old_timestamp, old_timestamp))
    clock = _MutableClock(START)
    lock = WorkspaceLock(
        workspace / ".locks",
        tenant_id="tenant-a",
        run_id="run-a",
        clock=clock,
    )
    lease = lock.acquire()
    try:
        removed = lock.cleanup_orphan_staging(
            workspace,
            lease=lease,
            minimum_age_seconds=300,
        )
    finally:
        lease.release()

    assert removed == (old,)
    assert not old.exists()
    assert recent.exists()

    with pytest.raises(WorkspaceLockError, match="active lease"):
        lock.cleanup_orphan_staging(
            workspace,
            lease=lease,
            minimum_age_seconds=300,
        )


def test_recover_workspace_cli_audits_takeover_and_removes_orphans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    orphan = workspace / ".intake.staging-crashed"
    orphan.mkdir(parents=True)
    old_timestamp = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(orphan, (old_timestamp, old_timestamp))

    lock = WorkspaceLock(
        workspace / ".locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    lock.lock_path.parent.mkdir(parents=True)
    stale = _build_metadata(
        tenant_id="tenant-a",
        run_id="run-a",
        lease_id="00000000-0000-4000-8000-000000000003",
        host="crashed-worker.invalid",
        pid=999_999,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        renewed_at=datetime(2020, 1, 1, tzinfo=UTC),
        lease_seconds=30,
    )
    lock.lock_path.write_text(
        canonical_json(stale.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "recover-workspace",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--actor",
            "operator@example.test",
            "--reason",
            "confirmed crashed worker",
            "--minimum-age-seconds",
            "300",
        ],
    )

    bfcl_author.main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "workspace_recovered"
    assert result["removed_staging_directories"] == [str(orphan)]
    audit = RecoveryAuditRecord.model_validate_json(
        lock.recovery_audit_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert audit.previous_lease_id == stale.lease_id
    assert audit.recovered_by == "operator@example.test"
