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

"""Single-writer workspace leases with explicit, audited stale recovery."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import stat
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on unsupported platforms
    fcntl = None  # type: ignore[assignment]

LOCK_METADATA_VERSION: Literal["bfcl-workspace-lock-v1"] = "bfcl-workspace-lock-v1"
RECOVERY_AUDIT_VERSION: Literal["bfcl-workspace-lock-recovery-v1"] = "bfcl-workspace-lock-recovery-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

Clock = Callable[[], datetime]

_RECOVERY_BY_CODE = {
    "workspace_locked": "wait for the live owner; never steal its lock",
    "stale_lock_recovery_required": "perform explicit recovery with an actor and reason",
    "orphan_lease_not_expired": "wait for lease expiry before audited recovery",
    "lock_metadata_invalid": "preserve the lock artifact and recover manually",
    "lock_path_unsafe": "replace the lock path with a regular workspace-owned file",
    "workspace_namespace_invalid": "use bounded tenant and run identifiers",
}


class WorkspaceLockError(ValueError):
    """A stable, machine-classifiable workspace-lock refusal."""

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
            "preserve lock state and retry only after operator review",
        )
        super().__init__(f"{code}: {detail}; recovery: {self.recovery}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LockMetadata(_StrictModel):
    schema_version: Literal["bfcl-workspace-lock-v1"]
    tenant_id: StrictStr
    run_id: StrictStr
    lease_id: StrictStr
    host: StrictStr
    pid: StrictInt
    created_at: datetime
    renewed_at: datetime
    lease_expires_at: datetime
    metadata_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> LockMetadata:
        _validate_identifier(self.tenant_id, "tenant_id")
        _validate_identifier(self.run_id, "run_id")
        try:
            uuid.UUID(self.lease_id)
        except ValueError as exc:
            raise ValueError("lease_id must be a UUID") from exc
        if not self.host or self.pid <= 0:
            raise ValueError("lock host and pid must identify an owner")
        timestamps = (self.created_at, self.renewed_at, self.lease_expires_at)
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("lock timestamps must be timezone-aware")
        if self.created_at > self.renewed_at:
            raise ValueError("lock renewed_at cannot precede created_at")
        if self.lease_expires_at <= self.renewed_at:
            raise ValueError("lock lease must expire after renewed_at")
        unsigned = self.model_dump(mode="json", exclude={"metadata_digest"})
        if self.metadata_digest != sha256_json(unsigned):
            raise ValueError("lock metadata digest mismatch")
        return self


class RecoveryAuditRecord(_StrictModel):
    schema_version: Literal["bfcl-workspace-lock-recovery-v1"]
    tenant_id: StrictStr
    run_id: StrictStr
    previous_lease_id: StrictStr
    previous_metadata_digest: StrictStr
    recovered_by: StrictStr
    reason: StrictStr
    recovered_at: datetime
    record_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RecoveryAuditRecord:
        _validate_identifier(self.tenant_id, "tenant_id")
        _validate_identifier(self.run_id, "run_id")
        if not self.recovered_by.strip() or not self.reason.strip():
            raise ValueError("stale recovery requires an actor and reason")
        if self.recovered_at.tzinfo is None:
            raise ValueError("recovered_at must be timezone-aware")
        unsigned = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("recovery audit digest mismatch")
        return self


def _validate_identifier(value: str, field: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise WorkspaceLockError(
            "workspace_namespace_invalid",
            f"{field} must match {_IDENTIFIER.pattern}",
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalized_now(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None:
        raise WorkspaceLockError(
            "clock_invalid",
            "workspace lock clock must return a timezone-aware datetime",
        )
    return value.astimezone(timezone.utc)


def _json_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    os.fsync(descriptor)


def _append_descriptor(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    os.fsync(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_metadata(payload: bytes) -> LockMetadata | None:
    if not payload:
        return None
    try:
        document = json.loads(payload.decode("utf-8"))
        return LockMetadata.model_validate(document)
    except Exception as exc:
        raise WorkspaceLockError(
            "lock_metadata_invalid",
            f"cannot verify existing lock metadata: {type(exc).__name__}: {exc}",
        ) from exc


def _build_metadata(
    *,
    tenant_id: str,
    run_id: str,
    lease_id: str,
    host: str,
    pid: int,
    created_at: datetime,
    renewed_at: datetime,
    lease_seconds: float,
) -> LockMetadata:
    unsigned: dict[str, Any] = {
        "schema_version": LOCK_METADATA_VERSION,
        "tenant_id": tenant_id,
        "run_id": run_id,
        "lease_id": lease_id,
        "host": host,
        "pid": pid,
        "created_at": _json_datetime(created_at),
        "renewed_at": _json_datetime(renewed_at),
        "lease_expires_at": _json_datetime(renewed_at + timedelta(seconds=lease_seconds)),
    }
    return LockMetadata.model_validate({**unsigned, "metadata_digest": sha256_json(unsigned)})


def _lock_exclusive_nonblocking(descriptor: int) -> None:
    if fcntl is None or platform.system() not in {"Darwin", "Linux"}:
        raise WorkspaceLockError(
            "lock_backend_unsupported",
            f"no approved workspace-lock backend for {platform.system()}",
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise WorkspaceLockError(
            "workspace_locked",
            "workspace is held by a live owner",
        ) from exc


def _unlock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _open_regular(path: Path, flags: int, mode: int = 0o600) -> int:
    safe_flags = flags
    if hasattr(os, "O_NOFOLLOW"):
        safe_flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, safe_flags, mode)
    except OSError as exc:
        raise WorkspaceLockError(
            "lock_path_unsafe",
            f"cannot safely open lock artifact {path.name}: {type(exc).__name__}",
        ) from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise WorkspaceLockError(
            "lock_path_unsafe",
            f"lock artifact must be a regular file: {path.name}",
        )
    return descriptor


class WorkspaceLease:
    """An acquired OS lock plus renewable, digest-bound owner metadata."""

    def __init__(
        self,
        *,
        descriptor: int,
        path: Path,
        metadata: LockMetadata,
        lease_seconds: float,
        clock: Clock,
    ) -> None:
        self._descriptor = descriptor
        self.path = path
        self.metadata = metadata
        self._lease_seconds = lease_seconds
        self._clock = clock
        self._released = False
        self._metadata_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_error: BaseException | None = None
        self._heartbeat = threading.Thread(
            target=self._renew_loop,
            name="bfcl-workspace-lease-renewal",
            daemon=True,
        )
        self._heartbeat.start()

    @property
    def active(self) -> bool:
        return not self._released

    def _renew_loop(self) -> None:
        interval = max(0.1, self._lease_seconds / 3.0)
        while not self._heartbeat_stop.wait(interval):
            try:
                self.renew()
            except BaseException as exc:
                self._heartbeat_error = exc
                return

    def renew(self) -> LockMetadata:
        with self._metadata_lock:
            if self._released:
                raise WorkspaceLockError(
                    "lease_released",
                    "cannot renew a released workspace lease",
                )
            observed = _parse_metadata(_read_descriptor(self._descriptor))
            if observed is None or observed.lease_id != self.metadata.lease_id:
                raise WorkspaceLockError(
                    "lease_identity_mismatch",
                    "lock metadata no longer belongs to this lease",
                )
            now = _normalized_now(self._clock)
            renewed = _build_metadata(
                tenant_id=self.metadata.tenant_id,
                run_id=self.metadata.run_id,
                lease_id=self.metadata.lease_id,
                host=self.metadata.host,
                pid=self.metadata.pid,
                created_at=self.metadata.created_at,
                renewed_at=now,
                lease_seconds=self._lease_seconds,
            )
            _write_descriptor(
                self._descriptor,
                (canonical_json(renewed.model_dump(mode="json")) + "\n").encode("utf-8"),
            )
            self.metadata = renewed
            return renewed

    def release(self) -> None:
        if self._released:
            return
        self._heartbeat_stop.set()
        if threading.current_thread() is not self._heartbeat:
            self._heartbeat.join()
        try:
            with self._metadata_lock:
                observed = _parse_metadata(_read_descriptor(self._descriptor))
                if observed is None or observed.lease_id != self.metadata.lease_id:
                    raise WorkspaceLockError(
                        "lease_identity_mismatch",
                        "refusing to clear metadata owned by another lease",
                    )
                _write_descriptor(self._descriptor, b"")
        finally:
            _unlock(self._descriptor)
            os.close(self._descriptor)
            self._released = True

    def __enter__(self) -> WorkspaceLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class WorkspaceLock:
    """Resolve one tenant/run namespace and acquire its single-writer lease."""

    def __init__(
        self,
        root: Path,
        *,
        tenant_id: str,
        run_id: str,
        lease_seconds: float = 60.0,
        clock: Clock = _utc_now,
        host: str | None = None,
        pid: int | None = None,
    ) -> None:
        _validate_identifier(tenant_id, "tenant_id")
        _validate_identifier(run_id, "run_id")
        if lease_seconds <= 0:
            raise WorkspaceLockError(
                "lease_duration_invalid",
                "lease_seconds must be positive",
            )
        self.root = root.expanduser().absolute()
        self.tenant_id = tenant_id
        self.run_id = run_id
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.host = host or socket.gethostname()
        self.pid = os.getpid() if pid is None else pid

    @property
    def lock_path(self) -> Path:
        return self.root / self.tenant_id / f"{self.run_id}.lock"

    @property
    def recovery_audit_path(self) -> Path:
        return self.root / self.tenant_id / f"{self.run_id}.recovery.jsonl"

    def _prepare_namespace(self) -> None:
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise WorkspaceLockError(
                "workspace_namespace_invalid",
                f"lock root must be a real directory: {self.root}",
            )
        self.root.mkdir(parents=True, exist_ok=True)
        tenant_root = self.root / self.tenant_id
        if tenant_root.exists() and (tenant_root.is_symlink() or not tenant_root.is_dir()):
            raise WorkspaceLockError(
                "workspace_namespace_invalid",
                f"tenant lock root must be a real directory: {tenant_root}",
            )
        tenant_root.mkdir(exist_ok=True)

    def _append_recovery_audit(
        self,
        previous: LockMetadata,
        *,
        recovered_by: str,
        reason: str,
        recovered_at: datetime,
    ) -> RecoveryAuditRecord:
        unsigned: dict[str, Any] = {
            "schema_version": RECOVERY_AUDIT_VERSION,
            "tenant_id": self.tenant_id,
            "run_id": self.run_id,
            "previous_lease_id": previous.lease_id,
            "previous_metadata_digest": previous.metadata_digest,
            "recovered_by": recovered_by.strip(),
            "reason": reason.strip(),
            "recovered_at": _json_datetime(recovered_at),
        }
        record = RecoveryAuditRecord.model_validate({**unsigned, "record_digest": sha256_json(unsigned)})
        payload = (canonical_json(record.model_dump(mode="json")) + "\n").encode("utf-8")
        descriptor = _open_regular(
            self.recovery_audit_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        )
        try:
            _append_descriptor(descriptor, payload)
        finally:
            os.close(descriptor)
        _fsync_directory(self.recovery_audit_path.parent)
        return record

    def acquire(
        self,
        *,
        recover_stale: bool = False,
        recovered_by: str | None = None,
        recovery_reason: str | None = None,
    ) -> WorkspaceLease:
        self._prepare_namespace()
        descriptor = _open_regular(self.lock_path, os.O_RDWR | os.O_CREAT)
        try:
            _lock_exclusive_nonblocking(descriptor)
            observed = _parse_metadata(_read_descriptor(descriptor))
            now = _normalized_now(self.clock)
            if observed is not None:
                if observed.tenant_id != self.tenant_id or observed.run_id != self.run_id:
                    raise WorkspaceLockError(
                        "lock_metadata_mismatch",
                        "existing lock metadata names another workspace",
                    )
                if observed.lease_expires_at > now:
                    raise WorkspaceLockError(
                        "orphan_lease_not_expired",
                        "an orphaned lease cannot be recovered before expiry",
                    )
                if not recover_stale:
                    raise WorkspaceLockError(
                        "stale_lock_recovery_required",
                        "expired lock metadata requires explicit audited recovery",
                    )
                if not recovered_by or not recovery_reason:
                    raise WorkspaceLockError(
                        "stale_recovery_context_required",
                        "stale recovery requires recovered_by and recovery_reason",
                    )
                self._append_recovery_audit(
                    observed,
                    recovered_by=recovered_by,
                    reason=recovery_reason,
                    recovered_at=now,
                )

            metadata = _build_metadata(
                tenant_id=self.tenant_id,
                run_id=self.run_id,
                lease_id=str(uuid.uuid4()),
                host=self.host,
                pid=self.pid,
                created_at=now,
                renewed_at=now,
                lease_seconds=self.lease_seconds,
            )
            _write_descriptor(
                descriptor,
                (canonical_json(metadata.model_dump(mode="json")) + "\n").encode("utf-8"),
            )
            return WorkspaceLease(
                descriptor=descriptor,
                path=self.lock_path,
                metadata=metadata,
                lease_seconds=self.lease_seconds,
                clock=self.clock,
            )
        except Exception:
            _unlock(descriptor)
            os.close(descriptor)
            raise

    def cleanup_orphan_staging(
        self,
        workspace: Path,
        *,
        lease: WorkspaceLease,
        minimum_age_seconds: float,
    ) -> tuple[Path, ...]:
        """Remove only old staging trees while this namespace lease is held."""
        if minimum_age_seconds <= 0:
            raise WorkspaceLockError(
                "staging_age_invalid",
                "minimum_age_seconds must be positive",
            )
        if (
            not lease.active
            or lease.path != self.lock_path
            or lease.metadata.tenant_id != self.tenant_id
            or lease.metadata.run_id != self.run_id
        ):
            raise WorkspaceLockError(
                "lease_identity_mismatch",
                "orphan staging cleanup requires this workspace's active lease",
            )
        root = workspace.expanduser().absolute()
        if root.is_symlink() or not root.is_dir():
            raise WorkspaceLockError(
                "workspace_namespace_invalid",
                "staging cleanup root must be a real directory",
            )
        cutoff = _normalized_now(self.clock).timestamp() - minimum_age_seconds
        removed: list[Path] = []
        for candidate in sorted(root.rglob(".*.staging-*")):
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            entries = [candidate, *candidate.rglob("*")]
            if any(path.is_symlink() for path in entries):
                raise WorkspaceLockError(
                    "staging_path_unsafe",
                    f"orphan staging tree contains a symbolic link: {candidate}",
                )
            if max(path.stat().st_mtime for path in entries) > cutoff:
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
        return tuple(removed)
