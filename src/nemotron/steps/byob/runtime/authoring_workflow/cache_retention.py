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

"""Reference-aware retention and auditable purge for authoring model I/O caches."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from nemotron.steps.byob.runtime.authoring_workflow.resume import (
    SESSION_STORE_DIRECTORY,
    ArtifactBinding,
    AuthoringResumeGate,
    AuthoringSessionState,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
    exclusive_model_io_cache_lock,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.provenance import (
    DraftProvenance,
    ProvenanceError,
)

CACHE_RETENTION_POLICY_VERSION: Literal["bfcl-cache-retention-policy-v1"] = (
    "bfcl-cache-retention-policy-v1"
)
CACHE_PURGE_PLAN_VERSION: Literal["bfcl-cache-purge-plan-v1"] = (
    "bfcl-cache-purge-plan-v1"
)
CACHE_PURGE_AUDIT_VERSION: Literal["bfcl-cache-purge-audit-v1"] = (
    "bfcl-cache-purge-audit-v1"
)
AUTHORING_CACHE_FILE_NAME = "authoring_io_cache.jsonl"
CACHE_PURGE_AUDIT_FILE_NAME = "cache_purge_audit.jsonl"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TERMINAL_PHASES = frozenset({"published", "refused"})


class CacheRetentionError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class CacheRetentionPolicy(_StrictModel):
    schema_version: Literal["bfcl-cache-retention-policy-v1"] = (
        CACHE_RETENTION_POLICY_VERSION
    )
    protect_referenced_requests: Literal[True] = True
    protect_uncommitted_active_heads: Literal[True] = True
    require_private_mode: Literal[True] = True

    @property
    def policy_digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class CachePurgePlan(_StrictModel):
    schema_version: Literal["bfcl-cache-purge-plan-v1"]
    cache_path_digest: StrictStr
    cache_content_hash_before: StrictStr
    retention_policy_digest: StrictStr
    session_scan_digest: StrictStr
    protected_all: StrictBool
    retained_request_hashes: tuple[StrictStr, ...]
    eligible_request_hashes: tuple[StrictStr, ...]
    plan_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> CachePurgePlan:
        for value in (
            self.cache_path_digest,
            self.cache_content_hash_before,
            self.retention_policy_digest,
            self.session_scan_digest,
            self.plan_digest,
            *self.retained_request_hashes,
            *self.eligible_request_hashes,
        ):
            _require_digest(value)
        _require_sorted_unique(self.retained_request_hashes)
        _require_sorted_unique(self.eligible_request_hashes)
        if set(self.retained_request_hashes) & set(self.eligible_request_hashes):
            raise ValueError("retained and eligible request hashes overlap")
        unsigned = self.model_dump(mode="json", exclude={"plan_digest"})
        if self.plan_digest != sha256_json(unsigned):
            raise ValueError("cache purge plan digest mismatch")
        return self


class CachePurgeAuditRecord(_StrictModel):
    schema_version: Literal["bfcl-cache-purge-audit-v1"]
    tenant_id: StrictStr
    run_id: StrictStr
    actor: StrictStr
    reason_code: StrictStr
    executed_at: StrictStr
    dry_run: StrictBool
    plan_digest: StrictStr
    cache_content_hash_before: StrictStr
    cache_content_hash_after: StrictStr
    retained_count: StrictInt
    purged_count: StrictInt
    eligible_request_hashes: tuple[StrictStr, ...]
    record_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> CachePurgeAuditRecord:
        for value in (self.tenant_id, self.run_id, self.actor, self.reason_code):
            if _IDENTIFIER.fullmatch(value) is None:
                raise ValueError("cache purge audit identifiers must be safe")
        for value in (
            self.plan_digest,
            self.cache_content_hash_before,
            self.cache_content_hash_after,
            self.record_digest,
            *self.eligible_request_hashes,
        ):
            _require_digest(value)
        _require_sorted_unique(self.eligible_request_hashes)
        if self.retained_count < 0 or self.purged_count < 0:
            raise ValueError("cache purge counts cannot be negative")
        try:
            timestamp = datetime.fromisoformat(self.executed_at)
        except ValueError as exc:
            raise ValueError("cache purge timestamp must be ISO-8601") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("cache purge timestamp must be timezone-aware")
        unsigned = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("cache purge audit digest mismatch")
        return self


def plan_authoring_cache_purge(
    workspace: Path,
    cache_path: Path,
    *,
    tenant_id: str,
    run_id: str,
    policy: CacheRetentionPolicy | None = None,
) -> CachePurgePlan:
    root, cache = _validate_scope(workspace, cache_path)
    selected_policy = policy or CacheRetentionPolicy()
    entries = ImmutableModelIOCache(cache).entry_documents()
    request_hashes = tuple(sorted(str(entry["request_hash"]) for entry in entries))
    protected, protected_all, session_scan_digest = _collect_protected_requests(
        root,
        cache,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    retained = request_hashes if protected_all else tuple(
        request_hash for request_hash in request_hashes if request_hash in protected
    )
    eligible = () if protected_all else tuple(
        request_hash for request_hash in request_hashes if request_hash not in protected
    )
    unsigned: dict[str, Any] = {
        "schema_version": CACHE_PURGE_PLAN_VERSION,
        "cache_path_digest": sha256_json(
            {"workspace_relative_path": cache.relative_to(root).as_posix()}
        ),
        "cache_content_hash_before": _digest_bytes(cache.read_bytes()),
        "retention_policy_digest": selected_policy.policy_digest,
        "session_scan_digest": session_scan_digest,
        "protected_all": protected_all,
        "retained_request_hashes": retained,
        "eligible_request_hashes": eligible,
    }
    return CachePurgePlan.model_validate(
        {**unsigned, "plan_digest": sha256_json(unsigned)}
    )


def purge_authoring_cache(
    workspace: Path,
    cache_path: Path,
    *,
    tenant_id: str,
    run_id: str,
    actor: str,
    reason_code: str,
    dry_run: bool,
    expected_plan_digest: str | None = None,
    policy: CacheRetentionPolicy | None = None,
) -> tuple[CachePurgePlan, CachePurgeAuditRecord]:
    gate = AuthoringResumeGate(
        workspace,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    _, validated_cache = _validate_scope(workspace, cache_path)
    with gate.workspace_lock.acquire():
        with exclusive_model_io_cache_lock(validated_cache):
            plan = plan_authoring_cache_purge(
                workspace,
                cache_path,
                tenant_id=tenant_id,
                run_id=run_id,
                policy=policy,
            )
            if (
                expected_plan_digest is not None
                and plan.plan_digest != expected_plan_digest
            ):
                raise CacheRetentionError(
                    "cache_purge_plan_stale",
                    "cache or session references changed after purge planning",
                )
            root, cache = _validate_scope(workspace, cache_path)
            if dry_run:
                after_hash = plan.cache_content_hash_before
            else:
                entries = {
                    str(entry["request_hash"]): entry
                    for entry in ImmutableModelIOCache(cache).entry_documents()
                }
                retained = [
                    entries[request_hash]
                    for request_hash in plan.retained_request_hashes
                ]
                _replace_cache_atomically(cache, retained)
                ImmutableModelIOCache(cache)
                after_hash = _digest_bytes(cache.read_bytes())
            audit = _build_audit(
                plan,
                tenant_id=tenant_id,
                run_id=run_id,
                actor=actor,
                reason_code=reason_code,
                dry_run=dry_run,
                after_hash=after_hash,
            )
            _append_audit(root / ".events" / CACHE_PURGE_AUDIT_FILE_NAME, audit)
            return plan, audit


def infer_authoring_cache_path(
    workspace: Path,
    *,
    tenant_id: str,
    run_id: str,
) -> Path:
    root = workspace.resolve()
    states = _load_session_states(root, tenant_id=tenant_id, run_id=run_id)
    states = tuple(
        state
        for state in states
        if state.tenant_id == tenant_id and state.run_id == run_id
    )
    parent_digests = {
        state.parent_session_digest
        for state in states
        if state.parent_session_digest is not None
    }
    heads = sorted(
        (state for state in states if state.session_digest not in parent_digests),
        key=lambda state: state.session_digest,
    )
    paths = {
        _resolve_binding(root, state.bindings.draft_provenance).parent
        / AUTHORING_CACHE_FILE_NAME
        for state in heads
        if state.bindings.draft_provenance is not None
    }
    if len(paths) != 1:
        raise CacheRetentionError(
            "cache_path_ambiguous",
            "provide --cache when the active session does not identify exactly one cache",
        )
    return next(iter(paths))


def load_cache_purge_audit(path: Path) -> tuple[CachePurgeAuditRecord, ...]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                raise CacheRetentionError(
                    "cache_purge_audit_invalid",
                    "audit stream contains an empty line",
                )
            records.append(
                CachePurgeAuditRecord.model_validate(
                    json.loads(line, object_pairs_hook=_unique_mapping)
                )
            )
    except CacheRetentionError:
        raise
    except Exception as exc:
        raise CacheRetentionError(
            "cache_purge_audit_invalid",
            f"cannot verify cache purge audit: {type(exc).__name__}",
        ) from exc
    return tuple(records)


def _collect_protected_requests(
    workspace: Path,
    cache_path: Path,
    *,
    tenant_id: str,
    run_id: str,
) -> tuple[set[str], bool, str]:
    states = _load_session_states(
        workspace,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    parent_digests = {
        state.parent_session_digest
        for state in states
        if state.parent_session_digest is not None
    }
    heads = [
        state for state in states if state.session_digest not in parent_digests
    ]
    protected: set[str] = set()
    provenance_records: list[dict[str, Any]] = []
    for state in states:
        binding = state.bindings.draft_provenance
        if binding is None:
            continue
        provenance_path = _resolve_binding(workspace, binding)
        if (
            provenance_path.parent / AUTHORING_CACHE_FILE_NAME
        ).resolve() != cache_path:
            continue
        document = _load_json(provenance_path)
        provenance = DraftProvenance(document=document)
        try:
            provenance.verify_digest()
        except ProvenanceError as exc:
            raise CacheRetentionError(
                "draft_provenance_invalid",
                "bound draft provenance failed digest verification",
            ) from exc
        calls = document.get("calls")
        if not isinstance(calls, list):
            raise CacheRetentionError(
                "draft_provenance_invalid",
                "bound draft provenance calls must be a list",
            )
        request_hashes = []
        for call in calls:
            request_hash = call.get("request_hash") if isinstance(call, dict) else None
            if not isinstance(request_hash, str):
                raise CacheRetentionError(
                    "draft_provenance_invalid",
                    "draft call lacks request_hash",
                )
            _require_digest(request_hash)
            request_hashes.append(request_hash)
        protected.update(request_hashes)
        provenance_records.append(
            {
                "session_digest": state.session_digest,
                "binding_digest": binding.digest,
                "request_hashes": sorted(set(request_hashes)),
            }
        )
    protected_all = any(
        state.phase not in _TERMINAL_PHASES
        and state.bindings.draft_provenance is None
        for state in heads
    )
    scan_document = {
        "session_digests": sorted(state.session_digest for state in states),
        "provenance": sorted(
            provenance_records,
            key=lambda item: str(item["session_digest"]),
        ),
        "protected_all": protected_all,
    }
    return protected, protected_all, sha256_json(scan_document)


def _load_session_states(
    workspace: Path,
    *,
    tenant_id: str,
    run_id: str,
) -> tuple[AuthoringSessionState, ...]:
    store = workspace / SESSION_STORE_DIRECTORY
    if not store.exists():
        return ()
    if store.is_symlink() or not store.is_dir():
        raise CacheRetentionError(
            "session_store_invalid",
            "session store must be a regular workspace directory",
        )
    gate = AuthoringResumeGate(
        workspace,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    states = []
    for candidate in sorted(store.iterdir(), key=lambda item: item.name):
        if not candidate.is_dir() or candidate.is_symlink():
            raise CacheRetentionError(
                "session_store_invalid",
                "session store contains a non-revision entry",
            )
        digest = f"sha256:{candidate.name}"
        try:
            state = gate.load_state(digest)
        except ValueError as exc:
            raise CacheRetentionError(
                "session_store_invalid",
                "session store contains an unverifiable revision",
            ) from exc
        states.append(state)
    return tuple(states)


def _resolve_binding(workspace: Path, binding: ArtifactBinding) -> Path:
    unresolved = workspace / binding.path
    current = workspace
    for part in Path(binding.path).parts:
        current /= part
        if current.is_symlink():
            raise CacheRetentionError(
                "cache_path_escape",
                "bound draft provenance traverses a symlink",
            )
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise CacheRetentionError(
            "cache_path_escape",
            "bound draft provenance escapes the workspace",
        ) from exc
    if not resolved.is_file():
        raise CacheRetentionError(
            "draft_provenance_missing",
            "bound draft provenance is not a regular file",
        )
    payload = resolved.read_bytes()
    observed = (
        sha256_json(_load_json(resolved))
        if binding.digest_kind == "canonical_json"
        else _digest_bytes(payload)
    )
    if observed != binding.digest:
        raise CacheRetentionError(
            "draft_provenance_drift",
            "bound draft provenance changed after session commit",
        )
    return resolved


def _validate_scope(workspace: Path, cache_path: Path) -> tuple[Path, Path]:
    root = workspace.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise CacheRetentionError(
            "workspace_invalid",
            "workspace must be a regular directory",
        )
    root = root.resolve()
    unresolved = cache_path.expanduser().absolute()
    try:
        relative = unresolved.relative_to(root)
    except ValueError as exc:
        raise CacheRetentionError(
            "cache_path_escape",
            "cache must be inside the authoring workspace",
        ) from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise CacheRetentionError(
                "cache_path_escape",
                "cache path must not traverse a symlink",
            )
    cache = unresolved.resolve()
    if cache.name != AUTHORING_CACHE_FILE_NAME:
        raise CacheRetentionError(
            "cache_kind_unsupported",
            "only authoring_io_cache.jsonl is eligible for this retention policy",
        )
    if not cache.is_file():
        raise CacheRetentionError(
            "cache_missing",
            "authoring model I/O cache is not a regular file",
        )
    if cache.stat().st_mode & 0o077:
        raise CacheRetentionError(
            "cache_access_too_broad",
            "authoring model I/O cache must not grant group or other access",
        )
    return root, cache


def _replace_cache_atomically(
    path: Path,
    retained_entries: list[dict[str, Any]],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.purge-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            for entry in retained_entries:
                encoded = (
                    json.dumps(
                        entry,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _build_audit(
    plan: CachePurgePlan,
    *,
    tenant_id: str,
    run_id: str,
    actor: str,
    reason_code: str,
    dry_run: bool,
    after_hash: str,
) -> CachePurgeAuditRecord:
    unsigned = {
        "schema_version": CACHE_PURGE_AUDIT_VERSION,
        "tenant_id": tenant_id,
        "run_id": run_id,
        "actor": actor,
        "reason_code": reason_code,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "plan_digest": plan.plan_digest,
        "cache_content_hash_before": plan.cache_content_hash_before,
        "cache_content_hash_after": after_hash,
        "retained_count": len(plan.retained_request_hashes),
        "purged_count": 0 if dry_run else len(plan.eligible_request_hashes),
        "eligible_request_hashes": plan.eligible_request_hashes,
    }
    return CachePurgeAuditRecord.model_validate(
        {**unsigned, "record_digest": sha256_json(unsigned)}
    )


def _append_audit(path: Path, audit: CachePurgeAuditRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise CacheRetentionError(
            "cache_purge_audit_invalid",
            "cache purge audit directory must not be a symlink",
        )
    payload = (
        json.dumps(
            audit.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.fstat(descriptor).st_mode & 0o077:
            raise CacheRetentionError(
                "cache_purge_audit_access_too_broad",
                "cache purge audit must not grant group or other access",
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_mapping,
        )
    except CacheRetentionError:
        raise
    except Exception as exc:
        raise CacheRetentionError(
            "artifact_invalid",
            f"cannot parse bound JSON artifact: {type(exc).__name__}",
        ) from exc
    if not isinstance(document, dict):
        raise CacheRetentionError(
            "artifact_invalid",
            "bound JSON artifact must be an object",
        )
    return document


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheRetentionError(
                "artifact_invalid",
                f"duplicate JSON key {key!r}",
            )
        result[key] = value
    return result


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _require_digest(value: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError("cache retention digest must be sha256:<64 lowercase hex>")


def _require_sorted_unique(values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError("cache retention hashes must be sorted and unique")
