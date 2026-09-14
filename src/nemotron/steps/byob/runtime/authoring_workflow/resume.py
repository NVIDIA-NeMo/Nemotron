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

"""Fail-closed authoring resume over immutable session states."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.authoring_workflow.refusal import (
    RefusalRecordError,
    load_refusal_record,
    load_revision_authorization,
    verify_next_revision_authorization,
)
from nemotron.steps.byob.runtime.authoring_workflow.revision_store import (
    RevisionStore,
    RevisionStoreError,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import (
    WorkspaceLease,
    WorkspaceLock,
    WorkspaceLockError,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.provenance import DraftProvenance
from nemotron.steps.byob.runtime.source_adapters.evidence import load_source_evidence

SESSION_VERSION: Literal["bfcl-authoring-session-v2"] = "bfcl-authoring-session-v2"
LEGACY_SESSION_VERSION = "bfcl-authoring-session-v1"
SESSION_FILE_NAME = "session.json"
SESSION_STORE_DIRECTORY = "session_states"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

AuthoringPhase = Literal[
    "initialized",
    "intake_complete",
    "questions_open",
    "evidence_revised",
    "exposure_authorized",
    "evidence_approved",
    "draft_complete",
    "pack_assembled",
    "review_ready",
    "release_approved",
    "frozen",
    "published",
    "refused",
]
AuthoringCommand = Literal[
    "intake",
    "answer",
    "apply_policy",
    "authorize_exposure",
    "approve_evidence",
    "draft",
    "assemble",
    "review",
    "approve_release",
    "freeze",
    "publish",
    "revise",
]

RESUMABILITY_MATRIX: Mapping[AuthoringPhase, tuple[AuthoringCommand, ...]] = {
    "initialized": ("intake",),
    "intake_complete": ("answer", "apply_policy", "authorize_exposure"),
    "questions_open": ("answer",),
    "evidence_revised": (
        "answer",
        "authorize_exposure",
    ),
    "exposure_authorized": ("approve_evidence",),
    "evidence_approved": ("draft",),
    # Assembly is where drafts become a loadable pack. A pack assembled outside the guided
    # workspace stays legal — review binds the pack it is handed either way — so `review`
    # remains reachable directly rather than making one command the only way in.
    "draft_complete": ("assemble", "review"),
    # Candidate validation is deliberately outside assembly and can expose reviewed
    # supplement defects (for example, missing negative cases).  Keep the previous
    # candidate immutable, but allow assembly to bind a new output as the next session
    # revision before review.
    "pack_assembled": ("assemble", "review"),
    "review_ready": ("approve_release",),
    "release_approved": ("freeze",),
    "frozen": ("publish",),
    "published": (),
    "refused": (),
}


class AuthoringResumeError(ValueError):
    """A stable resume refusal plus its safe operator recovery."""

    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactBinding(_StrictModel):
    path: StrictStr
    digest: StrictStr
    digest_kind: Literal["bytes", "canonical_json"] = "bytes"

    @model_validator(mode="after")
    def _validate(self) -> ArtifactBinding:
        _validate_relative_path(self.path)
        _validate_digest(self.digest, "artifact digest")
        return self


class ApprovalBinding(_StrictModel):
    artifact: ArtifactBinding
    evidence_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> ApprovalBinding:
        _validate_digest(self.evidence_digest, "approval evidence digest")
        return self


class SessionBindings(_StrictModel):
    source: ArtifactBinding
    evidence: ArtifactBinding
    resolved_config: ArtifactBinding
    source_identity_digest: StrictStr
    evidence_bundle_digest: StrictStr
    revision_content_address: StrictStr | None = None
    approval: ApprovalBinding | None = None
    draft_root: StrictStr | None = None
    draft_provenance: ArtifactBinding | None = None
    candidate_pack: ArtifactBinding | None = None
    exposure_authorization: ArtifactBinding | None = None
    review_packet: ArtifactBinding | None = None
    release_approval: ArtifactBinding | None = None
    frozen_manifest: ArtifactBinding | None = None
    publication_manifest: ArtifactBinding | None = None

    @model_validator(mode="after")
    def _validate(self) -> SessionBindings:
        _validate_digest(self.source_identity_digest, "source identity digest")
        _validate_digest(self.evidence_bundle_digest, "evidence bundle digest")
        if self.revision_content_address is not None:
            _validate_digest(self.revision_content_address, "revision content address")
            if self.revision_content_address != self.evidence_bundle_digest:
                raise ValueError("revision content address must equal evidence bundle digest")
        if self.draft_root is not None:
            _validate_relative_path(self.draft_root)
        if self.draft_provenance is not None and self.draft_root is None:
            raise ValueError("draft provenance requires a draft root")
        return self


class AuthoringSessionState(_StrictModel):
    schema_version: Literal[
        "bfcl-authoring-session-v1",
        "bfcl-authoring-session-v2",
    ]
    tenant_id: StrictStr
    run_id: StrictStr
    phase: AuthoringPhase
    bindings: SessionBindings
    parent_session_digest: StrictStr | None = None
    session_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> AuthoringSessionState:
        _validate_identifier(self.tenant_id, "tenant_id")
        _validate_identifier(self.run_id, "run_id")
        if self.parent_session_digest is not None:
            _validate_digest(self.parent_session_digest, "parent session digest")
            if self.parent_session_digest == self.session_digest:
                raise ValueError("session cannot name itself as parent")
        unsigned = self.model_dump(mode="json", exclude={"session_digest"})
        if self.schema_version == LEGACY_SESSION_VERSION:
            for field in (
                "candidate_pack",
                "exposure_authorization",
                "review_packet",
                "release_approval",
                "frozen_manifest",
                "publication_manifest",
            ):
                if getattr(self.bindings, field) is not None:
                    raise ValueError(f"legacy session cannot carry {field}")
                unsigned["bindings"].pop(field, None)
        if self.session_digest != sha256_json(unsigned):
            raise ValueError("authoring session digest mismatch")
        approval_phases = {
            "evidence_approved",
            "draft_complete",
            "pack_assembled",
            "review_ready",
            "release_approved",
            "frozen",
            "published",
        }
        if (
            self.schema_version == SESSION_VERSION
            and self.phase in {"exposure_authorized"}
            and self.bindings.exposure_authorization is None
        ):
            raise ValueError("exposure_authorized phase requires exposure authorization")
        if self.phase in approval_phases and self.bindings.approval is None:
            raise ValueError(f"phase {self.phase!r} requires evidence approval")
        drafted_phases = approval_phases - {"evidence_approved"}
        if self.phase in drafted_phases and self.bindings.draft_root is None:
            raise ValueError(f"phase {self.phase!r} requires a declared draft root")
        completed_draft_phases = drafted_phases
        if self.phase in completed_draft_phases and self.bindings.draft_provenance is None:
            raise ValueError(f"phase {self.phase!r} requires draft provenance")
        if (
            self.schema_version == SESSION_VERSION
            and self.phase == "pack_assembled"
            and self.bindings.candidate_pack is None
        ):
            raise ValueError("pack_assembled phase requires a candidate pack record")
        if self.schema_version == SESSION_VERSION and self.phase in {
            "review_ready",
            "release_approved",
            "frozen",
            "published",
        }:
            if self.bindings.review_packet is None:
                raise ValueError(f"phase {self.phase!r} requires review packet")
        if self.schema_version == SESSION_VERSION and self.phase in {
            "release_approved",
            "frozen",
            "published",
        }:
            if self.bindings.release_approval is None:
                raise ValueError(f"phase {self.phase!r} requires release approval")
        if (
            self.schema_version == SESSION_VERSION
            and self.phase in {"frozen", "published"}
            and self.bindings.frozen_manifest is None
        ):
            raise ValueError(f"phase {self.phase!r} requires frozen manifest")
        if (
            self.schema_version == SESSION_VERSION
            and self.phase == "published"
            and self.bindings.publication_manifest is None
        ):
            raise ValueError("published phase requires publication manifest")
        return self


class ResumeVerdict(_StrictModel):
    session_digest: StrictStr
    phase: AuthoringPhase
    command: AuthoringCommand
    permitted_commands: tuple[AuthoringCommand, ...]
    authorized_action: StrictStr | None = None


class ResumedAuthoringSession:
    """A verified verdict that retains exclusive workspace ownership."""

    def __init__(self, verdict: ResumeVerdict, lease: WorkspaceLease) -> None:
        self.verdict = verdict
        self.lease = lease

    def __enter__(self) -> ResumedAuthoringSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lease.release()


def _validate_identifier(value: str, field: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe workspace identifier")


def _validate_digest(value: str, field: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"artifact path must be a safe workspace-relative path: {value!r}")


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthoringResumeError(
                "session_invalid",
                f"duplicate JSON key {key!r}",
                recovery="preserve the workspace and resume from the last verified session",
            )
        result[key] = value
    return result


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _resolve_artifact(workspace: Path, binding: ArtifactBinding) -> Path:
    root = workspace.resolve()
    unresolved = root / binding.path
    relative = PurePosixPath(binding.path)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise AuthoringResumeError(
                "artifact_path_escape",
                f"bound artifact path contains a symlink: {binding.path}",
                recovery="restore the reviewed artifact as a regular workspace file",
            )
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AuthoringResumeError(
            "artifact_path_escape",
            f"artifact escapes the workspace: {binding.path}",
            recovery="restore the artifact under the workspace or create a fresh workspace",
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise AuthoringResumeError(
            "artifact_missing",
            f"bound artifact is not a regular file: {binding.path}",
            recovery="restore the exact bound artifact or resume from an earlier session",
        )
    return candidate


def bind_artifact(
    workspace: Path,
    path: Path,
    *,
    digest_kind: Literal["bytes", "canonical_json"] = "bytes",
) -> ArtifactBinding:
    root = workspace.resolve()
    unresolved = path.expanduser().absolute()
    try:
        relative_unresolved = unresolved.relative_to(root)
    except ValueError as exc:
        raise AuthoringResumeError(
            "artifact_path_escape",
            f"cannot bind artifact outside workspace: {unresolved}",
            recovery="copy the reviewed artifact into the workspace",
        ) from exc
    current = root
    for part in relative_unresolved.parts:
        current /= part
        if current.is_symlink():
            raise AuthoringResumeError(
                "artifact_path_escape",
                f"cannot bind artifact through a symlink: {unresolved}",
                recovery="copy the reviewed artifact into the workspace",
            )
    candidate = unresolved.resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise AuthoringResumeError(
            "artifact_path_escape",
            f"cannot bind artifact outside workspace: {candidate}",
            recovery="copy the reviewed artifact into the workspace",
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise AuthoringResumeError(
            "artifact_missing",
            f"cannot bind non-regular artifact: {relative}",
            recovery="write the complete artifact before committing session state",
        )
    payload = candidate.read_bytes()
    digest = _canonical_json_digest(payload, relative) if digest_kind == "canonical_json" else _digest_bytes(payload)
    return ArtifactBinding(path=relative, digest=digest, digest_kind=digest_kind)


def _canonical_json_digest(payload: bytes, label: str) -> str:
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_mapping)
    except AuthoringResumeError:
        raise
    except Exception as exc:
        raise AuthoringResumeError(
            "artifact_invalid",
            f"{label} is not strict UTF-8 JSON: {type(exc).__name__}: {exc}",
            recovery="regenerate the artifact from reviewed inputs",
        ) from exc
    return sha256_json(document)


def build_session_state(
    *,
    tenant_id: str,
    run_id: str,
    phase: AuthoringPhase,
    bindings: SessionBindings,
    parent_session_digest: str | None = None,
) -> AuthoringSessionState:
    unsigned = {
        "schema_version": SESSION_VERSION,
        "tenant_id": tenant_id,
        "run_id": run_id,
        "phase": phase,
        "bindings": bindings.model_dump(mode="json"),
        "parent_session_digest": parent_session_digest,
    }
    return AuthoringSessionState.model_validate({**unsigned, "session_digest": sha256_json(unsigned)})


def _approval_evidence_digest(path: Path) -> str:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_mapping,
        )
    except AuthoringResumeError:
        raise
    except Exception as exc:
        raise AuthoringResumeError(
            "approval_stale",
            f"approval cannot be parsed: {type(exc).__name__}: {exc}",
            recovery="obtain a new approval for the current evidence",
        ) from exc
    if not isinstance(document, dict):
        value = None
    else:
        value = document.get("normalized_bundle_digest", document.get("bundle_digest"))
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AuthoringResumeError(
            "approval_stale",
            "approval does not bind a recognized evidence digest",
            recovery="obtain a new approval for the current evidence",
        )
    return value


class AuthoringResumeGate:
    """Commit and verify immutable session states for one tenant/run workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        tenant_id: str,
        run_id: str,
        lease_seconds: float = 60.0,
    ) -> None:
        self.workspace = workspace.resolve()
        self.tenant_id = tenant_id
        self.run_id = run_id
        self.session_store = RevisionStore(self.workspace / SESSION_STORE_DIRECTORY)
        self.revision_store = RevisionStore(self.workspace / "revisions")
        self.workspace_lock = WorkspaceLock(
            self.workspace / ".locks",
            tenant_id=tenant_id,
            run_id=run_id,
            lease_seconds=lease_seconds,
        )

    def commit_state(
        self,
        state: AuthoringSessionState,
        *,
        lease: WorkspaceLease,
    ) -> Path:
        if (
            not lease.active
            or lease.metadata.tenant_id != self.tenant_id
            or lease.metadata.run_id != self.run_id
            or state.tenant_id != self.tenant_id
            or state.run_id != self.run_id
        ):
            raise AuthoringResumeError(
                "session_namespace_mismatch",
                "session, gate, and lease must name the same tenant/run",
                recovery="acquire the matching workspace lock",
            )
        if state.parent_session_digest is not None:
            parent = self.load_state(state.parent_session_digest)
            if parent.tenant_id != state.tenant_id or parent.run_id != state.run_id:
                raise AuthoringResumeError(
                    "session_namespace_mismatch",
                    "parent session belongs to another tenant/run",
                    recovery="link only to a verified session in the same namespace",
                )
        return self.session_store.put_json(
            state.session_digest,
            {SESSION_FILE_NAME: state.model_dump(mode="json")},
        )

    def load_state(self, session_digest: str) -> AuthoringSessionState:
        try:
            manifest = self.session_store.verify(session_digest)
        except RevisionStoreError as exc:
            raise AuthoringResumeError(
                "session_invalid",
                exc.detail,
                recovery="preserve the workspace and resume from the last verified session",
            ) from exc
        if tuple(item.path for item in manifest.artifacts) != (SESSION_FILE_NAME,):
            raise AuthoringResumeError(
                "session_invalid",
                "session revision must contain only session.json",
                recovery="resume from the last session with a closed manifest",
            )
        path = self.session_store.root / session_digest.removeprefix("sha256:") / SESSION_FILE_NAME
        try:
            document = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_mapping,
            )
            state = AuthoringSessionState.model_validate(document)
        except AuthoringResumeError:
            raise
        except Exception as exc:
            raise AuthoringResumeError(
                "session_invalid",
                f"cannot verify session state: {type(exc).__name__}: {exc}",
                recovery="resume from the last verified session",
            ) from exc
        if state.session_digest != session_digest:
            raise AuthoringResumeError(
                "session_invalid",
                "session content address does not match its signed state",
                recovery="resume from the last verified session",
            )
        return state

    def _verify_artifact(self, binding: ArtifactBinding) -> Path:
        path = _resolve_artifact(self.workspace, binding)
        payload = path.read_bytes()
        observed = (
            _canonical_json_digest(payload, binding.path)
            if binding.digest_kind == "canonical_json"
            else _digest_bytes(payload)
        )
        if observed != binding.digest:
            raise AuthoringResumeError(
                "session_binding_drift",
                f"artifact digest changed: {binding.path}",
                recovery="rerun the owning upstream step in a new session revision",
            )
        return path

    def _verify_bindings(self, state: AuthoringSessionState) -> None:
        bindings = state.bindings
        self._verify_artifact(bindings.source)
        evidence_path = self._verify_artifact(bindings.evidence)
        self._verify_artifact(bindings.resolved_config)
        try:
            evidence = load_source_evidence(evidence_path)
        except Exception as exc:
            raise AuthoringResumeError(
                "evidence_digest_mismatch",
                f"evidence cannot be verified: {type(exc).__name__}: {exc}",
                recovery="restore the immutable evidence or rerun intake",
            ) from exc
        if evidence.bundle_digest != bindings.evidence_bundle_digest:
            raise AuthoringResumeError(
                "evidence_digest_mismatch",
                "session names a different semantic evidence digest",
                recovery="commit a new session from the current verified evidence",
            )
        identity_digest = sha256_json(evidence.identity.model_dump(mode="json"))
        if identity_digest != bindings.source_identity_digest:
            raise AuthoringResumeError(
                "source_identity_drift",
                "evidence source identity differs from the session binding",
                recovery="rerun source intake and certification",
            )
        if bindings.revision_content_address is not None:
            if evidence.revision is None:
                raise AuthoringResumeError(
                    "revision_binding_mismatch",
                    "session names a revision but evidence has no revision lineage",
                    recovery="commit a session against the matching revised evidence",
                )
            try:
                revision_manifest = self.revision_store.verify(bindings.revision_content_address)
            except RevisionStoreError as exc:
                raise AuthoringResumeError(
                    "revision_unverified",
                    exc.detail,
                    recovery="re-apply answers from the last verified evidence revision",
                ) from exc
            paths = {item.path for item in revision_manifest.artifacts}
            if "evidence.json" not in paths:
                raise AuthoringResumeError(
                    "revision_binding_mismatch",
                    "revision store entry does not contain evidence.json",
                    recovery="re-apply answers into a complete revision",
                )
            try:
                stored_evidence = load_source_evidence(
                    self.revision_store.root
                    / bindings.revision_content_address.removeprefix("sha256:")
                    / "evidence.json"
                )
            except Exception as exc:
                raise AuthoringResumeError(
                    "revision_unverified",
                    f"stored revision evidence is invalid: {type(exc).__name__}: {exc}",
                    recovery="re-apply answers from the last verified evidence revision",
                ) from exc
            if stored_evidence != evidence:
                raise AuthoringResumeError(
                    "revision_binding_mismatch",
                    "workspace evidence differs from its content-addressed revision",
                    recovery="restore evidence from the verified revision store",
                )
        elif evidence.revision is not None:
            raise AuthoringResumeError(
                "revision_unverified",
                "revised evidence lacks a revision-store content address",
                recovery="commit the answered evidence through RevisionStore",
            )
        if bindings.approval is not None:
            approval_path = self._verify_artifact(bindings.approval.artifact)
            observed_approval_digest = _approval_evidence_digest(approval_path)
            if (
                bindings.approval.evidence_digest != evidence.bundle_digest
                or observed_approval_digest != evidence.bundle_digest
            ):
                raise AuthoringResumeError(
                    "approval_stale",
                    "approval does not cover the current evidence revision",
                    recovery="obtain a new approval for the current evidence",
                )
        for artifact in (
            bindings.candidate_pack,
            bindings.exposure_authorization,
            bindings.review_packet,
            bindings.release_approval,
            bindings.frozen_manifest,
            bindings.publication_manifest,
        ):
            if artifact is not None:
                self._verify_artifact(artifact)

    def _verify_workspace_shape(self, state: AuthoringSessionState) -> None:
        staging = [path for path in self.workspace.rglob(".*.staging-*") if path.is_dir()]
        if staging:
            raise AuthoringResumeError(
                "partial_workspace_write",
                f"workspace contains {len(staging)} incomplete staging directorie(s)",
                recovery="preserve the workspace for audit and resume in a fresh workspace",
            )
        bindings = state.bindings
        if bindings.draft_root is None:
            return
        unresolved_draft_root = self.workspace / bindings.draft_root
        if unresolved_draft_root.is_symlink():
            raise AuthoringResumeError(
                "artifact_path_escape",
                "draft root cannot be a symlink",
                recovery="use a real directory inside a fresh workspace",
            )
        draft_root = unresolved_draft_root.resolve()
        try:
            draft_root.relative_to(self.workspace)
        except ValueError as exc:
            raise AuthoringResumeError(
                "artifact_path_escape",
                "draft root escapes the workspace",
                recovery="use a fresh workspace with relative draft paths",
            ) from exc
        if draft_root.exists() and not draft_root.is_dir():
            raise AuthoringResumeError(
                "partial_draft_output",
                "declared draft root is not a directory",
                recovery="preserve the output and draft in a fresh workspace",
            )
        has_drafts = draft_root.exists() and any(draft_root.iterdir())
        if state.phase == "evidence_approved":
            if has_drafts or bindings.draft_provenance is not None:
                raise AuthoringResumeError(
                    "partial_draft_output",
                    "draft output exists before a committed draft-complete session",
                    recovery="preserve the partial output and draft in a fresh workspace",
                )
            return
        if state.phase in {
            "draft_complete",
            "pack_assembled",
            "review_ready",
            "release_approved",
            "frozen",
            "published",
        }:
            if not has_drafts or bindings.draft_provenance is None:
                raise AuthoringResumeError(
                    "partial_draft_output",
                    "completed phase lacks its draft tree or provenance",
                    recovery="resume from the evidence-approved session in a fresh workspace",
                )
            provenance_path = self._verify_artifact(bindings.draft_provenance)
            try:
                document = json.loads(
                    provenance_path.read_text(encoding="utf-8"),
                    object_pairs_hook=_unique_mapping,
                )
                DraftProvenance(document=document).verify_digest()
            except AuthoringResumeError:
                raise
            except Exception as exc:
                raise AuthoringResumeError(
                    "partial_draft_output",
                    f"draft provenance is invalid: {type(exc).__name__}: {exc}",
                    recovery="preserve the output and draft in a fresh workspace",
                ) from exc

    def open(
        self,
        session_digest: str,
        *,
        command: AuthoringCommand,
        recover_stale: bool = False,
        recovered_by: str | None = None,
        recovery_reason: str | None = None,
    ) -> ResumedAuthoringSession:
        try:
            lease = self.workspace_lock.acquire(
                recover_stale=recover_stale,
                recovered_by=recovered_by,
                recovery_reason=recovery_reason,
            )
        except WorkspaceLockError as exc:
            code = "concurrent_run_refused" if exc.code == "workspace_locked" else "workspace_lock_refused"
            raise AuthoringResumeError(
                code,
                exc.detail,
                recovery="wait for the live owner or perform explicit audited stale recovery",
            ) from exc
        try:
            state = self.load_state(session_digest)
            if state.tenant_id != self.tenant_id or state.run_id != self.run_id:
                raise AuthoringResumeError(
                    "session_namespace_mismatch",
                    "session belongs to another tenant/run",
                    recovery="open the session through its original namespace",
                )
            self._verify_bindings(state)
            self._verify_workspace_shape(state)
            permitted = RESUMABILITY_MATRIX[state.phase]
            if command not in permitted:
                raise AuthoringResumeError(
                    "resume_command_not_permitted",
                    f"{command!r} is not permitted from phase {state.phase!r}",
                    recovery=("run one of: " + ", ".join(permitted) if permitted else "start a new session revision"),
                )
            verdict = ResumeVerdict(
                session_digest=state.session_digest,
                phase=state.phase,
                command=command,
                permitted_commands=permitted,
            )
            return ResumedAuthoringSession(verdict, lease)
        except Exception:
            lease.release()
            raise

    def open_authorized_revision(
        self,
        session_digest: str,
        *,
        refusal_digest: str,
        authorization_digest: str,
    ) -> ResumedAuthoringSession:
        try:
            lease = self.workspace_lock.acquire()
        except WorkspaceLockError as exc:
            raise AuthoringResumeError(
                "concurrent_run_refused",
                exc.detail,
                recovery="wait for the live owner before creating the next revision",
            ) from exc
        try:
            state = self.load_state(session_digest)
            if state.phase != "refused":
                raise AuthoringResumeError(
                    "resume_command_not_permitted",
                    "authorized revision creation requires a refused session",
                    recovery="use the normal resumability matrix for this phase",
                )
            if state.tenant_id != self.tenant_id or state.run_id != self.run_id:
                raise AuthoringResumeError(
                    "session_namespace_mismatch",
                    "session belongs to another tenant/run",
                    recovery="open the session through its original namespace",
                )
            self._verify_bindings(state)
            self._verify_workspace_shape(state)
            try:
                record = load_refusal_record(
                    self.workspace / "refusals",
                    refusal_digest,
                )
                authorization = load_revision_authorization(
                    self.workspace / "refusal_authorizations",
                    authorization_digest,
                )
            except RefusalRecordError as exc:
                raise AuthoringResumeError(
                    "revision_authorization_required",
                    exc.detail,
                    recovery="persist operator authorization for this exact refusal",
                ) from exc
            if (
                record.tenant_id != self.tenant_id
                or record.run_id != self.run_id
                or record.session_digest != state.session_digest
            ):
                raise AuthoringResumeError(
                    "revision_authorization_stale",
                    "refusal does not bind this refused session",
                    recovery="classify and authorize the current refused session",
                )
            try:
                action = verify_next_revision_authorization(
                    record,
                    authorization,
                    parent_session_digest=state.session_digest,
                )
            except Exception as exc:
                raise AuthoringResumeError(
                    "revision_authorization_stale",
                    str(exc),
                    recovery="obtain fresh operator authorization for this refusal",
                ) from exc
            verdict = ResumeVerdict(
                session_digest=state.session_digest,
                phase=state.phase,
                command="revise",
                permitted_commands=("revise",),
                authorized_action=action.value,
            )
            return ResumedAuthoringSession(verdict, lease)
        except Exception:
            lease.release()
            raise
