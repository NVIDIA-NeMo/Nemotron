from __future__ import annotations

from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.authoring_workflow import revision_store
from nemotron.steps.byob.runtime.authoring_workflow.resume import (
    RESUMABILITY_MATRIX,
    ApprovalBinding,
    AuthoringPhase,
    AuthoringResumeError,
    AuthoringResumeGate,
    AuthoringSessionState,
    SessionBindings,
    bind_artifact,
    build_session_state,
)
from nemotron.steps.byob.runtime.authoring_workflow.revision_store import (
    MANIFEST_FILE_NAME,
    REVISION_MANIFEST_VERSION,
    RevisionStore,
    RevisionStoreError,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    AnswerSubmission,
    apply_answers,
    build_answer_set,
    build_open_questions,
    write_evidence_revision,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    SourceEvidenceDocument,
    write_source_evidence,
)
from tests.steps.byob.test_bfcl_authoring_questions import (
    _candidate,
    _evidence,
)

ADDRESS = "sha256:" + "a" * 64


def _committed_session(
    tmp_path: Path,
    *,
    phase: AuthoringPhase = "evidence_approved",
    evidence: SourceEvidenceDocument | None = None,
    revision_content_address: str | None = None,
    approval_evidence_digest: str | None = None,
    source_identity_digest: str | None = None,
    with_approval: bool = True,
) -> tuple[Path, AuthoringResumeGate, str, dict[str, Path]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    observed_evidence = evidence or _evidence()
    paths = {
        "source": workspace / "source-declaration.yaml",
        "evidence": workspace / "evidence.json",
        "config": workspace / "resolved_authoring_config.json",
        "approval": workspace / "approval.json",
    }
    paths["source"].write_text("adapter: reviewed\n", encoding="utf-8")
    write_source_evidence(observed_evidence, paths["evidence"])
    write_canonical_json({"model": "recorded", "seed": 0}, paths["config"])
    write_canonical_json(
        {
            "approval_version": "bfcl-authoring-approval-v1",
            "approved_by": "reviewer",
            "bundle_digest": approval_evidence_digest
            or observed_evidence.bundle_digest,
            "acknowledged_findings": [],
            "note": None,
        },
        paths["approval"],
    )
    approval = (
        ApprovalBinding(
            artifact=bind_artifact(
                workspace,
                paths["approval"],
                digest_kind="canonical_json",
            ),
            evidence_digest=observed_evidence.bundle_digest,
        )
        if with_approval
        else None
    )
    bindings = SessionBindings(
        source=bind_artifact(workspace, paths["source"]),
        evidence=bind_artifact(workspace, paths["evidence"]),
        resolved_config=bind_artifact(
            workspace,
            paths["config"],
            digest_kind="canonical_json",
        ),
        source_identity_digest=source_identity_digest
        or sha256_json(observed_evidence.identity.model_dump(mode="json")),
        evidence_bundle_digest=observed_evidence.bundle_digest,
        revision_content_address=revision_content_address,
        approval=approval,
        draft_root="drafts",
    )
    state = build_session_state(
        tenant_id="tenant-a",
        run_id="run-a",
        phase=phase,
        bindings=bindings,
    )
    gate = AuthoringResumeGate(
        workspace,
        tenant_id="tenant-a",
        run_id="run-a",
    )
    lease = gate.workspace_lock.acquire()
    try:
        gate.commit_state(state, lease=lease)
    finally:
        lease.release()
    return workspace, gate, state.session_digest, paths


def test_complete_revision_is_manifest_bound_and_immutable(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "revisions")
    target = store.put(
        ADDRESS,
        {
            "evidence.json": b'{"evidence":true}\n',
            "revision_record.json": b'{"revision":1}\n',
        },
    )

    manifest = store.verify(ADDRESS)
    assert target.name == "a" * 64
    assert manifest.schema_version == REVISION_MANIFEST_VERSION
    assert manifest.content_address == ADDRESS
    assert tuple(item.path for item in manifest.artifacts) == (
        "evidence.json",
        "revision_record.json",
    )
    assert (target / MANIFEST_FILE_NAME).is_file()

    with pytest.raises(RevisionStoreError) as duplicate:
        store.put(ADDRESS, {"evidence.json": b"different"})
    assert duplicate.value.code == "revision_already_exists"


def test_exclusive_directory_rename_never_replaces_a_racing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    (destination / "winner.txt").write_text("winner", encoding="utf-8")

    with pytest.raises(FileExistsError):
        revision_store._rename_directory(source, destination)

    assert (source / "source.txt").read_text(encoding="utf-8") == "source"
    assert (destination / "winner.txt").read_text(encoding="utf-8") == "winner"


@pytest.mark.parametrize(
    "failure_point",
    [
        "after_root_fsync",
        "after_artifact_write:evidence.json",
        "after_artifact_fsync:evidence.json",
        "after_artifacts_directory_fsync",
        "after_manifest_write",
        "after_manifest_fsync",
        "before_rename",
    ],
)
def test_crash_before_rename_leaves_revision_absent(
    tmp_path: Path,
    failure_point: str,
) -> None:
    store = RevisionStore(tmp_path / "revisions")

    def crash(point: str) -> None:
        if point == failure_point:
            raise RuntimeError(f"crash at {point}")

    with pytest.raises(RuntimeError, match="crash at"):
        store.put(
            ADDRESS,
            {"evidence.json": b"complete payload"},
            crash_hook=crash,
        )

    target = store.root / ("a" * 64)
    assert not target.exists()
    assert not list(store.root.glob(".*.staging-*"))


@pytest.mark.parametrize("failure_point", ["after_rename", "after_parent_fsync"])
def test_failure_after_rename_leaves_only_a_complete_revision(
    tmp_path: Path,
    failure_point: str,
) -> None:
    store = RevisionStore(tmp_path / "revisions")

    def crash(point: str) -> None:
        if point == failure_point:
            raise RuntimeError(f"crash at {point}")

    with pytest.raises(RuntimeError, match="crash at"):
        store.put(
            ADDRESS,
            {"evidence.json": b"complete payload"},
            crash_hook=crash,
        )

    assert store.verify(ADDRESS).content_address == ADDRESS
    assert not list(store.root.glob(".*.staging-*"))


def test_commit_orders_payload_manifest_rename_and_parent_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real_write = revision_store._write_durable
    real_fsync_directory = revision_store._fsync_directory
    real_rename = revision_store._rename_directory

    def observed_write(
        path: Path,
        payload: bytes,
        *,
        crash_hook: revision_store.CrashHook | None,
        write_event: str,
        fsync_event: str,
    ) -> None:
        events.append(f"write:{path.name}")
        real_write(
            path,
            payload,
            crash_hook=crash_hook,
            write_event=write_event,
            fsync_event=fsync_event,
        )

    def observed_fsync(path: Path) -> None:
        events.append(f"fsync:{path.name}")
        real_fsync_directory(path)

    def observed_rename(source: Path, destination: Path) -> None:
        events.append("rename")
        real_rename(source, destination)

    monkeypatch.setattr(revision_store, "_write_durable", observed_write)
    monkeypatch.setattr(revision_store, "_fsync_directory", observed_fsync)
    monkeypatch.setattr(revision_store, "_rename_directory", observed_rename)

    store = RevisionStore(tmp_path / "revisions")
    store.put(ADDRESS, {"evidence.json": b"payload"})

    payload_write = events.index("write:evidence.json")
    manifest_write = events.index(f"write:{MANIFEST_FILE_NAME}")
    rename = events.index("rename")
    assert payload_write < manifest_write < rename
    assert events[manifest_write + 1].startswith("fsync:.")
    assert events[rename + 1] == "fsync:revisions"


def test_unsupported_filesystem_refuses_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(revision_store, "_filesystem_kind", lambda _path: "nfs")
    store = RevisionStore(tmp_path / "revisions")

    with pytest.raises(RevisionStoreError) as refused:
        store.put(ADDRESS, {"evidence.json": b"payload"})

    assert refused.value.code == "unsupported_filesystem"
    assert not store.root.exists()


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra"])
def test_verification_rejects_incomplete_or_changed_revisions(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = RevisionStore(tmp_path / "revisions")
    target = store.put(ADDRESS, {"evidence.json": b"payload"})
    if mutation == "tamper":
        (target / "evidence.json").write_bytes(b"changed")
        expected_code = "artifact_digest_mismatch"
    elif mutation == "missing":
        (target / "evidence.json").unlink()
        expected_code = "revision_incomplete"
    else:
        (target / "unexpected.json").write_text("{}", encoding="utf-8")
        expected_code = "revision_incomplete"

    with pytest.raises(RevisionStoreError) as refused:
        store.verify(ADDRESS)
    assert refused.value.code == expected_code


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "revisions")
    target = store.put(ADDRESS, {"evidence.json": b"payload"})
    (target / MANIFEST_FILE_NAME).write_text(
        '{"schema_version":"bfcl-revision-manifest-v1",'
        f'"content_address":"{ADDRESS}",'
        '"artifacts":[],"manifest_digest":"sha256:'
        + "b" * 64
        + '","manifest_digest":"sha256:'
        + "c" * 64
        + '"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RevisionStoreError) as refused:
        store.load_manifest(ADDRESS)
    assert refused.value.code == "manifest_invalid"


def test_resume_reverifies_all_bindings_and_retains_workspace_lock(
    tmp_path: Path,
) -> None:
    _workspace, gate, session_digest, _paths = _committed_session(tmp_path)

    resumed = gate.open(session_digest, command="draft")
    try:
        assert resumed.verdict.phase == "evidence_approved"
        assert resumed.verdict.permitted_commands == ("draft",)
        with pytest.raises(AuthoringResumeError) as concurrent:
            gate.open(session_digest, command="draft")
        assert concurrent.value.code == "concurrent_run_refused"
    finally:
        resumed.lease.release()


@pytest.mark.parametrize("artifact", ["source", "evidence", "config"])
def test_resume_refuses_bound_artifact_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    _workspace, gate, session_digest, paths = _committed_session(tmp_path)
    paths[artifact].write_text(
        '{"changed":true}\n' if artifact == "config" else "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "session_binding_drift"
    assert refused.value.recovery


@pytest.mark.parametrize("artifact", ["source", "evidence", "config"])
def test_resume_refuses_missing_bound_artifact(
    tmp_path: Path,
    artifact: str,
) -> None:
    _workspace, gate, session_digest, paths = _committed_session(tmp_path)
    paths[artifact].unlink()

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "artifact_missing"
    assert refused.value.recovery


def test_resume_refuses_an_unknown_session_digest(tmp_path: Path) -> None:
    _workspace, gate, _session_digest, _paths = _committed_session(tmp_path)

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open("sha256:" + "e" * 64, command="draft")
    assert refused.value.code == "session_invalid"
    assert refused.value.recovery


def test_resume_refuses_stale_approval(tmp_path: Path) -> None:
    _workspace, gate, session_digest, _paths = _committed_session(
        tmp_path,
        approval_evidence_digest=ADDRESS,
    )

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "approval_stale"


def test_resume_refuses_source_identity_drift(tmp_path: Path) -> None:
    _workspace, gate, session_digest, _paths = _committed_session(
        tmp_path,
        source_identity_digest="sha256:" + "f" * 64,
    )

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "source_identity_drift"


def test_resume_refuses_partial_draft_output_without_restart_in_place(
    tmp_path: Path,
) -> None:
    workspace, gate, session_digest, _paths = _committed_session(tmp_path)
    drafts = workspace / "drafts"
    drafts.mkdir()
    (drafts / "manifest.yaml").write_text("partial: true\n", encoding="utf-8")

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "partial_draft_output"
    assert "fresh workspace" in refused.value.recovery


def test_resume_refuses_incomplete_staging_directory(tmp_path: Path) -> None:
    workspace, gate, session_digest, _paths = _committed_session(tmp_path)
    (workspace / ".draft.staging-crash").mkdir()

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "partial_workspace_write"


def test_resume_refuses_unverified_content_addressed_revision(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    questions = build_open_questions(
        evidence_digest=evidence.bundle_digest,
        candidates=(_candidate(),),
    )
    answers = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )
    revised = apply_answers(evidence, questions, answers)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    revision_root = write_evidence_revision(revised, workspace / "revisions")
    revision_record = revision_root / "revision_record.json"
    revision_record.write_text('{"tampered":true}\n', encoding="utf-8")

    _workspace, gate, session_digest, _paths = _committed_session(
        tmp_path,
        evidence=revised.evidence,
        revision_content_address=revised.evidence.bundle_digest,
    )
    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "revision_unverified"


def test_tampered_immutable_session_is_not_resumed(tmp_path: Path) -> None:
    _workspace, gate, session_digest, _paths = _committed_session(tmp_path)
    session_path = (
        gate.session_store.root
        / session_digest.removeprefix("sha256:")
        / "session.json"
    )
    session_path.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="draft")
    assert refused.value.code == "session_invalid"


def test_session_commit_requires_an_active_matching_lease(tmp_path: Path) -> None:
    _workspace, gate, session_digest, _paths = _committed_session(tmp_path)
    state = gate.load_state(session_digest)
    released = gate.workspace_lock.acquire()
    released.release()

    with pytest.raises(AuthoringResumeError) as refused:
        gate.commit_state(state, lease=released)
    assert refused.value.code == "session_namespace_mismatch"


def test_resume_command_matrix_is_closed_for_every_phase() -> None:
    assert RESUMABILITY_MATRIX == {
        "initialized": ("intake",),
        "intake_complete": (
            "answer",
            "authorize_exposure",
        ),
        "questions_open": ("answer",),
        "evidence_revised": (
            "answer",
            "authorize_exposure",
        ),
        "exposure_authorized": ("approve_evidence",),
        "evidence_approved": ("draft",),
        "draft_complete": ("assemble", "review"),
        "pack_assembled": ("assemble", "review"),
        "review_ready": ("approve_release",),
        "release_approved": ("freeze",),
        "frozen": ("publish",),
        "published": (),
        "refused": (),
    }


def test_legacy_session_digest_remains_verifiable(tmp_path: Path) -> None:
    _, gate, digest, _ = _committed_session(tmp_path)
    current = gate.load_state(digest).model_dump(mode="json")
    current["schema_version"] = "bfcl-authoring-session-v1"
    for field in (
        "candidate_pack",
        "exposure_authorization",
        "review_packet",
        "release_approval",
        "frozen_manifest",
        "publication_manifest",
    ):
        current["bindings"].pop(field)
    unsigned = {key: value for key, value in current.items() if key != "session_digest"}
    current["session_digest"] = sha256_json(unsigned)

    loaded = AuthoringSessionState.model_validate(current)

    assert loaded.schema_version == "bfcl-authoring-session-v1"


def test_resume_rejects_command_outside_phase_matrix(tmp_path: Path) -> None:
    _workspace, gate, session_digest, _paths = _committed_session(tmp_path)

    with pytest.raises(AuthoringResumeError) as refused:
        gate.open(session_digest, command="freeze")
    assert refused.value.code == "resume_command_not_permitted"
    assert refused.value.recovery == "run one of: draft"
