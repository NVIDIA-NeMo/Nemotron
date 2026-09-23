from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.authoring_workflow.cache_retention import (
    AUTHORING_CACHE_FILE_NAME,
    CACHE_PURGE_AUDIT_FILE_NAME,
    CacheRetentionError,
    infer_authoring_cache_path,
    load_cache_purge_audit,
    plan_authoring_cache_purge,
    purge_authoring_cache,
)
from nemotron.steps.byob.runtime.authoring_workflow.resume import (
    ApprovalBinding,
    AuthoringResumeGate,
    SessionBindings,
    bind_artifact,
    build_session_state,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import (
    WorkspaceLockError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.scripts import bfcl_author

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _cache(path: Path, entries: tuple[tuple[str, str], ...]) -> Path:
    cache = ImmutableModelIOCache(path)
    for request_hash, response in entries:
        cache.put(
            request_hash,
            {"answer": response},
            model_canonical="test-model",
            input_hash=SHA_C,
        )
    return path


def _base_bindings(workspace: Path) -> SessionBindings:
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    source = write_canonical_json({"source": True}, artifacts / "source.json")
    evidence = write_canonical_json({"evidence": True}, artifacts / "evidence.json")
    config = write_canonical_json({"config": True}, artifacts / "config.json")
    approval = write_canonical_json({"approval": True}, artifacts / "approval.json")
    return SessionBindings(
        source=bind_artifact(workspace, source, digest_kind="canonical_json"),
        evidence=bind_artifact(workspace, evidence, digest_kind="canonical_json"),
        resolved_config=bind_artifact(
            workspace,
            config,
            digest_kind="canonical_json",
        ),
        source_identity_digest=SHA_A,
        evidence_bundle_digest=SHA_B,
        approval=ApprovalBinding(
            artifact=bind_artifact(
                workspace,
                approval,
                digest_kind="canonical_json",
            ),
            evidence_digest=SHA_B,
        ),
    )


def _commit_draft_session(
    workspace: Path,
    request_hashes: tuple[str, ...],
) -> Path:
    output = workspace / "draft-output"
    output.mkdir(parents=True, exist_ok=True)
    unsigned = {
        "schema_version": "bfcl-authoring-draft-provenance-v1",
        "calls": [
            {"request_hash": request_hash} for request_hash in request_hashes
        ],
    }
    provenance = write_canonical_json(
        {**unsigned, "record_digest": sha256_json(unsigned)},
        output / "draft_provenance.json",
    )
    bindings = _base_bindings(workspace).model_copy(
        update={
            "draft_root": "draft-output/drafts",
            "draft_provenance": bind_artifact(
                workspace,
                provenance,
                digest_kind="canonical_json",
            ),
        }
    )
    state = build_session_state(
        tenant_id="tenant-a",
        run_id="run-a",
        phase="draft_complete",
        bindings=bindings,
    )
    gate = AuthoringResumeGate(
        workspace,
        tenant_id="tenant-a",
        run_id="run-a",
    )
    with gate.workspace_lock.acquire() as lease:
        gate.commit_state(state, lease=lease)
    return output / AUTHORING_CACHE_FILE_NAME


def _commit_active_session(workspace: Path) -> None:
    state = build_session_state(
        tenant_id="tenant-a",
        run_id="run-a",
        phase="evidence_approved",
        bindings=_base_bindings(workspace),
    )
    gate = AuthoringResumeGate(
        workspace,
        tenant_id="tenant-a",
        run_id="run-a",
    )
    with gate.workspace_lock.acquire() as lease:
        gate.commit_state(state, lease=lease)


def test_dry_run_and_execute_share_eligible_plan_and_retain_references(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = _commit_draft_session(workspace, (SHA_A,))
    secret_response = "model-response-must-not-enter-audit"
    _cache(cache_path, ((SHA_A, "referenced"), (SHA_B, secret_response)))

    dry_plan, dry_audit = purge_authoring_cache(
        workspace,
        cache_path,
        tenant_id="tenant-a",
        run_id="run-a",
        actor="retention-bot",
        reason_code="retention_expired",
        dry_run=True,
    )
    execute_plan, execute_audit = purge_authoring_cache(
        workspace,
        cache_path,
        tenant_id="tenant-a",
        run_id="run-a",
        actor="retention-bot",
        reason_code="retention_expired",
        dry_run=False,
        expected_plan_digest=dry_plan.plan_digest,
    )

    assert dry_plan.plan_digest == execute_plan.plan_digest
    assert dry_plan.eligible_request_hashes == execute_plan.eligible_request_hashes == (
        SHA_B,
    )
    assert dry_audit.purged_count == 0
    assert execute_audit.purged_count == 1
    reloaded = ImmutableModelIOCache(cache_path)
    assert reloaded.get(SHA_A) == {"answer": "referenced"}
    assert reloaded.get(SHA_B) is None
    audit_path = workspace / ".events" / CACHE_PURGE_AUDIT_FILE_NAME
    serialized_audit = audit_path.read_text(encoding="utf-8")
    assert secret_response not in serialized_audit
    assert [record.dry_run for record in load_cache_purge_audit(audit_path)] == [
        True,
        False,
    ]


def test_active_uncommitted_session_retains_the_entire_cache(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _commit_active_session(workspace)
    cache_path = _cache(
        workspace / "draft-output" / AUTHORING_CACHE_FILE_NAME,
        ((SHA_A, "orphan"),),
    )

    plan = plan_authoring_cache_purge(
        workspace,
        cache_path,
        tenant_id="tenant-a",
        run_id="run-a",
    )

    assert plan.protected_all is True
    assert plan.retained_request_hashes == (SHA_A,)
    assert plan.eligible_request_hashes == ()


def test_stale_plan_and_concurrent_workspace_are_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = _cache(
        workspace / "draft-output" / AUTHORING_CACHE_FILE_NAME,
        ((SHA_A, "first"),),
    )
    plan = plan_authoring_cache_purge(
        workspace,
        cache_path,
        tenant_id="tenant-a",
        run_id="run-a",
    )
    _cache(cache_path, ((SHA_B, "changed-after-plan"),))
    with pytest.raises(CacheRetentionError) as stale:
        purge_authoring_cache(
            workspace,
            cache_path,
            tenant_id="tenant-a",
            run_id="run-a",
            actor="retention-bot",
            reason_code="retention_expired",
            dry_run=False,
            expected_plan_digest=plan.plan_digest,
        )
    assert stale.value.code == "cache_purge_plan_stale"

    gate = AuthoringResumeGate(
        workspace,
        tenant_id="tenant-a",
        run_id="run-a",
    )
    with gate.workspace_lock.acquire():
        with pytest.raises(WorkspaceLockError):
            purge_authoring_cache(
                workspace,
                cache_path,
                tenant_id="tenant-a",
                run_id="run-a",
                actor="retention-bot",
                reason_code="retention_expired",
                dry_run=True,
            )


def test_cache_scope_and_private_access_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = _cache(
        tmp_path / AUTHORING_CACHE_FILE_NAME,
        ((SHA_A, "outside"),),
    )
    with pytest.raises(CacheRetentionError) as escaped:
        plan_authoring_cache_purge(
            workspace,
            outside,
            tenant_id="tenant-a",
            run_id="run-a",
        )
    assert escaped.value.code == "cache_path_escape"

    cache_path = _cache(
        workspace / AUTHORING_CACHE_FILE_NAME,
        ((SHA_A, "broad"),),
    )
    cache_path.chmod(0o644)
    with pytest.raises(CacheRetentionError) as broad:
        plan_authoring_cache_purge(
            workspace,
            cache_path,
            tenant_id="tenant-a",
            run_id="run-a",
        )
    assert broad.value.code == "cache_access_too_broad"


def test_cli_infers_cache_and_defaults_to_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = _commit_draft_session(workspace, (SHA_A,))
    _cache(cache_path, ((SHA_A, "referenced"), (SHA_B, "eligible")))
    assert (
        infer_authoring_cache_path(
            workspace,
            tenant_id="tenant-a",
            run_id="run-a",
        )
        == cache_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "purge-cache",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--actor",
            "retention-bot",
            "--reason-code",
            "retention_expired",
        ],
    )

    bfcl_author.main()

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry_run"
    assert output["eligible_request_hashes"] == [SHA_B]
    assert ImmutableModelIOCache(cache_path).get(SHA_B) == {
        "answer": "eligible"
    }
