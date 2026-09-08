from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.authoring_workflow.events import (
    AdapterIdentityPayload,
    AuthoringEventError,
    CertificationPayload,
    FileAuthoringEventSink,
    RefusalPayload,
    ReleaseFrozenPayload,
    RevisionAuthorizationPayload,
    ValidationVerdictPayload,
    build_authoring_event,
    emit_authoring_event,
    load_authoring_events,
)
from nemotron.steps.byob.runtime.authoring_workflow.refusal import (
    REQUIRED_ACTION,
    RefusalClassification,
    authorize_next_revision,
    build_refusal_record,
    build_sanitized_finding,
    persist_refusal_record,
    persist_revision_authorization,
)
from nemotron.steps.byob.runtime.authoring_workflow.workspace_lock import WorkspaceLock

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
NOW = datetime(2026, 8, 29, 13, 0, tzinfo=timezone.utc)


def _payloads() -> tuple[tuple[str, object], ...]:
    return (
        (
            "adapter_identity_bound",
            AdapterIdentityPayload(
                adapter_kind="local_python",
                source_identity_digest=SHA_A,
                evidence_bundle_digest=SHA_B,
                descriptor_digest=SHA_C,
            ),
        ),
        (
            "certification_verified",
            CertificationPayload(
                adapter_kind="local_python",
                attained_tier="A2",
                required_tier="A2",
                profile_id="local-python-v1",
                report_digest=SHA_A,
            ),
        ),
        (
            "refusal_recorded",
            RefusalPayload(
                refusal_digest=SHA_A,
                primary_classification="model_owned_proposal",
                finding_codes=("assertion_compilation_blocked",),
                reason_codes=("unresolved_behavior",),
            ),
        ),
        (
            "revision_authorized",
            RevisionAuthorizationPayload(
                authorization_digest=SHA_A,
                refusal_digest=SHA_B,
                parent_session_digest=SHA_C,
                action="retry_model",
                authorization_code="reviewed_retry",
            ),
        ),
        (
            "validation_verdict",
            ValidationVerdictPayload(
                stage="review",
                tier="gold",
                gold_eligible=True,
                pack_fingerprint=SHA_A,
                validation_report_digest=SHA_B,
                validation_config_fingerprint=SHA_C,
            ),
        ),
        (
            "release_frozen",
            ReleaseFrozenPayload(
                adapter_kind="local_python",
                manifest_digest=SHA_A,
                frozen_pack_fingerprint=SHA_B,
                review_packet_digest=SHA_C,
                review_approval_digest=SHA_A,
            ),
        ),
    )


def test_event_payloads_are_strict_allowlists() -> None:
    with pytest.raises(ValidationError):
        AdapterIdentityPayload.model_validate(
            {
                "adapter_kind": "local_python",
                "source_identity_digest": SHA_A,
                "evidence_bundle_digest": SHA_B,
                "source_subject": "private customer inventory",
            }
        )
    with pytest.raises(ValidationError):
        ValidationVerdictPayload.model_validate(
            {
                "stage": "review",
                "tier": "gold",
                "gold_eligible": True,
                "pack_fingerprint": SHA_A,
                "validation_report_digest": SHA_B,
                "endpoint_metadata": {"authorization": "secret"},
            }
        )


def test_every_required_event_round_trips_with_its_digest(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = FileAuthoringEventSink(path)
    for event_type, payload in _payloads():
        event = build_authoring_event(
            event_type,  # type: ignore[arg-type]
            payload,  # type: ignore[arg-type]
            tenant_id="tenant-a",
            run_id="run-a",
            session_digest=SHA_A,
            emitted_at=NOW,
        )
        sink.emit(event)

    loaded = load_authoring_events(path)
    assert [event.event_type for event in loaded] == [
        event_type for event_type, _ in _payloads()
    ]
    assert all(event.event_digest.startswith("sha256:") for event in loaded)


def test_event_stream_never_reads_or_serializes_sensitive_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = (
        "customer prose do not disclose",
        "fixture-account-991",
        "credential-super-secret",
        "model response private answer",
    )
    (tmp_path / "domain_brief.txt").write_text(forbidden[0], encoding="utf-8")
    (tmp_path / "fixtures.json").write_text(forbidden[1], encoding="utf-8")
    (tmp_path / "model_io_cache.jsonl").write_text(forbidden[3], encoding="utf-8")
    monkeypatch.setenv("BFCL_TEST_CREDENTIAL", forbidden[2])
    path = tmp_path / "events.jsonl"
    sink = FileAuthoringEventSink(path)
    for event_type, payload in _payloads():
        emit_authoring_event(
            sink,
            event_type,  # type: ignore[arg-type]
            payload,  # type: ignore[arg-type]
            tenant_id="tenant-a",
            run_id="run-a",
            session_digest=SHA_A,
        )

    serialized = path.read_text(encoding="utf-8")
    assert all(value not in serialized for value in forbidden)


def test_refusal_and_revision_persistence_emit_sanitized_events(
    tmp_path: Path,
) -> None:
    finding = build_sanitized_finding(
        finding_code="assertion_compilation_blocked",
        classification=RefusalClassification.MODEL_OWNED_PROPOSAL,
        reason_code="unresolved_behavior",
        artifact_role="assertions",
        evidence_digests=(SHA_B,),
    )
    record = build_refusal_record(
        tenant_id="tenant-a",
        run_id="run-a",
        session_digest=SHA_A,
        primary_classification=RefusalClassification.MODEL_OWNED_PROPOSAL,
        findings=(finding,),
        refused_at=NOW,
    )
    authorization = authorize_next_revision(
        record,
        action=REQUIRED_ACTION[record.primary_classification],
        authorized_by="reviewer",
        authorization_code="reviewed_retry",
        authorized_at=NOW,
    )
    event_path = tmp_path / "events.jsonl"
    sink = FileAuthoringEventSink(event_path)
    lock = WorkspaceLock(
        tmp_path / "locks",
        tenant_id="tenant-a",
        run_id="run-a",
    )
    with lock.acquire() as lease:
        persist_refusal_record(
            record,
            tmp_path / "refusals",
            lease=lease,
            event_sink=sink,
        )
        persist_revision_authorization(
            record,
            authorization,
            tmp_path / "authorizations",
            lease=lease,
            event_sink=sink,
        )

    events = load_authoring_events(event_path)
    assert [event.event_type for event in events] == [
        "refusal_recorded",
        "revision_authorized",
    ]
    serialized = json.dumps([event.payload for event in events])
    assert "reviewer" not in serialized
    assert "authorized_at" not in serialized


def test_tampered_or_duplicate_event_stream_fails_closed(tmp_path: Path) -> None:
    event = build_authoring_event(
        "refusal_recorded",
        RefusalPayload(
            primary_classification="command_refused",
            reason_codes=("quota_exhausted",),
        ),
        tenant_id="tenant-a",
        run_id="run-a",
        session_digest=None,
        emitted_at=NOW,
    )
    document = event.model_dump(mode="json")
    document["payload"]["reason_codes"] = ["changed"]
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(AuthoringEventError):
        load_authoring_events(path)

    path.write_text('{"event_type":"x","event_type":"y"}\n', encoding="utf-8")
    with pytest.raises(AuthoringEventError, match="duplicate"):
        load_authoring_events(path)
