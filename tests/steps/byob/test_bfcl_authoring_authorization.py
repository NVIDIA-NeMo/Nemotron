from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.pack_authoring.artifacts import write_canonical_json
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    AuthorizationError,
    ExposureSubject,
    authorize_model_exposure_by_human,
    authorize_model_exposure_by_policy,
    load_exposure_authorization,
    verify_exposure_authorization,
    write_exposure_authorization,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
SHA_E = "sha256:" + "e" * 64
SHA_F = "sha256:" + "f" * 64


def _subject() -> ExposureSubject:
    return ExposureSubject(
        evidence_digest=SHA_A,
        domain_brief_content_digest=SHA_B,
        domain_brief_source_digest=SHA_C,
        domain_brief_redaction_report_digest=SHA_D,
        held_out_decision_digest=SHA_E,
        held_out_policy_digest=None,
        held_out_redaction_report_digest=SHA_F,
    )


def test_named_human_authorizes_one_exact_subject(tmp_path: Path) -> None:
    subject = _subject()
    authorization = authorize_model_exposure_by_human(
        subject,
        authorized_by="reviewer@example.test",
    )
    path = write_exposure_authorization(
        authorization,
        tmp_path / "authorization.json",
    )

    loaded = load_exposure_authorization(path)
    verify_exposure_authorization(loaded, expected_subject=subject)

    for field in (
        "evidence_digest",
        "domain_brief_content_digest",
        "domain_brief_source_digest",
        "domain_brief_redaction_report_digest",
        "held_out_decision_digest",
        "held_out_policy_digest",
        "held_out_redaction_report_digest",
    ):
        changed = subject.model_copy(update={field: SHA_A if field != "evidence_digest" else SHA_B})
        with pytest.raises(AuthorizationError, match="does not cover"):
            verify_exposure_authorization(
                loaded,
                expected_subject=changed,
            )


def test_organizational_authorization_requires_current_policy_digest() -> None:
    authorization = authorize_model_exposure_by_policy(
        _subject(),
        organizational_policy_digest=SHA_A,
    )

    verify_exposure_authorization(
        authorization,
        expected_subject=_subject(),
        expected_organizational_policy_digest=SHA_A,
    )
    with pytest.raises(AuthorizationError, match="policy digest mismatch"):
        verify_exposure_authorization(
            authorization,
            expected_subject=_subject(),
            expected_organizational_policy_digest=SHA_B,
        )
    with pytest.raises(AuthorizationError, match="policy digest mismatch"):
        verify_exposure_authorization(
            authorization,
            expected_subject=_subject(),
        )


def test_final_semantic_approval_cannot_substitute_for_exposure_authorization(
    tmp_path: Path,
) -> None:
    final_approval = {
        "approval_version": "bfcl-source-evidence-approval-v2",
        "approved_by": "reviewer@example.test",
        "source_bundle_digest": SHA_A,
        "normalized_bundle_digest": SHA_A,
        "migration_record_digest": None,
        "acknowledged_warnings": [],
        "acknowledged_findings": [],
        "note": None,
    }
    path = write_canonical_json(final_approval, tmp_path / "final-approval.json")

    with pytest.raises(AuthorizationError, match="cannot load"):
        load_exposure_authorization(path)


def test_authorization_tampering_and_duplicate_keys_fail_closed(
    tmp_path: Path,
) -> None:
    authorization = authorize_model_exposure_by_human(
        _subject(),
        authorized_by="reviewer",
    )
    document = authorization.model_dump(mode="json")
    document["subject"]["evidence_digest"] = SHA_B
    tampered = write_canonical_json(document, tmp_path / "tampered.json")
    with pytest.raises(AuthorizationError, match="digest mismatch"):
        load_exposure_authorization(tampered)

    valid = authorization.model_dump(mode="json")
    duplicate = tmp_path / "duplicate.json"
    encoded = json.dumps(valid, separators=(",", ":"))
    duplicate.write_text(
        encoded[:-1] + ',"mode":"organizational_policy"' + "}",
        encoding="utf-8",
    )
    with pytest.raises(AuthorizationError, match="repeats JSON key"):
        load_exposure_authorization(duplicate)
