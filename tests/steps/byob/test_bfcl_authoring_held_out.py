from __future__ import annotations

import asyncio
import json
from base64 import b64encode
from pathlib import Path
from urllib.parse import quote

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from nemotron.steps.byob.runtime.mcp.authoring.runner import run_intake
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    CertificationAuthority,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutDecision,
    HeldOutError,
    HeldOutLeakageError,
    build_held_out_redaction_report,
    build_not_applicable_decision,
    load_held_out_sensitive_terms,
    load_required_held_out_policy,
    verify_held_out_redaction_report,
)


def test_required_and_not_applicable_are_the_only_valid_states(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "held_out.yaml"
    policy.write_text(
        "version: 1\nfixtures:\n  users: [customer-id]\n",
        encoding="utf-8",
    )

    required = load_required_held_out_policy(
        policy,
        reviewed_by="reviewer@example.test",
    )
    not_applicable = build_not_applicable_decision(
        "This synthetic benchmark has no private evaluation partition.",
        reviewed_by="reviewer@example.test",
    )

    assert required.status == "required"
    assert required.policy_digest is not None
    assert required.reviewed_reason is None
    assert not_applicable.status == "not_applicable"
    assert not_applicable.policy_digest is None
    assert not_applicable.reviewed_reason is not None


def test_held_out_policy_digest_and_decision_are_tamper_evident(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "held_out.yaml"
    policy.write_text("version: 1\nfixtures:\n  users: [private]\n", encoding="utf-8")
    first = load_required_held_out_policy(policy, reviewed_by="reviewer")
    policy.write_text("version: 1\nfixtures:\n  users: [public]\n", encoding="utf-8")
    second = load_required_held_out_policy(policy, reviewed_by="reviewer")

    assert first.policy_digest != second.policy_digest
    assert first.decision_digest != second.decision_digest

    document = first.model_dump(mode="json")
    document["reviewed_by"] = "attacker"
    with pytest.raises(ValidationError, match="decision digest mismatch"):
        HeldOutDecision.model_validate(document)


def test_policy_rejects_duplicate_keys_and_not_applicable_needs_review() -> None:
    with pytest.raises(ValidationError, match="reviewed reason"):
        build_not_applicable_decision("   ", reviewed_by="reviewer")
    with pytest.raises(ValidationError, match="reviewer"):
        build_not_applicable_decision("No held-out data.", reviewed_by=" ")


def test_duplicate_policy_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("version: 1\nversion: 2\n", encoding="utf-8")
    with pytest.raises(HeldOutError, match="repeats YAML key"):
        load_required_held_out_policy(path, reviewed_by="reviewer")


def test_policy_schema_and_reserved_content_are_fail_closed(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("version: 1\nselectors: [private]\n", encoding="utf-8")
    with pytest.raises(HeldOutError, match="unsupported fields"):
        load_required_held_out_policy(unknown, reviewed_by="reviewer")

    reserved = tmp_path / "reserved.yaml"
    reserved.write_text(
        "version: 1\nfixtures:\n  users: [private-7]\n",
        encoding="utf-8",
    )
    with pytest.raises(HeldOutError, match="requires runtime-only reserved content"):
        load_held_out_sensitive_terms(reserved)


def test_mcp_v2_intake_requires_held_out_before_reading_source(
    tmp_path: Path,
) -> None:
    brief = tmp_path / "brief.txt"
    brief.write_text("This must not be read yet.", encoding="utf-8")
    authority = CertificationAuthority(
        key_id="test-root",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x04" * 32),
    )

    with pytest.raises(ValueError, match="explicit held-out decision"):
        asyncio.run(
            run_intake(
                tmp_path / "missing-intake.yaml",
                tmp_path / "output",
                domain_brief_path=brief,
                certification_authority=authority,
            )
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("term", "payload", "variant"),
    [
        ("secret42", {"value": "prefix-secret42-suffix"}, "exact"),
        ("Café", {"value": "CAFE\u0301"}, "unicode_nfkc_casefold"),
        ("secret/42", {"value": quote("secret/42", safe="")}, "url_percent"),
        (
            "secret42",
            {"value": b64encode(b"secret42").decode("ascii")},
            "base64",
        ),
        (
            "secret42",
            {"value": json.dumps({"nested": "secret42"})},
            "nested_json",
        ),
    ],
)
def test_redaction_proof_detects_exact_normalized_encoded_and_nested_leaks(
    tmp_path: Path,
    term: str,
    payload: dict[str, str],
    variant: str,
) -> None:
    policy = tmp_path / "held_out.yaml"
    policy.write_text("version: 1\nfixtures: {}\n", encoding="utf-8")
    decision = load_required_held_out_policy(policy, reviewed_by="reviewer")
    authority = CertificationAuthority(
        key_id="test-root",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x05" * 32),
    )

    evidence = {
        **payload,
        "bundle_digest": sha256_json(payload),
    }
    with pytest.raises(HeldOutLeakageError) as captured:
        build_held_out_redaction_report(
            evidence,
            decision=decision,
            sensitive_terms=(term,),
            authority=authority,
        )

    assert variant in {finding.variant for finding in captured.value.findings}
    assert term not in str(captured.value)


def test_clean_redaction_proof_is_signed_and_evidence_bound(tmp_path: Path) -> None:
    policy = tmp_path / "held_out.yaml"
    policy.write_text("version: 1\nfixtures:\n  users: [private-7]\n", encoding="utf-8")
    decision = load_required_held_out_policy(policy, reviewed_by="reviewer")
    authority = CertificationAuthority(
        key_id="test-root",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x06" * 32),
    )
    unsigned_evidence = {
        "tools": [{"name": "lookup", "description": "Public lookup."}]
    }
    evidence = {
        **unsigned_evidence,
        "bundle_digest": sha256_json(unsigned_evidence),
    }

    report = build_held_out_redaction_report(
        evidence,
        decision=decision,
        sensitive_terms=("private-7",),
        authority=authority,
    )
    verify_held_out_redaction_report(
        report,
        decision=decision,
        evidence_digest=report.evidence_digest,
        trusted_public_keys={"test-root": authority.public_key},
        sensitive_terms=("private-7",),
        evidence=evidence,
    )
    with pytest.raises(HeldOutError, match="binding mismatch"):
        verify_held_out_redaction_report(
            report,
            decision=decision,
            evidence_digest="sha256:" + "0" * 64,
            trusted_public_keys={"test-root": authority.public_key},
            sensitive_terms=("private-7",),
            evidence=evidence,
        )
    leaked_evidence = {
        **evidence,
        "tools": [{"name": "lookup", "description": "private-7"}],
    }
    with pytest.raises(HeldOutError, match="fresh leakage scan"):
        verify_held_out_redaction_report(
            report,
            decision=decision,
            evidence_digest=report.evidence_digest,
            trusted_public_keys={"test-root": authority.public_key},
            sensitive_terms=("private-7",),
            evidence=leaked_evidence,
        )


def test_sensitive_terms_include_policy_ids_and_optional_content(tmp_path: Path) -> None:
    policy = tmp_path / "held_out.yaml"
    policy.write_text(
        "version: 1\nfixtures:\n  users: [private-7]\ntemplates: [secret-task]\n",
        encoding="utf-8",
    )
    content = tmp_path / "held_out_content.yaml"
    content.write_text(
        "rows:\n  - customer_name: Alice Reserved\n    balance: 7301\n",
        encoding="utf-8",
    )

    assert load_held_out_sensitive_terms(
        policy,
        content_path=content,
    ) == ("7301", "Alice Reserved", "private-7", "secret-task")
