from __future__ import annotations

import json
import sys
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.authoring_release import handoff as handoff_module
from nemotron.steps.byob.runtime.authoring_release.freeze import FrozenReleaseV2
from nemotron.steps.byob.runtime.authoring_release.handoff import (
    handoff_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.revocation import (
    ReleaseRevocationError,
    RevocationAuthority,
    RevocationTarget,
    build_revocation_record,
    build_revocation_registry,
    load_revocation_registry,
    verify_release_revocation,
    write_revocation_registry,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.scripts import publish_authoring_release as publish_script

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
NOW = datetime(2026, 8, 29, 16, 0, tzinfo=timezone.utc)


def _authority(
    issuer: str = "bfcl-release-operations",
    key_id: str = "release-key-1",
) -> RevocationAuthority:
    return RevocationAuthority(
        issuer=issuer,
        key_id=key_id,
        private_key=Ed25519PrivateKey.generate(),
    )


def _target(fingerprint: str = SHA_A) -> RevocationTarget:
    return RevocationTarget(
        frozen_pack_fingerprint=fingerprint,
        freeze_manifest_digest=SHA_C,
        adapter_kind="local_python",
    )


def _registry(
    authority: RevocationAuthority,
    *,
    valid_until: datetime | None = None,
) -> Any:
    record = build_revocation_record(
        _target(),
        authority=authority,
        action="revoke",
        reason_code="invalid_oracle_behavior",
        effective_at=NOW,
    )
    return build_revocation_registry(
        (record,),
        authority=authority,
        generation=3,
        generated_at=NOW,
        valid_until=valid_until or NOW + timedelta(days=1),
    )


def test_signed_registry_blocks_publication_and_consumer_reject_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    registry = _registry(authority)
    path = write_revocation_registry(registry, tmp_path / "registry.json")
    loaded = load_revocation_registry(
        path,
        expected_issuer=authority.issuer,
        trusted_public_keys={authority.key_id: authority.public_key},
        now=NOW,
        minimum_generation=3,
    )
    with pytest.raises(ReleaseRevocationError) as rejected:
        verify_release_revocation(SHA_A, loaded, policy="reject", now=NOW)
    assert rejected.value.code == "release_revoked"

    release = FrozenReleaseV2(
        root=tmp_path / "release",
        pack_root=tmp_path / "release" / "pack",
        manifest={
            "frozen_pack_fingerprint": SHA_A,
            "adapter_kind": "local_python",
        },
    )
    monkeypatch.setattr(
        handoff_module,
        "load_frozen_release",
        lambda _path, **_kwargs: release,
    )
    with pytest.raises(ReleaseRevocationError, match="release_revoked"):
        handoff_frozen_release(
            release.root,
            tmp_path / "config.yaml",
            adapter=cast(Any, object()),
            revocation_check=lambda fingerprint: verify_release_revocation(
                fingerprint,
                loaded,
                policy="reject",
                now=NOW,
            ),
        )


def test_consumer_warn_policy_reports_revocation_without_trusting_replacement() -> None:
    authority = _authority()
    revoke = build_revocation_record(
        _target(),
        authority=authority,
        action="revoke",
        reason_code="invalid_oracle_behavior",
        effective_at=NOW,
    )
    supersede = build_revocation_record(
        _target(),
        authority=authority,
        action="supersede",
        reason_code="replacement_published",
        effective_at=NOW + timedelta(minutes=1),
        prior=revoke,
        replacement_frozen_pack_fingerprint=SHA_B,
    )
    registry = build_revocation_registry(
        (revoke, supersede),
        authority=authority,
        generation=2,
        generated_at=NOW,
        valid_until=NOW + timedelta(days=1),
    )

    warning = verify_release_revocation(
        SHA_A,
        registry,
        policy="warn",
        now=NOW + timedelta(minutes=2),
    )

    assert warning.revoked is True
    assert warning.action == "supersede"
    assert warning.replacement_frozen_pack_fingerprint == SHA_B
    assert warning.warnings == ("release_revoked",)
    assert (
        verify_release_revocation(
            SHA_B,
            registry,
            policy="reject",
            now=NOW + timedelta(minutes=2),
        ).revoked
        is False
    )


def test_stale_unsigned_and_wrong_issuer_registries_fail_closed(
    tmp_path: Path,
) -> None:
    authority = _authority()
    expired = _registry(authority, valid_until=NOW + timedelta(seconds=1))
    path = write_revocation_registry(expired, tmp_path / "expired.json")
    with pytest.raises(ReleaseRevocationError) as stale:
        load_revocation_registry(
            path,
            expected_issuer=authority.issuer,
            trusted_public_keys={authority.key_id: authority.public_key},
            now=NOW + timedelta(seconds=2),
        )
    assert stale.value.code == "revocation_registry_stale"

    unsigned = expired.model_dump(mode="json")
    unsigned["records"][0].pop("signature")
    (tmp_path / "unsigned.json").write_text(
        json.dumps(unsigned),
        encoding="utf-8",
    )
    with pytest.raises(ReleaseRevocationError) as missing_signature:
        load_revocation_registry(
            tmp_path / "unsigned.json",
            expected_issuer=authority.issuer,
            trusted_public_keys={authority.key_id: authority.public_key},
            now=NOW,
        )
    assert missing_signature.value.code == "revocation_registry_invalid"

    wrong_issuer = _authority("other-release-operations")
    wrong_path = write_revocation_registry(
        _registry(wrong_issuer),
        tmp_path / "wrong-issuer.json",
    )
    with pytest.raises(ReleaseRevocationError) as untrusted:
        load_revocation_registry(
            wrong_path,
            expected_issuer=authority.issuer,
            trusted_public_keys={wrong_issuer.key_id: wrong_issuer.public_key},
            now=NOW,
        )
    assert untrusted.value.code == "revocation_issuer_untrusted"


def test_wrong_signing_key_and_registry_rollback_fail_closed(tmp_path: Path) -> None:
    authority = _authority()
    registry = _registry(authority)
    path = write_revocation_registry(registry, tmp_path / "registry.json")
    other = _authority(key_id=authority.key_id)
    with pytest.raises(ReleaseRevocationError) as wrong_key:
        load_revocation_registry(
            path,
            expected_issuer=authority.issuer,
            trusted_public_keys={other.key_id: other.public_key},
            now=NOW,
        )
    assert wrong_key.value.code == "revocation_signature_invalid"

    with pytest.raises(ReleaseRevocationError) as rollback:
        load_revocation_registry(
            path,
            expected_issuer=authority.issuer,
            trusted_public_keys={authority.key_id: authority.public_key},
            now=NOW,
            minimum_generation=4,
        )
    assert rollback.value.code == "revocation_registry_stale"


def _live_registry(authority: RevocationAuthority) -> Any:
    """Build a registry valid under the wall clock the publish CLI actually uses."""
    now = datetime.now(timezone.utc)
    record = build_revocation_record(
        _target(),
        authority=authority,
        action="revoke",
        reason_code="invalid_oracle_behavior",
        effective_at=now - timedelta(hours=1),
    )
    return build_revocation_registry(
        (record,),
        authority=authority,
        generation=3,
        generated_at=now - timedelta(minutes=1),
        valid_until=now + timedelta(days=1),
    )


def test_publish_cli_blocks_a_revoked_release_from_registry_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority = _authority()
    registry_path = write_revocation_registry(
        _live_registry(authority),
        tmp_path / "registry.json",
    )
    key_path = tmp_path / "revocation-key.pem"
    key_path.write_bytes(
        authority.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    release = FrozenReleaseV2(
        root=tmp_path / "release",
        pack_root=tmp_path / "release" / "pack",
        manifest={
            "frozen_pack_fingerprint": SHA_A,
            "adapter_kind": "local_python",
        },
    )
    monkeypatch.setattr(
        handoff_module,
        "load_frozen_release",
        lambda _path, **_kwargs: release,
    )
    monkeypatch.setattr(
        publish_script,
        "publication_adapter_for_release",
        lambda _path, **_kwargs: cast(Any, object()),
    )
    argv = [
        "publish_authoring_release",
        "--release",
        str(release.root),
        "--config",
        str(tmp_path / "config.yaml"),
        "--seal-issuer",
        authority.issuer,
        "--seal-public-key",
        str(key_path),
        "--seal-key-id",
        authority.key_id,
        "--revocation-registry",
        str(registry_path),
        "--revocation-issuer",
        authority.issuer,
        "--revocation-public-key",
        str(key_path),
        "--revocation-key-id",
        authority.key_id,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exited:
        publish_script.main()

    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().err)["code"] == "release_revoked"


def test_publish_cli_refuses_partial_revocation_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority = _authority()
    key_path = tmp_path / "seal-key.pem"
    key_path.write_bytes(
        authority.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_authoring_release",
            "--release",
            str(tmp_path / "release"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--seal-issuer",
            authority.issuer,
            "--seal-public-key",
            str(key_path),
            "--seal-key-id",
            authority.key_id,
            "--revocation-registry",
            str(tmp_path / "registry.json"),
        ],
    )

    with pytest.raises(SystemExit) as exited:
        publish_script.main()

    assert exited.value.code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error_type"] == "ValueError"
    assert "must be supplied together" in payload["reason"]


def test_conflicting_revocation_chain_fails_closed(tmp_path: Path) -> None:
    authority = _authority()
    first = build_revocation_record(
        _target(),
        authority=authority,
        action="revoke",
        reason_code="first_reason",
        effective_at=NOW,
    )
    conflict = build_revocation_record(
        _target(),
        authority=authority,
        action="revoke",
        reason_code="conflicting_reason",
        effective_at=NOW + timedelta(minutes=1),
    )
    records = sorted(
        [first.model_dump(mode="json"), conflict.model_dump(mode="json")],
        key=lambda record: (
            record["target"]["frozen_pack_fingerprint"],
            record["sequence"],
        ),
    )
    unsigned = {
        "schema_version": "bfcl-release-revocation-registry-v1",
        "issuer": authority.issuer,
        "signing_key_id": authority.key_id,
        "generation": 2,
        "generated_at": NOW.isoformat(),
        "valid_until": (NOW + timedelta(days=1)).isoformat(),
        "records": records,
    }
    digest = sha256_json(unsigned)
    document = {
        **unsigned,
        "registry_digest": digest,
        "signature": b64encode(authority.private_key.sign(digest.encode("ascii"))).decode("ascii"),
    }
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReleaseRevocationError) as rejected:
        load_revocation_registry(
            path,
            expected_issuer=authority.issuer,
            trusted_public_keys={authority.key_id: authority.public_key},
            now=NOW,
        )
    assert rejected.value.code == "revocation_registry_invalid"
