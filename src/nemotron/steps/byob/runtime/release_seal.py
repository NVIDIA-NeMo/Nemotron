# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted Ed25519 release-seal authorities."""

from __future__ import annotations

import re
from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ReleaseSealAuthority:
    """A release authority compatible with certification/revocation key stores."""

    issuer: str
    key_id: str
    private_key: Ed25519PrivateKey = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _identifier(self.issuer, "issuer")
        _identifier(self.key_id, "key_id")

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()


def load_release_seal_authority(
    path: Path,
    *,
    issuer: str,
    key_id: str,
    password: bytes | None = None,
) -> ReleaseSealAuthority:
    try:
        key = load_pem_private_key(path.resolve().read_bytes(), password=password)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("cannot load release-seal private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("release-seal private key must be Ed25519")
    return ReleaseSealAuthority(issuer=issuer, key_id=key_id, private_key=key)


def load_trusted_release_seal_key(
    path: Path,
    *,
    key_id: str,
) -> dict[str, Ed25519PublicKey]:
    try:
        key = load_pem_public_key(path.resolve().read_bytes())
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("cannot load trusted release-seal public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("trusted release-seal key must be Ed25519")
    _identifier(key_id, "key_id")
    return {key_id: key}


def sign_release_digest(authority: ReleaseSealAuthority, digest: str) -> str:
    return b64encode(authority.private_key.sign(digest.encode("ascii"))).decode("ascii")


def verify_release_digest(
    *,
    issuer: str,
    expected_issuer: str,
    key_id: str,
    digest: str,
    signature: str,
    trusted_public_keys: Mapping[str, Ed25519PublicKey],
) -> None:
    if issuer != expected_issuer:
        raise ValueError("release seal issuer is not trusted")
    key = trusted_public_keys.get(key_id)
    if key is None:
        raise ValueError("release seal signing key is not trusted")
    try:
        decoded = b64decode(signature, validate=True)
        if len(decoded) != 64 or b64encode(decoded).decode("ascii") != signature:
            raise ValueError("release seal signature is not canonical")
        key.verify(decoded, digest.encode("ascii"))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("release seal signature is invalid") from exc


def _identifier(value: str, label: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"release seal {label} must be a safe identifier")
