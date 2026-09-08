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

"""Signed release revocations, supersession chains, and consumer policy."""

from __future__ import annotations

import fcntl
import json
import os
import re
from base64 import b64decode, b64encode
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.authoring_release.freeze import (
    FrozenReleaseV2,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

REVOCATION_RECORD_VERSION: Literal["bfcl-release-revocation-v1"] = "bfcl-release-revocation-v1"
REVOCATION_REGISTRY_VERSION: Literal["bfcl-release-revocation-registry-v1"] = "bfcl-release-revocation-registry-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ReleaseRevocationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class RevocationAuthority:
    issuer: str
    key_id: str
    private_key: Ed25519PrivateKey = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_identifier(self.issuer, "issuer")
        _require_identifier(self.key_id, "key_id")

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()


@dataclass(frozen=True)
class RevocationRegistryVerifier:
    path: Path
    expected_issuer: str
    trusted_public_keys: Mapping[str, Ed25519PublicKey]
    policy: Literal["reject", "warn"] = "reject"
    minimum_generation: int = 1
    clock: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc),
        repr=False,
        compare=False,
    )

    def __call__(self, frozen_pack_fingerprint: str) -> RevocationVerdict:
        now = self.clock()
        registry = load_revocation_registry(
            self.path,
            expected_issuer=self.expected_issuer,
            trusted_public_keys=self.trusted_public_keys,
            now=now,
            minimum_generation=self.minimum_generation,
        )
        return verify_release_revocation(
            frozen_pack_fingerprint,
            registry,
            policy=self.policy,
            now=now,
        )


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class RevocationTarget(_StrictModel):
    frozen_pack_fingerprint: StrictStr
    freeze_manifest_digest: StrictStr
    adapter_kind: Literal["local_python", "http_package", "mcp_mode_a"]

    @field_validator("frozen_pack_fingerprint", "freeze_manifest_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _require_digest(value)


class ReleaseRevocationRecord(_StrictModel):
    schema_version: Literal["bfcl-release-revocation-v1"]
    issuer: StrictStr
    signing_key_id: StrictStr
    sequence: StrictInt
    action: Literal["revoke", "supersede"]
    target: RevocationTarget
    replacement_frozen_pack_fingerprint: StrictStr | None
    reason_code: StrictStr
    effective_at: StrictStr
    supersedes_record_digest: StrictStr | None
    record_digest: StrictStr
    signature: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> ReleaseRevocationRecord:
        _require_identifier(self.issuer, "issuer")
        _require_identifier(self.signing_key_id, "signing_key_id")
        _require_identifier(self.reason_code, "reason_code")
        if self.sequence <= 0:
            raise ValueError("revocation sequence must be positive")
        _require_timestamp(self.effective_at, "effective_at")
        if self.supersedes_record_digest is not None:
            _require_digest(self.supersedes_record_digest)
        if self.action == "supersede":
            if self.replacement_frozen_pack_fingerprint is None:
                raise ValueError("supersede requires a replacement fingerprint")
            _require_digest(self.replacement_frozen_pack_fingerprint)
            if self.replacement_frozen_pack_fingerprint == self.target.frozen_pack_fingerprint:
                raise ValueError("a release cannot supersede itself")
        elif self.replacement_frozen_pack_fingerprint is not None:
            raise ValueError("revoke cannot declare a replacement fingerprint")
        _require_digest(self.record_digest)
        _require_signature(self.signature)
        unsigned = self.model_dump(
            mode="json",
            exclude={"record_digest", "signature"},
        )
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("revocation record digest mismatch")
        return self


class ReleaseRevocationRegistry(_StrictModel):
    schema_version: Literal["bfcl-release-revocation-registry-v1"]
    issuer: StrictStr
    signing_key_id: StrictStr
    generation: StrictInt
    generated_at: StrictStr
    valid_until: StrictStr
    records: tuple[ReleaseRevocationRecord, ...]
    registry_digest: StrictStr
    signature: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> ReleaseRevocationRegistry:
        _require_identifier(self.issuer, "issuer")
        _require_identifier(self.signing_key_id, "signing_key_id")
        if self.generation <= 0:
            raise ValueError("revocation registry generation must be positive")
        generated = _require_timestamp(self.generated_at, "generated_at")
        valid_until = _require_timestamp(self.valid_until, "valid_until")
        if valid_until <= generated:
            raise ValueError("revocation registry valid_until must follow generated_at")
        keys = tuple((record.target.frozen_pack_fingerprint, record.sequence) for record in self.records)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("revocation records must be sorted and unique")
        _validate_chains(self.records)
        _require_digest(self.registry_digest)
        _require_signature(self.signature)
        unsigned = self.model_dump(
            mode="json",
            exclude={"registry_digest", "signature"},
        )
        if self.registry_digest != sha256_json(unsigned):
            raise ValueError("revocation registry digest mismatch")
        return self

    def latest_by_fingerprint(self) -> dict[str, ReleaseRevocationRecord]:
        latest: dict[str, ReleaseRevocationRecord] = {}
        for record in self.records:
            latest[record.target.frozen_pack_fingerprint] = record
        return latest


class RevocationVerdict(_StrictModel):
    policy: Literal["reject", "warn"]
    revoked: bool
    action: Literal["revoke", "supersede"] | None
    revocation_record_digest: StrictStr | None
    replacement_frozen_pack_fingerprint: StrictStr | None
    warnings: tuple[StrictStr, ...]


def load_revocation_authority(
    path: Path,
    *,
    issuer: str,
    key_id: str,
    password: bytes | None = None,
) -> RevocationAuthority:
    try:
        key = load_pem_private_key(path.resolve().read_bytes(), password=password)
    except (OSError, ValueError, TypeError) as exc:
        raise ReleaseRevocationError(
            "revocation_key_invalid",
            "cannot load revocation private key",
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ReleaseRevocationError(
            "revocation_key_invalid",
            "revocation private key must be Ed25519",
        )
    return RevocationAuthority(issuer=issuer, key_id=key_id, private_key=key)


def load_trusted_revocation_key(
    path: Path,
    *,
    key_id: str,
) -> dict[str, Ed25519PublicKey]:
    try:
        key = load_pem_public_key(path.resolve().read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise ReleaseRevocationError(
            "revocation_key_invalid",
            "cannot load trusted revocation public key",
        ) from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ReleaseRevocationError(
            "revocation_key_invalid",
            "trusted revocation key must be Ed25519",
        )
    _require_identifier(key_id, "key_id")
    return {key_id: key}


def revocation_target_from_release(
    release_root: Path,
    *,
    expected_seal_issuer: str | None = None,
    trusted_seal_keys: Mapping[str, Ed25519PublicKey] | None = None,
) -> RevocationTarget:
    release = load_frozen_release(
        release_root,
        expected_seal_issuer=expected_seal_issuer,
        trusted_seal_keys=trusted_seal_keys,
    )
    if isinstance(release, FrozenReleaseV2):
        return RevocationTarget(
            frozen_pack_fingerprint=release.pack_fingerprint,
            freeze_manifest_digest=str(release.manifest["manifest_digest"]),
            adapter_kind=cast(
                Literal["local_python", "http_package", "mcp_mode_a"],
                release.adapter_kind,
            ),
        )
    return RevocationTarget(
        frozen_pack_fingerprint=release.pack_fingerprint,
        freeze_manifest_digest=str(release.manifest["record_digest"]),
        adapter_kind="mcp_mode_a",
    )


def build_revocation_record(
    target: RevocationTarget,
    *,
    authority: RevocationAuthority,
    action: Literal["revoke", "supersede"],
    reason_code: str,
    effective_at: datetime,
    prior: ReleaseRevocationRecord | None = None,
    replacement_frozen_pack_fingerprint: str | None = None,
) -> ReleaseRevocationRecord:
    if prior is not None and prior.target != target:
        raise ReleaseRevocationError(
            "revocation_target_mismatch",
            "supersession chain must retain the exact release target",
        )
    unsigned = {
        "schema_version": REVOCATION_RECORD_VERSION,
        "issuer": authority.issuer,
        "signing_key_id": authority.key_id,
        "sequence": 1 if prior is None else prior.sequence + 1,
        "action": action,
        "target": target.model_dump(mode="json"),
        "replacement_frozen_pack_fingerprint": (replacement_frozen_pack_fingerprint),
        "reason_code": reason_code,
        "effective_at": _timestamp(effective_at),
        "supersedes_record_digest": (None if prior is None else prior.record_digest),
    }
    digest = sha256_json(unsigned)
    return cast(
        ReleaseRevocationRecord,
        ReleaseRevocationRecord.model_validate(
            {
                **unsigned,
                "record_digest": digest,
                "signature": _sign(authority.private_key, digest),
            }
        ),
    )


def build_revocation_registry(
    records: tuple[ReleaseRevocationRecord, ...],
    *,
    authority: RevocationAuthority,
    generation: int,
    generated_at: datetime,
    valid_until: datetime,
) -> ReleaseRevocationRegistry:
    if any(record.issuer != authority.issuer for record in records):
        raise ReleaseRevocationError(
            "revocation_issuer_untrusted",
            "registry cannot include a record from another issuer",
        )
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.target.frozen_pack_fingerprint,
                record.sequence,
            ),
        )
    )
    unsigned = {
        "schema_version": REVOCATION_REGISTRY_VERSION,
        "issuer": authority.issuer,
        "signing_key_id": authority.key_id,
        "generation": generation,
        "generated_at": _timestamp(generated_at),
        "valid_until": _timestamp(valid_until),
        "records": [record.model_dump(mode="json") for record in ordered],
    }
    digest = sha256_json(unsigned)
    return cast(
        ReleaseRevocationRegistry,
        ReleaseRevocationRegistry.model_validate(
            {
                **unsigned,
                "registry_digest": digest,
                "signature": _sign(authority.private_key, digest),
            }
        ),
    )


def write_revocation_registry(
    registry: ReleaseRevocationRegistry,
    path: Path,
) -> Path:
    return cast(
        Path,
        write_canonical_json(registry.model_dump(mode="json"), path),
    )


@contextmanager
def exclusive_revocation_registry(path: Path) -> Iterator[None]:
    lock_path = path.expanduser().absolute().parent / f".{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.parent.is_symlink():
        raise ReleaseRevocationError(
            "revocation_registry_invalid",
            "revocation registry directory must not be a symlink",
        )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if os.fstat(descriptor).st_mode & 0o077:
            raise ReleaseRevocationError(
                "revocation_registry_access_too_broad",
                "revocation registry lock must be private",
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load_revocation_registry(
    path: Path,
    *,
    expected_issuer: str,
    trusted_public_keys: Mapping[str, Ed25519PublicKey],
    now: datetime,
    minimum_generation: int = 1,
) -> ReleaseRevocationRegistry:
    try:
        document = json.loads(
            path.resolve().read_text(encoding="utf-8"),
            object_pairs_hook=_unique_mapping,
        )
        registry = ReleaseRevocationRegistry.model_validate(document)
    except ReleaseRevocationError:
        raise
    except Exception as exc:
        raise ReleaseRevocationError(
            "revocation_registry_invalid",
            f"cannot verify revocation registry: {type(exc).__name__}",
        ) from exc
    if registry.issuer != expected_issuer:
        raise ReleaseRevocationError(
            "revocation_issuer_untrusted",
            "revocation registry issuer is not trusted",
        )
    if registry.generation < minimum_generation:
        raise ReleaseRevocationError(
            "revocation_registry_stale",
            "revocation registry generation is older than policy",
        )
    current = _aware_utc(now)
    if _require_timestamp(registry.generated_at, "generated_at") > current:
        raise ReleaseRevocationError(
            "revocation_registry_stale",
            "revocation registry was generated in the future",
        )
    if current > _require_timestamp(registry.valid_until, "valid_until"):
        raise ReleaseRevocationError(
            "revocation_registry_stale",
            "revocation registry freshness window expired",
        )
    _verify_signature(
        registry.signing_key_id,
        registry.registry_digest,
        registry.signature,
        trusted_public_keys,
    )
    for record in registry.records:
        if record.issuer != expected_issuer:
            raise ReleaseRevocationError(
                "revocation_issuer_untrusted",
                "revocation record issuer differs from the trusted registry issuer",
            )
        _verify_signature(
            record.signing_key_id,
            record.record_digest,
            record.signature,
            trusted_public_keys,
        )
    return cast(ReleaseRevocationRegistry, registry)


def verify_release_revocation(
    frozen_pack_fingerprint: str,
    registry: ReleaseRevocationRegistry,
    *,
    policy: Literal["reject", "warn"],
    now: datetime | None = None,
) -> RevocationVerdict:
    _require_digest(frozen_pack_fingerprint)
    record = registry.latest_by_fingerprint().get(frozen_pack_fingerprint)
    if record is None:
        return RevocationVerdict(
            policy=policy,
            revoked=False,
            action=None,
            revocation_record_digest=None,
            replacement_frozen_pack_fingerprint=None,
            warnings=(),
        )
    current = _aware_utc(now or datetime.now(timezone.utc))
    if _require_timestamp(record.effective_at, "effective_at") > current:
        return RevocationVerdict(
            policy=policy,
            revoked=False,
            action=None,
            revocation_record_digest=None,
            replacement_frozen_pack_fingerprint=None,
            warnings=(),
        )
    if policy == "reject":
        raise ReleaseRevocationError(
            "release_revoked",
            "release identity is revoked by the authenticated registry",
        )
    return RevocationVerdict(
        policy=policy,
        revoked=True,
        action=record.action,
        revocation_record_digest=record.record_digest,
        replacement_frozen_pack_fingerprint=(record.replacement_frozen_pack_fingerprint),
        warnings=("release_revoked",),
    )


def _validate_chains(records: tuple[ReleaseRevocationRecord, ...]) -> None:
    grouped: dict[str, list[ReleaseRevocationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.target.frozen_pack_fingerprint].append(record)
    for chain in grouped.values():
        for index, record in enumerate(chain):
            expected_sequence = index + 1
            expected_prior = None if index == 0 else chain[index - 1].record_digest
            if record.sequence != expected_sequence or record.supersedes_record_digest != expected_prior:
                raise ValueError("revocation chain is stale, conflicting, or incomplete")
            if index and _require_timestamp(
                record.effective_at,
                "effective_at",
            ) <= _require_timestamp(chain[index - 1].effective_at, "effective_at"):
                raise ValueError("revocation chain effective times must increase")


def _verify_signature(
    key_id: str,
    digest: str,
    signature: str,
    trusted_public_keys: Mapping[str, Ed25519PublicKey],
) -> None:
    key = trusted_public_keys.get(key_id)
    if key is None:
        raise ReleaseRevocationError(
            "revocation_signing_key_untrusted",
            "revocation signing key is not trusted",
        )
    try:
        key.verify(
            b64decode(signature, validate=True),
            digest.encode("ascii"),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ReleaseRevocationError(
            "revocation_signature_invalid",
            "revocation signature is invalid",
        ) from exc


def _sign(key: Ed25519PrivateKey, digest: str) -> str:
    return b64encode(key.sign(digest.encode("ascii"))).decode("ascii")


def _require_digest(value: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError("release revocation digest must be sha256:<64 lowercase hex>")
    return value


def _require_identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"release revocation {label} must be a safe identifier")
    return value


def _require_signature(value: str) -> None:
    try:
        decoded = b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError("revocation signature must be canonical base64") from exc
    if len(decoded) != 64 or b64encode(decoded).decode("ascii") != value:
        raise ValueError("revocation signature must encode 64 bytes")


def _timestamp(value: datetime) -> str:
    return _aware_utc(value).isoformat()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("release revocation timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"release revocation {label} must be ISO-8601") from exc
    return _aware_utc(parsed)


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseRevocationError(
                "revocation_registry_invalid",
                f"duplicate JSON key {key!r}",
            )
        result[key] = value
    return result
