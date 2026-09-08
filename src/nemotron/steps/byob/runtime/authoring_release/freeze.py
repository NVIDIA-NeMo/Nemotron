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

"""Atomic v2 release sealing over typed adapter hooks."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.authoring_release.contracts import (
    FreezeHookContext,
    ReleaseAdapter,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    ReviewApprovalV2,
    ReviewPacketV2,
    load_json_mapping,
    load_review_approval,
    load_review_packet,
)
from nemotron.steps.byob.runtime.authoring_release.versions import (
    FREEZE_MANIFEST_VERSION_V2,
    FREEZE_MANIFEST_VERSION_V3,
    MCP_FREEZE_MANIFEST_VERSION_V1,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.origin_provenance import (
    AUTHORING_LINEAGE_RELATIVE_PATH,
    AUTHORING_LINEAGE_VERSION,
)
from nemotron.steps.byob.runtime.mcp.config import load_unique_yaml_mapping
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.release_seal import (
    ReleaseSealAuthority,
    sign_release_digest,
    verify_release_digest,
)

PACK_DIRECTORY_NAME = "pack"
FREEZE_MANIFEST_NAME = "freeze_manifest.json"
PROVENANCE_DIRECTORY_NAME = "provenance"
REVIEW_PACKET_PATH = Path(PROVENANCE_DIRECTORY_NAME) / "review_packet.json"
REVIEW_APPROVAL_PATH = Path(PROVENANCE_DIRECTORY_NAME) / "review_approval.json"
SOURCE_RECORDS_DIRECTORY = Path(PROVENANCE_DIRECTORY_NAME) / "source_records"
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "adapter_kind",
        "frozen_pack_fingerprint",
        "review_packet_digest",
        "review_approval_digest",
        "source_digests",
        "source_records",
        "adapter_sidecars",
        "pack_files",
        "seal_issuer",
        "signing_key_id",
        "manifest_digest",
        "signature",
    }
)


class AuthoringFreezeError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


@dataclass(frozen=True)
class FreezeInputsV2:
    pack_root: Path
    review_packet_path: Path
    review_approval_path: Path
    source_records: Mapping[str, Path]
    seal_authority: ReleaseSealAuthority | None = None


@dataclass(frozen=True)
class FrozenReleaseV2:
    root: Path
    pack_root: Path
    manifest: dict[str, Any]

    @property
    def pack_fingerprint(self) -> str:
        return str(self.manifest["frozen_pack_fingerprint"])

    @property
    def adapter_kind(self) -> str:
        return str(self.manifest["adapter_kind"])


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _install_authoring_lineage(
    pack_root: Path,
    packet: ReviewPacketV2,
    approval: ReviewApprovalV2,
) -> None:
    provenance = pack_root / PROVENANCE_DIRECTORY_NAME
    packet_path = pack_root / REVIEW_PACKET_PATH
    approval_path = pack_root / REVIEW_APPROVAL_PATH
    lineage_path = pack_root / AUTHORING_LINEAGE_RELATIVE_PATH
    if any(path.exists() for path in (packet_path, approval_path, lineage_path)):
        raise AuthoringFreezeError(
            "release_path_collision",
            "candidate pack occupies reserved authoring provenance paths",
            recovery="remove generated release provenance from the candidate pack",
        )
    provenance.mkdir(parents=True, exist_ok=True)
    write_canonical_json(packet.document, packet_path)
    write_canonical_json(approval.document, approval_path)
    manifest = load_unique_yaml_mapping(
        pack_root / "manifest.yaml",
        "candidate pack manifest",
    )
    document: dict[str, Any] = {
        "schema_version": AUTHORING_LINEAGE_VERSION,
        "provider_kind": "bfcl_authoring",
        "origin": "unified_authoring_release",
        "profile_version": "bfcl-authoring-release-v2",
        "adapter_kind": packet.document["adapter_kind"],
        "pack": {
            "pack_id": manifest.get("pack_id"),
            "version": manifest.get("version"),
        },
        "identity": {
            "source_pack_fingerprint": packet.document["candidate_pack"]["fingerprint"],
        },
        "review": {
            "packet_digest": packet.digest,
            "approval_digest": approval.digest,
        },
    }
    document["record_digest"] = sha256_json(document)
    write_canonical_json(document, lineage_path)


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuthoringFreezeError(
            "pack_entry_unsafe",
            f"cannot open pack entry without following links: {path}: {exc}",
            recovery="replace the entry with a reviewed regular file",
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AuthoringFreezeError(
                "pack_entry_unsafe",
                f"pack entry is not a regular file: {path}",
                recovery="replace the entry with a reviewed regular file",
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise AuthoringFreezeError(
                "pack_entry_unsafe",
                f"release input contains symbolic link {relative}",
                recovery="replace symbolic links with reviewed regular files",
            )
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_read_regular(path))
        else:
            raise AuthoringFreezeError(
                "pack_entry_unsafe",
                f"release input contains unsupported entry {relative}",
                recovery="retain only directories and regular files",
            )


def _safe_relative(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise AuthoringFreezeError(
            "release_path_unsafe",
            f"{label} path is unsafe: {value!r}",
            recovery="use a bounded relative POSIX path",
        )
    return Path(*pure.parts)


def _write_source_records(
    records: Mapping[str, Path],
    root: Path,
    expected_digests: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    sealed: dict[str, dict[str, str]] = {}
    for name, source in sorted(records.items()):
        relative = _safe_relative(name, "source record")
        data = _read_regular(source.resolve())
        digest = _digest_bytes(data)
        if digest != expected_digests[name]:
            raise AuthoringFreezeError(
                "source_record_digest_mismatch",
                f"source record {name!r} differs from the reviewed bytes",
                recovery="freeze the exact records used to build the review packet",
            )
        target = root / SOURCE_RECORDS_DIRECTORY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        sealed[name] = {
            "path": target.relative_to(root).as_posix(),
            "digest": digest,
        }
    return sealed


def _pack_file_records(pack_root: Path, release_root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(pack_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(release_root).as_posix()
            records[relative] = _digest_bytes(_read_regular(path))
    return records


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def freeze_canonical_pack(
    inputs: FreezeInputsV2,
    destination: Path,
    *,
    adapter: ReleaseAdapter,
) -> FrozenReleaseV2:
    if inputs.seal_authority is None:
        raise AuthoringFreezeError(
            "release_seal_required",
            "freeze requires a trusted Ed25519 release-seal authority",
            recovery="provide the reviewed certification/revocation authority key",
        )
    packet = load_review_packet(inputs.review_packet_path)
    approval = load_review_approval(inputs.review_approval_path)
    if not isinstance(packet, ReviewPacketV2) or not isinstance(approval, ReviewApprovalV2):
        raise AuthoringFreezeError(
            "release_version_mismatch",
            "v2 freeze requires a v2 review packet and approval",
            recovery="use the MCP compatibility freeze for v1 records",
        )
    packet.verify()
    approval.verify()
    if approval.document["review_packet_digest"] != packet.digest:
        raise AuthoringFreezeError(
            "review_approval_stale",
            "approval does not bind the current review packet",
            recovery="review and approve the exact current packet",
        )
    if packet.document["blockers"]:
        raise AuthoringFreezeError(
            "review_packet_blocked",
            "a review packet with blockers cannot be frozen",
            recovery="resolve blockers and build a new packet",
        )
    required_risks = {str(risk["risk_id"]) for risk in packet.document["risks"]}
    if set(approval.document["acknowledged_risks"]) != required_risks:
        raise AuthoringFreezeError(
            "review_risks_unacknowledged",
            "approval does not acknowledge the current packet risk set",
            recovery="approve every current risk ID and no stale IDs",
        )
    if packet.document["adapter_kind"] != adapter.kind:
        raise AuthoringFreezeError(
            "release_adapter_mismatch",
            "review packet names a different adapter",
            recovery="use the adapter that built the packet",
        )
    if packet.document.get("certification_tier") != "A2":
        raise AuthoringFreezeError(
            "adapter_under_certified",
            "v2 release freeze requires independently verified A2 certification",
            recovery="complete A2 probes and build a new review packet",
        )
    if set(inputs.source_records) != set(packet.document["source_digests"]):
        raise AuthoringFreezeError(
            "source_record_set_mismatch",
            "frozen source record names must exactly match reviewed source digests",
            recovery="supply every reviewed source record and no unreviewed records",
        )
    source_fingerprint = adapter.validate_pack(inputs.pack_root)
    if source_fingerprint != packet.document["candidate_pack"]["fingerprint"]:
        raise AuthoringFreezeError(
            "candidate_pack_drift",
            "candidate pack differs from the approved review packet",
            recovery="build and approve a new review packet",
        )
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"frozen release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
        )
    )
    try:
        pack_root = staging / PACK_DIRECTORY_NAME
        _copy_tree(inputs.pack_root.resolve(), pack_root)
        write_canonical_json(packet.document, staging / REVIEW_PACKET_PATH)
        write_canonical_json(approval.document, staging / REVIEW_APPROVAL_PATH)
        sealed_records = _write_source_records(
            inputs.source_records,
            staging,
            packet.document["source_digests"],
        )
        context = FreezeHookContext(
            adapter_kind=adapter.kind,
            packet=packet.document,
            approval=approval.document,
            source_digests=packet.document["source_digests"],
        )
        sidecars: dict[str, dict[str, str]] = {}
        reviewed_sidecars = packet.document["adapter_review"].get(
            "freeze_sidecars",
            {},
        )
        if not isinstance(reviewed_sidecars, Mapping):
            raise AuthoringFreezeError(
                "adapter_sidecars_unreviewed",
                "adapter_review.freeze_sidecars must be a digest mapping",
                recovery="build a new review packet that binds every sidecar",
            )
        supplied_sidecars = dict(adapter.freeze_sidecars(context))
        if set(supplied_sidecars) != set(reviewed_sidecars):
            raise AuthoringFreezeError(
                "adapter_sidecar_set_mismatch",
                "frozen sidecar names must exactly match the reviewed sidecar set",
                recovery="supply every reviewed sidecar and no unreviewed sidecars",
            )
        for name, data in sorted(supplied_sidecars.items()):
            if reviewed_sidecars.get(name) != _digest_bytes(data):
                raise AuthoringFreezeError(
                    "adapter_sidecars_unreviewed",
                    f"adapter sidecar {name!r} was not digest-bound during review",
                    recovery="build and approve a packet containing the sidecar digest",
                )
            relative = _safe_relative(name, "adapter sidecar")
            target = pack_root / relative
            if target.exists():
                raise AuthoringFreezeError(
                    "release_path_collision",
                    f"adapter sidecar collides with pack entry {name!r}",
                    recovery="use a reserved adapter provenance path",
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            sidecars[name] = {
                "path": (Path(PACK_DIRECTORY_NAME) / relative).as_posix(),
                "digest": _digest_bytes(data),
            }
        _install_authoring_lineage(pack_root, packet, approval)
        frozen_fingerprint = adapter.validate_pack(pack_root)
        if adapter.validate_pack(inputs.pack_root) != source_fingerprint:
            raise AuthoringFreezeError(
                "candidate_pack_drift",
                "candidate pack changed while it was being frozen",
                recovery="retry from a stable reviewed workspace",
            )
        manifest: dict[str, Any] = {
            "schema_version": FREEZE_MANIFEST_VERSION_V3,
            "adapter_kind": adapter.kind,
            "frozen_pack_fingerprint": frozen_fingerprint,
            "review_packet_digest": packet.digest,
            "review_approval_digest": approval.digest,
            "source_digests": dict(packet.document["source_digests"]),
            "source_records": sealed_records,
            "adapter_sidecars": sidecars,
            "pack_files": _pack_file_records(pack_root, staging),
            "seal_issuer": inputs.seal_authority.issuer,
            "signing_key_id": inputs.seal_authority.key_id,
        }
        manifest["manifest_digest"] = sha256_json(manifest)
        manifest["signature"] = sign_release_digest(
            inputs.seal_authority,
            manifest["manifest_digest"],
        )
        write_canonical_json(manifest, staging / FREEZE_MANIFEST_NAME)
        staging.replace(destination)
        _make_read_only(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return FrozenReleaseV2(
        root=destination,
        pack_root=destination / PACK_DIRECTORY_NAME,
        manifest=manifest,
    )


def _verify_file_digest(root: Path, record: Mapping[str, Any], label: str) -> None:
    path = record.get("path")
    expected = record.get("digest")
    if not isinstance(path, str) or not isinstance(expected, str):
        raise AuthoringFreezeError(
            "freeze_manifest_invalid",
            f"{label} record is incomplete",
            recovery="restore the immutable frozen release",
        )
    observed = _digest_bytes(_read_regular(root / _safe_relative(path, label)))
    if observed != expected:
        raise AuthoringFreezeError(
            "frozen_release_tampered",
            f"{label} digest mismatch",
            recovery="restore the immutable frozen release",
        )


def load_frozen_release(
    root: Path,
    *,
    adapter: ReleaseAdapter | None = None,
    trusted_seal_keys: Mapping[str, Ed25519PublicKey] | None = None,
    expected_seal_issuer: str | None = None,
    allow_legacy_unsigned: bool = False,
) -> Any:
    release_root = root.resolve()
    manifest = load_json_mapping(
        release_root / FREEZE_MANIFEST_NAME,
        "freeze manifest",
    )
    version = manifest.get("schema_version")
    if version == MCP_FREEZE_MANIFEST_VERSION_V1:
        from nemotron.steps.byob.runtime.mcp.release.freeze import (
            load_frozen_release as load_mcp_frozen_release,
        )

        return load_mcp_frozen_release(root)
    if version == FREEZE_MANIFEST_VERSION_V2 and not allow_legacy_unsigned:
        raise AuthoringFreezeError(
            "unsigned_release_unsupported",
            "legacy v2 freeze manifests have no authenticated seal",
            recovery="re-freeze under v3 with a trusted Ed25519 authority",
        )
    if version not in {FREEZE_MANIFEST_VERSION_V2, FREEZE_MANIFEST_VERSION_V3}:
        raise AuthoringFreezeError(
            "freeze_manifest_version_unsupported",
            f"unsupported freeze manifest version {version!r}",
            recovery="use a v1 MCP, explicitly migrated v2, or signed v3 release",
        )
    expected_keys = (
        _MANIFEST_KEYS
        if version == FREEZE_MANIFEST_VERSION_V3
        else _MANIFEST_KEYS - {"seal_issuer", "signing_key_id", "signature"}
    )
    if set(manifest) != expected_keys:
        raise AuthoringFreezeError(
            "freeze_manifest_invalid",
            "freeze manifest fields do not match the v2 contract",
            recovery="restore the immutable frozen release",
        )
    claimed = manifest.get("manifest_digest")
    unsigned = {key: value for key, value in manifest.items() if key not in {"manifest_digest", "signature"}}
    if claimed != sha256_json(unsigned):
        raise AuthoringFreezeError(
            "freeze_manifest_tampered",
            "freeze manifest digest mismatch",
            recovery="restore the immutable frozen release",
        )
    if version == FREEZE_MANIFEST_VERSION_V3:
        if trusted_seal_keys is None or expected_seal_issuer is None:
            raise AuthoringFreezeError(
                "release_seal_trust_required",
                "signed release verification requires an external issuer and trust store",
                recovery="configure the release authority public key outside the release",
            )
        try:
            verify_release_digest(
                issuer=str(manifest["seal_issuer"]),
                expected_issuer=expected_seal_issuer,
                key_id=str(manifest["signing_key_id"]),
                digest=str(claimed),
                signature=str(manifest["signature"]),
                trusted_public_keys=trusted_seal_keys,
            )
        except ValueError as exc:
            raise AuthoringFreezeError(
                "release_seal_invalid",
                str(exc),
                recovery="restore a release signed by a trusted, non-revoked authority",
            ) from exc
    for name, record in manifest.get("source_records", {}).items():
        _verify_file_digest(release_root, record, f"source record {name}")
    for name, record in manifest.get("adapter_sidecars", {}).items():
        _verify_file_digest(release_root, record, f"adapter sidecar {name}")
    pack_files = manifest.get("pack_files")
    if not isinstance(pack_files, Mapping):
        raise AuthoringFreezeError(
            "freeze_manifest_invalid",
            "freeze manifest omitted pack file seals",
            recovery="restore the immutable frozen release",
        )
    observed_files = _pack_file_records(
        release_root / PACK_DIRECTORY_NAME,
        release_root,
    )
    if observed_files != dict(pack_files):
        raise AuthoringFreezeError(
            "frozen_release_tampered",
            "frozen pack files differ from the sealed file inventory",
            recovery="restore the immutable frozen release",
        )
    packet = load_review_packet(release_root / REVIEW_PACKET_PATH)
    approval = load_review_approval(release_root / REVIEW_APPROVAL_PATH)
    if not isinstance(packet, ReviewPacketV2) or not isinstance(approval, ReviewApprovalV2):
        raise AuthoringFreezeError(
            "release_version_mismatch",
            "v2 manifest sealed non-v2 review records",
            recovery="restore the version-consistent frozen release",
        )
    if manifest.get("adapter_kind") != packet.document["adapter_kind"]:
        raise AuthoringFreezeError(
            "release_adapter_mismatch",
            "manifest adapter differs from its review packet",
            recovery="restore the immutable frozen release",
        )
    if manifest.get("source_digests") != packet.document["source_digests"]:
        raise AuthoringFreezeError(
            "frozen_release_tampered",
            "manifest source digests differ from its review packet",
            recovery="restore the immutable frozen release",
        )
    if set(manifest["source_records"]) != set(manifest["source_digests"]):
        raise AuthoringFreezeError(
            "freeze_manifest_invalid",
            "manifest source records do not cover the reviewed digest set",
            recovery="restore the immutable frozen release",
        )
    if approval.document["review_packet_digest"] != packet.digest:
        raise AuthoringFreezeError(
            "review_approval_stale",
            "sealed approval covers a different review packet",
            recovery="restore the immutable frozen release",
        )
    if packet.digest != manifest.get("review_packet_digest"):
        raise AuthoringFreezeError(
            "frozen_release_tampered",
            "sealed review packet does not match the manifest",
            recovery="restore the immutable frozen release",
        )
    if approval.digest != manifest.get("review_approval_digest"):
        raise AuthoringFreezeError(
            "frozen_release_tampered",
            "sealed review approval does not match the manifest",
            recovery="restore the immutable frozen release",
        )
    release = FrozenReleaseV2(
        root=release_root,
        pack_root=release_root / PACK_DIRECTORY_NAME,
        manifest=manifest,
    )
    if adapter is not None:
        if adapter.kind != release.adapter_kind:
            raise AuthoringFreezeError(
                "release_adapter_mismatch",
                "loader adapter does not match the frozen release",
                recovery="load with the recorded adapter kind",
            )
        if adapter.validate_pack(release.pack_root) != release.pack_fingerprint:
            raise AuthoringFreezeError(
                "frozen_release_tampered",
                "frozen pack fingerprint mismatch",
                recovery="restore the immutable frozen release",
            )
    return release
