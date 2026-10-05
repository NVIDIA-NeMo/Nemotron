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

"""Freeze one approved canonical pack into a sealed, content-addressed release.

The release is a directory rather than an archive so the existing BFCL loader can consume it
without a second execution path. The copy is assembled under a private staging name, every
source file is opened with ``O_NOFOLLOW``, source drift is checked before and after copying,
and the finished tree is renamed into place only after its final fingerprint is recorded.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import PackTrustError
from nemotron.steps.byob.runtime.benchmark_families.bfcl.origin_provenance import (
    MCP_LINEAGE_RELATIVE_PATH,
    MCP_LINEAGE_VERSION,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    FORBIDDEN_PACK_BYTECODE_SUFFIXES,
    IGNORED_PACK_DIRS,
    PACK_FINGERPRINT_CONTRACT,
    ResolvedPackPaths,
    pack_fingerprint,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.mcp.config import load_mcp_oracle_config
from nemotron.steps.byob.runtime.mcp.release.review import (
    ReviewPacket,
    load_json_mapping,
    load_review_approval,
    load_review_packet,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.pack_authoring.provenance import (
    DraftProvenance,
    ProvenanceError,
)
from nemotron.steps.byob.runtime.source_adapters.certification import AdapterTier

FREEZE_MANIFEST_VERSION = "bfcl-mcp-frozen-release-v1"
PACK_DIRECTORY_NAME = "pack"
FREEZE_MANIFEST_NAME = "freeze_manifest.json"
LINEAGE_PATH = MCP_LINEAGE_RELATIVE_PATH
REVIEW_PACKET_PATH = Path("provenance") / "review_packet.json"
REVIEW_APPROVAL_PATH = Path("provenance") / "review_approval.json"
_RESERVED_PATHS = frozenset({"mcp_oracle.yaml", "provenance"})


class FreezeError(ValueError):
    """Raised when reviewed bytes cannot be frozen without changing their meaning."""


def _sealed_digest(path: Path, label: str, field: str) -> str:
    """Recompute a record's own digest field instead of believing it."""
    document = load_json_mapping(path, label)
    claimed = document.get(field)
    unsigned = {key: value for key, value in document.items() if key != field}
    if claimed != sha256_json(unsigned):
        raise FreezeError(f"{label} {field} mismatch")
    return str(claimed)


@dataclass(frozen=True)
class FreezeInputs:
    pack_root: Path
    mcp_config_path: Path
    evidence_path: Path
    intake_provenance_path: Path
    gateway_attestation_path: Path
    draft_provenance_path: Path
    validation_report_path: Path
    review_packet_path: Path
    review_approval_path: Path
    certification_report_path: Path | None = None
    trusted_certification_keys: Mapping[str, Ed25519PublicKey] | None = None
    domain_brief_source_path: Path | None = None
    domain_brief_report_path: Path | None = None
    held_out_redaction_report_path: Path | None = None
    held_out_policy_path: Path | None = None
    held_out_content_path: Path | None = None
    source_bundle_path: Path | None = None
    migration_record_path: Path | None = None
    source_observations_path: Path | None = None


@dataclass(frozen=True)
class FrozenRelease:
    root: Path
    pack_root: Path
    manifest: dict[str, Any]

    @property
    def pack_fingerprint(self) -> str:
        return str(self.manifest["frozen_pack_fingerprint"])


def _resolved_paths(pack_root: Path) -> ResolvedPackPaths:
    root = pack_root.resolve()
    return resolve_declared_pack_paths(
        OraclePackRef(manifest_path=root / "manifest.yaml"),
        (root,),
    )


def _require_canonical_endpoint_pack(pack_root: Path) -> tuple[ResolvedPackPaths, str]:
    """Resolve the pack and assert the one property freeze itself owns.

    Only the endpoint requirement is enforced here, because an MCP release is endpoint-backed
    by definition. Everything else a pack needs is the Gold Gate's decision; re-litigating it
    with slightly different rules would create a second gate that can drift from the first.
    """
    paths = _resolved_paths(pack_root)
    if paths.endpoint_config_path is None or paths.backend_path is not None:
        raise FreezeError("an MCP frozen pack must use endpoint_config.yaml, not backend.py")
    try:
        fingerprint = pack_fingerprint(paths)
    except PackTrustError as exc:
        raise FreezeError(str(exc)) from exc
    return paths, fingerprint


def _read_regular_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FreezeError(f"cannot open pack file without following links: {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FreezeError(f"pack entry is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _copy_pack_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix.lower() in FORBIDDEN_PACK_BYTECODE_SUFFIXES:
            raise FreezeError(
                f"canonical pack must not contain executable Python bytecode (__pycache__, .pyc, or .pyo): {path}"
            )
        if any(part in IGNORED_PACK_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise FreezeError(f"canonical pack must not contain symbolic links: {path}")
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        data = _read_regular_no_follow(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _require_source_digest(
    packet: ReviewPacket,
    name: str,
    observed: str | None,
) -> None:
    expected = packet.document["source_digests"].get(name)
    if observed != expected:
        raise FreezeError(
            f"{name} differs from the approved review packet: expected {expected!r}, observed {observed!r}"
        )


def _load_and_verify_inputs(
    inputs: FreezeInputs,
) -> tuple[
    ReviewPacket,
    dict[str, dict[str, Any]],
    str,
    dict[str, Any],
]:
    packet = load_review_packet(inputs.review_packet_path)
    approval = load_review_approval(inputs.review_approval_path, packet)
    if packet.document.get("status") != "ready_for_approval":
        raise FreezeError("review packet is not ready for approval")

    evidence = load_evidence_bundle(
        inputs.evidence_path,
        certification_report_path=inputs.certification_report_path,
        trusted_certification_keys=inputs.trusted_certification_keys,
        domain_brief_source_path=inputs.domain_brief_source_path,
        domain_brief_report_path=inputs.domain_brief_report_path,
        held_out_redaction_report_path=inputs.held_out_redaction_report_path,
        held_out_policy_path=inputs.held_out_policy_path,
        held_out_content_path=inputs.held_out_content_path,
        source_bundle_path=inputs.source_bundle_path,
        migration_record_path=inputs.migration_record_path,
        source_observations_path=inputs.source_observations_path,
        required_certification_tier=AdapterTier.A2,
    )
    intake = load_json_mapping(inputs.intake_provenance_path, "intake provenance")
    gateway_attestation = load_json_mapping(
        inputs.gateway_attestation_path,
        "gateway attestation",
    )
    draft = load_json_mapping(inputs.draft_provenance_path, "draft provenance")
    try:
        DraftProvenance(document=draft).verify_digest()
    except ProvenanceError as exc:
        raise FreezeError(str(exc)) from exc
    validation = load_json_mapping(
        inputs.validation_report_path,
        "oracle validation report",
    )
    profile = load_mcp_oracle_config(inputs.mcp_config_path)

    _require_source_digest(packet, "evidence_bundle", evidence.digest)
    _require_source_digest(packet, "intake_provenance", str(intake.get("record_digest")))
    _require_source_digest(
        packet,
        "gateway_attestation",
        sha256_json(gateway_attestation),
    )
    _require_source_digest(packet, "draft_provenance", str(draft.get("record_digest")))
    _require_source_digest(packet, "validation_report", sha256_json(validation))
    _require_source_digest(packet, "mcp_config", sha256_json(profile.raw_document))
    if evidence.is_v2:
        assert evidence.certification_report is not None
        assert evidence.domain_brief_report is not None
        assert evidence.held_out_redaction_report is not None
        assert inputs.domain_brief_source_path is not None
        _require_source_digest(
            packet,
            "source_evidence_bundle",
            str(evidence.source_digest),
        )
        _require_source_digest(
            packet,
            "certification_report",
            evidence.certification_report.report_digest,
        )
        _require_source_digest(
            packet,
            "migration_record",
            evidence.migration.record_digest if evidence.migration is not None else None,
        )
        _require_source_digest(
            packet,
            "domain_brief_source",
            "sha256:" + hashlib.sha256(inputs.domain_brief_source_path.read_bytes()).hexdigest(),
        )
        _require_source_digest(
            packet,
            "domain_brief_report",
            evidence.domain_brief_report.record_digest,
        )
        _require_source_digest(
            packet,
            "held_out_redaction_report",
            evidence.held_out_redaction_report.report_digest,
        )

    _, source_fingerprint = _require_canonical_endpoint_pack(inputs.pack_root)
    _require_source_digest(
        packet,
        "canonical_pack",
        f"sha256:{source_fingerprint}",
    )
    records = {
        "evidence_bundle.json": evidence.document,
        "intake_provenance.json": intake,
        "gateway_attestation.json": gateway_attestation,
        "draft_provenance.json": draft,
        "oracle_validation_report.json": validation,
        "review_packet.json": packet.document,
        "review_approval.json": approval.document,
    }
    if evidence.is_v2:
        assert evidence.source_document is not None
        assert evidence.certification_report is not None
        assert evidence.domain_brief_report is not None
        assert evidence.held_out_redaction_report is not None
        assert evidence.migration is not None
        records.update(
            {
                "source_evidence_bundle.json": evidence.source_document,
                "adapter_certification.json": (evidence.certification_report.model_dump(mode="json")),
                "evidence_migration.json": evidence.migration.model_dump(mode="json"),
                "domain_brief_redaction.json": (evidence.domain_brief_report.model_dump(mode="json")),
                "held_out_redaction.json": (evidence.held_out_redaction_report.model_dump(mode="json")),
            }
        )
    return packet, records, source_fingerprint, profile.raw_document


def _lineage_document(
    packet: ReviewPacket,
    records: dict[str, dict[str, Any]],
    source_fingerprint: str,
) -> dict[str, Any]:
    endpoint = packet.document["canonical_pack"]["endpoint_config"]
    document: dict[str, Any] = {
        "schema_version": MCP_LINEAGE_VERSION,
        "provider_kind": "mcp",
        "origin": "mcp_backed_endpoint",
        "profile_version": records["gateway_attestation.json"]["profile_version"],
        "mode": packet.document["mode"],
        "pack": dict(packet.document["pack"]),
        "identity": {
            "source_pack_fingerprint": f"sha256:{source_fingerprint}",
            "effective_content_digest": packet.document["identity"]["effective_content_digest"],
            "tool_catalog_digest": packet.document["identity"]["tool_catalog_digest"],
            "server_content_digest": packet.document["identity"]["server_content_digest"],
            "gateway_artifact_digest": packet.document["identity"]["gateway_artifact_digest"],
            "shim_artifact_digest": packet.document["identity"]["shim_artifact_digest"],
            "snapshot_digest": packet.document["identity"]["snapshot_digest"],
            "conformance_digest": endpoint["attestation"]["expected_digest"],
        },
        "review": {
            "packet_digest": packet.digest,
            "approval_digest": records["review_approval.json"]["approval_digest"],
            "approved_by": records["review_approval.json"]["approved_by"],
            "reviewed_at": records["review_approval.json"]["reviewed_at"],
        },
        "provenance": {
            "evidence_bundle_digest": packet.document["source_digests"]["evidence_bundle"],
            "intake_provenance_digest": packet.document["source_digests"]["intake_provenance"],
            "draft_provenance_digest": packet.document["source_digests"]["draft_provenance"],
            "validation_report_digest": packet.document["source_digests"]["validation_report"],
            "mcp_config_digest": packet.document["source_digests"]["mcp_config"],
            "gateway_attestation_digest": packet.document["source_digests"]["gateway_attestation"],
        },
    }
    document["record_digest"] = sha256_json(document)
    return document


def _make_read_only(root: Path) -> None:
    files = [path for path in root.rglob("*") if path.is_file()]
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in files:
        path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def freeze_canonical_pack(inputs: FreezeInputs, output_root: Path) -> FrozenRelease:
    """Atomically seal an approved pack and its final lineage."""
    if inputs.pack_root.is_symlink():
        raise FreezeError("canonical pack root must not be a symbolic link")
    source = inputs.pack_root.resolve()
    for reserved in _RESERVED_PATHS:
        if (source / reserved).exists():
            raise FreezeError(f"canonical pack uses reserved release path {reserved!r}")
    packet, records, source_fingerprint, profile_document = _load_and_verify_inputs(inputs)

    destination = output_root.resolve()
    if destination.exists():
        raise FreezeError(f"frozen release destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise FreezeError(f"frozen release parent does not exist: {destination.parent}")
    staging = destination.parent / f".{destination.name}.freeze-{uuid.uuid4().hex}"
    pack_destination = staging / PACK_DIRECTORY_NAME
    try:
        pack_destination.mkdir(parents=True)
        _copy_pack_tree(source, pack_destination)
        _, source_after = _require_canonical_endpoint_pack(source)
        if source_after != source_fingerprint:
            raise FreezeError("canonical pack changed while it was being frozen")

        (pack_destination / "mcp_oracle.yaml").write_text(
            yaml.safe_dump(
                profile_document,
                sort_keys=True,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        provenance_root = pack_destination / "provenance"
        for name, document in sorted(records.items()):
            write_canonical_json(document, provenance_root / name)
        lineage = _lineage_document(packet, records, source_fingerprint)
        write_canonical_json(lineage, pack_destination / LINEAGE_PATH)

        frozen_paths, frozen_fingerprint = _require_canonical_endpoint_pack(pack_destination)
        # Resolve once more after every sidecar exists; this also rejects links introduced
        # by a concurrent writer before the seal is committed.
        if pack_fingerprint(frozen_paths) != frozen_fingerprint:
            raise FreezeError("frozen pack fingerprint was not stable")
        manifest: dict[str, Any] = {
            "schema_version": FREEZE_MANIFEST_VERSION,
            "pack_path": PACK_DIRECTORY_NAME,
            "pack": dict(packet.document["pack"]),
            "pack_fingerprint_contract": PACK_FINGERPRINT_CONTRACT,
            "source_pack_fingerprint": f"sha256:{source_fingerprint}",
            "frozen_pack_fingerprint": f"sha256:{frozen_fingerprint}",
            "effective_content_digest": packet.document["identity"]["effective_content_digest"],
            "conformance_digest": lineage["identity"]["conformance_digest"],
            "tool_catalog_digest": packet.document["identity"]["tool_catalog_digest"],
            "lineage_record_digest": lineage["record_digest"],
            "review_packet_digest": packet.digest,
            "review_approval_digest": records["review_approval.json"]["approval_digest"],
        }
        manifest["record_digest"] = sha256_json(manifest)
        write_canonical_json(manifest, staging / FREEZE_MANIFEST_NAME)
        _make_read_only(staging)
        staging.rename(destination)
    except Exception:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            staging.chmod(0o755)
            shutil.rmtree(staging)
        raise
    return FrozenRelease(
        root=destination,
        pack_root=destination / PACK_DIRECTORY_NAME,
        manifest=manifest,
    )


def load_frozen_release(root: Path) -> FrozenRelease:
    """Verify the seal and recompute the final pack fingerprint."""
    release_root = root.resolve()
    manifest = load_json_mapping(
        release_root / FREEZE_MANIFEST_NAME,
        "freeze manifest",
    )
    if manifest.get("schema_version") != FREEZE_MANIFEST_VERSION:
        raise FreezeError(f"freeze manifest must use schema {FREEZE_MANIFEST_VERSION!r}")
    claimed = manifest.get("record_digest")
    unsigned = {key: value for key, value in manifest.items() if key != "record_digest"}
    if claimed != sha256_json(unsigned):
        raise FreezeError("freeze manifest record_digest mismatch")
    pack_root = release_root / str(manifest.get("pack_path"))
    _, observed = _require_canonical_endpoint_pack(pack_root)
    if manifest.get("frozen_pack_fingerprint") != f"sha256:{observed}":
        raise FreezeError("frozen pack fingerprint mismatch")
    lineage_digest = _sealed_digest(pack_root / LINEAGE_PATH, "MCP lineage", "record_digest")
    if manifest.get("lineage_record_digest") != lineage_digest:
        raise FreezeError("freeze manifest pins a different MCP lineage record")
    # The manifest sits outside the fingerprinted tree, so the records it cites are verified
    # here rather than trusted. A seal that only checks itself proves nothing about the pack.
    for relative, label, field, pinned in (
        (REVIEW_PACKET_PATH, "review packet", "packet_digest", "review_packet_digest"),
        (REVIEW_APPROVAL_PATH, "review approval", "approval_digest", "review_approval_digest"),
    ):
        if manifest.get(pinned) != _sealed_digest(pack_root / relative, label, field):
            raise FreezeError(f"freeze manifest pins a different {label}")
    return FrozenRelease(root=release_root, pack_root=pack_root, manifest=manifest)
