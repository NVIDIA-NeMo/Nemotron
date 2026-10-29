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

"""Secret-free MCP origin provenance carried by a frozen oracle pack.

The lineage record is one file inside the pack, so on its own it proves nothing: it can be
copied into any other endpoint pack. Publication therefore accepts it only when it names this
pack and agrees with the review records sealed beside it. Field names are read structurally so
this module stays free of any import from the MCP release path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import EndpointConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    ResolvedPackPaths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

MCP_LINEAGE_VERSION = "bfcl-mcp-publication-lineage-v1"
MCP_LINEAGE_RELATIVE_PATH = Path("provenance") / "mcp_lineage.json"
AUTHORING_LINEAGE_VERSION = "bfcl-authoring-publication-lineage-v2"
AUTHORING_LINEAGE_RELATIVE_PATH = Path("provenance") / "authoring_lineage.json"
_REVIEW_PACKET_RELATIVE_PATH = Path("provenance") / "review_packet.json"
_REVIEW_APPROVAL_RELATIVE_PATH = Path("provenance") / "review_approval.json"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class OriginProvenanceError(ValueError):
    """Raised when pack origin fields contradict executable pack identity."""


@dataclass(frozen=True)
class OriginProfile:
    """Inert contract for one provider lineage record."""

    schema_version: str
    provider_kind: str
    origin: str
    lineage_relative_path: Path
    label: str
    allowed_profile_versions: tuple[str, ...]
    identity_digest_fields: tuple[str, ...]
    endpoint_content_field: str | None
    endpoint_attestation_field: str | None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.schema_version, self.provider_kind, self.origin)

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.schema_version,
                self.provider_kind,
                self.origin,
                self.label,
            )
        ):
            raise ValueError("origin profile text fields must be non-empty")
        if self.lineage_relative_path.is_absolute() or ".." in self.lineage_relative_path.parts:
            raise ValueError("origin lineage path must stay inside the pack")
        if not self.allowed_profile_versions or len(self.allowed_profile_versions) != len(
            set(self.allowed_profile_versions)
        ):
            raise ValueError("origin profile versions must be non-empty and unique")
        if len(self.identity_digest_fields) != len(set(self.identity_digest_fields)):
            raise ValueError("origin identity digest fields must be unique")


MCP_ORIGIN_PROFILE = OriginProfile(
    schema_version=MCP_LINEAGE_VERSION,
    provider_kind="mcp",
    origin="mcp_backed_endpoint",
    lineage_relative_path=MCP_LINEAGE_RELATIVE_PATH,
    label="MCP lineage",
    allowed_profile_versions=("bfcl-mcp-oracle-v1",),
    identity_digest_fields=(
        "source_pack_fingerprint",
        "effective_content_digest",
        "conformance_digest",
        "tool_catalog_digest",
    ),
    endpoint_content_field="effective_content_digest",
    endpoint_attestation_field="conformance_digest",
)
AUTHORING_ORIGIN_PROFILE = OriginProfile(
    schema_version=AUTHORING_LINEAGE_VERSION,
    provider_kind="bfcl_authoring",
    origin="unified_authoring_release",
    lineage_relative_path=AUTHORING_LINEAGE_RELATIVE_PATH,
    label="authoring lineage",
    allowed_profile_versions=("bfcl-authoring-release-v2",),
    identity_digest_fields=("source_pack_fingerprint",),
    endpoint_content_field=None,
    endpoint_attestation_field=None,
)
DEFAULT_ORIGIN_PROFILES: Mapping[tuple[str, str, str], OriginProfile] = (
    MappingProxyType(
        {
            MCP_ORIGIN_PROFILE.key: MCP_ORIGIN_PROFILE,
            AUTHORING_ORIGIN_PROFILE.key: AUTHORING_ORIGIN_PROFILE,
        }
    )
)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OriginProvenanceError(f"MCP lineage repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise OriginProvenanceError(f"MCP lineage contains {token}")


def _digest(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise OriginProvenanceError(f"{label} must be sha256:<64 lowercase hex>")
    return value


def _sealed_record(paths: ResolvedPackPaths, relative: Path, label: str) -> dict[str, Any]:
    """Read one provenance record from inside the fingerprinted tree."""
    path = paths.pack_root / relative
    if path.is_symlink() or not path.is_file():
        raise OriginProvenanceError(f"{label} must be a regular file inside the pack")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OriginProvenanceError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise OriginProvenanceError(f"{label} must be a JSON object")
    return document


def _self_digest(document: Mapping[str, Any], field: str, label: str) -> str:
    """Recompute a record's own digest field so an edited record cannot keep its name."""
    claimed = _digest(document.get(field), f"{label} {field}")
    unsigned = {key: value for key, value in document.items() if key != field}
    observed = "sha256:" + hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    if claimed != observed:
        raise OriginProvenanceError(f"{label} {field} mismatch")
    return str(claimed)


def load_origin_provenance(
    paths: ResolvedPackPaths,
    endpoint_config: EndpointConfig | None,
    *,
    pack_fingerprint: str,
    pack_id: str,
    pack_version: str,
    profiles: Mapping[
        tuple[str, str, str], OriginProfile
    ] = DEFAULT_ORIGIN_PROFILES,
) -> dict[str, Any] | None:
    """Resolve exactly one strict profile and return its publication-safe projection."""
    present = [
        profile
        for profile in profiles.values()
        if (paths.pack_root / profile.lineage_relative_path).exists()
    ]
    if not present:
        return None
    unique_paths = {profile.lineage_relative_path for profile in present}
    if len(present) != 1 or len(unique_paths) != 1:
        raise OriginProvenanceError(
            "origin_profile_ambiguous: multiple provider lineage profiles are present"
        )
    profile = present[0]
    if profiles.get(profile.key) != profile:
        raise OriginProvenanceError(
            "origin_profile_registry_mismatch: profile key does not match its contract"
        )
    document = _sealed_record(
        paths,
        profile.lineage_relative_path,
        profile.label,
    )
    observed_key = (
        document.get("schema_version"),
        document.get("provider_kind"),
        document.get("origin"),
    )
    if observed_key != profile.key:
        raise OriginProvenanceError(
            "origin_profile_mismatch: lineage does not match its selected profile"
        )
    profile_version = document.get("profile_version")
    if profile_version not in profile.allowed_profile_versions:
        raise OriginProvenanceError(
            "origin_profile_version_unknown: lineage profile_version is unsupported"
        )
    claimed = _self_digest(document, "record_digest", profile.label)
    if (
        profile.endpoint_content_field is not None
        or profile.endpoint_attestation_field is not None
    ) and endpoint_config is None:
        raise OriginProvenanceError(
            f"{profile.label} is present on a non-endpoint pack"
        )

    identity = document.get("identity")
    review = document.get("review")
    declared = document.get("pack")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(review, Mapping)
        or not isinstance(declared, Mapping)
    ):
        raise OriginProvenanceError(
            f"{profile.label} pack, identity, and review must be objects"
        )
    if str(declared.get("pack_id")) != pack_id or str(declared.get("version")) != pack_version:
        raise OriginProvenanceError(
            f"{profile.label} names a different pack than manifest.yaml, so it was not "
            "produced by freezing this pack"
        )
    identity_digests = {
        field: _digest(identity.get(field), f"{profile.label} {field}")
        for field in profile.identity_digest_fields
    }
    source_pack = identity_digests.get("source_pack_fingerprint")
    if source_pack is None:
        raise OriginProvenanceError(
            "origin_profile_invalid: source_pack_fingerprint is required"
        )
    packet = _sealed_record(paths, _REVIEW_PACKET_RELATIVE_PATH, "review packet")
    approval = _sealed_record(paths, _REVIEW_APPROVAL_RELATIVE_PATH, "review approval")
    packet_digest_field = (
        "record_digest"
        if packet.get("schema_version") == "bfcl-authoring-review-packet-v2"
        else "packet_digest"
    )
    packet_digest = _self_digest(packet, packet_digest_field, "review packet")
    approval_digest = _self_digest(approval, "approval_digest", "review approval")
    if review.get("packet_digest") != packet_digest:
        raise OriginProvenanceError("MCP lineage cites a review packet the pack does not carry")
    if review.get("approval_digest") != approval_digest:
        raise OriginProvenanceError("MCP lineage cites an approval the pack does not carry")
    if approval.get("review_packet_digest") != packet_digest:
        raise OriginProvenanceError("the sealed approval covers a different review packet")
    packet_sources = packet.get("source_digests")
    candidate_pack = packet.get("candidate_pack")
    reviewed_pack = (
        candidate_pack.get("fingerprint")
        if isinstance(candidate_pack, Mapping)
        else packet_sources.get("canonical_pack")
        if isinstance(packet_sources, Mapping)
        else None
    )
    if reviewed_pack != source_pack:
        raise OriginProvenanceError(
            "the approved review packet covers a different pre-freeze pack than the lineage"
        )
    if (
        profile.endpoint_content_field is not None
        and endpoint_config is not None
        and endpoint_config.expected.content_digest
        != identity_digests.get(profile.endpoint_content_field)
    ):
        raise OriginProvenanceError(
            f"{profile.label} effective digest differs from endpoint_config.yaml"
        )
    if profile.endpoint_attestation_field is not None and (
        endpoint_config is None
        or endpoint_config.attestation is None
        or endpoint_config.attestation.expected_digest
        != identity_digests.get(profile.endpoint_attestation_field)
    ):
        raise OriginProvenanceError(
            f"{profile.label} conformance digest differs from endpoint_config.yaml"
        )
    projection: dict[str, Any] = {
        "provider_kind": profile.provider_kind,
        "origin": profile.origin,
        "profile_version": profile_version,
        "mode": document.get("mode"),
        "frozen_pack_fingerprint": (
            pack_fingerprint
            if pack_fingerprint.startswith("sha256:")
            else f"sha256:{pack_fingerprint}"
        ),
        "source_pack_fingerprint": source_pack,
        "lineage_record_digest": claimed,
        "review_packet_digest": _digest(
            review.get("packet_digest"),
            "MCP review packet digest",
        ),
        "review_approval_digest": _digest(
            review.get("approval_digest"),
            "MCP review approval digest",
        ),
        # Reviewer identity is intentionally omitted. The frozen lineage retains it for
        # audit, while publication needs the decision digest rather than personal data.
    }
    if document.get("adapter_kind") is not None:
        projection["adapter_kind"] = document.get("adapter_kind")
    for field, digest in identity_digests.items():
        if field != "source_pack_fingerprint":
            projection[field] = digest
    return projection


def load_mcp_origin(
    paths: ResolvedPackPaths,
    endpoint_config: EndpointConfig | None,
    *,
    pack_fingerprint: str,
    pack_id: str,
    pack_version: str,
) -> dict[str, Any] | None:
    """Compatibility wrapper for the legacy MCP-only import surface."""
    return load_origin_provenance(
        paths,
        endpoint_config,
        pack_fingerprint=pack_fingerprint,
        pack_id=pack_id,
        pack_version=pack_version,
        profiles={MCP_ORIGIN_PROFILE.key: MCP_ORIGIN_PROFILE},
    )
