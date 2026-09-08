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

"""Versioned, adapter-neutral review packets and explicit approval."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.authoring_release.contracts import ReleaseAdapter
from nemotron.steps.byob.runtime.authoring_release.versions import (
    MCP_REVIEW_APPROVAL_VERSION_V1,
    MCP_REVIEW_PACKET_VERSION_V1,
    REVIEW_APPROVAL_VERSION_V2,
    REVIEW_PACKET_VERSION_V2,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

REQUIRED_CHECKLIST_V2 = frozenset(
    {
        "semantics",
        "descriptions_and_snapshots",
        "held_out_policy",
        "assumptions",
        "validation_evidence",
        "independently_verified_certification",
        "pre_model_authorization",
        "answered_questions",
    }
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PACKET_KEYS = frozenset(
    {
        "schema_version",
        "adapter_kind",
        "source_identity_digest",
        "certification_tier",
        "candidate_pack",
        "source_digests",
        "adapter_review",
        "blockers",
        "risks",
        "record_digest",
    }
)
_APPROVAL_KEYS = frozenset(
    {
        "schema_version",
        "review_packet_digest",
        "approved_by",
        "reviewed_at",
        "checklist",
        "acknowledged_risks",
        "note",
        "approval_digest",
    }
)


class AuthoringReviewError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    source = path.resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuthoringReviewError(
            "release_record_invalid",
            f"cannot read {label} {source}: {exc}",
            recovery="restore the immutable reviewed record",
        ) from exc
    if not isinstance(value, dict):
        raise AuthoringReviewError(
            "release_record_invalid",
            f"{label} must be a JSON object",
            recovery="restore the immutable reviewed record",
        )
    return value


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AuthoringReviewError(
            "release_digest_invalid",
            f"{label} must be sha256:<64 lowercase hex characters>",
            recovery="rebuild the record from verified inputs",
        )
    return value


def _verify_record_digest(
    document: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    claimed = _validate_digest(document.get(field), f"{label}.{field}")
    unsigned = {key: value for key, value in document.items() if key != field}
    if claimed != sha256_json(unsigned):
        raise AuthoringReviewError(
            "release_digest_mismatch",
            f"{label} changed after {field} was computed",
            recovery="restore or rebuild the immutable record",
        )
    return claimed


def _canonical_mappings(
    values: Any,
    *,
    label: str,
    identity_field: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        raise AuthoringReviewError(
            "release_record_invalid",
            f"{label} must be a sequence",
            recovery="return canonical adapter review records",
        )
    documents: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise AuthoringReviewError(
                "release_record_invalid",
                f"{label} entries must be mappings",
                recovery="return canonical adapter review records",
            )
        document = dict(value)
        if identity_field not in document:
            document[identity_field] = (
                f"{label.removesuffix('s')}:"
                f"{sha256_json(document).removeprefix('sha256:')[:16]}"
            )
        documents.append(document)
    identities = [str(value[identity_field]) for value in documents]
    if len(identities) != len(set(identities)):
        raise AuthoringReviewError(
            "release_record_invalid",
            f"{label} identities must be unique",
            recovery="deduplicate adapter review records",
        )
    return sorted(documents, key=lambda value: str(value[identity_field]))


@dataclass(frozen=True)
class ReviewPacketV2:
    document: dict[str, Any]

    @property
    def digest(self) -> str:
        return str(self.document["record_digest"])

    def verify(self) -> None:
        if self.document.get("schema_version") != REVIEW_PACKET_VERSION_V2:
            raise AuthoringReviewError(
                "review_packet_version_unsupported",
                "review packet is not bfcl-authoring-review-packet-v2",
                recovery="use a supported versioned loader",
            )
        if set(self.document) != _PACKET_KEYS:
            raise AuthoringReviewError(
                "release_record_invalid",
                "review packet fields do not match the v2 contract",
                recovery="rebuild the packet with the versioned kernel",
            )
        if (
            not isinstance(self.document["adapter_kind"], str)
            or not self.document["adapter_kind"]
        ):
            raise AuthoringReviewError(
                "release_record_invalid",
                "review packet adapter_kind must be non-empty",
                recovery="rebuild the packet with a registered adapter",
            )
        _validate_digest(
            self.document["source_identity_digest"],
            "source_identity_digest",
        )
        if self.document["certification_tier"] not in {"A0", "A1", "A2"}:
            raise AuthoringReviewError(
                "certification_tier_invalid",
                "review packet certification tier is invalid",
                recovery="rebuild from an independently verified certification",
            )
        candidate = self.document["candidate_pack"]
        if not isinstance(candidate, Mapping) or set(candidate) != {"fingerprint"}:
            raise AuthoringReviewError(
                "release_record_invalid",
                "candidate_pack must contain only its fingerprint",
                recovery="rebuild the packet with the versioned kernel",
            )
        _validate_digest(candidate["fingerprint"], "candidate pack fingerprint")
        source_digests = self.document["source_digests"]
        if not isinstance(source_digests, Mapping) or not source_digests:
            raise AuthoringReviewError(
                "release_record_invalid",
                "review packet requires source digests",
                recovery="bind every reviewed release input",
            )
        for name, digest in source_digests.items():
            if not isinstance(name, str) or not name:
                raise AuthoringReviewError(
                    "release_record_invalid",
                    "source digest names must be non-empty strings",
                    recovery="rebuild the packet with canonical source names",
                )
            _validate_digest(digest, f"source_digests.{name}")
        if not isinstance(self.document["adapter_review"], Mapping):
            raise AuthoringReviewError(
                "release_record_invalid",
                "adapter_review must be a mapping",
                recovery="rebuild the packet with a registered adapter",
            )
        for label, identity in (("blockers", "blocker_id"), ("risks", "risk_id")):
            canonical = _canonical_mappings(
                self.document[label],
                label=label,
                identity_field=identity,
            )
            if canonical != self.document[label]:
                raise AuthoringReviewError(
                    "release_record_invalid",
                    f"{label} are not canonical",
                    recovery="rebuild the packet with the versioned kernel",
                )
        _verify_record_digest(
            self.document,
            field="record_digest",
            label="review packet",
        )


@dataclass(frozen=True)
class ReviewApprovalV2:
    document: dict[str, Any]

    @property
    def digest(self) -> str:
        return str(self.document["approval_digest"])

    def verify(self) -> None:
        if self.document.get("schema_version") != REVIEW_APPROVAL_VERSION_V2:
            raise AuthoringReviewError(
                "review_approval_version_unsupported",
                "review approval is not bfcl-authoring-review-approval-v2",
                recovery="use a supported versioned loader",
            )
        if set(self.document) != _APPROVAL_KEYS:
            raise AuthoringReviewError(
                "release_record_invalid",
                "review approval fields do not match the v2 contract",
                recovery="rebuild approval from the exact packet",
            )
        _validate_digest(
            self.document["review_packet_digest"],
            "review_packet_digest",
        )
        if (
            not isinstance(self.document["approved_by"], str)
            or not self.document["approved_by"].strip()
        ):
            raise AuthoringReviewError(
                "reviewer_missing",
                "approval must name its reviewer",
                recovery="record a stable reviewer identity",
            )
        try:
            datetime.fromisoformat(
                str(self.document["reviewed_at"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise AuthoringReviewError(
                "review_timestamp_invalid",
                "approval reviewed_at is not ISO-8601",
                recovery="record the explicit review time",
            ) from exc
        checklist = self.document["checklist"]
        if (
            not isinstance(checklist, Mapping)
            or set(checklist) != REQUIRED_CHECKLIST_V2
            or not all(value is True for value in checklist.values())
        ):
            raise AuthoringReviewError(
                "review_checklist_incomplete",
                "approval checklist is incomplete",
                recovery="review and accept each named checklist item",
            )
        acknowledged = self.document["acknowledged_risks"]
        if (
            not isinstance(acknowledged, list)
            or not all(isinstance(value, str) for value in acknowledged)
            or acknowledged != sorted(set(acknowledged))
        ):
            raise AuthoringReviewError(
                "release_record_invalid",
                "acknowledged risks must be sorted unique strings",
                recovery="rebuild approval from current risk IDs",
            )
        if self.document["note"] is not None and not isinstance(
            self.document["note"], str
        ):
            raise AuthoringReviewError(
                "release_record_invalid",
                "approval note must be text or null",
                recovery="rebuild approval with a textual note",
            )
        _verify_record_digest(
            self.document,
            field="approval_digest",
            label="review approval",
        )


def build_review_packet(
    *,
    adapter: ReleaseAdapter,
    pack_root: Path,
    source_digests: Mapping[str, str],
) -> ReviewPacketV2:
    canonical_digests = {
        key: _validate_digest(value, f"source_digests.{key}")
        for key, value in sorted(source_digests.items())
    }
    fingerprint = adapter.validate_pack(pack_root)
    _validate_digest(fingerprint, "candidate pack fingerprint")
    contribution = adapter.review(pack_root, canonical_digests)
    _validate_digest(contribution.identity_digest, "adapter identity digest")
    if contribution.certification_tier not in {"A0", "A1", "A2"}:
        raise AuthoringReviewError(
            "certification_tier_invalid",
            f"adapter returned invalid certification tier {contribution.certification_tier!r}",
            recovery="use an independently verified adapter certification report",
        )
    blockers = _canonical_mappings(
        contribution.blockers,
        label="blockers",
        identity_field="blocker_id",
    )
    risks = _canonical_mappings(
        contribution.risks,
        label="risks",
        identity_field="risk_id",
    )
    document: dict[str, Any] = {
        "schema_version": REVIEW_PACKET_VERSION_V2,
        "adapter_kind": adapter.kind,
        "source_identity_digest": contribution.identity_digest,
        "certification_tier": contribution.certification_tier,
        "candidate_pack": {
            "fingerprint": fingerprint,
        },
        "source_digests": canonical_digests,
        "adapter_review": dict(contribution.review_data),
        "blockers": blockers,
        "risks": risks,
    }
    document["record_digest"] = sha256_json(document)
    packet = ReviewPacketV2(document)
    packet.verify()
    return packet


def write_review_packet(packet: ReviewPacketV2, path: Path) -> Path:
    packet.verify()
    return write_canonical_json(packet.document, path)


def load_review_packet(path: Path) -> Any:
    document = load_json_mapping(path, "review packet")
    version = document.get("schema_version")
    if version == REVIEW_PACKET_VERSION_V2:
        packet = ReviewPacketV2(document)
        packet.verify()
        return packet
    if version == MCP_REVIEW_PACKET_VERSION_V1:
        from nemotron.steps.byob.runtime.mcp.release.review import (
            load_review_packet as load_mcp_review_packet,
        )

        return load_mcp_review_packet(path)
    raise AuthoringReviewError(
        "review_packet_version_unsupported",
        f"unsupported review packet version {version!r}",
        recovery="use a v1 MCP or v2 authoring review packet",
    )


def build_review_approval(
    packet: ReviewPacketV2,
    *,
    approved_by: str,
    reviewed_at: str,
    checklist: Mapping[str, bool],
    acknowledged_risks: list[str] | tuple[str, ...] = (),
    note: str | None = None,
) -> ReviewApprovalV2:
    packet.verify()
    adapter_review = packet.document["adapter_review"]
    authoring = adapter_review.get("authoring")
    certification = adapter_review.get("certification")
    if (
        not isinstance(certification, Mapping)
        or not isinstance(certification.get("report_digest"), str)
        or "certification_report" not in packet.document["source_digests"]
    ):
        raise AuthoringReviewError(
            "independent_certification_missing",
            "final approval requires the independently verified certification record",
            recovery="rebuild review from verified certification evidence",
        )
    if (
        not isinstance(authoring, Mapping)
        or not isinstance(
            authoring.get("model_exposure_authorization_digest"),
            str,
        )
        or "model_exposure_authorization"
        not in packet.document["source_digests"]
    ):
        raise AuthoringReviewError(
            "pre_model_authorization_missing",
            "final approval cannot substitute for pre-model authorization",
            recovery="authorize model exposure, redraft, and rebuild review",
        )
    questions_status = authoring.get("questions_status")
    if questions_status not in {"answered", "not_required"}:
        raise AuthoringReviewError(
            "answered_questions_missing",
            "review does not establish the question/answer state",
            recovery="replay answered evidence or record that no questions were required",
        )
    if questions_status == "answered" and not {
        "parent_evidence",
        "open_questions",
        "answer_set",
    }.issubset(packet.document["source_digests"]):
        raise AuthoringReviewError(
            "answered_questions_missing",
            "answered revision artifacts are not all digest-bound",
            recovery="bind parent evidence, open questions, and answer set",
        )
    if packet.document["blockers"]:
        raise AuthoringReviewError(
            "review_packet_blocked",
            "a packet with blockers cannot be approved",
            recovery="resolve blockers and build a new packet",
        )
    if not approved_by.strip():
        raise AuthoringReviewError(
            "reviewer_missing",
            "approval must name a stable reviewer",
            recovery="provide --approved-by",
        )
    try:
        datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthoringReviewError(
            "review_timestamp_invalid",
            "reviewed_at must be an ISO-8601 timestamp",
            recovery="record the explicit review time",
        ) from exc
    if set(checklist) != REQUIRED_CHECKLIST_V2 or not all(checklist.values()):
        raise AuthoringReviewError(
            "review_checklist_incomplete",
            "every v2 release checklist item must be explicitly accepted",
            recovery="review and accept each named checklist item",
        )
    required_risks = {str(risk["risk_id"]) for risk in packet.document["risks"]}
    acknowledged = set(acknowledged_risks)
    if acknowledged != required_risks:
        raise AuthoringReviewError(
            "review_risks_unacknowledged",
            "acknowledged risks must exactly match the current packet",
            recovery="acknowledge every stable risk ID and no stale IDs",
        )
    document: dict[str, Any] = {
        "schema_version": REVIEW_APPROVAL_VERSION_V2,
        "review_packet_digest": packet.digest,
        "approved_by": approved_by.strip(),
        "reviewed_at": reviewed_at,
        "checklist": dict(sorted(checklist.items())),
        "acknowledged_risks": sorted(acknowledged),
        "note": note,
    }
    document["approval_digest"] = sha256_json(document)
    approval = ReviewApprovalV2(document)
    approval.verify()
    return approval


def write_review_approval(approval: ReviewApprovalV2, path: Path) -> Path:
    approval.verify()
    return write_canonical_json(approval.document, path)


def load_review_approval(path: Path) -> Any:
    document = load_json_mapping(path, "review approval")
    version = document.get("schema_version")
    if version == REVIEW_APPROVAL_VERSION_V2:
        approval = ReviewApprovalV2(document)
        approval.verify()
        return approval
    if version == MCP_REVIEW_APPROVAL_VERSION_V1:
        from nemotron.steps.byob.runtime.mcp.release.review import (
            load_review_approval as load_mcp_review_approval,
        )
        from nemotron.steps.byob.runtime.mcp.release.review import (
            load_review_packet as load_mcp_review_packet,
        )

        packet = load_mcp_review_packet(path.resolve().with_name("review_packet.json"))
        return load_mcp_review_approval(path, packet)
    raise AuthoringReviewError(
        "review_approval_version_unsupported",
        f"unsupported review approval version {version!r}",
        recovery="use a v1 MCP or v2 authoring review approval",
    )
