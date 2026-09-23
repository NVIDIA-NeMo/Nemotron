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

"""The endpoint conformance attestation, and why an oracle cannot certify itself.

An oracle that reports its own trustworthiness is not evidence, it is a claim. This module
exists to keep that distinction enforceable. The document a gateway serves at
`GET /v1/conformance` says what it believes about itself; verification here decides what BFCL
is willing to act on, and the two answers are allowed to differ.

Two rules carry most of the weight. A `level` field is never read as a verdict — the level a
document may keep is re-derived from the evidence beside it, so a gateway that writes `"L2"`
without complete state observability or without BFCL having run the conformance suite itself
cannot publish. And the digests in the attestation must agree with the
digests the pack pinned and with what `GET /v1/metadata` reports live; a document that
describes a different build than the one answering calls is the drift this exists to catch.

Findings accumulate instead of raising at the first problem. A reviewer asking why a pack was
refused wants the whole picture, and a single mismatch is rarely the only one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.release_seal import verify_release_digest

ATTESTATION_KIND = "bfcl-endpoint-conformance-v1"
MCP_PROFILE_VERSION = "bfcl-mcp-oracle-v1"
HTTP_PROFILE_VERSION = "bfcl-http-oracle-v1"
PROVIDER_KIND_MCP = "mcp"
PROVIDER_KIND_HTTP = "http"

LEVELS = ("L0", "L1", "L2")
REQUIREMENTS = frozenset({"required", "conditional"})
CHECK_STATUSES = frozenset({"pass", "not_applicable", "fail", "skipped"})
EVIDENCE_KINDS = frozenset({"locally_verified", "signed_release"})
STATE_OBSERVABILITY = frozenset({"complete", "diagnostic"})
# A mode-C attestation has to name its boundary from a closed set. Free-form prose here would
# be a claim no verifier can check, which is the same as no boundary at all.
READ_ONLY_BOUNDARIES = frozenset({"upstream_authorization", "immutable_snapshot_sandbox"})
MCP_L1_PROBES = tuple(f"P{index}" for index in range(1, 5))
MCP_L2_PROBES = tuple(f"P{index}" for index in range(1, 12))
HTTP_L1_PROBES = tuple(f"H{index}" for index in range(1, 5))
HTTP_L2_PROBES = tuple(f"H{index}" for index in range(1, 12))

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

# Digests that must be present, and those that carry an explicit null rather than being
# omitted, so two different deployments cannot hash to the same document by accident.
_REQUIRED_DIGESTS = (
    "effective_content_digest",
    "gateway_artifact_digest",
    "tool_catalog_digest",
    "probe_report_digest",
    "gateway_conformance_report_digest",
)
_NULLABLE_DIGESTS = ("shim_artifact_digest", "server_content_digest", "snapshot_digest")

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "provider_kind",
        "profile_version",
        "level",
        "gateway_evidence_kind",
        "gateway_evidence_issuer",
        "state_observability",
        "read_only_boundary",
        "checks",
        *_REQUIRED_DIGESTS,
        *_NULLABLE_DIGESTS,
    }
)


class AttestationError(ValueError):
    """Raised when a document cannot be read as a conformance attestation at all."""

    def __init__(self, message: str, *, code: str = "malformed") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ConformanceProfile:
    """Strict, inert rules used to interpret one provider attestation profile."""

    schema_version: str
    provider_kind: str
    profile_version: str
    levels: tuple[str, ...]
    publishable_level: str
    required_probes: tuple[tuple[str, tuple[str, ...]], ...]
    report_suite_kind: str
    report_suite_version: str
    timeout_probe_requirements: tuple[tuple[str, Any], ...]
    enforce_snapshot_identity: bool = False
    cap_without_server_content: bool = True
    cap_without_complete_state: bool = True

    def __post_init__(self) -> None:
        text_fields = (
            self.schema_version,
            self.provider_kind,
            self.profile_version,
            self.publishable_level,
            self.report_suite_kind,
            self.report_suite_version,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("conformance profile text fields must be non-empty")
        if not self.levels or len(self.levels) != len(set(self.levels)):
            raise ValueError("conformance profile levels must be non-empty and unique")
        if self.publishable_level not in self.levels:
            raise ValueError("publishable level must belong to the profile")
        probe_levels = [level for level, _ in self.required_probes]
        if len(probe_levels) != len(set(probe_levels)) or any(level not in self.levels for level in probe_levels):
            raise ValueError("required probe levels must be unique profile levels")
        for _, probes in self.required_probes:
            if len(probes) != len(set(probes)) or any(not probe.strip() for probe in probes):
                raise ValueError("required probe identifiers must be non-empty and unique")

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider_kind, self.profile_version)

    def probes_for(self, level: str) -> tuple[str, ...]:
        return dict(self.required_probes).get(level, ())


MCP_CONFORMANCE_PROFILE = ConformanceProfile(
    schema_version=ATTESTATION_KIND,
    provider_kind=PROVIDER_KIND_MCP,
    profile_version=MCP_PROFILE_VERSION,
    levels=LEVELS,
    publishable_level="L2",
    required_probes=(
        ("L1", MCP_L1_PROBES),
        ("L2", MCP_L2_PROBES),
    ),
    report_suite_kind="gateway",
    report_suite_version="bfcl-mcp-gateway-conformance-v1",
    timeout_probe_requirements=(
        ("timeout_observed", True),
        ("business_call_attempts", 1),
        ("episode_poisoned", True),
        ("transport_cleanup_completed", True),
        ("unknown_commit_state_preserved", True),
    ),
    enforce_snapshot_identity=True,
)
HTTP_CONFORMANCE_PROFILE = ConformanceProfile(
    schema_version=ATTESTATION_KIND,
    provider_kind=PROVIDER_KIND_HTTP,
    profile_version=HTTP_PROFILE_VERSION,
    levels=LEVELS,
    publishable_level="L2",
    required_probes=(
        ("L1", HTTP_L1_PROBES),
        ("L2", HTTP_L2_PROBES),
    ),
    report_suite_kind="endpoint",
    report_suite_version="bfcl-http-conformance-v1",
    timeout_probe_requirements=(
        ("timeout_observed", True),
        ("business_call_attempts", 1),
        ("episode_poisoned", True),
        ("transport_cleanup_completed", True),
        ("unknown_commit_state_preserved", True),
    ),
    enforce_snapshot_identity=False,
    cap_without_server_content=False,
)
DEFAULT_CONFORMANCE_PROFILES: Mapping[tuple[str, str], ConformanceProfile] = MappingProxyType(
    {
        HTTP_CONFORMANCE_PROFILE.key: HTTP_CONFORMANCE_PROFILE,
        MCP_CONFORMANCE_PROFILE.key: MCP_CONFORMANCE_PROFILE,
    }
)


def resolve_conformance_profile(
    *,
    provider_kind: Any,
    profile_version: Any,
    profiles: Mapping[tuple[str, str], ConformanceProfile] = DEFAULT_CONFORMANCE_PROFILES,
) -> ConformanceProfile:
    if not isinstance(provider_kind, str) or not isinstance(profile_version, str):
        raise AttestationError(
            "endpoint conformance provider_kind and profile_version must be strings",
            code="profile_identity_invalid",
        )
    profile = profiles.get((provider_kind, profile_version))
    if profile is None:
        raise AttestationError(
            "endpoint conformance names an unknown provider/profile pair",
            code="unknown_profile",
        )
    if profile.key != (provider_kind, profile_version):
        raise AttestationError(
            "conformance profile registry key does not match its profile",
            code="profile_registry_mismatch",
        )
    return profile


def attestation_digest(document: Mapping[str, Any]) -> str:
    """Digest the attestation exactly as the endpoint config pins it."""
    return "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def _digest_field(value: Any, source: str, *, nullable: bool) -> str | None:
    if value is None:
        if nullable:
            return None
        raise AttestationError(f"{source} must be sha256:<64 hex characters>, not null")
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AttestationError(f"{source} must be sha256:<64 lowercase hex characters>")
    return value


def _enum_field(value: Any, allowed: frozenset[str] | tuple[str, ...], source: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise AttestationError(f"{source} must be one of {sorted(allowed)}, got {value!r}")
    return value


@dataclass(frozen=True)
class ConformanceCheck:
    """One probe result as the endpoint reports it."""

    id: str
    requirement: str
    status: str
    reason: str | None

    @classmethod
    def from_mapping(cls, value: Any, *, source: str) -> ConformanceCheck:
        if not isinstance(value, dict):
            raise AttestationError(f"{source} must be a JSON object")
        allowed = {"id", "requirement", "status", "reason"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise AttestationError(f"{source} has unknown fields: {', '.join(unknown)}")
        missing = sorted(allowed - set(value))
        if missing:
            raise AttestationError(f"{source} is missing: {', '.join(missing)}")
        identifier = value["id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise AttestationError(f"{source}.id must be a non-empty string")
        reason = value["reason"]
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise AttestationError(f"{source}.reason must be null or a non-empty string")
        return cls(
            id=identifier.strip(),
            requirement=_enum_field(value["requirement"], REQUIREMENTS, f"{source}.requirement"),
            status=_enum_field(value["status"], CHECK_STATUSES, f"{source}.status"),
            reason=reason,
        )


@dataclass(frozen=True)
class ConformanceAttestation:
    """A parsed attestation, plus the verbatim document its digest was taken over."""

    document: dict[str, Any]
    profile: ConformanceProfile
    level: str
    effective_content_digest: str
    gateway_artifact_digest: str
    tool_catalog_digest: str
    probe_report_digest: str
    gateway_conformance_report_digest: str
    shim_artifact_digest: str | None
    server_content_digest: str | None
    snapshot_digest: str | None
    gateway_evidence_kind: str
    gateway_evidence_issuer: str
    state_observability: str
    read_only_boundary: str | None
    checks: tuple[ConformanceCheck, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        source: str = "endpoint conformance",
        profile: ConformanceProfile | None = None,
        profiles: Mapping[tuple[str, str], ConformanceProfile] = DEFAULT_CONFORMANCE_PROFILES,
    ) -> ConformanceAttestation:
        if not isinstance(value, dict):
            raise AttestationError(f"{source} must be a JSON object")
        unknown = sorted(set(value) - _REQUIRED_FIELDS)
        if unknown:
            # An unknown field may be a newer profile or a smuggled one; either way this
            # verifier cannot say what it means, so it refuses rather than ignoring it.
            raise AttestationError(f"{source} has unknown fields: {', '.join(unknown)}")
        missing = sorted(_REQUIRED_FIELDS - set(value))
        if missing:
            raise AttestationError(f"{source} is missing: {', '.join(missing)}")

        resolved_profile = resolve_conformance_profile(
            provider_kind=value["provider_kind"],
            profile_version=value["profile_version"],
            profiles=profiles,
        )
        if profile is not None and profile != resolved_profile:
            raise AttestationError(
                f"{source} does not match the selected conformance profile",
                code="profile_mismatch",
            )
        for field, expected in (
            ("schema_version", resolved_profile.schema_version),
            ("provider_kind", resolved_profile.provider_kind),
            ("profile_version", resolved_profile.profile_version),
        ):
            if value[field] != expected:
                raise AttestationError(f"{source}.{field} must be {expected!r}, got {value[field]!r}")

        issuer = value["gateway_evidence_issuer"]
        if not isinstance(issuer, str) or not issuer.strip():
            raise AttestationError(f"{source}.gateway_evidence_issuer must be a non-empty string")

        raw_checks = value["checks"]
        if not isinstance(raw_checks, list) or not raw_checks:
            raise AttestationError(f"{source}.checks must be a non-empty array")
        checks = tuple(
            ConformanceCheck.from_mapping(entry, source=f"{source}.checks[{index}]")
            for index, entry in enumerate(raw_checks)
        )
        seen = [check.id for check in checks]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise AttestationError(f"{source}.checks repeats {', '.join(duplicates)}")

        boundary = value["read_only_boundary"]
        if boundary is not None:
            boundary = _enum_field(boundary, READ_ONLY_BOUNDARIES, f"{source}.read_only_boundary")

        def required(name: str) -> str:
            digest = _digest_field(value[name], f"{source}.{name}", nullable=False)
            assert digest is not None  # _digest_field raises rather than returning None here
            return digest

        def optional(name: str) -> str | None:
            return _digest_field(value[name], f"{source}.{name}", nullable=True)

        return cls(
            document=dict(value),
            profile=resolved_profile,
            level=_enum_field(value["level"], resolved_profile.levels, f"{source}.level"),
            effective_content_digest=required("effective_content_digest"),
            gateway_artifact_digest=required("gateway_artifact_digest"),
            tool_catalog_digest=required("tool_catalog_digest"),
            probe_report_digest=required("probe_report_digest"),
            gateway_conformance_report_digest=required("gateway_conformance_report_digest"),
            shim_artifact_digest=optional("shim_artifact_digest"),
            server_content_digest=optional("server_content_digest"),
            snapshot_digest=optional("snapshot_digest"),
            gateway_evidence_kind=_enum_field(
                value["gateway_evidence_kind"], EVIDENCE_KINDS, f"{source}.gateway_evidence_kind"
            ),
            gateway_evidence_issuer=issuer.strip(),
            state_observability=_enum_field(
                value["state_observability"], STATE_OBSERVABILITY, f"{source}.state_observability"
            ),
            read_only_boundary=boundary,
            checks=checks,
        )


@dataclass(frozen=True)
class ConformanceVerdict:
    """What BFCL is willing to believe, as opposed to what the endpoint claimed."""

    attested_level: str
    effective_level: str
    publishable: bool
    findings: tuple[str, ...]
    caps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "attested_level": self.attested_level,
            "effective_level": self.effective_level,
            "publishable": self.publishable,
            "findings": list(self.findings),
            "caps": list(self.caps),
        }


def _document_digest(document: Mapping[str, Any]) -> str:
    return attestation_digest(document)


def _evidence_findings(
    attestation: ConformanceAttestation,
    *,
    probe_report: Mapping[str, Any] | None,
    gateway_conformance_report: Mapping[str, Any] | None,
    trusted_evidence_keys: Mapping[str, Ed25519PublicKey] | None,
    expected_evidence_issuer: str | None,
) -> list[str]:
    """Verify the artifacts whose digests the attestation cites.

    A digest-shaped string is a reference, not evidence.  The referenced document has to be
    present and hash to that value before a Gold verdict can rely on it.
    """
    findings: list[str] = []
    if probe_report is None:
        findings.append("probe_report_missing")
    elif _document_digest(probe_report) != attestation.probe_report_digest:
        findings.append("probe_report_digest_mismatch")
    else:
        expected_checks = [
            {
                "id": check.id,
                "requirement": check.requirement,
                "status": check.status,
                "reason": check.reason,
            }
            for check in attestation.checks
        ]
        if probe_report.get("probes") != expected_checks:
            findings.append("probe_report_checks_mismatch")

    if gateway_conformance_report is None:
        findings.append("gateway_conformance_report_missing")
    elif _document_digest(gateway_conformance_report) != attestation.gateway_conformance_report_digest:
        findings.append("gateway_conformance_report_digest_mismatch")
    else:
        if gateway_conformance_report.get("gateway_artifact_digest") != attestation.gateway_artifact_digest:
            findings.append("gateway_report_artifact_mismatch")
        if gateway_conformance_report.get("effective_content_digest") != attestation.effective_content_digest:
            findings.append("gateway_report_effective_digest_mismatch")
        if gateway_conformance_report.get("tool_catalog_digest") != attestation.tool_catalog_digest:
            findings.append("gateway_report_catalog_digest_mismatch")
        if gateway_conformance_report.get("issuer") != attestation.gateway_evidence_issuer:
            findings.append("gateway_report_issuer_mismatch")
        suite = gateway_conformance_report.get("suite")
        p9 = suite.get("p9") if isinstance(suite, Mapping) else None
        if (
            not isinstance(suite, Mapping)
            or suite.get("kind") != attestation.profile.report_suite_kind
            or suite.get("profile_version") != attestation.profile.report_suite_version
        ):
            findings.append("gateway_report_suite_invalid")
        required_p9 = dict(attestation.profile.timeout_probe_requirements)
        if required_p9 and (
            not isinstance(p9, Mapping) or any(p9.get(field) != expected for field, expected in required_p9.items())
        ):
            findings.append("gateway_report_timeout_probe_invalid")
        producer = suite.get("producer") if isinstance(suite, Mapping) else None
        if not isinstance(producer, Mapping) or trusted_evidence_keys is None or expected_evidence_issuer is None:
            findings.append("gateway_evidence_signature_missing")
        else:
            unsigned_suite = {
                key: value for key, value in suite.items() if key not in {"evidence_digest", "signature"}
            }
            digest = attestation_digest(unsigned_suite)
            if (
                producer.get("gateway_artifact_digest") != attestation.gateway_artifact_digest
                or suite.get("evidence_digest") != digest
            ):
                findings.append("gateway_evidence_binding_mismatch")
            else:
                try:
                    verify_release_digest(
                        issuer=str(producer.get("issuer")),
                        expected_issuer=expected_evidence_issuer,
                        key_id=str(producer.get("signing_key_id")),
                        digest=digest,
                        signature=str(suite.get("signature")),
                        trusted_public_keys=trusted_evidence_keys,
                    )
                except ValueError:
                    findings.append("gateway_evidence_signature_invalid")
    return findings


def _identity_semantic_findings(attestation: ConformanceAttestation) -> list[str]:
    """Enforce the mode information encoded by optional identity components."""
    if not attestation.profile.enforce_snapshot_identity:
        return []
    findings: list[str] = []
    is_snapshot = attestation.snapshot_digest is not None
    if is_snapshot and attestation.shim_artifact_digest is not None:
        findings.append("snapshot_and_shim_digests_both_present")
    if is_snapshot and attestation.read_only_boundary is None:
        findings.append("snapshot_read_only_boundary_missing")
    if not is_snapshot and attestation.read_only_boundary is not None:
        findings.append("read_only_boundary_without_snapshot")
    return findings


def _check_findings(attestation: ConformanceAttestation) -> list[str]:
    findings: list[str] = []
    for check in attestation.checks:
        if check.status == "skipped":
            # A skipped probe is the absence of evidence. Treating it as satisfaction is
            # how an untested oracle acquires a passing record.
            findings.append(f"check_skipped:{check.id}")
        elif check.status == "fail":
            findings.append(f"check_failed:{check.id}")
        elif check.requirement == "required" and check.status != "pass":
            findings.append(f"required_check_not_passed:{check.id}")
        elif check.status == "not_applicable":
            if check.requirement != "conditional":
                findings.append(f"required_check_not_passed:{check.id}")
            elif check.reason is None:
                findings.append(f"conditional_check_without_reason:{check.id}")
    return findings


def _probe_coverage_findings(attestation: ConformanceAttestation) -> list[str]:
    """Independently enforce the selected profile's level-to-probe contract."""
    expected = attestation.profile.probes_for(attestation.level)
    present = {check.id for check in attestation.checks}
    return [
        f"probe_missing_for_{attestation.level.lower()}:{identifier}"
        for identifier in expected
        if identifier not in present
    ]


def verify_conformance(
    document: Any,
    *,
    expected_digest: str,
    metadata_content_digest: str,
    expected_identity: Mapping[str, str | None] | None = None,
    probe_report: Mapping[str, Any] | None = None,
    gateway_conformance_report: Mapping[str, Any] | None = None,
    trusted_evidence_keys: Mapping[str, Ed25519PublicKey] | None = None,
    expected_evidence_issuer: str | None = None,
    profile: ConformanceProfile | None = None,
    profiles: Mapping[tuple[str, str], ConformanceProfile] = DEFAULT_CONFORMANCE_PROFILES,
) -> ConformanceVerdict:
    """Decide which level an attestation earns, and whether it may publish.

    `expected_digest` is what the pack pinned, `metadata_content_digest` is what the live
    endpoint reports now, and `expected_identity` is the set of digests the pack recorded at
    intake. The two report documents are required evidence, not optional digest strings:
    callers cannot upgrade an endpoint by repeating a hash the endpoint supplied.
    """
    findings: list[str] = []
    caps: list[str] = []

    try:
        attestation = ConformanceAttestation.from_mapping(
            document,
            profile=profile,
            profiles=profiles,
        )
    except AttestationError as exc:
        # Unreadable is strictly worse than low: nothing below can be evaluated.
        return ConformanceVerdict(
            attested_level="unknown",
            effective_level="L0",
            publishable=False,
            findings=(f"attestation_malformed:{exc.code}",),
            caps=(f"attestation_malformed:{exc.code}",),
        )

    observed = attestation_digest(attestation.document)
    if observed != expected_digest:
        findings.append("attestation_digest_mismatch")
    if attestation.effective_content_digest != metadata_content_digest:
        # The document describes a different build than the one answering calls.
        findings.append("effective_digest_differs_from_metadata")

    for name, value in dict(expected_identity or {}).items():
        actual = getattr(attestation, name, None)
        if actual != value:
            findings.append(f"identity_mismatch:{name}")

    findings.extend(_check_findings(attestation))
    findings.extend(_probe_coverage_findings(attestation))
    findings.extend(_identity_semantic_findings(attestation))

    findings.extend(
        _evidence_findings(
            attestation,
            probe_report=probe_report,
            gateway_conformance_report=gateway_conformance_report,
            trusted_evidence_keys=trusted_evidence_keys,
            expected_evidence_issuer=expected_evidence_issuer,
        )
    )

    immutable_snapshot = (
        attestation.snapshot_digest is not None and attestation.read_only_boundary == "immutable_snapshot_sandbox"
    )
    if (
        attestation.profile.cap_without_server_content
        and attestation.server_content_digest is None
        and not immutable_snapshot
    ):
        # Without a server content statement the effective digest still catches adapter and
        # catalog drift, but not the server's own business logic changing underneath. An
        # immutable local snapshot is the one contract-defined exception.
        caps.append("server_content_digest_absent")
    if attestation.profile.cap_without_complete_state and attestation.state_observability != "complete":
        caps.append("state_observability_incomplete")

    effective = attestation.level
    if caps and effective == attestation.profile.publishable_level:
        publishable_index = attestation.profile.levels.index(attestation.profile.publishable_level)
        effective = attestation.profile.levels[max(0, publishable_index - 1)]
    if findings:
        effective = attestation.profile.levels[0]
    return ConformanceVerdict(
        attested_level=attestation.level,
        effective_level=effective,
        publishable=effective == attestation.profile.publishable_level and not findings,
        findings=tuple(findings),
        caps=tuple(caps),
    )
