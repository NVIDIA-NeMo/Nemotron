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

"""Producing the attestation the gateway serves at `GET /v1/conformance`.

This side states what the gateway can actually demonstrate, and nothing more. The temptation
in a module like this is to write `"L2"` because the deployment is intended to be certifiable;
the discipline is that the level is computed from the checks that were really run, so an
unimplemented probe lowers the claim instead of being assumed to pass.

Nothing here decides whether the pack may publish. The gateway is the subject under test, and
a subject grading itself is not evidence, so it declares `locally_verified` and leaves the
consequence to `benchmark_families/bfcl/conformance.py`, which requires the independently
supplied reports whose digests this document cites. Both sides share one schema so the document
cannot be valid to produce and invalid to read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    MCP_PROFILE_VERSION,
    PROVIDER_KIND_MCP,
    ConformanceAttestation,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.config import McpOracleConfig
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    GatewayArtifacts,
    GatewayIdentity,
)
from nemotron.steps.byob.runtime.release_seal import (
    ReleaseSealAuthority,
    sign_release_digest,
    verify_release_digest,
)

# The issuer name identifies which suite produced the evidence, so a verifier configured to
# trust a signed release can tell one issuer from another.
GATEWAY_EVIDENCE_ISSUER = "bfcl-mcp-conformance-v1"
GATEWAY_SUITE_VERSION = "bfcl-mcp-gateway-conformance-v1"

# P1-P3 establish discovery, P4 establishes executable L1, and P5-P11 establish L2.
# P7 and P8 are conditional, but they still have to be present as either pass or an explicit
# not_applicable with a reason. Absence never means not applicable.
DISCOVERY_PROBES = ("P1", "P2", "P3")
EXECUTABLE_PROBE = "P4"
L2_PROBES = tuple(f"P{index}" for index in range(5, 12))
CONDITIONAL_PROBES = frozenset({"P7", "P8"})
_PROBE_STATUSES = frozenset({"pass", "fail", "skipped", "not_applicable"})
_PROBE_REQUIREMENTS = frozenset({"required", "conditional"})


@dataclass(frozen=True)
class ProbeOutcome:
    """One probe result, as observed rather than as intended."""

    id: str
    requirement: str
    status: str
    reason: str | None = None

    def as_check(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "requirement": self.requirement,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConformanceEvidence:
    """Everything the gateway actually ran, plus the artifact digests it can prove."""

    probes: tuple[ProbeOutcome, ...]
    # A separate document from the probe results: it records the suite, its version, and the
    # exact gateway artifact it exercised, which is what stops a passing report from being
    # reused for a different build.
    suite: Mapping[str, Any] = field(default_factory=dict)
    trusted_producer: bool = False

    @property
    def probe_report(self) -> dict[str, Any]:
        return {"probes": [probe.as_check() for probe in self.probes]}

    @property
    def report_digest(self) -> str:
        return attestation_digest(self.probe_report)

    def gateway_report(
        self,
        gateway_artifact_digest: str,
        *,
        effective_content_digest: str,
        tool_catalog_digest: str,
    ) -> dict[str, Any]:
        return {
            "issuer": GATEWAY_EVIDENCE_ISSUER,
            "gateway_artifact_digest": gateway_artifact_digest,
            "effective_content_digest": effective_content_digest,
            "tool_catalog_digest": tool_catalog_digest,
            "suite": dict(self.suite),
        }

    def suite_digest(
        self,
        gateway_artifact_digest: str,
        *,
        effective_content_digest: str,
        tool_catalog_digest: str,
    ) -> str:
        return attestation_digest(
            self.gateway_report(
                gateway_artifact_digest,
                effective_content_digest=effective_content_digest,
                tool_catalog_digest=tool_catalog_digest,
            )
        )


def discovery_evidence(report: DiscoveryReport) -> ConformanceEvidence:
    """Lift the L0 discovery attestation into conformance evidence, unchanged.

    Discovery already recorded `P1` to `P3` against a real server. Re-deriving them here
    would let two code paths disagree about the same observation, so they are copied.
    """
    checks = report.document.get("checks", [])
    if not isinstance(checks, list) or not checks:
        raise ValueError(
            "the discovery report carries no probe results, so there is no evidence to "
            "attest to; refusing to serve an attestation with an empty check list"
        )
    probes = tuple(
        ProbeOutcome(
            id=str(entry["id"]),
            requirement=str(entry.get("requirement", "required")),
            status=str(entry["status"]),
            reason=entry.get("reason"),
        )
        for entry in checks
    )
    return ConformanceEvidence(
        probes=probes,
        suite={
            "kind": "discovery",
            "profile_version": report.document.get("profile_version"),
            "discovery_report_digest": report.document.get("report_digest"),
        },
    )


def load_conformance_evidence(
    probe_report: Any,
    gateway_suite: Any,
    *,
    trusted_public_keys: Mapping[str, Ed25519PublicKey] | None = None,
    expected_issuer: str | None = None,
    gateway_artifact_digest: str | None = None,
) -> ConformanceEvidence:
    """Strictly load BFCL-produced evidence before a gateway may advertise it."""
    if not isinstance(probe_report, dict) or set(probe_report) != {"probes"}:
        raise ValueError("probe report must contain exactly the 'probes' field")
    raw_probes = probe_report["probes"]
    if not isinstance(raw_probes, list):
        raise ValueError("probe report.probes must be an array")
    probes: list[ProbeOutcome] = []
    for index, raw in enumerate(raw_probes):
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "requirement",
            "status",
            "reason",
        }:
            raise ValueError(f"probe report.probes[{index}] has an invalid field set")
        identifier = raw["id"]
        requirement = raw["requirement"]
        status = raw["status"]
        reason = raw["reason"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"probe report.probes[{index}].id must be non-empty")
        if requirement not in _PROBE_REQUIREMENTS:
            raise ValueError(f"probe report.probes[{index}].requirement is invalid")
        if status not in _PROBE_STATUSES:
            raise ValueError(f"probe report.probes[{index}].status is invalid")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"probe report.probes[{index}].reason must be null or non-empty")
        probes.append(
            ProbeOutcome(
                id=identifier,
                requirement=requirement,
                status=status,
                reason=reason,
            )
        )
    identifiers = [probe.id for probe in probes]
    expected = [f"P{index}" for index in range(1, 12)]
    if identifiers != expected:
        raise ValueError("probe report must contain P1 through P11 in order")

    if not isinstance(gateway_suite, dict) or set(gateway_suite) not in {
        frozenset(
            {
                "kind",
                "profile_version",
                "p9",
            }
        ),
        frozenset(
            {
                "kind",
                "profile_version",
                "p9",
                "producer",
                "evidence_digest",
                "signature",
            }
        ),
    }:
        raise ValueError("gateway suite has an invalid field set")
    p9 = gateway_suite.get("p9")
    required_p9 = {
        "timeout_observed": True,
        "business_call_attempts": 1,
        "episode_poisoned": True,
        "transport_cleanup_completed": True,
        "unknown_commit_state_preserved": True,
    }
    if (
        gateway_suite.get("kind") != "gateway"
        or gateway_suite.get("profile_version") != GATEWAY_SUITE_VERSION
        or not isinstance(p9, dict)
        or p9 != required_p9
    ):
        raise ValueError("gateway suite does not contain a passing P9 observation")
    trusted_producer = False
    producer = gateway_suite.get("producer")
    if producer is not None:
        if (
            not isinstance(producer, dict)
            or set(producer) != {"issuer", "signing_key_id", "gateway_artifact_digest"}
            or trusted_public_keys is None
            or expected_issuer is None
            or gateway_artifact_digest is None
            or producer.get("gateway_artifact_digest") != gateway_artifact_digest
        ):
            raise ValueError("gateway suite producer binding is invalid or untrusted")
        unsigned = {key: value for key, value in gateway_suite.items() if key not in {"evidence_digest", "signature"}}
        digest = attestation_digest(unsigned)
        if gateway_suite.get("evidence_digest") != digest:
            raise ValueError("gateway suite evidence digest mismatch")
        verify_release_digest(
            issuer=str(producer["issuer"]),
            expected_issuer=expected_issuer,
            key_id=str(producer["signing_key_id"]),
            digest=digest,
            signature=str(gateway_suite.get("signature")),
            trusted_public_keys=trusted_public_keys,
        )
        trusted_producer = True
    return ConformanceEvidence(
        probes=tuple(probes),
        suite=dict(gateway_suite),
        trusted_producer=trusted_producer,
    )


async def run_gateway_timeout_conformance(
    service: Any,
    *,
    context: Mapping[str, Any],
    business_tool: str,
    arguments: Mapping[str, Any],
    business_call_attempts: Callable[[], int],
    transport_cleanup_completed: Callable[[], bool],
    fixtures: Mapping[str, Any] | None = None,
    authority: ReleaseSealAuthority | None = None,
    gateway_artifact_digest: str | None = None,
) -> dict[str, Any]:
    """Run P9 against a controlled hanging MCP fixture through the real gateway core."""
    timeout_observed = False
    episode_poisoned = False
    session_id: str | None = None
    await service.start()
    try:
        created = await service.create_session(
            context=dict(context),
            fixtures=None if fixtures is None else dict(fixtures),
        )
        session_id = str(created["session_id"])
        try:
            await service.call_tool(
                session_id,
                name=business_tool,
                arguments=dict(arguments),
                turn_index=0,
            )
        except Exception as exc:  # noqa: BLE001 - the stable gateway code is inspected below
            timeout_observed = getattr(exc, "code", None) == "mcp_call_timeout"
        try:
            await service.call_tool(
                session_id,
                name=business_tool,
                arguments=dict(arguments),
                turn_index=1,
            )
        except Exception as exc:  # noqa: BLE001 - the stable gateway code is inspected below
            episode_poisoned = getattr(exc, "code", None) == "mcp_session_poisoned"
    finally:
        if session_id is not None:
            try:
                await service.delete_session(session_id)
            except Exception:  # noqa: BLE001 - shutdown below is the final bounded cleanup
                pass
        await service.shutdown()

    attempts = business_call_attempts()
    cleanup = transport_cleanup_completed()
    document: dict[str, Any] = {
        "kind": "gateway",
        "profile_version": GATEWAY_SUITE_VERSION,
        "p9": {
            "timeout_observed": timeout_observed,
            "business_call_attempts": attempts,
            "episode_poisoned": episode_poisoned,
            "transport_cleanup_completed": cleanup,
            # A timed-out mutation has unknown commit state. Poisoning the episode and refusing
            # a second call is the observable proof that the gateway did not invent rollback.
            "unknown_commit_state_preserved": timeout_observed and episode_poisoned,
        },
    }
    if (authority is None) != (gateway_artifact_digest is None):
        raise ValueError("gateway conformance signing requires both authority and artifact digest")
    if authority is not None and gateway_artifact_digest is not None:
        document["producer"] = {
            "issuer": authority.issuer,
            "signing_key_id": authority.key_id,
            "gateway_artifact_digest": gateway_artifact_digest,
        }
        document["evidence_digest"] = attestation_digest(document)
        document["signature"] = sign_release_digest(
            authority,
            str(document["evidence_digest"]),
        )
    return document


def _attained_level(probes: Sequence[ProbeOutcome], *, state_observability: str) -> str:
    """Derive the level from evidence, never from intent."""
    by_id = {probe.id: probe for probe in probes}

    def satisfies(identifier: str) -> bool:
        probe = by_id.get(identifier)
        if probe is None:
            return False
        if probe.status == "pass":
            return True
        return (
            identifier in CONDITIONAL_PROBES
            and probe.requirement == "conditional"
            and probe.status == "not_applicable"
            and isinstance(probe.reason, str)
            and bool(probe.reason.strip())
        )

    if not all(satisfies(identifier) for identifier in DISCOVERY_PROBES):
        return "L0"
    if not satisfies(EXECUTABLE_PROBE):
        return "L0"
    if state_observability != "complete":
        # Without complete observable state, replay and isolation cannot be proven, which is
        # the whole basis of a publishable verdict.
        return "L1"
    if not all(satisfies(identifier) for identifier in L2_PROBES):
        return "L1"
    return "L2"


def state_observability_for(
    config: McpOracleConfig,
    evidence: ConformanceEvidence,
) -> str:
    """Derive completeness from the control plane or from the probes that prove a projection."""
    if config.control.state_strategy == "control_tool":
        return "complete"
    passed = {probe.id for probe in evidence.probes if probe.status == "pass"}
    return "complete" if {"P6", "P10", "P11"} <= passed else "diagnostic"


def read_only_boundary_for(
    config: McpOracleConfig,
    evidence: ConformanceEvidence,
) -> str | None:
    """Name a mode-C boundary only after P6 verified that exact mechanism."""
    if config.mode != "C":
        return None
    p6_passed = any(probe.id == "P6" and probe.status == "pass" for probe in evidence.probes)
    boundary = evidence.suite.get("read_only_boundary")
    if p6_passed and boundary in {
        "upstream_authorization",
        "immutable_snapshot_sandbox",
    }:
        return str(boundary)
    return None


def build_attestation(
    config: McpOracleConfig,
    report: DiscoveryReport,
    artifacts: GatewayArtifacts,
    identity: GatewayIdentity,
    evidence: ConformanceEvidence,
) -> dict[str, Any]:
    """Build the exact document `GET /v1/conformance` returns."""
    validated = artifacts.validated()
    observability = state_observability_for(config, evidence)
    document: dict[str, Any] = {
        "schema_version": ATTESTATION_KIND,
        "provider_kind": PROVIDER_KIND_MCP,
        "profile_version": MCP_PROFILE_VERSION,
        "level": _attained_level(evidence.probes, state_observability=observability),
        # The same digest `GET /v1/metadata` serves, from the same function, so the two
        # routes cannot describe different builds.
        "effective_content_digest": identity.content_digest,
        "gateway_artifact_digest": validated.gateway_artifact_digest,
        "shim_artifact_digest": validated.shim_artifact_digest,
        "tool_catalog_digest": report.tool_catalog_digest,
        "server_content_digest": report.document["identity"].get("server_content_digest"),
        "snapshot_digest": validated.snapshot_digest,
        "probe_report_digest": evidence.report_digest,
        "gateway_conformance_report_digest": evidence.suite_digest(
            validated.gateway_artifact_digest,
            effective_content_digest=identity.content_digest,
            tool_catalog_digest=report.tool_catalog_digest,
        ),
        "gateway_evidence_kind": ("signed_release" if evidence.trusted_producer else "locally_verified"),
        "gateway_evidence_issuer": GATEWAY_EVIDENCE_ISSUER,
        "state_observability": observability,
        "read_only_boundary": read_only_boundary_for(config, evidence),
        "checks": [probe.as_check() for probe in evidence.probes],
    }
    # Parsing what we just built is cheap and catches a schema drift here before it reaches a
    # pack that pinned the digest of a document BFCL can no longer read.
    ConformanceAttestation.from_mapping(document, source="gateway conformance")
    return document


def attestation_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize canonically, because the pinned digest is over these exact bytes."""
    return str(canonical_json(dict(document))).encode("utf-8")
