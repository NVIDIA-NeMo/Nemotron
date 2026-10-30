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

"""The Gold Gate check that decides whether an attested endpoint may publish.

The check runs for every endpoint oracle. Omitting an attestation is legal for smoke execution
but is an explicit non-Gold result; otherwise deleting the block from a generated pack would
remove the check that was supposed to protect it. Local Python oracles remain outside this
endpoint-specific gate.

Three digests have to agree before anything is published: the one the pack pinned at intake,
the one the live endpoint reports at `GET /v1/metadata`, and the one inside the attestation
itself. Each is produced by a different party at a different time, so agreement is the only
evidence that the build being certified is the build answering calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    DEFAULT_CONFORMANCE_PROFILES,
    ConformanceProfile,
    ConformanceVerdict,
    attestation_digest,
    verify_conformance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointOracleClient,
    resolve_endpoint_headers,
)

CHECK_ID = "A1"
CHECK_NAME = "endpoint_conformance"

AttestationFetcher = Callable[[EndpointConfig], Any]


def _fetch_over_https(config: EndpointConfig, *, timeout_s: float) -> Any:
    # Read-only and callable before any session exists, so it needs none of the process
    # isolation that importing pack code would: there is no pack code on this path.
    client = EndpointOracleClient(
        config,
        headers=resolve_endpoint_headers(config),
        timeout_s=timeout_s,
    )
    return client.conformance()


def run_endpoint_conformance_check(
    endpoint_config: EndpointConfig | None,
    endpoint_metadata: Mapping[str, str] | None,
    *,
    fetch: AttestationFetcher | None = None,
    timeout_s: float = 30.0,
    probe_report: Mapping[str, Any] | None = None,
    gateway_conformance_report: Mapping[str, Any] | None = None,
    trusted_evidence_keys: Mapping[str, Ed25519PublicKey] | None = None,
    expected_evidence_issuer: str | None = None,
    profile: ConformanceProfile | None = None,
    profiles: Mapping[
        tuple[str, str], ConformanceProfile
    ] = DEFAULT_CONFORMANCE_PROFILES,
) -> dict[str, Any] | None:
    """Return the `A1` check entry, or None for a local Python oracle."""
    if endpoint_config is None:
        return None

    def _entry(
        status: str,
        failures: list[dict[str, Any]],
        verdict: ConformanceVerdict | None,
        *,
        document: Any | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": CHECK_ID,
            "name": CHECK_NAME,
            "status": status,
            "failures": failures,
        }
        if verdict is not None:
            entry["conformance"] = verdict.as_dict()
        if document is not None:
            # Preserve exactly what was judged in oracle_validation_report.json. A summary
            # cannot later prove which check list or identity components were verified.
            entry["attestation_document"] = document
            if isinstance(document, Mapping):
                entry["attestation_digest"] = attestation_digest(document)
        return entry

    if endpoint_config.attestation is None:
        # Omitting the pin is legal for smoke execution, but never for publication.
        return _entry("fail", [{"reason": "endpoint_attestation_missing"}], None)

    if endpoint_metadata is None:
        # Without live metadata there is nothing to compare the attestation against, and a
        # comparison against the pack's pin alone would accept a stale document.
        return _entry("fail", [{"reason": "endpoint_metadata_unavailable"}], None)

    try:
        document = (
            fetch(endpoint_config)
            if fetch is not None
            else _fetch_over_https(endpoint_config, timeout_s=timeout_s)
        )
    except Exception as exc:
        return _entry(
            "fail",
            [{"reason": "conformance_unavailable", "detail": f"{type(exc).__name__}: {exc}"}],
            None,
        )

    verdict = verify_conformance(
        document,
        expected_digest=endpoint_config.attestation.expected_digest,
        metadata_content_digest=str(endpoint_metadata.get("content_digest")),
        expected_identity={
            # Closes the chain: what the pack pinned must be what the attestation attests.
            "effective_content_digest": endpoint_config.expected.content_digest,
        },
        probe_report=probe_report,
        gateway_conformance_report=gateway_conformance_report,
        trusted_evidence_keys=trusted_evidence_keys,
        expected_evidence_issuer=expected_evidence_issuer,
        profile=profile,
        profiles=profiles,
    )
    if verdict.publishable:
        return _entry("pass", [], verdict, document=document)

    failures = [{"reason": finding} for finding in verdict.findings]
    failures.extend(
        # A cap is not a defect in the pack; it is missing evidence, and it still blocks
        # publication, so it has to appear as a reason rather than as a silent downgrade.
        {"reason": "level_capped", "detail": cap, "effective_level": verdict.effective_level}
        for cap in verdict.caps
    )
    if not failures:
        failures = [{"reason": "level_below_l2", "effective_level": verdict.effective_level}]
    return _entry("fail", failures, verdict, document=document)
