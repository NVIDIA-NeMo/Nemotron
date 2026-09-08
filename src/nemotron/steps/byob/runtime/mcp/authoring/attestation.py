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

"""Bind a live gateway attestation to reviewed intake identity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ConformanceAttestation,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointIdentity,
    EndpointOracleClient,
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.mcp.authoring.intake import LoadedMcpIntake
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.errors import McpProtocolError
from nemotron.steps.byob.runtime.mcp.gateway.identity import GatewayIdentity

AttestationFetcher = Callable[[EndpointConfig], Any]


def temporary_endpoint_config(
    intake: LoadedMcpIntake,
    identity: GatewayIdentity,
) -> EndpointConfig:
    """Build the secret-free endpoint declaration used only to fetch conformance."""
    gateway = intake.value.gateway
    return EndpointConfig(
        path=Path("<mcp-intake>"),
        base_url=gateway.base_url.rstrip("/"),
        expected=EndpointIdentity.from_mapping(identity.as_dict(), source="gateway identity"),
        bearer_token_env=gateway.auth.bearer_token_env,
        header_env=tuple(sorted(gateway.auth.headers.items())),
        bearer_token_ref=gateway.auth.bearer_token_ref,
        header_refs=tuple(sorted(gateway.auth.header_refs.items())),
        ca_bundle_path=gateway.ca_bundle_path,
        max_request_bytes=gateway.max_request_bytes,
        max_response_bytes=gateway.max_response_bytes,
    )


def fetch_gateway_attestation(
    config: EndpointConfig,
    *,
    environ: Mapping[str, str] | None,
    timeout_s: float,
) -> Any:
    """Fetch the live document without starting an oracle session."""
    client = EndpointOracleClient(
        config,
        headers=resolve_endpoint_headers(config, environ),
        timeout_s=timeout_s,
    )
    return client.conformance()


def validate_gateway_attestation(
    document: Any,
    *,
    intake: LoadedMcpIntake,
    report: DiscoveryReport,
    identity: GatewayIdentity,
) -> dict[str, Any]:
    """Prove the live document describes the gateway/catalog this intake reviewed."""
    try:
        attestation = ConformanceAttestation.from_mapping(
            document,
            source="gateway conformance at intake",
        )
    except ValueError as exc:
        raise McpProtocolError(f"gateway returned an invalid conformance attestation: {exc}") from exc

    gateway = intake.value.gateway
    expected = {
        "effective_content_digest": identity.content_digest,
        "gateway_artifact_digest": gateway.gateway_artifact_digest,
        "shim_artifact_digest": gateway.shim_artifact_digest,
        "tool_catalog_digest": report.tool_catalog_digest,
        "server_content_digest": report.document["identity"].get("server_content_digest"),
        "snapshot_digest": gateway.snapshot_digest,
    }
    mismatches = [
        f"{name}: expected {value!r}, observed {getattr(attestation, name)!r}"
        for name, value in expected.items()
        if getattr(attestation, name) != value
    ]
    if mismatches:
        raise McpProtocolError(
            "gateway conformance identity does not match the reviewed intake: " + "; ".join(mismatches)
        )
    return dict(attestation.document)
