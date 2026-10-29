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

"""Deterministic BFCL Oracle HTTP identity for an MCP-backed gateway."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.config import McpOracleConfig
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.gateway.errors import GatewayError

BFCL_ORACLE_PROTOCOL_VERSION = "bfcl-oracle-http-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _validated_digest(value: str | None, label: str, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise GatewayError(
                "mcp_gateway_identity_invalid",
                f"{label} is required to publish gateway metadata",
            )
        return None
    normalized = value.strip().lower()
    if _DIGEST.fullmatch(normalized) is None:
        raise GatewayError(
            "mcp_gateway_identity_invalid",
            f"{label} must be sha256:<64 lowercase hexadecimal characters>",
        )
    return normalized


@dataclass(frozen=True)
class GatewayArtifacts:
    """Exact gateway-owned artifacts included in the effective oracle identity."""

    gateway_artifact_digest: str
    shim_artifact_digest: str | None = None
    snapshot_digest: str | None = None

    def validated(self) -> GatewayArtifacts:
        return GatewayArtifacts(
            gateway_artifact_digest=_validated_digest(
                self.gateway_artifact_digest,
                "gateway_artifact_digest",
                required=True,
            )
            or "",
            shim_artifact_digest=_validated_digest(
                self.shim_artifact_digest,
                "shim_artifact_digest",
                required=False,
            ),
            snapshot_digest=_validated_digest(
                self.snapshot_digest,
                "snapshot_digest",
                required=False,
            ),
        )


@dataclass(frozen=True)
class GatewayIdentity:
    protocol_version: str
    oracle_id: str
    oracle_version: str
    content_digest: str
    principal_digest: str | None = None
    permission_digest: str | None = None
    authorization_context_digest: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {
            "protocol_version": self.protocol_version,
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "content_digest": self.content_digest,
        }
        for name, value in (
            ("principal_digest", self.principal_digest),
            ("permission_digest", self.permission_digest),
            ("authorization_context_digest", self.authorization_context_digest),
        ):
            if value is not None:
                result[name] = value
        return result


def effective_content_document(
    config: McpOracleConfig,
    report: DiscoveryReport,
    artifacts: GatewayArtifacts,
) -> dict[str, Any]:
    """Return the exact §9 document whose hash identifies the executable oracle."""
    artifacts = artifacts.validated()
    if config.mode == "A" and (artifacts.shim_artifact_digest is not None or artifacts.snapshot_digest is not None):
        raise GatewayError(
            "mcp_gateway_identity_invalid",
            "mode A forbids shim_artifact_digest and snapshot_digest",
        )
    server_digest = report.document["identity"].get("server_content_digest")
    if config.mode in {"A", "B"} and server_digest is None:
        raise GatewayError(
            "mcp_gateway_identity_invalid",
            "a live executable gateway requires a verified server_content_digest",
        )
    if config.mode == "C" and artifacts.snapshot_digest is None:
        raise GatewayError(
            "mcp_gateway_identity_invalid",
            "mode C requires snapshot_digest in the gateway identity",
        )
    return {
        "server_content_digest": server_digest,
        "tool_catalog_digest": report.tool_catalog_digest,
        "gateway_artifact_digest": artifacts.gateway_artifact_digest,
        "shim_artifact_digest": artifacts.shim_artifact_digest,
        "snapshot_digest": artifacts.snapshot_digest,
        "profile_version": config.profile_version,
        "negotiated_mcp_version": report.document["negotiated_mcp_version"],
    }


def build_gateway_identity(
    config: McpOracleConfig,
    report: DiscoveryReport,
    artifacts: GatewayArtifacts,
) -> GatewayIdentity:
    document = effective_content_document(config, report, artifacts)
    digest = "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
    return GatewayIdentity(
        protocol_version=BFCL_ORACLE_PROTOCOL_VERSION,
        oracle_id=config.expected.oracle_id,
        oracle_version=config.expected.oracle_version,
        content_digest=digest,
        principal_digest=config.expected.principal_digest,
        permission_digest=config.expected.permission_digest,
        authorization_context_digest=config.expected.authorization_context_digest,
    )
