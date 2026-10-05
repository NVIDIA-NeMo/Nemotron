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

"""Complete, paginated MCP catalog discovery and deterministic evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.client import (
    ConnectedMcpClient,
    open_mcp_connection,
)
from nemotron.steps.byob.runtime.mcp.config import (
    DISCOVERY_REPORT_VERSION,
    LoadedMcpOracleConfig,
    McpOracleConfig,
    StdioTransportConfig,
    StreamableHttpTransportConfig,
    TrustedExecutablePolicies,
)
from nemotron.steps.byob.runtime.mcp.errors import (
    McpCatalogError,
    McpConfigError,
    McpIdentityMismatchError,
    McpProtocolError,
)
from nemotron.steps.byob.runtime.mcp.normalization import (
    NormalizedCatalog,
    normalize_catalog,
)

ConnectionFactory = Callable[
    [McpOracleConfig],
    AbstractAsyncContextManager[ConnectedMcpClient],
]
MCP_DISCOVERY_ADAPTER_VERSION = "1.0.0"


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _resolve_loaded_paths(
    config: McpOracleConfig,
    *,
    source_directory: Path,
) -> McpOracleConfig:
    transport = config.transport
    if isinstance(transport, StdioTransportConfig):
        cwd = transport.cwd
        resolved_cwd = (cwd if cwd.is_absolute() else source_directory / cwd).resolve()
        return config.model_copy(update={"transport": transport.model_copy(update={"cwd": resolved_cwd})})
    if isinstance(transport, StreamableHttpTransportConfig):
        ca_bundle = transport.tls.ca_bundle_path
        if ca_bundle is None:
            return config
        resolved_ca = (ca_bundle if ca_bundle.is_absolute() else source_directory / ca_bundle).resolve()
        return config.model_copy(
            update={
                "transport": transport.model_copy(
                    update={"tls": transport.tls.model_copy(update={"ca_bundle_path": resolved_ca})}
                )
            }
        )
    raise McpConfigError(f"unsupported MCP transport model {type(transport).__name__}")


def _verify_loaded_config(loaded: LoadedMcpOracleConfig) -> None:
    """Prove the report source and the runtime model describe the same config."""
    try:
        parsed = McpOracleConfig.model_validate(loaded.raw_document)
    except ValueError as exc:
        raise McpConfigError("loaded MCP raw_document no longer satisfies the strict profile") from exc
    expected = _resolve_loaded_paths(
        parsed,
        source_directory=loaded.path.parent,
    )
    if expected != loaded.value:
        raise McpConfigError(
            "loaded MCP raw_document does not match the effective runtime config; "
            "refusing to hash one document while executing another"
        )


def catalog_identity_document(
    config: McpOracleConfig,
    *,
    negotiated_mcp_version: str,
    server_name: str,
    server_version: str,
    catalog: NormalizedCatalog,
) -> dict[str, Any]:
    """Build the exact §9 catalog identity document."""
    control = config.control
    return {
        "digest_schema_version": "bfcl-mcp-tool-catalog-digest-v2",
        "profile_version": config.profile_version,
        "mode": config.mode,
        "negotiated_mcp_version": negotiated_mcp_version,
        "server_name": server_name,
        "server_version": server_version,
        "tools": catalog.bfcl_tools,
        # The gateway validates results against outputSchema and may derive runtime
        # mutation behavior from annotations under an explicitly reviewed policy.
        # They are therefore contract identity even though they are not model-facing.
        "runtime_contracts": [
            {
                "published_name": tool.published_name,
                "output_schema": tool.output_schema,
                "annotations": tool.annotations,
                "mutation_source": tool.mutation_source,
            }
            for tool in catalog.tools
        ],
        "control": {
            "reset_strategy": control.reset_strategy,
            "state_strategy": control.state_strategy,
            "describe_oracle": control.describe_oracle,
            "reset_episode": control.reset_episode,
            "get_episode_state": control.get_episode_state,
            "end_episode": control.end_episode,
            "episode_binding": control.episode_binding,
            "episode_argument": control.episode_argument,
            "state_projection": [call.model_dump(mode="json") for call in control.state_projection],
        },
    }


def _extract_control_result(result: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    """Read a control result from structured content only.

    The profile deliberately refuses to parse free-text content: guessing a shape from
    prose would let an untrusted server steer BFCL's identity checks. Servers that only
    emit text must be fronted by a reviewed Mode B shim that declares the structure.
    """
    if result.get("isError", result.get("is_error", False)) is True:
        raise McpProtocolError(f"control tool {tool_name!r} returned isError=true")
    structured = result.get("structuredContent", result.get("structured_content"))
    if isinstance(structured, Mapping):
        return dict(structured)
    raise McpProtocolError(
        f"control tool {tool_name!r} must return structuredContent as a JSON object; "
        "front text-only servers with a reviewed Mode B shim instead"
    )


def _verify_described_identity(
    config: McpOracleConfig,
    described: Mapping[str, Any],
) -> str:
    # Required, not exhaustive: a server may report extra descriptive fields such as a
    # build id. Only the three pinned fields decide identity, and the whole declaration
    # is recorded in the report, so an added field is reviewable and drifts the report
    # digest rather than being silently dropped or hard-failing a conformant server.
    required = ["oracle_id", "oracle_version", "content_digest"]
    if (
        isinstance(config.transport, StreamableHttpTransportConfig)
        and config.transport.auth.credential_references()
        and config.expected.authorization_context_digest is not None
    ):
        required.extend(
            [
                "principal_digest",
                "permission_digest",
                "authorization_context_digest",
            ]
        )
    absent = [name for name in required if name not in described]
    if absent:
        raise McpProtocolError(f"describe_oracle omitted required identity field(s) {absent}")
    content_digest = described.get("content_digest")
    if not isinstance(content_digest, str):
        raise McpProtocolError("describe_oracle content_digest must be a string")
    # Hex digests are case insensitive, so only the casing is normalized before comparing.
    normalized_digest = content_digest.strip().lower()
    observed = {
        "oracle_id": described.get("oracle_id"),
        "oracle_version": described.get("oracle_version"),
        "content_digest": normalized_digest,
    }
    expected = {
        "oracle_id": config.expected.oracle_id,
        "oracle_version": config.expected.oracle_version,
        "content_digest": config.expected.server_content_digest,
    }
    for name in (
        "principal_digest",
        "permission_digest",
        "authorization_context_digest",
    ):
        expected_value = getattr(config.expected, name)
        if expected_value is not None:
            observed_value = described.get(name)
            observed[name] = observed_value.strip().lower() if isinstance(observed_value, str) else observed_value
            expected[name] = expected_value
    if observed != expected:
        raise McpIdentityMismatchError(
            f"describe_oracle identity mismatch: expected {canonical_json(expected)}, "
            f"observed {canonical_json(observed)}"
        )
    return normalized_digest


@dataclass(frozen=True)
class DiscoveryReport:
    document: dict[str, Any]

    @property
    def tool_catalog_digest(self) -> str:
        return str(self.document["identity"]["tool_catalog_digest"])

    def verify_digest(self) -> None:
        claimed = self.document.get("report_digest")
        unsigned = {key: value for key, value in self.document.items() if key != "report_digest"}
        observed = _sha256_json(unsigned)
        if claimed != observed:
            raise McpProtocolError(
                "discovery report was modified after report_digest was computed: "
                f"claimed {claimed!r}, observed {observed!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.verify_digest()
        copied = json.loads(canonical_json(self.document))
        if not isinstance(copied, dict):
            raise McpProtocolError("discovery report did not preserve its object shape")
        return copied


async def _list_complete_catalog(
    client: ConnectedMcpClient,
    config: McpOracleConfig,
) -> tuple[list[Any], int]:
    tools: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    pages = 0
    while True:
        if pages >= config.limits.max_catalog_pages:
            raise McpCatalogError(f"tools/list exceeded max_catalog_pages={config.limits.max_catalog_pages}")
        try:
            page = await asyncio.wait_for(
                client.list_tools(cursor),
                timeout=float(config.limits.tool_timeout_s),
            )
        except TimeoutError as exc:
            raise McpCatalogError(f"tools/list page {pages + 1} exceeded tool_timeout_s") from exc
        pages += 1
        tools.extend(page.tools)
        if len(tools) > config.limits.max_tools:
            raise McpCatalogError(f"tools/list exceeded max_tools={config.limits.max_tools}")
        next_cursor = page.next_cursor
        if next_cursor is None:
            return tools, pages
        if not isinstance(next_cursor, str) or not next_cursor:
            raise McpCatalogError("tools/list returned an invalid nextCursor")
        if next_cursor in seen_cursors:
            raise McpCatalogError("tools/list pagination cursor cycle detected")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


async def discover_mcp_oracle(
    loaded: LoadedMcpOracleConfig,
    *,
    environ: Mapping[str, str] | None = None,
    executable_policies: TrustedExecutablePolicies | None = None,
    connection_factory: ConnectionFactory | None = None,
    verify_catalog_digest: bool = True,
) -> DiscoveryReport:
    """Negotiate, retrieve all pages, normalize, pin identity, and attest L0."""
    _verify_loaded_config(loaded)
    config = loaded.value
    if connection_factory is None:

        def connection_factory(
            value: McpOracleConfig,
        ) -> AbstractAsyncContextManager[ConnectedMcpClient]:
            return open_mcp_connection(
                value,
                environ=environ,
                executable_policies=executable_policies,
            )

    async with connection_factory(config) as client:
        protocol_version = client.protocol_version
        if protocol_version not in config.mcp_protocol_versions:
            raise McpIdentityMismatchError(
                f"negotiated MCP version {protocol_version!r} is not in "
                f"mcp_protocol_versions={list(config.mcp_protocol_versions)!r}"
            )
        server = client.server_identity
        observed_identity = {"name": server.name, "version": server.version}
        expected_identity = {
            "name": config.expected.server_name,
            "version": config.expected.server_version,
        }
        if observed_identity != expected_identity:
            raise McpIdentityMismatchError(
                f"MCP server identity mismatch: expected {canonical_json(expected_identity)}, "
                f"observed {canonical_json(observed_identity)}"
            )
        capabilities = client.capabilities
        if capabilities.get("tools") is None:
            raise McpProtocolError("MCP server did not advertise the tools capability")
        raw_tools, page_count = await _list_complete_catalog(client, config)
        catalog = normalize_catalog(raw_tools, config)
        identity_document = catalog_identity_document(
            config,
            negotiated_mcp_version=protocol_version,
            # Identity equality above already proved these match the pinned strings.
            server_name=config.expected.server_name,
            server_version=config.expected.server_version,
            catalog=catalog,
        )
        catalog_digest = _sha256_json(identity_document)
        catalog_matches = catalog_digest == config.expected.tool_catalog_digest
        if not catalog_matches and verify_catalog_digest:
            raise McpIdentityMismatchError(
                "normalized MCP tool catalog digest does not match "
                f"expected.tool_catalog_digest: expected "
                f"{config.expected.tool_catalog_digest}, observed {catalog_digest}"
            )

        server_content_digest: str | None = None
        oracle_declaration: dict[str, Any] | None = None
        if config.control.describe_oracle is not None:
            try:
                raw_result = await asyncio.wait_for(
                    client.call_tool(config.control.describe_oracle, {}),
                    timeout=float(config.limits.tool_timeout_s),
                )
            except TimeoutError as exc:
                raise McpProtocolError("describe_oracle exceeded tool_timeout_s") from exc
            described = _extract_control_result(
                raw_result,
                config.control.describe_oracle,
            )
            server_content_digest = _verify_described_identity(config, described)
            oracle_declaration = described

    base_document = {
        "schema_version": DISCOVERY_REPORT_VERSION,
        "profile_version": config.profile_version,
        "status": "pass" if catalog_matches else "needs_catalog_pin",
        "attained_level": "L0" if catalog_matches else None,
        "mode": config.mode,
        "transport_kind": config.transport.kind,
        "implementation": {
            "adapter": "nemotron-bfcl-mcp-discovery",
            "adapter_version": MCP_DISCOVERY_ADAPTER_VERSION,
            "sdk_requirement": "mcp>=2,<3",
            "sdk_version": client.sdk_version,
        },
        # Digest the reviewed document, not the loaded model: resolved absolute paths
        # would otherwise make the same reviewed config hash differently per host.
        "source_config_digest": _sha256_json(loaded.raw_document),
        "negotiated_mcp_version": protocol_version,
        "server": observed_identity,
        "capabilities": capabilities,
        "oracle_declaration": oracle_declaration,
        "catalog": {
            "page_count": page_count,
            "discovered_tool_count": len(raw_tools),
            "selected_source_to_published": catalog.source_to_published,
            "trust_annotations": config.tools.trust_annotations,
            "tools": catalog.bfcl_tools,
            "evidence": [tool.evidence() for tool in catalog.tools],
            "exclusions": [issue.as_dict() for issue in catalog.exclusions],
            "warnings": [issue.as_dict() for issue in catalog.warnings],
        },
        "identity": {
            "tool_catalog_digest": catalog_digest,
            "server_content_digest": server_content_digest,
            "gateway_artifact_digest": None,
            "shim_artifact_digest": None,
            "snapshot_digest": None,
            "effective_content_digest": None,
        },
        "checks": [
            {
                "id": "P1",
                "requirement": "required",
                "status": "pass",
                "reason": None,
            },
            {
                "id": "P2",
                "requirement": "required",
                "status": "pass" if catalog_matches else "fail",
                "reason": (
                    None
                    if catalog_matches
                    else "copy the observed tool_catalog_digest into the reviewed config and rerun"
                ),
            },
            {
                "id": "P3",
                "requirement": "required",
                "status": "pass",
                "reason": None,
            },
        ],
        "deferred": [
            "gateway_artifact_digest",
            "snapshot_digest for mode C",
            "effective_content_digest",
            "P4-P11 executable and conformance probes",
        ],
        "assumptions": [
            "MCP descriptions are untrusted data and are never executed as instructions",
            "unselected tools are outside the benchmark surface",
            "L0 discovery does not claim reset, isolation, mutation, or replay conformance",
        ],
    }
    base_document["report_digest"] = _sha256_json(base_document)
    return DiscoveryReport(document=base_document)


def write_discovery_report(report: DiscoveryReport, path: Path) -> Path:
    """Atomically write canonical JSON so identical discovery yields identical bytes."""
    report.verify_digest()
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(canonical_json(report.document) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return destination
