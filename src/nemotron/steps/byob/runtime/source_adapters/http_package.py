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

"""Reviewed HTTP source packages for transport-neutral assisted authoring."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nemotron.steps.byob.runtime.authoring_workflow.credentials import CredentialResolver
from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ConformanceAttestation,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointOracleClient,
    load_endpoint_config,
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    PackTrustError,
    assert_pack_allowed,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterProbeObservation,
    CertificationProbe,
    CertificationRefusalCode,
    ProbeExecutionRecord,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapability,
    AdapterDescriptor,
    CleanupKind,
    CleanupSemantics,
    FixtureAccessKind,
    FixtureAccessPolicy,
    ProbeSafetyKind,
    ProbeSafetyPolicy,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    IdentityArtifact,
    SourceIdentity,
    ToolEvidence,
)
from nemotron.steps.byob.runtime.source_adapters.reviewed_catalog import (
    ReviewedCatalogError,
    load_reviewed_tool_catalog,
)


class HttpPackageError(ValueError):
    """Stable fail-closed error raised before HTTP evidence can be issued."""

    def __init__(self, code: str, detail: str) -> None:
        try:
            self.code = CertificationRefusalCode(code).value
        except ValueError as exc:
            raise ValueError(f"unknown HTTP package refusal code {code!r}") from exc
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class HttpInspectionClient(Protocol):
    def metadata(self) -> dict[str, str]: ...

    def list_tools(self) -> list[str]: ...

    def conformance(self) -> Any: ...

    def close(self, *, suppress_errors: bool = False) -> None: ...


HttpClientFactory = Callable[
    [EndpointConfig, Mapping[str, str], float],
    HttpInspectionClient,
]


@dataclass(frozen=True)
class HttpPackageInspection:
    package_root: Path
    endpoint_config: EndpointConfig
    descriptor: AdapterDescriptor
    identity: SourceIdentity
    tools: tuple[ToolEvidence, ...]
    execution_records: tuple[ProbeExecutionRecord, ...]

    @property
    def source_identity_digest(self) -> str:
        return sha256_json(self.identity.model_dump(mode="json"))


def _default_client_factory(
    config: EndpointConfig,
    headers: Mapping[str, str],
    timeout_s: float,
) -> HttpInspectionClient:
    return EndpointOracleClient(config, headers=headers, timeout_s=timeout_s)


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _redact_header_values(message: str, headers: Mapping[str, str]) -> str:
    redacted = message
    values = set(headers.values())
    authorization = headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        values.add(authorization.removeprefix("Bearer "))
    for value in sorted(values, key=len, reverse=True):
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _regular_reviewed_file(
    package_root: Path,
    relative_name: str,
    *,
    missing_code: str,
) -> Path:
    candidate = package_root / relative_name
    if candidate.is_symlink():
        raise HttpPackageError(
            "source_path_escape",
            f"{relative_name} must be a regular in-package file, not a symlink",
        )
    try:
        resolved = assert_pack_allowed(candidate, (package_root,))
    except PackTrustError as exc:
        raise HttpPackageError("source_path_escape", str(exc)) from exc
    if not resolved.is_file():
        raise HttpPackageError(missing_code, f"missing reviewed {relative_name}")
    return resolved


def _descriptor(timeout_s: float) -> AdapterDescriptor:
    return AdapterDescriptor(
        contract_version=ADAPTER_CONTRACT_VERSION,
        kind="http_package",
        implementation_name="bfcl.http_package",
        implementation_version="1.0.0",
        capabilities=tuple(sorted(AdapterCapability, key=lambda item: item.value)),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.PUSHED,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.RESET_ISOLATED,
            max_calls=24,
            timeout_s=timeout_s,
        ),
        cleanup=CleanupSemantics(kind=CleanupKind.SESSION, timeout_s=timeout_s),
    )


def _config_identity_document(
    config: EndpointConfig,
    *,
    ca_bundle_digest: str | None,
) -> dict[str, Any]:
    return {
        "protocol_version": config.expected.protocol_version,
        "base_url": config.base_url,
        "expected": config.expected.as_dict(),
        "attestation": (
            {
                "kind": config.attestation.kind,
                "expected_digest": config.attestation.expected_digest,
            }
            if config.attestation is not None
            else None
        ),
        "auth": {
            "bearer_token_env": config.bearer_token_env,
            "headers": list(config.header_env),
            "credential_references": [
                reference.model_dump(mode="json")
                for reference in (
                    *((config.bearer_token_ref,) if config.bearer_token_ref else ()),
                    *(reference for _, reference in config.header_refs),
                )
            ],
            "principal_digest": config.expected.principal_digest,
            "permission_digest": config.expected.permission_digest,
            "authorization_context_digest": (
                config.expected.authorization_context_digest
            ),
        },
        "tls": {"ca_bundle_digest": ca_bundle_digest},
        "max_request_bytes": config.max_request_bytes,
        "max_response_bytes": config.max_response_bytes,
    }


def inspect_http_package(
    package_path: Path,
    *,
    allowed_roots: tuple[Path, ...],
    environ: Mapping[str, str] | None = None,
    credential_resolver: CredentialResolver | None = None,
    timeout_s: float = 15.0,
    client_factory: HttpClientFactory | None = None,
) -> HttpPackageInspection:
    """Verify an HTTP package's reviewed schema against its live A0 identity."""
    if package_path.is_symlink():
        raise HttpPackageError(
            "source_path_escape",
            "HTTP package path must not be a symlink",
        )
    try:
        package_root = assert_pack_allowed(package_path, allowed_roots)
    except PackTrustError as exc:
        raise HttpPackageError("source_path_escape", str(exc)) from exc
    if package_root.is_symlink() or not package_root.is_dir():
        raise HttpPackageError(
            "source_package_invalid",
            "HTTP package path must be a regular directory",
        )
    endpoint_path = _regular_reviewed_file(
        package_root,
        "endpoint_config.yaml",
        missing_code="source_package_invalid",
    )
    tools_path = _regular_reviewed_file(
        package_root,
        "tools.json",
        missing_code="reviewed_schema_missing",
    )
    try:
        config = load_endpoint_config(endpoint_path, allowed_roots=(package_root,))
        headers = resolve_endpoint_headers(
            config,
            environ,
            credential_resolver=credential_resolver,
        )
    except (OSError, ValueError) as exc:
        detail = str(exc)
        code = (
            "unsupported_auth"
            if "auth" in detail
            or "credential" in detail
            or "environment variable" in detail
            else "source_package_invalid"
        )
        raise HttpPackageError(
            code,
            f"invalid endpoint declaration: {type(exc).__name__}: {exc}",
        ) from exc
    if config.attestation is None:
        raise HttpPackageError(
            "attestation_mismatch",
            "HTTP authoring packages require a pinned conformance attestation",
        )
    if (config.bearer_token_ref is not None or config.header_refs) and (
        config.expected.principal_digest is None
        or config.expected.permission_digest is None
        or config.expected.authorization_context_digest is None
    ):
        raise HttpPackageError(
            "unsupported_auth",
            "authenticated HTTP authoring packages require expected principal_digest, "
            "permission_digest, and authorization_context_digest",
        )
    try:
        reviewed_catalog = load_reviewed_tool_catalog(tools_path)
    except ReviewedCatalogError as exc:
        raise HttpPackageError(exc.code, exc.detail) from exc
    tools = reviewed_catalog.tools
    catalog_digest = reviewed_catalog.digest

    factory = client_factory or _default_client_factory
    client = factory(config, headers, timeout_s)
    identity_started = time.monotonic()
    try:
        metadata = client.metadata()
        live_names = client.list_tools()
        attestation_document = client.conformance()
    except TimeoutError as exc:
        raise HttpPackageError("probe_timeout", "HTTP A0 inspection timed out") from exc
    except Exception as exc:
        message = _redact_header_values(str(exc), headers)
        if "identity" in message or "metadata does not match" in message:
            code = "identity_drift"
        elif "HTTP 30" in message:
            code = "cross_origin_redirect"
        elif "exceeds max_response_bytes" in message:
            code = "response_too_large"
        else:
            code = "probe_failed"
        raise HttpPackageError(
            code,
            f"HTTP A0 inspection failed: {type(exc).__name__}: {message}",
        ) from None
    finally:
        try:
            client.close(suppress_errors=True)
        except Exception:
            pass
    elapsed = time.monotonic() - identity_started

    reviewed_names = [tool.published_name for tool in tools]
    if (
        len(live_names) != len(set(live_names))
        or sorted(live_names) != reviewed_names
    ):
        raise HttpPackageError(
            "catalog_mismatch",
            "live /v1/tools names do not exactly match reviewed tools.json",
        )
    if attestation_digest(attestation_document) != config.attestation.expected_digest:
        raise HttpPackageError(
            "attestation_mismatch",
            "live conformance document does not match its pinned digest",
        )
    try:
        parsed_attestation = ConformanceAttestation.from_mapping(
            attestation_document,
            source="HTTP package conformance",
        )
    except ValueError as exc:
        raise HttpPackageError(
            "attestation_mismatch",
            f"invalid HTTP conformance document: {exc}",
        ) from exc
    if (
        parsed_attestation.effective_content_digest
        != config.expected.content_digest
        or parsed_attestation.effective_content_digest != metadata["content_digest"]
        or parsed_attestation.tool_catalog_digest != catalog_digest
        or metadata.get("principal_digest") != config.expected.principal_digest
        or metadata.get("permission_digest") != config.expected.permission_digest
        or metadata.get("authorization_context_digest")
        != config.expected.authorization_context_digest
    ):
        raise HttpPackageError(
            "attestation_mismatch",
            "attestation identity or reviewed catalog digest does not match",
        )

    ca_digest = (
        _sha256_file(config.ca_bundle_path)
        if config.ca_bundle_path is not None
        else None
    )
    source_config_digest = sha256_json(
        _config_identity_document(config, ca_bundle_digest=ca_digest)
    )
    artifacts = [
        IdentityArtifact(
            role="attestation",
            digest=config.attestation.expected_digest,
        ),
        IdentityArtifact(role="endpoint_config", digest=source_config_digest),
        IdentityArtifact(role="reviewed_tool_catalog", digest=catalog_digest),
    ]
    if ca_digest is not None:
        artifacts.append(IdentityArtifact(role="ca_bundle", digest=ca_digest))
    if config.expected.authorization_context_digest is not None:
        artifacts.append(
            IdentityArtifact(
                role="authorization_context",
                digest=config.expected.authorization_context_digest,
            )
        )
    identity = SourceIdentity(
        subject=f"http:{config.expected.oracle_id}@{config.expected.oracle_version}",
        effective_content_digest=config.expected.content_digest,
        source_config_digest=source_config_digest,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.role)),
    )
    common_evidence = {
        "endpoint_identity": metadata,
        "attestation_digest": config.attestation.expected_digest,
        "reviewed_tool_catalog_digest": catalog_digest,
    }
    records = (
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=CertificationProbe.IDENTITY_INTEGRITY,
                status="pass",
                evidence=common_evidence,
            ),
            observed_calls=2,
            elapsed_s=float(elapsed),
            cleanup_status="passed",
        ),
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=CertificationProbe.CATALOG_INTEGRITY,
                status="pass",
                evidence={
                    "live_names": sorted(live_names),
                    "reviewed_tool_catalog_digest": catalog_digest,
                },
            ),
            observed_calls=2,
            elapsed_s=float(elapsed),
            cleanup_status="passed",
        ),
    )
    return HttpPackageInspection(
        package_root=package_root,
        endpoint_config=config,
        descriptor=_descriptor(timeout_s),
        identity=identity,
        tools=tools,
        execution_records=records,
    )
