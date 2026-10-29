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

"""The MCP intake declaration that stands in for a hand-authored tools.json and backend.py.

An operator using MCP as an authoring source still has to state three things no server
can state for them: which pack identity the draft carries, which reviewed MCP profile to
discover, and which gateway will serve the resulting pack as an oracle. Everything else
in the draft is derived. Keeping that list short is the point of the whole flow, and
keeping it *declared* is what makes the derivation reviewable.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from nemotron.steps.byob.runtime.authoring_workflow.credentials import (
    CredentialReference,
    build_authorization_context,
)
from nemotron.steps.byob.runtime.mcp.config import (
    LoadedMcpOracleConfig,
    load_mcp_oracle_config,
    load_unique_yaml_mapping,
    path_is_inside,
)
from nemotron.steps.byob.runtime.mcp.errors import McpConfigError

INTAKE_VERSION = "bfcl-mcp-intake-v1"

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
# RFC 7230 token, matching the endpoint loader that will read the emitted config.
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# A pack id becomes part of task ids, cache keys, and file names, so it is restricted to
# what all three can carry without escaping.
_PACK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class _StrictModel(BaseModel):
    # Mirrors the oracle profile: an unknown key is refused rather than ignored, so a
    # misspelled intake field cannot silently fall back to a default.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


def _digest(value: str, label: str) -> str:
    if not _SHA256.match(value):
        raise ValueError(f"{label} must look like sha256:<64 lowercase hex>")
    return value


class PackConfig(_StrictModel):
    """The pack identity the generated draft will carry."""

    pack_id: StrictStr
    version: StrictStr

    @model_validator(mode="after")
    def _check(self) -> PackConfig:
        if not _PACK_ID.match(self.pack_id):
            raise ValueError(
                "pack.pack_id must be lowercase alphanumeric with dashes or underscores"
            )
        if not self.version.strip():
            raise ValueError("pack.version must be a non-empty string")
        return self


class GatewayAuthConfig(_StrictModel):
    """Environment variable *names*; no intake file ever holds a credential."""

    bearer_token_env: StrictStr | None = None
    bearer_token_ref: CredentialReference | None = None
    headers: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    header_refs: dict[StrictStr, CredentialReference] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> GatewayAuthConfig:
        if self.bearer_token_env is not None and self.bearer_token_ref is not None:
            raise ValueError(
                "gateway.auth cannot set bearer_token_env and bearer_token_ref"
            )
        if self.bearer_token_env is not None and not _ENV_NAME.match(
            self.bearer_token_env
        ):
            raise ValueError(
                "gateway.auth.bearer_token_env must be an uppercase environment "
                "variable name, not a token value"
            )
        for header, env_name in self.headers.items():
            if not _HEADER_NAME.match(header):
                raise ValueError(f"gateway.auth.headers key {header!r} is not an HTTP token")
            # Refused here rather than by the endpoint loader reading the emitted file,
            # so the operator sees the problem in the document they are reviewing.
            if header.lower() in {"authorization", "host", "content-length"}:
                raise ValueError(
                    f"gateway.auth.headers key {header!r} is reserved; declare a bearer "
                    "token through bearer_token_env"
                )
            if not _ENV_NAME.match(env_name):
                raise ValueError(
                    f"gateway.auth.headers[{header!r}] must name an environment variable"
                )
        overlap = set(self.headers) & set(self.header_refs)
        if overlap:
            raise ValueError(f"gateway auth header is declared twice: {sorted(overlap)}")
        for header in self.header_refs:
            if not _HEADER_NAME.match(header):
                raise ValueError(
                    f"gateway.auth.header_refs key {header!r} is not an HTTP token"
                )
            if header.lower() in {"authorization", "host", "content-length"}:
                raise ValueError(
                    f"gateway.auth.header_refs key {header!r} is reserved"
                )
        return self

    def credential_references(self) -> tuple[CredentialReference, ...]:
        references = []
        if self.bearer_token_env is not None:
            references.append(CredentialReference.environment(self.bearer_token_env))
        elif self.bearer_token_ref is not None:
            references.append(self.bearer_token_ref)
        references.extend(
            CredentialReference.environment(name) for name in self.headers.values()
        )
        references.extend(self.header_refs.values())
        return tuple(references)


class GatewayConfig(_StrictModel):
    """Where the generated pack will reach its oracle, and which build serves it."""

    base_url: StrictStr
    auth: GatewayAuthConfig = Field(default_factory=GatewayAuthConfig)
    ca_bundle_path: Path | None = None
    gateway_artifact_digest: StrictStr
    principal_digest: StrictStr | None = None
    permission_digest: StrictStr | None = None
    authorization_context_digest: StrictStr | None = None
    # Applicable to modes B and C respectively. Which ones are required, and which are
    # forbidden, is decided by the gateway identity function so both sides cannot disagree.
    shim_artifact_digest: StrictStr | None = None
    snapshot_digest: StrictStr | None = None
    max_request_bytes: StrictInt = 10 * 1024 * 1024
    max_response_bytes: StrictInt = 10 * 1024 * 1024

    @model_validator(mode="after")
    def _check(self) -> GatewayConfig:
        # Validated here rather than only in the emitted file: the operator is reviewing
        # this document, so this is where a wrong URL should be named.
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "gateway.base_url must be an HTTPS origin without credentials, query, "
                "or fragment"
            )
        if parsed.hostname in _LOOPBACK:
            # A pack that pins a loopback oracle cannot be replayed by anyone else, and
            # a Gold verdict on it would mean nothing outside this host.
            raise ValueError(
                "gateway.base_url must not be loopback: a published pack has to name an "
                "oracle another host can reach"
            )
        _digest(self.gateway_artifact_digest, "gateway.gateway_artifact_digest")
        for label, value in (
            ("principal_digest", self.principal_digest),
            ("permission_digest", self.permission_digest),
            ("authorization_context_digest", self.authorization_context_digest),
            ("shim_artifact_digest", self.shim_artifact_digest),
            ("snapshot_digest", self.snapshot_digest),
        ):
            if value is not None:
                _digest(value, f"gateway.{label}")
        for name, limit in (
            ("max_request_bytes", self.max_request_bytes),
            ("max_response_bytes", self.max_response_bytes),
        ):
            if limit <= 0:
                raise ValueError(f"gateway.{name} must be a positive integer")
        references = self.auth.credential_references()
        if references:
            if (
                self.principal_digest is None
                or self.permission_digest is None
                or self.authorization_context_digest is None
            ):
                raise ValueError(
                    "authenticated gateway requires principal_digest, "
                    "permission_digest, and authorization_context_digest"
                )
            context = build_authorization_context(
                references,
                principal_digest=self.principal_digest,
                permission_digest=self.permission_digest,
            )
            if (
                context.authorization_context_digest
                != self.authorization_context_digest
            ):
                raise ValueError(
                    "gateway.authorization_context_digest does not match credential "
                    "references, principal, and permissions"
                )
        return self


class McpIntakeConfig(_StrictModel):
    """The whole reviewed intake surface for the MCP authoring flow."""

    intake_version: Literal["bfcl-mcp-intake-v1"]
    kind: Literal["mcp"]
    mcp_oracle_config: Path
    pack: PackConfig
    gateway: GatewayConfig


@dataclass(frozen=True)
class LoadedMcpIntake:
    """The intake document, the profile it names, and the bytes that were reviewed."""

    path: Path
    value: McpIntakeConfig
    # Digested instead of the resolved model, so the same reviewed document hashes
    # identically no matter where the tree is checked out.
    raw_document: dict[str, Any]
    oracle: LoadedMcpOracleConfig


def load_mcp_intake(
    path: Path,
    *,
    allow_insecure_localhost: bool = False,
) -> LoadedMcpIntake:
    """Load the intake declaration together with the MCP profile it points at."""
    source = path.resolve()
    raw = load_unique_yaml_mapping(source, "MCP intake config")
    try:
        config = McpIntakeConfig.model_validate(raw)
    except ValueError as exc:
        raise McpConfigError(f"invalid MCP intake config {source}: {exc}") from exc

    root = source.parent
    oracle_path = config.mcp_oracle_config
    resolved_oracle = (
        oracle_path if oracle_path.is_absolute() else root / oracle_path
    ).resolve()
    if not path_is_inside(resolved_oracle, root):
        raise McpConfigError(
            "mcp_oracle_config must stay inside the intake directory so review covers "
            f"both documents: {resolved_oracle}"
        )
    if not resolved_oracle.is_file():
        raise McpConfigError(f"mcp_oracle_config is not a file: {resolved_oracle}")

    ca_bundle = config.gateway.ca_bundle_path
    if ca_bundle is not None:
        resolved_ca = (ca_bundle if ca_bundle.is_absolute() else root / ca_bundle).resolve()
        if not path_is_inside(resolved_ca, root):
            raise McpConfigError(
                "gateway.ca_bundle_path must stay inside the intake directory"
            )
        if not resolved_ca.is_file():
            raise McpConfigError(f"gateway.ca_bundle_path is not a file: {resolved_ca}")

    oracle = load_mcp_oracle_config(
        resolved_oracle,
        allow_insecure_localhost=allow_insecure_localhost,
    )
    return LoadedMcpIntake(
        path=source,
        value=config,
        raw_document=raw,
        oracle=oracle,
    )
