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

"""Strict, secret-free configuration for BFCL MCP discovery."""

from __future__ import annotations

import math
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError

from nemotron.steps.byob.runtime.authoring_workflow.credentials import (
    CredentialReference,
    build_authorization_context,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.errors import McpConfigError

PROFILE_VERSION = "bfcl-mcp-oracle-v1"
DISCOVERY_REPORT_VERSION = "1.0"
_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_PUBLISHED_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RESERVED_HEADERS = frozenset({"authorization", "connection", "content-length", "host", "transfer-encoding"})
# A dotted path into structuredContent; segments may hold any non-dot, non-space key.
_RESULT_PATH = re.compile(r"^[^\s.]+(?:\.[^\s.]+)*$")

# Ceilings exist so an obviously wrong value is refused rather than silently accepted.
# They are sanity bounds, not policy: a limit far past these either disables itself or
# reflects a unit mistake, and both are worth failing at load time.
MAX_TIMEOUT_S = 3600.0
MAX_RESPONSE_BYTES_CEILING = 256 * 1024 * 1024
MAX_TOOLS_CEILING = 4096
MAX_CATALOG_PAGES_CEILING = 1000
MAX_CONCURRENT_EPISODES_CEILING = 256


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_unique_yaml_document(source: Path, label: str) -> Any:
    """Load YAML while rejecting duplicate keys at every mapping depth."""
    try:
        return yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, yaml.YAMLError) as exc:
        raise McpConfigError(f"cannot load {label} {source}: {exc}") from exc


def load_unique_yaml_mapping(source: Path, label: str) -> dict[str, Any]:
    raw = load_unique_yaml_document(source, label)
    if not isinstance(raw, dict):
        raise McpConfigError(f"{label} must be a YAML mapping: {source}")
    return raw


class _StrictModel(BaseModel):
    # Container conversion is intentional: YAML arrays become immutable tuples.
    # Scalar fields use Strict* annotations, so strings never become numbers/bools.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


def _nonempty(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def _digest(value: str, label: str) -> str:
    value = _nonempty(value, label).lower()
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be sha256:<64 hexadecimal characters>")
    return value


def _bounded_number(value: float | int, label: str, maximum: float) -> float:
    converted = float(value)
    if isinstance(value, bool) or not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    if converted > maximum:
        raise ValueError(
            f"{label} must be at most {maximum:g}; a larger value is usually a unit "
            "mistake such as milliseconds written where seconds are expected"
        )
    return converted


def _bounded_integer(value: int, label: str, maximum: int) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    if value > maximum:
        raise ValueError(
            f"{label} must be at most {maximum}; a larger value disables the limit "
            "while still looking configured"
        )
    return value


class StdioTransportConfig(_StrictModel):
    kind: Literal["stdio"]
    command: tuple[StrictStr, ...]
    cwd: Path
    env_passthrough: tuple[StrictStr, ...] = ()
    executable_policy: StrictStr

    @model_validator(mode="after")
    def _validate_stdio(self) -> StdioTransportConfig:
        if not self.command or any(not item.strip() for item in self.command):
            raise ValueError("transport.command must be a non-empty argument vector")
        if any("\x00" in item for item in self.command):
            raise ValueError("transport.command must not contain NUL characters")
        _nonempty(self.executable_policy, "transport.executable_policy")
        _unique(self.env_passthrough, "transport.env_passthrough")
        invalid = [name for name in self.env_passthrough if _ENV_NAME.fullmatch(name) is None]
        if invalid:
            raise ValueError(f"transport.env_passthrough contains invalid environment names: {invalid}")
        return self


class HttpAuthConfig(_StrictModel):
    bearer_token_env: StrictStr | None = None
    bearer_token_ref: CredentialReference | None = None
    headers: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    header_refs: dict[StrictStr, CredentialReference] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_auth(self) -> HttpAuthConfig:
        if self.bearer_token_env is not None and self.bearer_token_ref is not None:
            raise ValueError(
                "transport.auth cannot set bearer_token_env and bearer_token_ref"
            )
        if self.bearer_token_env is not None and _ENV_NAME.fullmatch(self.bearer_token_env) is None:
            raise ValueError("transport.auth.bearer_token_env must be an environment variable name")
        folded_headers: dict[str, str] = {}
        overlap = set(self.headers) & set(self.header_refs)
        if overlap:
            raise ValueError(
                f"transport.auth header is declared twice: {sorted(overlap)}"
            )
        for header, env_name in self.headers.items():
            if _HEADER_NAME.fullmatch(header) is None:
                raise ValueError(f"transport.auth.headers contains invalid HTTP field name {header!r}")
            folded = header.casefold()
            if folded in _RESERVED_HEADERS:
                raise ValueError(f"transport.auth.headers may not set reserved field {header!r}")
            previous = folded_headers.get(folded)
            if previous is not None:
                raise ValueError(
                    "transport.auth.headers contains case-insensitive duplicate fields "
                    f"{previous!r} and {header!r}"
                )
            folded_headers[folded] = header
            if _ENV_NAME.fullmatch(env_name) is None:
                raise ValueError(
                    f"transport.auth.headers.{header} must name an environment variable, not contain a value"
                )
        for header in self.header_refs:
            if _HEADER_NAME.fullmatch(header) is None:
                raise ValueError(
                    f"transport.auth.header_refs contains invalid HTTP field name {header!r}"
                )
            if header.casefold() in _RESERVED_HEADERS:
                raise ValueError(
                    f"transport.auth.header_refs may not set reserved field {header!r}"
                )
            folded = header.casefold()
            previous = folded_headers.get(folded)
            if previous is not None:
                raise ValueError(
                    "transport.auth contains case-insensitive duplicate fields "
                    f"{previous!r} and {header!r}"
                )
            folded_headers[folded] = header
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


class HttpTlsConfig(_StrictModel):
    ca_bundle_path: Path | None = None


class StreamableHttpTransportConfig(_StrictModel):
    kind: Literal["streamable_http"]
    url: StrictStr
    auth: HttpAuthConfig = Field(default_factory=HttpAuthConfig)
    tls: HttpTlsConfig = Field(default_factory=HttpTlsConfig)

    @model_validator(mode="after")
    def _validate_url_shape(self) -> StreamableHttpTransportConfig:
        parsed = urllib.parse.urlsplit(self.url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "transport.url must be an HTTP(S) URL without credentials, query, or fragment"
            )
        return self


TransportConfig = Annotated[
    StdioTransportConfig | StreamableHttpTransportConfig,
    Field(discriminator="kind"),
]


class ExpectedIdentityConfig(_StrictModel):
    server_name: StrictStr
    server_version: StrictStr
    tool_catalog_digest: StrictStr
    oracle_id: StrictStr
    oracle_version: StrictStr
    server_content_digest: StrictStr | None = None
    principal_digest: StrictStr | None = None
    permission_digest: StrictStr | None = None
    authorization_context_digest: StrictStr | None = None

    # Digests are normalized once here so no downstream comparison needs to re-case them.
    @field_validator(
        "tool_catalog_digest",
        "server_content_digest",
        "principal_digest",
        "permission_digest",
        "authorization_context_digest",
        mode="after",
    )
    @classmethod
    def _normalize_digest(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _digest(value, f"expected.{info.field_name}")

    @model_validator(mode="after")
    def _validate_identity(self) -> ExpectedIdentityConfig:
        for field_name in ("server_name", "server_version", "oracle_id", "oracle_version"):
            _nonempty(getattr(self, field_name), f"expected.{field_name}")
        return self


class CallSpec(_StrictModel):
    tool: StrictStr
    arguments: dict[StrictStr, Any] = Field(default_factory=dict)
    collection: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_call(self) -> CallSpec:
        if _TOOL_NAME.fullmatch(self.tool) is None:
            raise ValueError("call.tool is not a valid MCP tool name")
        if self.collection is not None:
            _nonempty(self.collection, "call.collection")
        canonical_json(self.arguments)
        return self


class ControlConfig(_StrictModel):
    reset_strategy: Literal["control_tool", "process_restart", "namespace", "no_op_verified"]
    state_strategy: Literal["control_tool", "read_only_projection"]
    describe_oracle: StrictStr | None = None
    reset_episode: StrictStr | None = None
    get_episode_state: StrictStr | None = None
    end_episode: StrictStr | None = None
    episode_binding: Literal["transport", "meta", "argument"]
    episode_argument: StrictStr | None = None
    state_projection: tuple[CallSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_control(self) -> ControlConfig:
        if self.reset_strategy == "control_tool" and self.reset_episode is None:
            raise ValueError("control.reset_episode is required for reset_strategy=control_tool")
        if self.state_strategy == "control_tool" and self.get_episode_state is None:
            raise ValueError(
                "control.get_episode_state is required for state_strategy=control_tool"
            )
        if self.state_strategy == "read_only_projection" and not self.state_projection:
            raise ValueError(
                "control.state_projection is required for state_strategy=read_only_projection"
            )
        if self.episode_binding == "argument":
            if self.episode_argument is None:
                raise ValueError(
                    "control.episode_argument is required for episode_binding=argument"
                )
            _nonempty(self.episode_argument, "control.episode_argument")
        elif self.episode_argument is not None:
            raise ValueError(
                "control.episode_argument is only valid for episode_binding=argument"
            )
        for field_name in (
            "describe_oracle",
            "reset_episode",
            "get_episode_state",
            "end_episode",
        ):
            value = getattr(self, field_name)
            if value is not None and _TOOL_NAME.fullmatch(value) is None:
                raise ValueError(f"control.{field_name} is not a valid MCP tool name")
        return self

    @property
    def reserved_tool_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in (
                self.describe_oracle,
                self.reset_episode,
                self.get_episode_state,
                self.end_episode,
            )
            if name is not None
        )


class FixturesConfig(_StrictModel):
    direction: Literal["pushed", "snapshot"]
    snapshot_calls: tuple[CallSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_fixtures(self) -> FixturesConfig:
        if self.direction == "snapshot" and not self.snapshot_calls:
            raise ValueError("fixtures.snapshot_calls is required for direction=snapshot")
        if self.direction == "pushed" and self.snapshot_calls:
            raise ValueError("fixtures.snapshot_calls is only valid for direction=snapshot")
        collections = [call.collection for call in self.snapshot_calls]
        if any(collection is None for collection in collections):
            raise ValueError("every fixtures.snapshot_calls entry needs collection")
        if len(collections) != len(set(collections)):
            raise ValueError("fixtures.snapshot_calls collection names must be unique")
        return self


class ToolsConfig(_StrictModel):
    include: tuple[StrictStr, ...]
    aliases: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    mutates: tuple[StrictStr, ...] = ()
    requires_confirmation: tuple[StrictStr, ...] = ()
    trust_annotations: StrictBool = False

    @model_validator(mode="after")
    def _validate_tools(self) -> ToolsConfig:
        if not self.include:
            raise ValueError("tools.include must select at least one business tool")
        _unique(self.include, "tools.include")
        _unique(self.mutates, "tools.mutates")
        _unique(self.requires_confirmation, "tools.requires_confirmation")
        for name in self.include:
            if _TOOL_NAME.fullmatch(name) is None:
                raise ValueError(f"tools.include contains invalid MCP tool name {name!r}")
        if set(self.aliases) - set(self.include):
            raise ValueError("tools.aliases keys must be selected names from tools.include")
        published_names = []
        for source in self.include:
            published = self.aliases.get(source, source)
            if _PUBLISHED_TOOL_NAME.fullmatch(published) is None:
                raise ValueError(
                    f"tool {source!r} needs an alias matching [A-Za-z0-9_-]{{1,64}}"
                )
            published_names.append(published)
        if len(published_names) != len(set(published_names)):
            raise ValueError("tools.aliases creates duplicate published tool names")
        published_set = set(published_names)
        if set(self.mutates) - published_set:
            raise ValueError("tools.mutates must reference published tool names")
        if set(self.requires_confirmation) - published_set:
            raise ValueError(
                "tools.requires_confirmation must reference published tool names"
            )
        return self

    def published_name(self, source_name: str) -> str:
        return self.aliases.get(source_name, source_name)


class ResultsConfig(_StrictModel):
    error_path: StrictStr = "error"
    status_field: StrictStr = "status"
    pending_status: StrictStr = "awaiting_confirmation"
    confirmation_parameter: StrictStr = "confirm"

    @model_validator(mode="after")
    def _validate_results(self) -> ResultsConfig:
        for name in ("error_path", "status_field", "pending_status", "confirmation_parameter"):
            _nonempty(getattr(self, name), f"results.{name}")
        if _RESULT_PATH.fullmatch(self.error_path) is None:
            raise ValueError(
                "results.error_path must be a dotted path into structuredContent, "
                "such as 'error' or 'result.error'"
            )
        return self


class LimitsConfig(_StrictModel):
    connect_timeout_s: StrictFloat | StrictInt = 5
    handshake_timeout_s: StrictFloat | StrictInt = 5
    tool_timeout_s: StrictFloat | StrictInt = 5
    reset_timeout_s: StrictFloat | StrictInt = 10
    episode_timeout_s: StrictFloat | StrictInt = 60
    max_response_bytes: StrictInt = 10 * 1024 * 1024
    max_tools: StrictInt = 64
    max_catalog_pages: StrictInt = 20
    max_concurrent_episodes: StrictInt = 4
    session_idle_ttl_s: StrictFloat | StrictInt = 300

    @model_validator(mode="after")
    def _validate_limits(self) -> LimitsConfig:
        for field_name in (
            "connect_timeout_s",
            "handshake_timeout_s",
            "tool_timeout_s",
            "reset_timeout_s",
            "episode_timeout_s",
            "session_idle_ttl_s",
        ):
            _bounded_number(
                getattr(self, field_name),
                f"limits.{field_name}",
                MAX_TIMEOUT_S,
            )
        for field_name, ceiling in (
            ("max_response_bytes", MAX_RESPONSE_BYTES_CEILING),
            ("max_tools", MAX_TOOLS_CEILING),
            ("max_catalog_pages", MAX_CATALOG_PAGES_CEILING),
            ("max_concurrent_episodes", MAX_CONCURRENT_EPISODES_CEILING),
        ):
            _bounded_integer(getattr(self, field_name), f"limits.{field_name}", ceiling)
        return self


class McpOracleConfig(_StrictModel):
    profile_version: Literal["bfcl-mcp-oracle-v1"]
    mode: Literal["A", "B", "C"]
    mcp_protocol_versions: tuple[StrictStr, ...]
    transport: TransportConfig
    expected: ExpectedIdentityConfig
    control: ControlConfig
    fixtures: FixturesConfig
    tools: ToolsConfig
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    isolation: Literal["process_per_episode", "namespace_per_episode", "read_only"]
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @model_validator(mode="after")
    def _validate_profile(self) -> McpOracleConfig:
        if not self.mcp_protocol_versions:
            raise ValueError("mcp_protocol_versions must not be empty")
        _unique(self.mcp_protocol_versions, "mcp_protocol_versions")
        for version in self.mcp_protocol_versions:
            _nonempty(version, "mcp_protocol_versions[]")
        if isinstance(self.transport, StreamableHttpTransportConfig):
            references = self.transport.auth.credential_references()
            context_fields = (
                self.expected.principal_digest,
                self.expected.permission_digest,
                self.expected.authorization_context_digest,
            )
            if references and any(value is not None for value in context_fields):
                if (
                    self.expected.principal_digest is None
                    or self.expected.permission_digest is None
                    or self.expected.authorization_context_digest is None
                ):
                    raise ValueError(
                        "MCP authorization context requires expected principal_digest, "
                        "permission_digest, and authorization_context_digest"
                    )
                context = build_authorization_context(
                    references,
                    principal_digest=self.expected.principal_digest,
                    permission_digest=self.expected.permission_digest,
                )
                if (
                    context.authorization_context_digest
                    != self.expected.authorization_context_digest
                ):
                    raise ValueError(
                        "expected.authorization_context_digest does not match credential "
                        "references, principal, and permissions"
                    )
        if self.control.reserved_tool_names & set(self.tools.include):
            raise ValueError("tools.include must not expose a control tool")
        if self.control.episode_argument == self.results.confirmation_parameter:
            raise ValueError(
                "control.episode_argument must differ from results.confirmation_parameter; "
                "the episode argument is stripped from published schemas"
            )
        published_names = {
            self.tools.published_name(source) for source in self.tools.include
        }
        if self.control.reserved_tool_names & published_names:
            raise ValueError("tools.aliases must not publish a control tool name")
        if (
            self.control.describe_oracle is None
            and self.expected.server_content_digest is not None
        ):
            raise ValueError(
                "expected.server_content_digest requires control.describe_oracle"
            )
        if (
            self.control.describe_oracle is not None
            and self.expected.server_content_digest is None
        ):
            raise ValueError(
                "control.describe_oracle requires expected.server_content_digest"
            )
        if self.mode == "A":
            if (
                self.control.reset_strategy != "control_tool"
                or self.control.state_strategy != "control_tool"
            ):
                raise ValueError("mode A requires control-tool reset and state strategies")
            if self.fixtures.direction != "pushed":
                raise ValueError("mode A requires fixtures.direction=pushed")
        elif self.mode == "C":
            if self.control.reset_strategy != "no_op_verified":
                raise ValueError("mode C requires reset_strategy=no_op_verified")
            if self.control.state_strategy != "read_only_projection":
                raise ValueError("mode C requires state_strategy=read_only_projection")
            if self.fixtures.direction != "snapshot":
                raise ValueError("mode C requires fixtures.direction=snapshot")
            if self.tools.mutates or self.tools.requires_confirmation:
                raise ValueError("mode C may not expose mutating or confirmation-gated tools")
            if self.tools.trust_annotations:
                raise ValueError(
                    "mode C may not derive mutation from server annotations; "
                    "read-only snapshots must declare the surface in reviewed config"
                )
            if self.isolation != "read_only":
                raise ValueError("mode C requires isolation=read_only")
        return self


@dataclass(frozen=True)
class LoadedMcpOracleConfig:
    path: Path
    value: McpOracleConfig
    # The reviewed document before host path resolution, so digests stay host independent.
    raw_document: dict[str, Any]


class TrustedExecutablePolicy(_StrictModel):
    executable: Path
    sha256: StrictStr
    allowed_argv: tuple[tuple[StrictStr, ...], ...]
    allowed_cwd_roots: tuple[Path, ...]

    @field_validator("sha256", mode="after")
    @classmethod
    def _normalize_sha256(cls, value: str) -> str:
        return _digest(value, "trusted executable sha256")

    @model_validator(mode="after")
    def _validate_policy(self) -> TrustedExecutablePolicy:
        if not self.allowed_argv:
            raise ValueError("trusted executable allowed_argv must not be empty")
        if len(self.allowed_argv) != len(set(self.allowed_argv)):
            raise ValueError("trusted executable allowed_argv must not contain duplicates")
        if not self.allowed_cwd_roots:
            raise ValueError("trusted executable allowed_cwd_roots must not be empty")
        return self


class TrustedExecutablePolicies(_StrictModel):
    schema_version: Literal["bfcl-trusted-executables-v1"]
    policies: dict[StrictStr, TrustedExecutablePolicy]


def _resolve_path(path: Path, base: Path) -> Path:
    return (path if path.is_absolute() else base / path).resolve()


def path_is_inside(path: Path, *roots: Path) -> bool:
    """Return whether ``path`` resolves to or under any of ``roots``."""
    resolved_path = path.resolve()
    for root in roots:
        resolved_root = root.resolve()
        if resolved_path == resolved_root or resolved_root in resolved_path.parents:
            return True
    return False


def load_mcp_oracle_config(
    path: Path,
    *,
    allow_insecure_localhost: bool = False,
) -> LoadedMcpOracleConfig:
    """Load one strict MCP profile without resolving any credentials."""
    source = path.resolve()
    raw = load_unique_yaml_mapping(source, "MCP oracle config")
    try:
        config = McpOracleConfig.model_validate(raw)
        transport = config.transport
        if isinstance(transport, StdioTransportConfig):
            resolved = transport.model_copy(
                update={"cwd": _resolve_path(transport.cwd, source.parent)}
            )
            if not resolved.cwd.is_dir():
                raise ValueError(f"transport.cwd is not a directory: {resolved.cwd}")
            config = config.model_copy(update={"transport": resolved})
        else:
            parsed = urllib.parse.urlsplit(transport.url)
            is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if parsed.scheme != "https" and not (
                allow_insecure_localhost and parsed.scheme == "http" and is_loopback
            ):
                raise ValueError(
                    "transport.url must use HTTPS; HTTP is debug-only for explicit loopback"
                )
            ca_bundle = transport.tls.ca_bundle_path
            if ca_bundle is not None:
                resolved_ca = _resolve_path(ca_bundle, source.parent)
                if not resolved_ca.is_file():
                    raise ValueError(
                        f"transport.tls.ca_bundle_path is not a file: {resolved_ca}"
                    )
                if not path_is_inside(resolved_ca, source.parent):
                    raise ValueError(
                        "transport.tls.ca_bundle_path must remain inside the MCP pack tree"
                    )
                transport = transport.model_copy(
                    update={
                        "tls": transport.tls.model_copy(
                            update={"ca_bundle_path": resolved_ca}
                        )
                    }
                )
                config = config.model_copy(update={"transport": transport})
    except (OSError, ValueError) as exc:
        if isinstance(exc, McpConfigError):
            raise
        raise McpConfigError(f"invalid MCP oracle config {source}: {exc}") from exc
    return LoadedMcpOracleConfig(path=source, value=config, raw_document=raw)


def load_trusted_executable_policies(path: Path) -> TrustedExecutablePolicies:
    """Load the host-owned stdio executable policy from outside the oracle pack."""
    source = path.resolve()
    raw = load_unique_yaml_mapping(source, "trusted executable policy")
    try:
        policies = TrustedExecutablePolicies.model_validate(raw)
        resolved = {
            name: policy.model_copy(
                update={
                    "executable": _resolve_path(policy.executable, source.parent),
                    "allowed_cwd_roots": tuple(
                        _resolve_path(root, source.parent)
                        for root in policy.allowed_cwd_roots
                    ),
                }
            )
            for name, policy in policies.policies.items()
        }
        return policies.model_copy(update={"policies": resolved})
    except (OSError, ValueError) as exc:
        raise McpConfigError(f"invalid trusted executable policy {source}: {exc}") from exc
