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

"""Bounded MCP client transport and protocol validation."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import io
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nemotron.steps.byob.runtime.authoring_workflow.credentials import (
    CredentialReference,
    CredentialResolver,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.config import (
    McpOracleConfig,
    StdioTransportConfig,
    StreamableHttpTransportConfig,
    TrustedExecutablePolicies,
    path_is_inside,
)
from nemotron.steps.byob.runtime.mcp.errors import (
    McpCredentialError,
    McpExecutablePolicyError,
    McpIntegrationError,
    McpProtocolError,
    McpTransportError,
)

_SAFE_STDIO_ENV = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "TEMP", "TMP", "TMPDIR", "TZ"})
_MAX_CAPTURED_STDERR_CHARS = 64 * 1024
_MIN_EMBEDDED_SECRET_CHARS = 8


class _BoundedRedactingTextSink(io.StringIO):
    """Capture bounded subprocess diagnostics without forwarding secrets to logs."""

    def __init__(self, secrets: tuple[str, ...]):
        super().__init__()
        self._secrets = secrets
        self._length = 0
        self.truncated = False

    def write(self, value: str) -> int:
        redacted = _redact(str(value), self._secrets)
        remaining = _MAX_CAPTURED_STDERR_CHARS - self._length
        if remaining > 0:
            kept = redacted[:remaining]
            super().write(kept)
            self._length += len(kept)
        if len(redacted) > remaining:
            self.truncated = True
        return len(value)


def _drain_stderr_fd(descriptor: int, sink: _BoundedRedactingTextSink) -> None:
    """Drain child stderr without ever storing its unredacted text."""
    with os.fdopen(
        descriptor,
        "r",
        encoding="utf-8",
        errors="replace",
        closefd=True,
    ) as source:
        while chunk := source.read(4096):
            sink.write(chunk)


@asynccontextmanager
async def _stdio_error_pipe(
    secrets: tuple[str, ...],
) -> AsyncIterator[tuple[io.TextIOWrapper, _BoundedRedactingTextSink]]:
    """Provide the real file descriptor required by MCP SDK v2's stdio transport.

    ``io.StringIO`` worked with mocked transports but cannot be passed to
    ``subprocess.Popen``. A pipe lets the child write to a real descriptor while a
    background reader redacts each bounded chunk before it reaches memory.
    """
    read_fd, write_fd = os.pipe()
    sink = _BoundedRedactingTextSink(secrets)
    writer = os.fdopen(write_fd, "w", encoding="utf-8", closefd=True)
    reader = asyncio.create_task(asyncio.to_thread(_drain_stderr_fd, read_fd, sink))
    try:
        yield writer, sink
    finally:
        writer.close()
        await reader


@dataclass(frozen=True)
class McpServerIdentity:
    name: str | None
    version: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class McpToolPage:
    tools: tuple[Any, ...]
    next_cursor: str | None


class ConnectedMcpClient(Protocol):
    @property
    def sdk_version(self) -> str: ...

    @property
    def protocol_version(self) -> str: ...

    @property
    def server_identity(self) -> McpServerIdentity: ...

    @property
    def capabilities(self) -> dict[str, Any]: ...

    async def list_tools(self, cursor: str | None = None) -> McpToolPage: ...

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_stdio_launch(
    config: StdioTransportConfig,
    policies: TrustedExecutablePolicies | None,
) -> tuple[str, list[str]]:
    """Replace the untrusted executable token with a host-pinned executable."""
    if policies is None:
        raise McpExecutablePolicyError("stdio transport requires a host-owned trusted executable policy file")
    policy = policies.policies.get(config.executable_policy)
    if policy is None:
        raise McpExecutablePolicyError(f"unknown trusted executable policy {config.executable_policy!r}")
    executable = policy.executable.resolve()
    if not executable.is_file():
        raise McpExecutablePolicyError(f"trusted executable is not a file: {executable}")
    configured = Path(config.command[0])
    if configured.is_absolute():
        if configured.resolve() != executable:
            raise McpExecutablePolicyError(f"configured executable {configured} does not match policy {executable}")
    elif configured.name != executable.name:
        raise McpExecutablePolicyError(
            f"configured executable {configured!s} does not name policy binary {executable.name!r}"
        )
    observed_digest = _sha256_file(executable)
    if observed_digest != policy.sha256:
        raise McpExecutablePolicyError(f"trusted executable digest mismatch for policy {config.executable_policy!r}")
    arguments = tuple(config.command[1:])
    if arguments not in policy.allowed_argv:
        raise McpExecutablePolicyError(f"stdio argv is not explicitly allowed by policy {config.executable_policy!r}")
    if not path_is_inside(config.cwd, *policy.allowed_cwd_roots):
        raise McpExecutablePolicyError(f"stdio cwd {config.cwd} is outside policy {config.executable_policy!r} roots")
    return str(executable), list(arguments)


def build_stdio_environment(
    config: StdioTransportConfig,
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Construct a minimal child environment without inheriting ambient secrets."""
    selected = _SAFE_STDIO_ENV | set(config.env_passthrough)
    missing = sorted(set(config.env_passthrough) - set(environ))
    if missing:
        raise McpCredentialError(f"stdio environment references missing variable(s): {missing}")
    return {name: environ[name] for name in sorted(selected) if name in environ}


def resolve_http_headers(
    config: StreamableHttpTransportConfig,
    environ: Mapping[str, str],
    *,
    credential_resolver: CredentialResolver | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Resolve named secret references without placing literals in config or reports."""
    headers: dict[str, str] = {}
    secrets: list[str] = []
    resolver = credential_resolver or CredentialResolver(environ=environ)
    token_reference = config.auth.bearer_token_ref
    if config.auth.bearer_token_env is not None:
        token_reference = CredentialReference.environment(config.auth.bearer_token_env)
    if token_reference is not None:
        try:
            token = resolver.resolve(token_reference).reveal()
        except ValueError as exc:
            raise McpCredentialError(str(exc)) from exc
        headers["Authorization"] = f"Bearer {token}"
        secrets.append(token)
    for header, env_name in sorted(config.auth.headers.items()):
        value = environ.get(env_name)
        if not value:
            raise McpCredentialError(
                f"HTTP header {header!r} references missing or empty environment variable {env_name!r}"
            )
        if "\r" in value or "\n" in value:
            raise McpCredentialError(f"HTTP header environment variable {env_name!r} contains a line break")
        headers[header] = value
        secrets.append(value)
    for header, reference in sorted(config.auth.header_refs.items()):
        try:
            value = resolver.resolve(reference).reveal()
        except ValueError as exc:
            raise McpCredentialError(str(exc)) from exc
        headers[header] = value
        secrets.append(value)
    return headers, tuple(secrets)


def _redact(message: str, secrets: tuple[str, ...]) -> str:
    redacted = message
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


@dataclass
class SecretRegistry:
    """Secrets learned while opening a transport, tracked only to redact diagnostics."""

    values: tuple[str, ...] = ()


@asynccontextmanager
async def mcp_error_boundary(secrets: SecretRegistry) -> AsyncIterator[None]:
    """Convert unexpected faults into McpTransportError while preserving typed failures.

    Discovery runs inside the open transport, so a naive boundary would relabel catalog,
    normalization, and identity failures as transport faults and destroy the taxonomy
    that operators and the Gold Gate rely on.
    """
    try:
        yield
    except McpIntegrationError as exc:
        redacted = _redact(str(exc), secrets.values)
        if redacted == str(exc):
            raise
        raise type(exc)(redacted) from exc
    except Exception as exc:
        raise McpTransportError(_redact(f"MCP transport failed: {type(exc).__name__}: {exc}", secrets.values)) from exc


def require_mcp_sdk_v2() -> str:
    """Fail closed unless the isolated bfcl-mcp runtime supplies the pinned SDK major."""
    try:
        version = importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError as exc:
        raise McpTransportError(
            "the MCP SDK is not installed; run BFCL MCP discovery inside the isolated "
            "bfcl-mcp runtime that provides mcp>=2,<3"
        ) from exc
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) != 2:
        raise McpTransportError(
            f"BFCL MCP runtime requires mcp>=2,<3, found {version}; use the isolated bfcl-mcp runtime"
        )
    return version


def _strict_json_size(
    value: Any,
    limit: int,
    label: str,
    *,
    secrets: tuple[str, ...] = (),
) -> None:
    try:
        encoded = canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise McpProtocolError(f"{label} is not strict JSON: {exc}") from exc
    if len(encoded) > limit:
        raise McpProtocolError(f"{label} exceeds configured max_response_bytes ({len(encoded)} > {limit})")
    if _reflects_secret(value, secrets):
        raise McpCredentialError(f"{label} reflected a configured credential; refusing to persist or report it")


def _reflects_secret(value: Any, secrets: tuple[str, ...]) -> bool:
    """Detect exact reflection, plus embedding for secrets long enough to be distinctive."""
    if isinstance(value, str):
        return any(
            secret and (value == secret or (len(secret) >= _MIN_EMBEDDED_SECRET_CHARS and secret in value))
            for secret in secrets
        )
    if isinstance(value, Mapping):
        return any(_reflects_secret(key, secrets) or _reflects_secret(child, secrets) for key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return any(_reflects_secret(child, secrets) for child in value)
    return False


def _reject_secret_text(
    value: str | None,
    *,
    secrets: tuple[str, ...],
    label: str,
) -> None:
    if value is not None and _reflects_secret(value, secrets):
        raise McpCredentialError(f"{label} reflected a configured credential; refusing to persist or report it")


_MISSING = object()


def _sdk_attr(value: Any, *names: str, label: str) -> Any:
    """Read one SDK field, accepting either spelling and refusing to guess a default.

    MCP models mirror the wire protocol, so fields arrive camelCase (``nextCursor``)
    while Python callers often expect snake_case. Defaulting a missing attribute to
    ``None`` would silently truncate pagination after the first page and pin a digest
    over an incomplete catalog, so an unrecognized surface fails loudly instead.
    """
    for name in names:
        found = getattr(value, name, _MISSING)
        if found is not _MISSING:
            return found
    raise McpProtocolError(
        f"MCP SDK does not expose {label} under any of {list(names)}; "
        "the pinned SDK surface changed and discovery cannot be trusted"
    )


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise McpProtocolError(f"MCP SDK returned unsupported value type {type(value).__name__}")


class SdkConnectedMcpClient:
    """Small SDK-independent facade used by discovery and tests."""

    def __init__(
        self,
        client: Any,
        *,
        max_response_bytes: int,
        sdk_version: str,
        secrets: SecretRegistry | None = None,
    ):
        self._client = client
        self._max_response_bytes = max_response_bytes
        self._sdk_version = sdk_version
        self._secrets = SecretRegistry() if secrets is None else secrets

    @property
    def sdk_version(self) -> str:
        return self._sdk_version

    @property
    def protocol_version(self) -> str:
        value = _sdk_attr(
            self._client,
            "protocol_version",
            "protocolVersion",
            label="the negotiated protocol version",
        )
        if not isinstance(value, str) or not value:
            raise McpProtocolError("MCP SDK did not expose a negotiated protocol version")
        _reject_secret_text(
            value,
            secrets=self._secrets.values,
            label="negotiated protocol version",
        )
        return value

    @property
    def server_identity(self) -> McpServerIdentity:
        info = _sdk_attr(
            self._client,
            "server_info",
            "serverInfo",
            label="the server implementation info",
        )
        if info is None:
            raise McpProtocolError("MCP server completed initialize without serverInfo")
        name = getattr(info, "name", None)
        version = getattr(info, "version", None)
        if not isinstance(name, str) or not name:
            raise McpProtocolError("MCP serverInfo.name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise McpProtocolError("MCP serverInfo.version must be a non-empty string")
        _reject_secret_text(
            name,
            secrets=self._secrets.values,
            label="server name",
        )
        _reject_secret_text(
            version,
            secrets=self._secrets.values,
            label="server version",
        )
        return McpServerIdentity(name=name, version=version)

    @property
    def capabilities(self) -> dict[str, Any]:
        capabilities = _sdk_attr(
            self._client,
            "server_capabilities",
            "serverCapabilities",
            label="the negotiated server capabilities",
        )
        if capabilities is None:
            raise McpProtocolError("MCP server completed initialize without capabilities")
        dumped = _model_dump(capabilities)
        _strict_json_size(
            dumped,
            self._max_response_bytes,
            "server capabilities",
            secrets=self._secrets.values,
        )
        return dumped

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        result = await self._client.list_tools(cursor=cursor)
        raw_tools = _sdk_attr(result, "tools", label="the tools/list page contents")
        if isinstance(raw_tools, str | bytes) or not isinstance(raw_tools, Sequence):
            raise McpProtocolError("tools/list did not return a sequence of tools")
        tools = tuple(raw_tools)
        next_cursor = _sdk_attr(
            result,
            "nextCursor",
            "next_cursor",
            label="the tools/list continuation cursor",
        )
        if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor):
            raise McpProtocolError("tools/list nextCursor must be a non-empty string or null")
        payload = {
            "tools": [_model_dump(tool) for tool in tools],
            "next_cursor": next_cursor,
        }
        _strict_json_size(
            payload,
            self._max_response_bytes,
            "tools/list page",
            secrets=self._secrets.values,
        )
        return McpToolPage(tools=tools, next_cursor=next_cursor)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # The high-level SDK method automatically drives InputRequiredResult and
        # may issue up to ten additional tools/call requests. BFCL v1 requires one
        # call and one result, so expose extension shapes to the gateway unchanged.
        result = await self._client.session.call_tool(
            name,
            dict(arguments),
            meta=None if meta is None else dict(meta),
            allow_input_required=True,
            allow_claimed=True,
        )
        dumped = _model_dump(result)
        _strict_json_size(
            dumped,
            self._max_response_bytes,
            f"tools/call {name!r}",
            secrets=self._secrets.values,
        )
        return dumped


@asynccontextmanager
async def open_mcp_connection(
    config: McpOracleConfig,
    *,
    environ: Mapping[str, str] | None = None,
    credential_resolver: CredentialResolver | None = None,
    executable_policies: TrustedExecutablePolicies | None = None,
) -> AsyncIterator[ConnectedMcpClient]:
    """Open one MCP SDK v2 connection for discovery, then close it deterministically."""
    environ = os.environ if environ is None else environ
    secrets = SecretRegistry()
    async with mcp_error_boundary(secrets):
        sdk_version = require_mcp_sdk_v2()
        from mcp import Client, StdioServerParameters

        operation_timeout = max(
            float(config.limits.tool_timeout_s),
            float(config.limits.reset_timeout_s),
        )
        handshake_timeout = float(config.limits.connect_timeout_s) + float(config.limits.handshake_timeout_s)
        outer_timeout_ceiling = max(operation_timeout, handshake_timeout)
        sdk_read_timeout = outer_timeout_ceiling + max(
            1.0,
            outer_timeout_ceiling * 0.1,
        )
        async with AsyncExitStack() as stack:
            if isinstance(config.transport, StdioTransportConfig):
                from mcp.client.stdio import stdio_client

                command, arguments = resolve_stdio_launch(
                    config.transport,
                    executable_policies,
                )
                child_env = build_stdio_environment(config.transport, environ)
                secrets.values = tuple(
                    child_env[name] for name in config.transport.env_passthrough if name in child_env
                )
                errlog, _captured_stderr = await stack.enter_async_context(_stdio_error_pipe(secrets.values))
                target = stdio_client(
                    StdioServerParameters(
                        command=command,
                        args=arguments,
                        env=child_env,
                        cwd=config.transport.cwd,
                    ),
                    errlog=errlog,
                )
            else:
                import httpx2
                from mcp.client.streamable_http import streamable_http_client

                headers, secrets.values = resolve_http_headers(
                    config.transport,
                    environ,
                    credential_resolver=credential_resolver,
                )
                # Every request is already bounded by asyncio.wait_for, so the transport
                # budget stays the looser one; otherwise raw httpx timeouts would preempt
                # the typed catalog and control failures operators need to diagnose.
                timeout = httpx2.Timeout(
                    float(config.limits.connect_timeout_s),
                    read=sdk_read_timeout,
                    write=sdk_read_timeout,
                    pool=float(config.limits.connect_timeout_s),
                )
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers=headers,
                        verify=(
                            str(config.transport.tls.ca_bundle_path)
                            if config.transport.tls.ca_bundle_path is not None
                            else True
                        ),
                        timeout=timeout,
                        follow_redirects=False,
                        trust_env=False,
                    )
                )
                target = streamable_http_client(
                    config.transport.url,
                    http_client=http_client,
                    terminate_on_close=True,
                )
            sdk_client = Client(
                target,
                read_timeout_seconds=sdk_read_timeout,
            )
            try:
                connected = await asyncio.wait_for(
                    stack.enter_async_context(sdk_client),
                    timeout=handshake_timeout,
                )
            except TimeoutError as exc:
                raise McpTransportError(f"MCP connect/handshake exceeded {handshake_timeout:g} seconds") from exc
            yield SdkConnectedMcpClient(
                connected,
                max_response_bytes=config.limits.max_response_bytes,
                sdk_version=sdk_version,
                secrets=secrets,
            )
