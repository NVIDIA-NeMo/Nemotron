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

"""HTTPS client and configuration for BFCL Oracle HTTP v1 endpoints."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.authoring_workflow.credentials import (
    CredentialReference,
    CredentialResolver,
    build_authorization_context,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    assert_pack_allowed,
)

PROTOCOL_VERSION = "bfcl-oracle-http-v1"
ENDPOINT_CREDENTIAL_POLICY_VERSION = "bfcl-endpoint-credential-policy-v1"
ENDPOINT_CREDENTIAL_POLICY_ENV = "BFCL_ENDPOINT_CREDENTIAL_POLICY"
_ATTESTATION_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONFIG_KEYS = frozenset(
    {
        "protocol_version",
        "base_url",
        "auth",
        "expected",
        "attestation",
        "tls",
        "max_request_bytes",
        "max_response_bytes",
    }
)
_AUTH_KEYS = frozenset({"bearer_token_env", "bearer_token_ref", "headers"})
_EXPECTED_KEYS = frozenset(
    {
        "oracle_id",
        "oracle_version",
        "content_digest",
        "principal_digest",
        "permission_digest",
        "authorization_context_digest",
    }
)
_ATTESTATION_KEYS = frozenset({"kind", "expected_digest"})
_TLS_KEYS = frozenset({"ca_bundle_path"})
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


@dataclass(frozen=True)
class EndpointIdentity:
    protocol_version: str
    oracle_id: str
    oracle_version: str
    content_digest: str
    principal_digest: str | None = None
    permission_digest: str | None = None
    authorization_context_digest: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, source: str) -> EndpointIdentity:
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be a JSON object")
        required = {
            "protocol_version",
            "oracle_id",
            "oracle_version",
            "content_digest",
        }
        allowed = required | {
            "principal_digest",
            "permission_digest",
            "authorization_context_digest",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"{source} has unknown fields: {', '.join(unknown)}")
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"{source} is missing: {', '.join(missing)}")
        fields = {name: value[name] for name in required}
        if any(not isinstance(item, str) or not item.strip() for item in fields.values()):
            raise ValueError(f"{source} identity fields must be non-empty strings")
        if fields["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(
                f"{source} protocol_version must be {PROTOCOL_VERSION!r}, got {fields['protocol_version']!r}"
            )
        digest = str(fields["content_digest"])
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError(f"{source} content_digest must be sha256:<64 hex characters>")
        try:
            int(digest.removeprefix("sha256:"), 16)
        except ValueError as exc:
            raise ValueError(f"{source} content_digest is not hexadecimal") from exc
        optional_digests = {
            name: value.get(name)
            for name in (
                "principal_digest",
                "permission_digest",
                "authorization_context_digest",
            )
        }
        for name, optional_digest in optional_digests.items():
            if optional_digest is not None:
                if not isinstance(optional_digest, str) or _ATTESTATION_DIGEST.fullmatch(optional_digest) is None:
                    raise ValueError(f"{source} {name} must be sha256:<64 lowercase hex characters>")
        return cls(
            protocol_version=fields["protocol_version"],
            oracle_id=fields["oracle_id"],
            oracle_version=fields["oracle_version"],
            content_digest=fields["content_digest"],
            principal_digest=optional_digests["principal_digest"],
            permission_digest=optional_digests["permission_digest"],
            authorization_context_digest=optional_digests["authorization_context_digest"],
        )

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


@dataclass(frozen=True)
class ExpectedAttestation:
    """The conformance document a pack pins, identified by digest rather than by trust."""

    kind: str
    expected_digest: str


@dataclass(frozen=True)
class EndpointConfig:
    path: Path
    base_url: str
    expected: EndpointIdentity
    # Absent means the pack is not claiming certifiable conformance. That is a legal
    # configuration for a smoke run, and it caps the endpoint below publication.
    attestation: ExpectedAttestation | None = None
    bearer_token_env: str | None = None
    header_env: tuple[tuple[str, str], ...] = ()
    bearer_token_ref: CredentialReference | None = None
    header_refs: tuple[tuple[str, CredentialReference], ...] = ()
    ca_bundle_path: Path | None = None
    max_request_bytes: int = 10 * 1024 * 1024
    max_response_bytes: int = 10 * 1024 * 1024


@dataclass(frozen=True)
class EndpointCredentialGrant:
    """Operator-owned permission to send named credentials to one exact destination."""

    base_url: str
    credential_references: tuple[CredentialReference, ...]


@dataclass(frozen=True)
class EndpointCredentialPolicy:
    """Credential grants loaded independently of the untrusted oracle pack."""

    grants: tuple[EndpointCredentialGrant, ...]

    def authorize(
        self,
        *,
        base_url: str,
        references: tuple[CredentialReference, ...],
    ) -> None:
        requested = {_reference_identity(reference) for reference in references}
        authorized = {
            _reference_identity(reference)
            for grant in self.grants
            if grant.base_url == base_url
            for reference in grant.credential_references
        }
        if not requested <= authorized:
            raise ValueError(
                "endpoint credential policy does not authorize every configured "
                f"credential reference for destination {base_url!r}"
            )

    @classmethod
    def from_mapping(cls, value: Any, *, source: str) -> EndpointCredentialPolicy:
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be a JSON object")
        _reject_unknown(
            value,
            frozenset({"schema_version", "grants"}),
            source,
        )
        if value.get("schema_version") != ENDPOINT_CREDENTIAL_POLICY_VERSION:
            raise ValueError(f"{source} schema_version must be {ENDPOINT_CREDENTIAL_POLICY_VERSION!r}")
        raw_grants = value.get("grants")
        if not isinstance(raw_grants, list) or not raw_grants:
            raise ValueError(f"{source} grants must be a non-empty list")
        grants: list[EndpointCredentialGrant] = []
        for index, raw_grant in enumerate(raw_grants):
            label = f"{source} grants[{index}]"
            if not isinstance(raw_grant, dict):
                raise ValueError(f"{label} must be a mapping")
            _reject_unknown(
                raw_grant,
                frozenset({"base_url", "credential_references"}),
                label,
            )
            base_url = _validate_base_url(raw_grant.get("base_url"), f"{label}.base_url")
            raw_references = raw_grant.get("credential_references")
            if not isinstance(raw_references, list) or not raw_references:
                raise ValueError(f"{label}.credential_references must be a non-empty list")
            references = tuple(
                _credential_reference(reference, f"{label}.credential_references") for reference in raw_references
            )
            if len({_reference_identity(reference) for reference in references}) != len(references):
                raise ValueError(f"{label} repeats a credential reference")
            grants.append(
                EndpointCredentialGrant(
                    base_url=base_url,
                    credential_references=references,
                )
            )
        return cls(grants=tuple(grants))


def _reject_unknown(value: Mapping[str, Any], allowed: frozenset[str], source: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{source} has unknown keys: {', '.join(unknown)}")


def _nonempty_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return value.strip()


def _credential_reference(value: Any, source: str) -> CredentialReference:
    if isinstance(value, str):
        return CredentialReference.environment(_nonempty_string(value, source))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an environment name or credential reference")
    try:
        return CredentialReference.model_validate(value)
    except ValueError as exc:
        raise ValueError(f"{source} is invalid: {exc}") from exc


def _reference_identity(reference: CredentialReference) -> str:
    return json.dumps(
        reference.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_base_url(value: Any, source: str) -> str:
    base_url = _nonempty_string(value, source).rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source} must be an HTTPS origin/path without credentials, query, or fragment")
    return base_url


def load_endpoint_credential_policy(path: Path) -> EndpointCredentialPolicy:
    """Load an operator policy from a JSON file outside pack-controlled configuration."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"endpoint credential policy is unreadable: {path}") from exc
    return EndpointCredentialPolicy.from_mapping(
        value,
        source=f"endpoint credential policy {path}",
    )


def load_endpoint_config(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
) -> EndpointConfig:
    """Load a strict, secret-free endpoint declaration."""
    path = assert_pack_allowed(path, allowed_roots)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"endpoint config must be a mapping: {path}")
    _reject_unknown(raw, _CONFIG_KEYS, "endpoint config")
    protocol = _nonempty_string(raw.get("protocol_version"), "endpoint protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError(f"endpoint protocol_version must be {PROTOCOL_VERSION!r}, got {protocol!r}")

    base_url = _validate_base_url(raw.get("base_url"), "endpoint base_url")

    auth = raw.get("auth") or {}
    if not isinstance(auth, dict):
        raise ValueError("endpoint auth must be a mapping")
    _reject_unknown(auth, _AUTH_KEYS, "endpoint auth")
    bearer = auth.get("bearer_token_env")
    bearer_ref_value = auth.get("bearer_token_ref")
    if bearer is not None and bearer_ref_value is not None:
        raise ValueError("endpoint auth must not set both bearer_token_env and bearer_token_ref")
    bearer_ref = None
    if bearer is not None:
        bearer = _nonempty_string(bearer, "endpoint auth.bearer_token_env")
        bearer_ref = CredentialReference.environment(bearer)
    elif bearer_ref_value is not None:
        bearer_ref = _credential_reference(
            bearer_ref_value,
            "endpoint auth.bearer_token_ref",
        )
        if bearer_ref.resolver == "environment":
            bearer = bearer_ref.name
    headers = auth.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("endpoint auth.headers must map HTTP header names to environment names")
    header_env: list[tuple[str, str]] = []
    header_refs: list[tuple[str, CredentialReference]] = []
    for header, reference_value in headers.items():
        header_name = _nonempty_string(header, "endpoint auth header name")
        if _HEADER_NAME.fullmatch(header_name) is None:
            raise ValueError(f"endpoint auth header {header_name!r} is not a valid HTTP field name")
        if header_name.lower() in {"authorization", "host", "content-length"}:
            raise ValueError(f"endpoint auth header {header_name!r} is reserved")
        reference = _credential_reference(
            reference_value,
            f"endpoint auth.headers.{header_name}",
        )
        header_refs.append((header_name, reference))
        if reference.resolver == "environment":
            header_env.append((header_name, reference.name))

    tls = raw.get("tls") or {}
    if not isinstance(tls, dict):
        raise ValueError("endpoint tls must be a mapping")
    _reject_unknown(tls, _TLS_KEYS, "endpoint tls")
    ca_bundle_path = None
    if tls.get("ca_bundle_path") is not None:
        candidate = Path(_nonempty_string(tls["ca_bundle_path"], "endpoint tls.ca_bundle_path"))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        ca_bundle_path = assert_pack_allowed(candidate, allowed_roots)
        if not ca_bundle_path.is_file():
            raise FileNotFoundError(f"endpoint CA bundle does not exist: {ca_bundle_path}")

    request_limit = raw.get("max_request_bytes", 10 * 1024 * 1024)
    if not isinstance(request_limit, int) or isinstance(request_limit, bool) or request_limit <= 0:
        raise ValueError("endpoint max_request_bytes must be a positive integer")
    response_limit = raw.get("max_response_bytes", 10 * 1024 * 1024)
    if not isinstance(response_limit, int) or isinstance(response_limit, bool) or response_limit <= 0:
        raise ValueError("endpoint max_response_bytes must be a positive integer")
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("endpoint expected must be a mapping")
    _reject_unknown(expected, _EXPECTED_KEYS, "endpoint expected")

    attestation = None
    if raw.get("attestation") is not None:
        declared = raw["attestation"]
        if not isinstance(declared, dict):
            raise ValueError("endpoint attestation must be a mapping")
        _reject_unknown(declared, _ATTESTATION_KEYS, "endpoint attestation")
        kind = _nonempty_string(declared.get("kind"), "endpoint attestation.kind")
        if kind != ATTESTATION_KIND:
            raise ValueError(f"endpoint attestation.kind must be {ATTESTATION_KIND!r}, got {kind!r}")
        digest = _nonempty_string(declared.get("expected_digest"), "endpoint attestation.expected_digest")
        if _ATTESTATION_DIGEST.fullmatch(digest) is None:
            raise ValueError("endpoint attestation.expected_digest must be sha256:<64 lowercase hex characters>")
        attestation = ExpectedAttestation(kind=kind, expected_digest=digest)

    expected_identity = EndpointIdentity.from_mapping(
        {"protocol_version": protocol, **expected},
        source="endpoint expected",
    )
    credential_refs = tuple(
        reference
        for reference in (
            bearer_ref,
            *(reference for _, reference in header_refs),
        )
        if reference is not None
    )
    if credential_refs and expected_identity.authorization_context_digest is not None:
        if expected_identity.principal_digest is None or expected_identity.permission_digest is None:
            raise ValueError("authorization_context_digest requires principal_digest and permission_digest")
        context = build_authorization_context(
            credential_refs,
            principal_digest=expected_identity.principal_digest,
            permission_digest=expected_identity.permission_digest,
        )
        if context.authorization_context_digest != expected_identity.authorization_context_digest:
            raise ValueError(
                "endpoint expected authorization_context_digest does not match "
                "credential references, principal, and permissions"
            )
    return EndpointConfig(
        path=path,
        base_url=base_url,
        expected=expected_identity,
        attestation=attestation,
        bearer_token_env=bearer,
        header_env=tuple(sorted(header_env)),
        bearer_token_ref=bearer_ref,
        header_refs=tuple(sorted(header_refs, key=lambda item: item[0])),
        ca_bundle_path=ca_bundle_path,
        max_request_bytes=request_limit,
        max_response_bytes=response_limit,
    )


def resolve_endpoint_headers(
    config: EndpointConfig,
    environ: Mapping[str, str] | None = None,
    *,
    credential_resolver: CredentialResolver | None = None,
    credential_policy: EndpointCredentialPolicy | None = None,
) -> dict[str, str]:
    """Resolve secrets only after an operator-owned destination grant authorizes them.

    Passing ``environ`` or ``credential_resolver`` explicitly is a one-call scoped
    operator grant for exactly the references and destination in ``config``. Ambient
    process credentials require a policy file named by
    ``BFCL_ENDPOINT_CREDENTIAL_POLICY``; the untrusted pack cannot select that path.
    """
    references = tuple(
        reference
        for reference in (
            config.bearer_token_ref,
            *(reference for _, reference in config.header_refs),
        )
        if reference is not None
    )
    if not references:
        return {}
    explicit_scope = environ is not None or credential_resolver is not None
    if credential_policy is None and not explicit_scope:
        policy_path = os.environ.get(ENDPOINT_CREDENTIAL_POLICY_ENV, "").strip()
        if not policy_path:
            raise ValueError(
                f"authenticated endpoint requires an operator credential policy; set {ENDPOINT_CREDENTIAL_POLICY_ENV}"
            )
        credential_policy = load_endpoint_credential_policy(Path(policy_path))
    if credential_policy is not None:
        credential_policy.authorize(
            base_url=config.base_url,
            references=references,
        )
    resolver = credential_resolver or CredentialResolver(environ=os.environ if environ is None else environ)
    headers: dict[str, str] = {}
    if config.bearer_token_ref is not None:
        token = resolver.resolve(config.bearer_token_ref)
        headers["Authorization"] = f"Bearer {token.reveal()}"
    for name, reference in config.header_refs:
        headers[name] = resolver.resolve(reference).reveal()
    return headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class EndpointOracleClient:
    """Stateful client implementing the same callable surface as a local backend."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> None:
        self.config = config
        self.headers = dict(headers)
        self.timeout_s = timeout_s
        self.session_id: str | None = None
        context = ssl.create_default_context(cafile=str(config.ca_bundle_path) if config.ca_bundle_path else None)
        self._opener = urllib.request.build_opener(
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=context),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if body is not None and len(body) > self.config.max_request_bytes:
            raise RuntimeError("endpoint request exceeds max_request_bytes")
        headers = {"Accept": "application/json", **self.headers}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > self.config.max_response_bytes:
                    raise RuntimeError("endpoint response exceeds max_response_bytes")
                raw = response.read(self.config.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"endpoint {method} {path} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"endpoint {method} {path} failed: {type(exc).__name__}") from exc
        if len(raw) > self.config.max_response_bytes:
            raise RuntimeError("endpoint response exceeds max_response_bytes")
        if not raw and method == "DELETE":
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"endpoint {method} {path} returned invalid JSON") from exc

    def metadata(self) -> dict[str, str]:
        identity = EndpointIdentity.from_mapping(
            self._request("GET", "/v1/metadata"),
            source="endpoint metadata",
        )
        if identity != self.config.expected:
            raise RuntimeError("endpoint metadata does not match the expected oracle identity or content digest")
        return identity.as_dict()

    def conformance(self) -> Any:
        """Fetch the attestation verbatim, without judging it.

        Returned unparsed on purpose. The digest the pack pinned is taken over the exact
        document the endpoint served, so anything this method normalised on the way through
        would be a digest over something the endpoint never said.
        """
        return self._request("GET", "/v1/conformance")

    def list_tools(self) -> list[str]:
        self.metadata()
        payload = self._request("GET", "/v1/tools")
        tools = payload.get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools):
            raise RuntimeError("endpoint /v1/tools must return {'tools': [<name>, ...]}")
        return tools

    def reset(self, *, ctx: Any, fixtures: dict[str, Any] | None = None) -> None:
        self.close()
        payload = self._request(
            "POST",
            "/v1/sessions",
            {
                "context": {
                    "clock": ctx.clock.isoformat(),
                    "seed": ctx.seed,
                    "timeout_s": ctx.timeout_s,
                    "task_id": ctx.task_id,
                },
                "fixtures": fixtures,
            },
        )
        if not isinstance(payload, dict):
            raise RuntimeError("endpoint session response must be an object")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("endpoint session response needs a non-empty session_id")
        self.session_id = session_id
        try:
            identity = EndpointIdentity.from_mapping(
                payload.get("oracle"),
                source="endpoint session oracle",
            )
            if identity != self.config.expected:
                raise RuntimeError("endpoint identity changed while creating a session")
        except Exception:
            self.close()
            raise

    def _session_path(self, suffix: str) -> str:
        if self.session_id is None:
            raise RuntimeError("endpoint session is not initialized; reset must run first")
        identifier = urllib.parse.quote(self.session_id, safe="")
        return f"/v1/sessions/{identifier}{suffix}"

    def call_tool(self, name: str, arguments: dict[str, Any], *, ctx: Any) -> Any:
        return self._request(
            "POST",
            self._session_path("/calls"),
            {
                "name": name,
                "arguments": arguments,
                "turn_index": ctx.turn_index,
            },
        )

    def get_state(self) -> dict[str, Any]:
        state = self._request("GET", self._session_path("/state"))
        if not isinstance(state, dict):
            raise RuntimeError("endpoint state response must be a JSON object")
        return state

    def close(self, *, suppress_errors: bool = False) -> None:
        if self.session_id is None:
            return
        path = self._session_path("")
        try:
            self._request("DELETE", path)
        except Exception:
            if not suppress_errors:
                raise
        else:
            self.session_id = None
