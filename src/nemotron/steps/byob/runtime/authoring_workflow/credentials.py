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

"""Secret-free credential references and authorization-context binding."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

CREDENTIAL_REFERENCE_VERSION: Literal["bfcl-credential-reference-v1"] = (
    "bfcl-credential-reference-v1"
)
AUTHORIZATION_CONTEXT_VERSION: Literal["bfcl-authorization-context-v1"] = (
    "bfcl-authorization-context-v1"
)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}$")
_PROVIDER = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class CredentialLifecycleError(ValueError):
    """Stable credential failure that never includes credential material."""

    def __init__(self, code: str, reference_name: str) -> None:
        self.code = code
        self.reference_name = reference_name
        super().__init__(f"{code}: credential reference {reference_name!r}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class CredentialReference(_StrictModel):
    schema_version: Literal["bfcl-credential-reference-v1"] = (
        CREDENTIAL_REFERENCE_VERSION
    )
    resolver: Literal["environment", "secret_manager"]
    name: StrictStr
    provider: StrictStr | None = None
    version: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> CredentialReference:
        if self.resolver == "environment":
            if _ENV_NAME.fullmatch(self.name) is None:
                raise ValueError("environment credential name must be uppercase")
            if self.provider is not None or self.version is not None:
                raise ValueError(
                    "environment credential cannot declare provider or version"
                )
        else:
            if _SAFE_NAME.fullmatch(self.name) is None:
                raise ValueError("secret-manager credential name is not safe")
            if self.provider is None or _PROVIDER.fullmatch(self.provider) is None:
                raise ValueError(
                    "secret-manager credential requires a safe provider name"
                )
            if self.version is not None and _SAFE_NAME.fullmatch(self.version) is None:
                raise ValueError("secret-manager credential version is not safe")
        return self

    @classmethod
    def environment(cls, name: str) -> CredentialReference:
        return cls(resolver="environment", name=name)


@dataclass(frozen=True, repr=False)
class ResolvedCredential:
    """Ephemeral credential bytes; repr and str never reveal the value."""

    reference: CredentialReference
    _value: str

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return (
            "ResolvedCredential("
            f"resolver={self.reference.resolver!r}, name={self.reference.name!r}, "
            "value=<redacted>)"
        )

    __str__ = __repr__


class SecretManagerBackend(Protocol):
    def resolve(self, name: str, *, version: str | None) -> str: ...


class CredentialResolver:
    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        secret_managers: Mapping[str, SecretManagerBackend] | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._secret_managers = dict(secret_managers or {})

    def resolve(self, reference: CredentialReference) -> ResolvedCredential:
        if reference.resolver == "environment":
            value = self._environ.get(reference.name)
        else:
            backend = self._secret_managers.get(reference.provider or "")
            if backend is None:
                raise CredentialLifecycleError(
                    "credential_provider_unavailable",
                    reference.name,
                )
            try:
                value = backend.resolve(reference.name, version=reference.version)
            except Exception:
                raise CredentialLifecycleError(
                    "credential_unresolved",
                    reference.name,
                ) from None
        if not value or "\r" in value or "\n" in value:
            raise CredentialLifecycleError(
                "credential_unresolved",
                reference.name,
            )
        return ResolvedCredential(reference=reference, _value=value)


class AuthorizationContext(_StrictModel):
    schema_version: Literal["bfcl-authorization-context-v1"]
    credential_references_digest: StrictStr
    principal_digest: StrictStr
    permission_digest: StrictStr
    authorization_context_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> AuthorizationContext:
        for value in (
            self.credential_references_digest,
            self.principal_digest,
            self.permission_digest,
            self.authorization_context_digest,
        ):
            if _DIGEST.fullmatch(value) is None:
                raise ValueError(
                    "authorization context fields must be sha256:<64 lowercase hex>"
                )
        unsigned = self.model_dump(
            mode="json",
            exclude={"authorization_context_digest"},
        )
        if self.authorization_context_digest != sha256_json(unsigned):
            raise ValueError("authorization context digest mismatch")
        return self


def build_authorization_context(
    references: tuple[CredentialReference, ...],
    *,
    principal_digest: str,
    permission_digest: str,
) -> AuthorizationContext:
    canonical_references = sorted(
        (reference.model_dump(mode="json") for reference in references),
        key=lambda item: (
            str(item["resolver"]),
            str(item.get("provider") or ""),
            str(item["name"]),
            str(item.get("version") or ""),
        ),
    )
    unsigned: dict[str, Any] = {
        "schema_version": AUTHORIZATION_CONTEXT_VERSION,
        "credential_references_digest": sha256_json(canonical_references),
        "principal_digest": principal_digest,
        "permission_digest": permission_digest,
    }
    return AuthorizationContext.model_validate(
        {
            **unsigned,
            "authorization_context_digest": sha256_json(unsigned),
        }
    )


def require_current_authorization_context(
    bound_digest: str,
    current: AuthorizationContext,
) -> None:
    if _DIGEST.fullmatch(bound_digest) is None:
        raise CredentialLifecycleError(
            "credential_context_invalid",
            "authorization_context",
        )
    if bound_digest != current.authorization_context_digest:
        raise CredentialLifecycleError(
            "credential_context_stale",
            "authorization_context",
        )
