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

"""Static, side-effect-free resolution of assisted-authoring source adapters.

Detection examines only a validated declaration already in memory.  It never
imports the declared path, opens a socket, starts a process, or probes a server.
Those operations belong to the selected adapter after orchestration has recorded
and authorized the resolution result.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

SOURCE_DECLARATION_VERSION: Literal[
    "bfcl-source-declaration-v1"
] = "bfcl-source-declaration-v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_NAMESPACE = re.compile(
    r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+$"
)
_MAX_EXTENSION_BYTES = 64 * 1024
_MAX_EXTENSION_DEPTH = 8
_SOURCE_FIELDS = frozenset({"local_python", "http_package", "mcp_mode_a"})
_AUTHORITY_KEYS = frozenset(
    {
        "approval",
        "approved",
        "attained_level",
        "attained_tier",
        "certification",
        "certification_report",
        "gold_eligible",
        "issuer",
        "publication_authorized",
        "report_digest",
        "release_approved",
        "trust_tier",
    }
)


class AdapterResolutionError(ValueError):
    """Stable failure returned before any adapter or model can be invoked."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_json_value(value: Any, *, path: str, depth: int = 0) -> None:
    if depth > _MAX_EXTENSION_DEPTH:
        raise ValueError(
            f"{path} exceeds the maximum extension nesting depth "
            f"{_MAX_EXTENSION_DEPTH}"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(
                child,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            normalized = key.casefold().replace("-", "_")
            if normalized in _AUTHORITY_KEYS:
                raise ValueError(
                    f"{path}.{key} is an authoritative field and cannot be "
                    "supplied by an extension"
                )
            _validate_json_value(
                child,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return
    raise ValueError(f"{path} must contain JSON values only")


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: _deep_freeze_json(child)
                for key, child in sorted(value.items())
            }
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


class SourceLocation(_StrictModel):
    """An inert path string; resolution deliberately does not inspect the path."""

    path: StrictStr

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source path must be non-empty")
        if "\x00" in value:
            raise ValueError("source path cannot contain NUL")
        if len(value) > 4096:
            raise ValueError("source path exceeds 4096 characters")
        return value


class SourceDeclaration(_StrictModel):
    """The complete source half of the normal ``source + domain brief`` input."""

    declaration_version: Literal["bfcl-source-declaration-v1"]
    adapter: StrictStr | None = None
    local_python: SourceLocation | None = None
    http_package: SourceLocation | None = None
    mcp_mode_a: SourceLocation | None = None
    extensions: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("adapter")
    @classmethod
    def _adapter(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER.fullmatch(value):
            raise ValueError("adapter must be a safe lowercase identifier")
        return value

    @field_validator("extensions")
    @classmethod
    def _extensions(cls, value: dict[str, Any]) -> dict[str, Any]:
        for namespace, payload in value.items():
            if not _NAMESPACE.fullmatch(namespace):
                raise ValueError(
                    "extension keys must be explicit dotted lowercase namespaces"
                )
            if namespace == "bfcl" or namespace.startswith("bfcl."):
                raise ValueError("the reserved bfcl namespace cannot be extended")
            _validate_json_value(payload, path=f"extensions.{namespace}")
        canonical = dict(sorted(value.items()))
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > _MAX_EXTENSION_BYTES:
            raise ValueError(
                f"extensions exceed the {_MAX_EXTENSION_BYTES}-byte limit"
            )
        return canonical

    @model_validator(mode="after")
    def _extension_authority_boundary(self) -> SourceDeclaration:
        # The recursive validator handles payload keys. This explicit model-level
        # check documents that extensions never affect adapter selection.
        if self.adapter is not None and self.adapter.startswith("extension."):
            raise ValueError("extensions cannot select or define an adapter")
        return self

    @property
    def digest(self) -> str:
        return str(sha256_json(self.model_dump(mode="json")))


@dataclass(frozen=True)
class AdapterRegistration:
    """Static metadata only; no module path, factory, callable, or user code."""

    adapter_id: str
    declaration_field: str
    descriptor_kind: str

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("declaration_field", self.declaration_field),
            ("descriptor_kind", self.descriptor_kind),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise AdapterResolutionError(
                    "adapter_registry_invalid",
                    f"{name} {value!r} is not a safe identifier",
                )
        if self.declaration_field not in _SOURCE_FIELDS:
            raise AdapterResolutionError(
                "adapter_registry_invalid",
                f"declaration field {self.declaration_field!r} is not built in",
            )


@dataclass(frozen=True)
class ResolvedAdapter:
    """A deterministic selection record, not an initialized adapter instance."""

    registration: AdapterRegistration
    source: SourceLocation
    declaration_digest: str
    extensions: Mapping[str, Any]

    @property
    def adapter_id(self) -> str:
        return self.registration.adapter_id

    @property
    def descriptor_kind(self) -> str:
        return self.registration.descriptor_kind


class AdapterRegistry:
    """Immutable registry whose matching logic reads declaration fields only."""

    __slots__ = ("_locked", "_registrations")
    _locked: bool
    _registrations: tuple[AdapterRegistration, ...]

    def __init__(self, registrations: Sequence[AdapterRegistration]) -> None:
        canonical = tuple(
            sorted(registrations, key=lambda item: item.adapter_id)
        )
        adapter_ids = [item.adapter_id for item in canonical]
        fields = [item.declaration_field for item in canonical]
        if (
            not canonical
            or len(adapter_ids) != len(set(adapter_ids))
            or len(fields) != len(set(fields))
        ):
            raise AdapterResolutionError(
                "adapter_registry_invalid",
                "registrations must be non-empty with unique adapter ids and fields",
            )
        object.__setattr__(self, "_registrations", canonical)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("adapter registry is immutable")
        object.__setattr__(self, name, value)

    @property
    def registrations(self) -> tuple[AdapterRegistration, ...]:
        return self._registrations

    def resolve(self, declaration: SourceDeclaration) -> ResolvedAdapter:
        # Revalidate even model instances: Pydantic's frozen setting is shallow,
        # so a caller may otherwise mutate a nested extension dictionary.
        try:
            declaration = SourceDeclaration.model_validate(
                declaration.model_dump(mode="python")
            )
        except ValueError as exc:
            raise AdapterResolutionError(
                "source_declaration_invalid",
                str(exc),
            ) from exc
        allowlisted = {
            registration.adapter_id: registration
            for registration in self._registrations
        }
        if (
            declaration.adapter is not None
            and declaration.adapter not in allowlisted
        ):
            raise AdapterResolutionError(
                "adapter_not_allowlisted",
                f"adapter {declaration.adapter!r} is not in the built-in allowlist",
            )

        candidates = [
            (registration, source)
            for registration in self._registrations
            if (
                source := getattr(declaration, registration.declaration_field)
            )
            is not None
        ]
        if not candidates:
            raise AdapterResolutionError(
                "adapter_not_detected",
                "declaration contains no recognized source block",
            )
        if len(candidates) > 1:
            raise AdapterResolutionError(
                "adapter_detection_ambiguous",
                "declaration matches multiple built-in adapters: "
                + ", ".join(item.adapter_id for item, _ in candidates),
            )

        registration, source = candidates[0]
        if (
            declaration.adapter is not None
            and declaration.adapter != registration.adapter_id
        ):
            raise AdapterResolutionError(
                "adapter_source_mismatch",
                f"explicit adapter {declaration.adapter!r} does not match "
                f"source block {registration.declaration_field!r}",
            )
        return ResolvedAdapter(
            registration=registration,
            source=source,
            declaration_digest=declaration.digest,
            # Copy so a caller cannot mutate the validated declaration through
            # the resolution record.
            extensions=_deep_freeze_json(declaration.extensions),
        )


BUILTIN_ADAPTER_REGISTRY = AdapterRegistry(
    (
        AdapterRegistration(
            adapter_id="http_package",
            declaration_field="http_package",
            descriptor_kind="http_package",
        ),
        AdapterRegistration(
            adapter_id="local_python",
            declaration_field="local_python",
            descriptor_kind="local_python",
        ),
        AdapterRegistration(
            adapter_id="mcp_mode_a",
            declaration_field="mcp_mode_a",
            descriptor_kind="mcp_mode_a",
        ),
    )
)


def resolve_source_adapter(
    declaration: SourceDeclaration | Mapping[str, Any],
) -> ResolvedAdapter:
    """Resolve through the fixed built-in allowlist and nothing else."""

    try:
        validated = SourceDeclaration.model_validate(
            declaration.model_dump(mode="python")
            if isinstance(declaration, SourceDeclaration)
            else declaration
        )
    except ValueError as exc:
        raise AdapterResolutionError(
            "source_declaration_invalid",
            str(exc),
        ) from exc
    return BUILTIN_ADAPTER_REGISTRY.resolve(validated)
