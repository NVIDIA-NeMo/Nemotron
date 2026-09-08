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

"""Transport-neutral contract for BFCL assisted-authoring sources.

An adapter declares what it can do.  It never declares what BFCL has proved about
it: certification is a separate, BFCL-owned artifact referenced by the evidence
schema.  Keeping those two concepts apart prevents a source plug-in from granting
itself publication authority.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

if TYPE_CHECKING:
    from nemotron.steps.byob.runtime.source_adapters.evidence import UnsignedSourceEvidence


ADAPTER_CONTRACT_VERSION: Literal[
    "bfcl-source-adapter-v1"
] = "bfcl-source-adapter-v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class AdapterContractError(ValueError):
    """Raised when an adapter cannot satisfy the shared authoring contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterCapability(str, Enum):
    """Operations an adapter may offer to BFCL-owned orchestration."""

    DESCRIBE_TOOLS = "describe_tools"
    DESCRIBE_STATE = "describe_state"
    PIN_IDENTITY = "pin_identity"
    OBSERVE = "observe"
    RESET_STATE = "reset_state"
    GET_STATE = "get_state"


class FixtureAccessKind(str, Enum):
    """How reviewed fixture data can cross the source boundary."""

    NONE = "none"
    READ_ONLY = "read_only"
    PUSHED = "pushed"
    SNAPSHOT = "snapshot"


class ProbeSafetyKind(str, Enum):
    """Maximum observation authority declared by the adapter."""

    IDENTITY_ONLY = "identity_only"
    READ_ONLY = "read_only"
    RESET_ISOLATED = "reset_isolated"


class CleanupKind(str, Enum):
    """Resource boundary BFCL must close after an observation."""

    NONE = "none"
    EPISODE = "episode"
    SESSION = "session"
    PROCESS = "process"


class FixtureAccessPolicy(_StrictModel):
    kind: FixtureAccessKind
    supports_redaction: StrictBool = False


class ProbeSafetyPolicy(_StrictModel):
    kind: ProbeSafetyKind
    max_calls: StrictInt
    timeout_s: StrictFloat

    @field_validator("max_calls")
    @classmethod
    def _positive_calls(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_calls must be positive")
        return value

    @field_validator("timeout_s")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_s must be positive")
        return value


class CleanupSemantics(_StrictModel):
    kind: CleanupKind
    timeout_s: StrictFloat

    @field_validator("timeout_s")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_s must be positive")
        return value


class AdapterDescriptor(_StrictModel):
    """Strict declarations supplied by an adapter implementation.

    There is deliberately no certification tier or probe verdict here.  A BFCL
    verifier will bind those to this descriptor's digest in a separate report.
    """

    contract_version: Literal["bfcl-source-adapter-v1"]
    kind: StrictStr
    implementation_name: StrictStr
    implementation_version: StrictStr
    capabilities: tuple[AdapterCapability, ...]
    fixture_access: FixtureAccessPolicy
    probe_safety: ProbeSafetyPolicy
    cleanup: CleanupSemantics

    @field_validator("kind", "implementation_name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(
                "adapter identifiers must be lowercase dotted, dashed, or underscored names"
            )
        return value

    @field_validator("implementation_version")
    @classmethod
    def _nonempty_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("implementation_version must be non-empty")
        return value

    @field_validator("capabilities")
    @classmethod
    def _canonical_capabilities(
        cls,
        value: tuple[AdapterCapability, ...],
    ) -> tuple[AdapterCapability, ...]:
        if not value:
            raise ValueError("an adapter must declare at least one capability")
        if len(value) != len(set(value)):
            raise ValueError("adapter capabilities must be unique")
        if tuple(sorted(value, key=lambda item: item.value)) != value:
            raise ValueError("adapter capabilities must be sorted by their serialized name")
        return value


class AdapterRequest(_StrictModel):
    """Digest-bound inputs supplied by orchestration to one adapter call."""

    request_version: Literal["bfcl-source-adapter-request-v1"]
    source_declaration_digest: StrictStr
    workspace_id: StrictStr

    @field_validator("source_declaration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("source_declaration_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("workspace_id")
    @classmethod
    def _workspace_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("workspace_id must be a safe lowercase identifier")
        return value


@runtime_checkable
class OracleSourceAdapter(Protocol):
    """Narrow interface implemented by an assisted-authoring source.

    ``collect_evidence`` returns unsigned observations.  It cannot return a
    certification report, approval, Gold verdict, or frozen release.
    """

    @property
    def descriptor(self) -> AdapterDescriptor:
        """Return immutable declarations about this implementation."""

    def collect_evidence(self, request: AdapterRequest) -> UnsignedSourceEvidence:
        """Collect source evidence under the bounds in ``request``."""
