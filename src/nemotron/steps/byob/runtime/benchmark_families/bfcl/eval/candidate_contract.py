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

"""One native function-calling request and what came back.

The client contract deliberately stops at the transport boundary. It preserves
the assistant message and every raw function argument exactly as the provider
returned them; the scoring component, not this module, decides whether a call is
correct.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    FrozenDict,
    freeze_json,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

CANDIDATE_CLIENT_CONTRACT_VERSION: Final = "1.0"
CANDIDATE_IO_CACHE_FILE: Final = "candidate_io_cache.jsonl"

ArgumentStatus = Literal["valid_object", "invalid_json", "json_not_object", "missing", "wrong_type"]
CallStatus = Literal[
    "completed",
    "malformed_response",
    "provider_rejected",
    "rate_limited",
    "timeout",
    "transport_error",
    "provider_unavailable",
    "authentication_failed",
    "cancelled",
    "retry_exhausted",
]


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class CandidateRequest(_Frozen):
    """The exact request body sent for one candidate assistant turn."""

    schema_version: Literal["1.0"] = CANDIDATE_CLIENT_CONTRACT_VERSION
    request_id: StrictStr
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    task_id: StrictStr
    turn_index: StrictInt = Field(ge=0)
    provider: StrictStr
    provider_api_version: StrictStr
    base_url: StrictStr
    model: StrictStr
    body: FrozenDict
    request_hash: ContentHash
    request_body_hash: ContentHash

    @model_validator(mode="before")
    @classmethod
    def _freeze_body(cls, value: Any) -> Any:
        if isinstance(value, dict) and "body" in value:
            value = dict(value)
            validate_json_value(value["body"], label="candidate request body")
            value["body"] = freeze_json(value["body"])
        return value

    @model_validator(mode="after")
    def _hashes_match(self) -> CandidateRequest:
        document = thaw_json(self.body)
        if self.request_body_hash != _sha256_json(document):
            raise ValueError("request_body_hash does not identify the request body")
        identity = self.identity_payload(document)
        if self.request_hash != _sha256_json(identity):
            raise ValueError("request_hash does not identify every semantic request input")
        return self

    def identity_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "candidate_alias": self.candidate_alias,
            "canonical_model_identity": self.canonical_model_identity,
            "task_id": self.task_id,
            "turn_index": self.turn_index,
            "provider": self.provider,
            "provider_api_version": self.provider_api_version,
            "base_url": self.base_url,
            "model": self.model,
            "body": body if body is not None else thaw_json(self.body),
        }

    def as_document(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "request_hash": self.request_hash,
            "request_body_hash": self.request_body_hash,
        }


class CandidateToolCall(_Frozen):
    """One provider tool call, including its unmodified argument string."""

    index: StrictInt = Field(ge=0)
    id: StrictStr | None
    type: StrictStr | None
    function_name: StrictStr | None
    raw_arguments: Any = None
    parsed_arguments: FrozenDict | None = None
    arguments_status: ArgumentStatus

    @model_validator(mode="before")
    @classmethod
    def _freeze_arguments(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            if isinstance(value.get("parsed_arguments"), dict):
                value["parsed_arguments"] = freeze_json(value["parsed_arguments"])
            value["raw_arguments"] = freeze_json(value.get("raw_arguments"))
        return value

    def as_document(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "type": self.type,
            "function_name": self.function_name,
            "raw_arguments": thaw_json(self.raw_arguments),
            "parsed_arguments": (
                thaw_json(self.parsed_arguments) if self.parsed_arguments is not None else None
            ),
            "arguments_status": self.arguments_status,
        }


class CandidateResponse(_Frozen):
    """A usable provider envelope, whether or not its function calls are valid."""

    schema_version: Literal["1.0"] = CANDIDATE_CLIENT_CONTRACT_VERSION
    provider_response_id: StrictStr | None = None
    provider_request_id: StrictStr | None = None
    assistant_content: Any = None
    tool_calls: tuple[CandidateToolCall, ...] = ()
    finish_reason: StrictStr | None = None
    usage: FrozenDict | None = None
    selected_attempt: StrictInt = Field(ge=0)
    raw_response_hash: ContentHash

    @model_validator(mode="before")
    @classmethod
    def _freeze_usage(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            if isinstance(value.get("usage"), dict):
                value["usage"] = freeze_json(value["usage"])
            value["assistant_content"] = freeze_json(value.get("assistant_content"))
        return value

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "provider_response_id": self.provider_response_id,
            "provider_request_id": self.provider_request_id,
            "assistant_content": thaw_json(self.assistant_content),
            "tool_calls": [call.as_document() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": thaw_json(self.usage) if self.usage is not None else None,
            "selected_attempt": self.selected_attempt,
            "raw_response_hash": self.raw_response_hash,
        }

    @property
    def response_hash(self) -> str:
        return _sha256_json(self.semantic_payload())

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "schema_version": self.schema_version, "response_hash": self.response_hash}


class CandidateAttempt(_Frozen):
    """One HTTP observation made while satisfying a logical request."""

    attempt_index: StrictInt = Field(ge=0)
    observed_at: StrictStr
    status: CallStatus
    retryable: StrictBool
    http_status: StrictInt | None = None
    latency_s: StrictFloat = Field(ge=0)
    # Kept only for HTTP 200 model output. Provider error bodies are arbitrary
    # diagnostics and may echo credentials, so the client records their shape,
    # never their bytes.
    raw_response: StrictStr | None = None
    raw_response_hash: ContentHash | None = None
    provider_request_id: StrictStr | None = None
    diagnostic: StrictStr | None = None
    retry_after_s: StrictFloat | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _raw_hash_matches(self) -> CandidateAttempt:
        if self.raw_response is None:
            return self
        if self.raw_response_hash != _sha256_bytes(self.raw_response.encode("utf-8")):
            raise ValueError("raw_response_hash does not identify raw_response")
        return self

    def as_document(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def as_summary(self) -> dict[str, Any]:
        """The attempt minus the model bytes its own durable record already holds.

        A logical call cites its attempts, and citing them by hash keeps one copy
        of every response body in the cache instead of one per citation.
        """
        return {key: value for key, value in self.as_document().items() if key != "raw_response"}


class CandidateCallOutcome(_Frozen):
    """The complete result of one logical candidate call."""

    schema_version: Literal["1.0"] = CANDIDATE_CLIENT_CONTRACT_VERSION
    request_hash: ContentHash
    status: CallStatus
    attempts: tuple[CandidateAttempt, ...]
    response: CandidateResponse | None = None
    replayed: StrictBool = False

    @model_validator(mode="after")
    def _coherent(self) -> CandidateCallOutcome:
        if not self.attempts:
            raise ValueError("a candidate outcome must carry at least one attempt")
        expected = tuple(range(len(self.attempts)))
        if tuple(attempt.attempt_index for attempt in self.attempts) != expected:
            raise ValueError("candidate attempts must be contiguous and zero-based")
        if (self.status == "completed") != (self.response is not None):
            raise ValueError("exactly a completed outcome carries a parsed response")
        if self.status == "cancelled":
            # Cancellation is the runner stopping, not the candidate answering.
            # It stays attempt evidence so a resumed run never reads a keyboard
            # interrupt as this task's observed behaviour.
            raise ValueError("a cancelled call has no outcome, only attempt evidence")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_hash": self.request_hash,
            "status": self.status,
            "attempts": [attempt.as_summary() for attempt in self.attempts],
            # Keep the nested object directly model-validatable for cache replay.
            # ``CandidateResponse.as_document`` adds its derived response_hash,
            # which is artifact evidence rather than an input field.
            "response": self.response.model_dump(mode="json") if self.response is not None else None,
        }

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "replayed": self.replayed}


def parse_function_arguments(value: Any) -> tuple[Any, FrozenDict | None, ArgumentStatus]:
    """Parse once without repair, preserving the provider's original value."""
    if value is None:
        return None, None, "missing"
    if not isinstance(value, str):
        return value, None, "wrong_type"
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError):
        return value, None, "invalid_json"
    if not isinstance(parsed, dict):
        return value, None, "json_not_object"
    validate_json_value(parsed, label="candidate function arguments")
    return value, freeze_json(parsed), "valid_object"


__all__ = [
    "ArgumentStatus",
    "CANDIDATE_CLIENT_CONTRACT_VERSION",
    "CANDIDATE_IO_CACHE_FILE",
    "CallStatus",
    "CandidateAttempt",
    "CandidateCallOutcome",
    "CandidateRequest",
    "CandidateResponse",
    "CandidateToolCall",
    "parse_function_arguments",
]
