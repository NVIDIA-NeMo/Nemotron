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

"""Allowlisted, secret-free structured events for BFCL authoring."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

EVENT_VERSION: Literal["bfcl-authoring-event-v1"] = "bfcl-authoring-event-v1"
EVENT_FILE_NAME = "authoring_events.jsonl"
AuthoringEventType = Literal[
    "adapter_identity_bound",
    "certification_verified",
    "refusal_recorded",
    "revision_authorized",
    "validation_verdict",
    "release_frozen",
]
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AuthoringEventError(ValueError):
    pass


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterIdentityPayload(_Payload):
    adapter_kind: Literal["local_python", "http_package", "mcp_mode_a"]
    source_identity_digest: StrictStr
    evidence_bundle_digest: StrictStr
    descriptor_digest: StrictStr | None = None
    authorization_context_digest: StrictStr | None = None

    @field_validator(
        "source_identity_digest",
        "evidence_bundle_digest",
        "descriptor_digest",
        "authorization_context_digest",
    )
    @classmethod
    def _digests(cls, value: str | None) -> str | None:
        return _require_digest(value) if value is not None else None


class CertificationPayload(_Payload):
    adapter_kind: Literal["local_python", "http_package", "mcp_mode_a"]
    attained_tier: Literal["A0", "A1", "A2"]
    required_tier: Literal["A0", "A1", "A2"]
    profile_id: StrictStr
    report_digest: StrictStr

    @field_validator("profile_id")
    @classmethod
    def _profile(cls, value: str) -> str:
        return _require_code(value)

    @field_validator("report_digest")
    @classmethod
    def _report_digest(cls, value: str) -> str:
        return _require_digest(value)


class RefusalPayload(_Payload):
    refusal_digest: StrictStr | None = None
    primary_classification: StrictStr
    finding_codes: tuple[StrictStr, ...] = ()
    reason_codes: tuple[StrictStr, ...] = ()

    @field_validator("refusal_digest")
    @classmethod
    def _refusal_digest(cls, value: str | None) -> str | None:
        return _require_digest(value) if value is not None else None

    @field_validator("primary_classification")
    @classmethod
    def _classification(cls, value: str) -> str:
        return _require_code(value)

    @field_validator("finding_codes", "reason_codes")
    @classmethod
    def _codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("event codes must be sorted and unique")
        return tuple(_require_code(item) for item in value)


class RevisionAuthorizationPayload(_Payload):
    authorization_digest: StrictStr
    refusal_digest: StrictStr
    parent_session_digest: StrictStr
    action: StrictStr
    authorization_code: StrictStr

    @field_validator(
        "authorization_digest",
        "refusal_digest",
        "parent_session_digest",
    )
    @classmethod
    def _digests(cls, value: str) -> str:
        return _require_digest(value)

    @field_validator("action", "authorization_code")
    @classmethod
    def _codes(cls, value: str) -> str:
        return _require_code(value)


class ValidationVerdictPayload(_Payload):
    stage: Literal["review", "publication"]
    tier: Literal["prototype", "silver", "gold"]
    gold_eligible: StrictBool
    pack_fingerprint: StrictStr
    validation_report_digest: StrictStr
    validation_config_fingerprint: StrictStr | None = None

    @field_validator(
        "pack_fingerprint",
        "validation_report_digest",
        "validation_config_fingerprint",
    )
    @classmethod
    def _digests(cls, value: str | None) -> str | None:
        return _require_digest(value) if value is not None else None


class ReleaseFrozenPayload(_Payload):
    adapter_kind: Literal["local_python", "http_package", "mcp_mode_a"]
    manifest_digest: StrictStr
    frozen_pack_fingerprint: StrictStr
    review_packet_digest: StrictStr
    review_approval_digest: StrictStr

    @field_validator(
        "manifest_digest",
        "frozen_pack_fingerprint",
        "review_packet_digest",
        "review_approval_digest",
    )
    @classmethod
    def _digests(cls, value: str) -> str:
        return _require_digest(value)


EventPayload = (
    AdapterIdentityPayload
    | CertificationPayload
    | RefusalPayload
    | RevisionAuthorizationPayload
    | ValidationVerdictPayload
    | ReleaseFrozenPayload
)
_PAYLOAD_TYPES: Mapping[str, type[_Payload]] = {
    "adapter_identity_bound": AdapterIdentityPayload,
    "certification_verified": CertificationPayload,
    "refusal_recorded": RefusalPayload,
    "revision_authorized": RevisionAuthorizationPayload,
    "validation_verdict": ValidationVerdictPayload,
    "release_frozen": ReleaseFrozenPayload,
}


class AuthoringEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bfcl-authoring-event-v1"]
    event_type: AuthoringEventType
    tenant_id: StrictStr
    run_id: StrictStr
    session_digest: StrictStr | None
    emitted_at: StrictStr
    payload: dict[str, Any]
    event_digest: StrictStr

    def model_post_init(self, __context: Any) -> None:
        _require_identifier(self.tenant_id)
        _require_identifier(self.run_id)
        if self.session_digest is not None:
            _require_digest(self.session_digest)
        try:
            timestamp = datetime.fromisoformat(self.emitted_at)
        except ValueError as exc:
            raise ValueError("event timestamp must be ISO-8601") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")
        payload_type = _PAYLOAD_TYPES[self.event_type]
        payload_type.model_validate(self.payload)
        _require_digest(self.event_digest)
        unsigned = self.model_dump(mode="json", exclude={"event_digest"})
        if self.event_digest != sha256_json(unsigned):
            raise ValueError("authoring event digest mismatch")


class AuthoringEventSink(Protocol):
    def emit(self, event: AuthoringEvent) -> None: ...


class NullAuthoringEventSink:
    def emit(self, event: AuthoringEvent) -> None:
        del event


class FileAuthoringEventSink:
    """Append complete canonical event lines under an exclusive advisory lock."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def emit(self, event: AuthoringEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                event.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise AuthoringEventError("incomplete structured event append")
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def build_authoring_event(
    event_type: AuthoringEventType,
    payload: EventPayload,
    *,
    tenant_id: str,
    run_id: str,
    session_digest: str | None,
    emitted_at: datetime | None = None,
) -> AuthoringEvent:
    expected = _PAYLOAD_TYPES[event_type]
    if type(payload) is not expected:
        raise AuthoringEventError(
            f"payload type {type(payload).__name__} does not match {event_type}"
        )
    timestamp = emitted_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise AuthoringEventError("event timestamp must be timezone-aware")
    unsigned = {
        "schema_version": EVENT_VERSION,
        "event_type": event_type,
        "tenant_id": tenant_id,
        "run_id": run_id,
        "session_digest": session_digest,
        "emitted_at": timestamp.isoformat(),
        "payload": payload.model_dump(mode="json"),
    }
    return AuthoringEvent.model_validate(
        {**unsigned, "event_digest": sha256_json(unsigned)}
    )


def emit_authoring_event(
    sink: AuthoringEventSink,
    event_type: AuthoringEventType,
    payload: EventPayload,
    *,
    tenant_id: str,
    run_id: str,
    session_digest: str | None,
) -> AuthoringEvent:
    event = build_authoring_event(
        event_type,
        payload,
        tenant_id=tenant_id,
        run_id=run_id,
        session_digest=session_digest,
    )
    sink.emit(event)
    return event


def load_authoring_events(path: Path) -> tuple[AuthoringEvent, ...]:
    source = path.resolve()
    events: list[AuthoringEvent] = []
    try:
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                raise AuthoringEventError(
                    f"empty structured event line {line_number}"
                )
            events.append(
                AuthoringEvent.model_validate(
                    json.loads(line, object_pairs_hook=_unique_mapping)
                )
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, AuthoringEventError):
            raise
        raise AuthoringEventError(f"invalid structured event stream: {exc}") from exc
    return tuple(events)


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthoringEventError(f"duplicate structured event key {key!r}")
        result[key] = value
    return result


def _require_digest(value: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError("event digest field must be sha256:<64 lowercase hex>")
    return value


def _require_code(value: str) -> str:
    if _CODE.fullmatch(value) is None:
        raise ValueError("event code must be a stable lowercase identifier")
    return value


def _require_identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("event namespace must be a safe identifier")
    return value
