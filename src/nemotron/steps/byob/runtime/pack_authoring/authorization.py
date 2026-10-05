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

"""Authorization to expose one exact assisted-authoring input to a model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import (
    DomainBriefRedactionReport,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import SourceEvidenceDocument
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutRedactionReport,
)

EXPOSURE_AUTHORIZATION_VERSION: Literal["bfcl-model-exposure-authorization-v1"] = (
    "bfcl-model-exposure-authorization-v1"
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_AUTHORIZER = re.compile(r"^[^\s@]+@[^\s@]+$|^[a-zA-Z0-9][a-zA-Z0-9._-]{1,127}$")
_MAX_BYTES = 64 * 1024


class AuthorizationError(ValueError):
    """Raised when model exposure has not been authorized exactly."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExposureSubject(_StrictModel):
    evidence_digest: StrictStr
    resolved_authoring_config_digest: StrictStr | None = None
    domain_brief_content_digest: StrictStr
    domain_brief_source_digest: StrictStr
    domain_brief_redaction_report_digest: StrictStr
    held_out_decision_digest: StrictStr
    held_out_policy_digest: StrictStr | None
    held_out_redaction_report_digest: StrictStr

    @field_validator("*")
    @classmethod
    def _digests(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST.fullmatch(value):
            raise ValueError("exposure subject values must be sha256 digests")
        return value


class ExposureAuthorization(_StrictModel):
    schema_version: Literal["bfcl-model-exposure-authorization-v1"]
    mode: Literal["named_human", "organizational_policy"]
    subject: ExposureSubject
    authorized_by: StrictStr | None = None
    organizational_policy_digest: StrictStr | None = None
    authorization_digest: StrictStr

    @field_validator("authorized_by")
    @classmethod
    def _authorizer(cls, value: str | None) -> str | None:
        if value is not None and not _AUTHORIZER.fullmatch(value):
            raise ValueError("exposure authorizer must be a stable name or email")
        return value

    @field_validator("organizational_policy_digest", "authorization_digest")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST.fullmatch(value):
            raise ValueError("authorization digest fields must be sha256 values")
        return value

    @model_validator(mode="after")
    def _mode_and_digest(self) -> ExposureAuthorization:
        if self.mode == "named_human":
            if self.authorized_by is None or self.organizational_policy_digest is not None:
                raise ValueError(
                    "named-human authorization requires only authorized_by"
                )
        elif self.authorized_by is not None or self.organizational_policy_digest is None:
            raise ValueError(
                "organizational authorization requires only its policy digest"
            )
        unsigned = self.model_dump(mode="json", exclude={"authorization_digest"})
        if self.authorization_digest != sha256_json(unsigned):
            raise ValueError("model exposure authorization digest mismatch")
        return self


def build_exposure_subject(
    evidence: SourceEvidenceDocument,
    *,
    domain_brief_report: DomainBriefRedactionReport,
    held_out_redaction_report: HeldOutRedactionReport,
    resolved_authoring_config_digest: str | None = None,
) -> ExposureSubject:
    brief = evidence.domain_brief
    held_out = evidence.fixtures.held_out
    if (
        domain_brief_report.record_digest != brief.redaction_report_digest
        or held_out_redaction_report.decision_digest != held_out.decision_digest
        or held_out_redaction_report.policy_digest != held_out.policy_digest
        or held_out_redaction_report.evidence_digest != evidence.bundle_digest
    ):
        raise AuthorizationError(
            "cannot authorize model exposure for mismatched evidence reports"
        )
    return ExposureSubject(
        evidence_digest=evidence.bundle_digest,
        resolved_authoring_config_digest=resolved_authoring_config_digest,
        domain_brief_content_digest=brief.content_digest,
        domain_brief_source_digest=brief.source_digest,
        domain_brief_redaction_report_digest=domain_brief_report.record_digest,
        held_out_decision_digest=held_out.decision_digest,
        held_out_policy_digest=held_out.policy_digest,
        held_out_redaction_report_digest=held_out_redaction_report.report_digest,
    )


def _build_authorization(
    *,
    mode: Literal["named_human", "organizational_policy"],
    subject: ExposureSubject,
    authorized_by: str | None = None,
    organizational_policy_digest: str | None = None,
) -> ExposureAuthorization:
    document: dict[str, Any] = {
        "schema_version": EXPOSURE_AUTHORIZATION_VERSION,
        "mode": mode,
        "subject": subject.model_dump(mode="json"),
        "authorized_by": authorized_by,
        "organizational_policy_digest": organizational_policy_digest,
    }
    document["authorization_digest"] = sha256_json(document)
    return ExposureAuthorization.model_validate(document)


def authorize_model_exposure_by_human(
    subject: ExposureSubject,
    *,
    authorized_by: str,
) -> ExposureAuthorization:
    return _build_authorization(
        mode="named_human",
        subject=subject,
        authorized_by=authorized_by,
    )


def authorize_model_exposure_by_policy(
    subject: ExposureSubject,
    *,
    organizational_policy_digest: str,
) -> ExposureAuthorization:
    return _build_authorization(
        mode="organizational_policy",
        subject=subject,
        organizational_policy_digest=organizational_policy_digest,
    )


def verify_exposure_authorization(
    authorization: ExposureAuthorization,
    *,
    expected_subject: ExposureSubject,
    expected_organizational_policy_digest: str | None = None,
) -> None:
    if authorization.subject != expected_subject:
        raise AuthorizationError(
            "model exposure authorization does not cover the current input"
        )
    if authorization.mode == "organizational_policy":
        if (
            expected_organizational_policy_digest is None
            or authorization.organizational_policy_digest
            != expected_organizational_policy_digest
        ):
            raise AuthorizationError(
                "model exposure organizational policy digest mismatch"
            )


def load_exposure_authorization(path: Path) -> ExposureAuthorization:
    source = path.resolve()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuthorizationError(
                    f"model exposure authorization repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = source.read_bytes()
        if len(raw) > _MAX_BYTES:
            raise AuthorizationError(
                "model exposure authorization exceeds 64 KiB"
            )
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        return ExposureAuthorization.model_validate(document)
    except AuthorizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuthorizationError(
            f"cannot load model exposure authorization {source}: {exc}"
        ) from exc


def write_exposure_authorization(
    authorization: ExposureAuthorization,
    path: Path,
) -> Path:
    return write_canonical_json(authorization.model_dump(mode="json"), path)
