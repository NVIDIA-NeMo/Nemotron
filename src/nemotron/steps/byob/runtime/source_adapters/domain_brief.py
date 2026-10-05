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

"""Bounded, reviewable domain intent for assisted authoring.

A domain brief is useful context, not oracle truth.  Its prose is scanned and
fenced as untrusted data, and every transformation is represented by two
digests: one for the user-owned source bytes and one for the sanitized text the
model may see.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json, sha256_text
from nemotron.steps.byob.runtime.pack_authoring.untrusted_text import (
    TextFinding,
    blocking,
    quote_untrusted,
    scan_text,
    sorted_findings,
)

DOMAIN_BRIEF_VERSION: Literal["bfcl-domain-brief-v1"] = "bfcl-domain-brief-v1"
DOMAIN_BRIEF_REDACTION_VERSION: Literal["bfcl-domain-brief-redaction-v1"] = (
    "bfcl-domain-brief-redaction-v1"
)
DEFAULT_MAX_BYTES = 16 * 1024
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_LANGUAGE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_SKELETON_MARKER = "[BFCL-SKELETON:"


class DomainBriefError(ValueError):
    """Raised when domain intent is unsafe or impossible to review exactly."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BriefFinding(_StrictModel):
    location: StrictStr
    code: StrictStr
    detail: StrictStr
    severity: Literal["review"]

    @classmethod
    def from_text_finding(cls, finding: TextFinding) -> BriefFinding:
        if finding.severity != "review":
            raise DomainBriefError("only review findings belong in a domain brief record")
        return cls(
            location=finding.location,
            code=finding.code,
            detail=finding.detail,
            severity="review",
        )


class RedactionSummary(_StrictModel):
    label: StrictStr
    replacements: StrictInt

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        if not _LABEL.fullmatch(value):
            raise ValueError("redaction labels must be safe lowercase identifiers")
        return value

    @field_validator("replacements")
    @classmethod
    def _replacements(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("redaction summaries must record a positive replacement count")
        return value


class DomainBriefRedactionReport(_StrictModel):
    schema_version: Literal["bfcl-domain-brief-redaction-v1"]
    source_digest: StrictStr
    sanitized_digest: StrictStr
    redactions: tuple[RedactionSummary, ...]
    advisory: tuple[BriefFinding, ...]
    record_digest: StrictStr

    @field_validator("source_digest", "sanitized_digest", "record_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("domain brief digests must be lowercase SHA-256 values")
        return value

    @field_validator("redactions")
    @classmethod
    def _canonical_redactions(
        cls,
        value: tuple[RedactionSummary, ...],
    ) -> tuple[RedactionSummary, ...]:
        labels = [item.label for item in value]
        if len(labels) != len(set(labels)):
            raise ValueError("domain brief redaction labels must be unique")
        if labels != sorted(labels):
            raise ValueError("domain brief redactions must be sorted by label")
        return value

    @field_validator("advisory")
    @classmethod
    def _canonical_advisory(
        cls,
        value: tuple[BriefFinding, ...],
    ) -> tuple[BriefFinding, ...]:
        keys = [(item.location, item.code, item.detail) for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("domain brief advisory findings must be unique")
        if keys != sorted(keys):
            raise ValueError("domain brief advisory findings must be sorted")
        return value

    @model_validator(mode="after")
    def _verify_record_digest(self) -> DomainBriefRedactionReport:
        unsigned = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("domain brief redaction report digest mismatch")
        return self


class DomainBriefEvidence(_StrictModel):
    """The exact sanitized brief allowed into a model request."""

    schema_version: Literal["bfcl-domain-brief-v1"]
    language: StrictStr
    untrusted_text: StrictStr
    source_digest: StrictStr
    content_digest: StrictStr
    redaction_report_digest: StrictStr

    @field_validator("language")
    @classmethod
    def _language(cls, value: str) -> str:
        if not _LANGUAGE.fullmatch(value):
            raise ValueError("domain brief language must be a BCP-47-style tag")
        return value

    @field_validator("source_digest", "content_digest", "redaction_report_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("domain brief digests must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def _verify_content_digest(self) -> DomainBriefEvidence:
        encoded = self.untrusted_text.encode("utf-8")
        if not self.untrusted_text.strip():
            raise ValueError("domain brief must contain non-whitespace text")
        if _SKELETON_MARKER in self.untrusted_text:
            raise ValueError(
                "domain brief still contains BFCL skeleton placeholders; replace every "
                "BFCL-SKELETON block before intake"
            )
        if len(encoded) > DEFAULT_MAX_BYTES:
            raise ValueError(
                f"domain brief exceeds the {DEFAULT_MAX_BYTES}-byte persisted limit"
            )
        refused = blocking(scan_text(self.untrusted_text, "domain_brief"))
        if refused:
            raise ValueError(
                "domain brief contains review-blocking text: "
                + "; ".join(item.code for item in refused)
            )
        if self.content_digest != sha256_text(self.untrusted_text):
            raise ValueError("domain brief content digest mismatch")
        return self

    def model_input(self) -> dict[str, str]:
        """Return model input with prose fenced as data, never instructions."""

        return {
            "schema_version": self.schema_version,
            "language": self.language,
            "content_digest": self.content_digest,
            "redaction_report_digest": self.redaction_report_digest,
            "content": quote_untrusted(self.untrusted_text),
        }


def _source_digest(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _validate_redactions(redactions: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    seen_values: set[str] = set()
    for label, value in sorted(redactions.items()):
        if not _LABEL.fullmatch(label):
            raise DomainBriefError(
                f"redaction label {label!r} must be a safe lowercase identifier"
            )
        if not value:
            raise DomainBriefError(f"redaction {label!r} must not be empty")
        if value in seen_values:
            raise DomainBriefError("one redaction literal must not have multiple labels")
        seen_values.add(value)
        normalized[label] = value
    return normalized


def _redact_once(text: str, redactions: Mapping[str, str]) -> tuple[str, tuple[RedactionSummary, ...]]:
    if not redactions:
        return text, ()
    by_value = {value: label for label, value in redactions.items()}
    pattern = re.compile(
        "|".join(
            re.escape(value)
            for value in sorted(by_value, key=lambda item: (-len(item), item))
        )
    )
    counts = {label: 0 for label in redactions}

    def replace(match: re.Match[str]) -> str:
        label = by_value[match.group(0)]
        counts[label] += 1
        return f"<redacted:{label}>"

    sanitized = pattern.sub(replace, text)
    summaries = tuple(
        RedactionSummary(label=label, replacements=count)
        for label, count in sorted(counts.items())
        if count
    )
    return sanitized, summaries


def load_domain_brief(
    path: Path,
    *,
    language: str,
    redactions: Mapping[str, str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[DomainBriefEvidence, DomainBriefRedactionReport]:
    """Load, sanitize, scan, and bind one user-owned domain brief."""

    if max_bytes <= 0:
        raise DomainBriefError("max_bytes must be positive")
    source = path.resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise DomainBriefError(f"cannot read domain brief {source}: {exc}") from exc
    if len(raw) > max_bytes:
        raise DomainBriefError(
            f"domain brief {source} is {len(raw)} bytes; limit is {max_bytes}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DomainBriefError(f"domain brief {source} is not valid UTF-8") from exc
    if not text.strip():
        raise DomainBriefError("domain brief must contain non-whitespace text")
    if _SKELETON_MARKER in text:
        raise DomainBriefError(
            "domain brief still contains BFCL skeleton placeholders; replace every "
            "BFCL-SKELETON block before intake"
        )

    declared = _validate_redactions(redactions or {})
    sanitized, summaries = _redact_once(text, declared)
    findings = sorted_findings(scan_text(sanitized, "domain_brief"))
    refused = blocking(findings)
    if refused:
        details = "; ".join(f"{item.code}: {item.detail}" for item in refused)
        raise DomainBriefError(f"domain brief cannot be reviewed safely: {details}")
    advisory = tuple(
        BriefFinding.from_text_finding(finding)
        for finding in findings
        if finding.severity == "review"
    )

    report_document = {
        "schema_version": DOMAIN_BRIEF_REDACTION_VERSION,
        "source_digest": _source_digest(raw),
        "sanitized_digest": sha256_text(sanitized),
        "redactions": [item.model_dump(mode="json") for item in summaries],
        "advisory": [item.model_dump(mode="json") for item in advisory],
    }
    report = DomainBriefRedactionReport.model_validate(
        {
            **report_document,
            "record_digest": sha256_json(report_document),
        }
    )
    brief = DomainBriefEvidence(
        schema_version=DOMAIN_BRIEF_VERSION,
        language=language,
        untrusted_text=sanitized,
        source_digest=report.source_digest,
        content_digest=report.sanitized_digest,
        redaction_report_digest=report.record_digest,
    )
    return brief, report


def load_domain_brief_redaction_report(
    path: Path,
    *,
    brief: DomainBriefEvidence,
    source_path: Path,
) -> DomainBriefRedactionReport:
    """Verify the report and original bytes behind persisted brief evidence."""

    import json

    source = path.resolve()

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DomainBriefError(
                    f"domain brief report repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
        report = DomainBriefRedactionReport.model_validate(document)
        original = source_path.resolve().read_bytes()
    except DomainBriefError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DomainBriefError(
            f"cannot verify domain brief report {source}: {exc}"
        ) from exc
    if report.record_digest != brief.redaction_report_digest:
        raise DomainBriefError("domain brief evidence references a different report")
    if report.source_digest != brief.source_digest:
        raise DomainBriefError("domain brief report source digest mismatch")
    if report.sanitized_digest != brief.content_digest:
        raise DomainBriefError("domain brief report sanitized digest mismatch")
    if report.source_digest != _source_digest(original):
        raise DomainBriefError("domain brief source bytes do not match the report")
    return report
