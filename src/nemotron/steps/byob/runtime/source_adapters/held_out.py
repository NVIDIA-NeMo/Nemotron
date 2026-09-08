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

"""Explicit, digest-bound held-out decisions for assisted authoring."""

from __future__ import annotations

import json
import re
import unicodedata
from base64 import b64decode, b64encode, urlsafe_b64encode
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, quote_plus

import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HeldOutPolicy,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    CertificationAuthority,
)

HELD_OUT_DECISION_VERSION: Literal["bfcl-held-out-decision-v1"] = (
    "bfcl-held-out-decision-v1"
)
_SAFE_REVIEWER = re.compile(r"^[^\s@]+@[^\s@]+$|^[a-zA-Z0-9][a-zA-Z0-9._-]{1,127}$")
_MAX_REASON_BYTES = 2048
HELD_OUT_REDACTION_VERSION: Literal["bfcl-held-out-redaction-v1"] = (
    "bfcl-held-out-redaction-v1"
)
LeakVariant = Literal[
    "base64",
    "exact",
    "nested_json",
    "unicode_nfkc_casefold",
    "url_percent",
]
_VARIANTS: tuple[LeakVariant, ...] = (
    "base64",
    "exact",
    "nested_json",
    "unicode_nfkc_casefold",
    "url_percent",
)


class HeldOutError(ValueError):
    """Raised when assisted authoring has no reviewable held-out decision."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeldOutDecision(_StrictModel):
    """One reviewed decision; source paths never become persisted authority."""

    schema_version: Literal["bfcl-held-out-decision-v1"]
    status: Literal["required", "not_applicable"]
    policy_digest: StrictStr | None = None
    reviewed_reason: StrictStr | None = None
    reviewed_by: StrictStr
    decision_digest: StrictStr

    @field_validator("reviewed_by")
    @classmethod
    def _reviewer(cls, value: str) -> str:
        if not _SAFE_REVIEWER.fullmatch(value):
            raise ValueError("held-out reviewer must be a stable name or email")
        return value

    @field_validator("reviewed_reason")
    @classmethod
    def _reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("held-out reviewed reason must be non-empty")
        if len(normalized.encode("utf-8")) > _MAX_REASON_BYTES:
            raise ValueError("held-out reviewed reason exceeds 2048 bytes")
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise ValueError("held-out reviewed reason contains control characters")
        return normalized

    @model_validator(mode="after")
    def _state_and_digest(self) -> HeldOutDecision:
        if self.status == "required":
            if self.policy_digest is None or self.reviewed_reason is not None:
                raise ValueError(
                    "required held-out status needs policy_digest and no reason"
                )
        elif self.policy_digest is not None or self.reviewed_reason is None:
            raise ValueError(
                "not_applicable held-out status needs a reviewed reason and no policy"
            )
        unsigned = self.model_dump(mode="json", exclude={"decision_digest"})
        if self.decision_digest != sha256_json(unsigned):
            raise ValueError("held-out decision digest mismatch")
        return self


class HeldOutLeakFinding(_StrictModel):
    location: StrictStr
    variant: LeakVariant
    term_digest: StrictStr


class HeldOutRedactionReport(_StrictModel):
    schema_version: Literal["bfcl-held-out-redaction-v1"]
    decision_digest: StrictStr
    policy_digest: StrictStr | None
    evidence_digest: StrictStr
    terms_commitment_digest: StrictStr
    term_count: int
    variants_checked: tuple[StrictStr, ...]
    findings: tuple[HeldOutLeakFinding, ...] = ()
    signing_key_id: StrictStr
    report_digest: StrictStr
    signature: StrictStr

    @model_validator(mode="after")
    def _report_contract(self) -> HeldOutRedactionReport:
        if self.term_count < 0:
            raise ValueError("held-out term_count cannot be negative")
        if self.variants_checked != _VARIANTS:
            raise ValueError("held-out redaction variants are incomplete")
        unsigned = self.model_dump(
            mode="json",
            exclude={"report_digest", "signature"},
        )
        if self.report_digest != sha256_json(unsigned):
            raise ValueError("held-out redaction report digest mismatch")
        try:
            signature = b64decode(self.signature, validate=True)
        except ValueError as exc:
            raise ValueError("held-out redaction signature must be base64") from exc
        if len(signature) != 64 or b64encode(signature).decode("ascii") != self.signature:
            raise ValueError("held-out redaction signature must encode 64 bytes")
        return self


class HeldOutLeakageError(HeldOutError):
    """Raised with sanitized locations; reserved values are never echoed."""

    def __init__(self, findings: Sequence[HeldOutLeakFinding]) -> None:
        self.findings = tuple(findings)
        locations = ", ".join(sorted({finding.location for finding in findings}))
        super().__init__(f"held-out leakage detected at: {locations}")


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise HeldOutError(f"held-out policy repeats YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _build_decision(
    *,
    status: Literal["required", "not_applicable"],
    reviewed_by: str,
    policy_digest: str | None = None,
    reviewed_reason: str | None = None,
) -> HeldOutDecision:
    document: dict[str, Any] = {
        "schema_version": HELD_OUT_DECISION_VERSION,
        "status": status,
        "policy_digest": policy_digest,
        "reviewed_reason": reviewed_reason,
        "reviewed_by": reviewed_by,
    }
    document["decision_digest"] = sha256_json(document)
    return HeldOutDecision.model_validate(document)


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    source = path.resolve()
    try:
        document = yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_UniqueSafeLoader,
        )
    except HeldOutError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HeldOutError(f"cannot load {label} {source}: {exc}") from exc
    if not isinstance(document, dict) or not document:
        raise HeldOutError(f"{label} must be a non-empty YAML mapping")
    if any(not isinstance(key, str) for key in document):
        raise HeldOutError(f"{label} keys must be strings")
    return document


def _validated_policy(document: dict[str, Any]) -> HeldOutPolicy:
    allowed = {"fixtures", "policy", "source", "templates", "version"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise HeldOutError(
            "held-out policy contains unsupported fields: " + ", ".join(unknown)
        )
    try:
        return HeldOutPolicy.from_normalized(document)
    except ValueError as exc:
        raise HeldOutError(f"invalid held-out policy: {exc}") from exc


def load_required_held_out_policy(
    path: Path,
    *,
    reviewed_by: str,
) -> HeldOutDecision:
    """Load a strict policy mapping and retain only its canonical digest."""

    document = _load_yaml_mapping(path, label="held-out policy")
    _validated_policy(document)
    try:
        policy_digest = sha256_json(document)
    except (TypeError, ValueError) as exc:
        raise HeldOutError("held-out policy must contain canonical JSON values") from exc
    return _build_decision(
        status="required",
        reviewed_by=reviewed_by,
        policy_digest=policy_digest,
    )


def build_not_applicable_decision(
    reason: str,
    *,
    reviewed_by: str,
) -> HeldOutDecision:
    return _build_decision(
        status="not_applicable",
        reviewed_by=reviewed_by,
        reviewed_reason=reason,
    )


def _collect_scalar_terms(value: Any, terms: set[str]) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _collect_scalar_terms(child, terms)
    elif isinstance(value, list):
        for child in value:
            _collect_scalar_terms(child, terms)
    elif value is not None and not isinstance(value, bool):
        normalized = str(value).strip()
        if normalized:
            terms.add(normalized)


def load_held_out_sensitive_terms(
    policy_path: Path,
    *,
    content_path: Path | None = None,
) -> tuple[str, ...]:
    """Load runtime-only identifiers/content; no cleartext enters persisted reports."""

    policy = _load_yaml_mapping(policy_path, label="held-out policy")
    validated = _validated_policy(policy)
    terms: set[str] = set()
    for reference in validated.fixture_refs:
        collection, identifier = json.loads(reference)
        terms.add(str(identifier))
    _collect_scalar_terms(list(validated.template_ids), terms)
    if not validated.reserves_nothing and content_path is None:
        raise HeldOutError(
            "a non-empty held-out policy requires runtime-only reserved content"
        )
    if content_path is not None:
        content = _load_yaml_mapping(content_path, label="held-out content")
        _collect_scalar_terms(content, terms)
    return tuple(sorted(terms))


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _string_variants(term: str) -> dict[LeakVariant, tuple[str, ...]]:
    encoded = term.encode("utf-8")
    return {
        "exact": (term,),
        "unicode_nfkc_casefold": (_normalize(term),),
        "url_percent": tuple(sorted({quote(term, safe=""), quote_plus(term, safe="")})),
        "base64": tuple(
            sorted(
                {
                    b64encode(encoded).decode("ascii"),
                    urlsafe_b64encode(encoded).decode("ascii"),
                }
            )
        ),
    }


def _scan_string(
    text: str,
    *,
    location: str,
    term: str,
    term_digest: str,
    findings: list[HeldOutLeakFinding],
    nested_depth: int = 0,
) -> None:
    variants = _string_variants(term)
    for variant, needles in variants.items():
        haystack = _normalize(text) if variant == "unicode_nfkc_casefold" else text
        if any(needle and needle in haystack for needle in needles):
            findings.append(
                HeldOutLeakFinding(
                    location=location,
                    variant=variant,
                    term_digest=term_digest,
                )
            )
    if nested_depth >= 3:
        return
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return
    try:
        nested = json.loads(stripped)
    except json.JSONDecodeError:
        return
    before = len(findings)
    _scan_value(
        nested,
        location=f"{location}.$json",
        term=term,
        term_digest=term_digest,
        findings=findings,
        nested_depth=nested_depth + 1,
    )
    for index in range(before, len(findings)):
        finding = findings[index]
        findings[index] = finding.model_copy(update={"variant": "nested_json"})


def _scan_value(
    value: Any,
    *,
    location: str,
    term: str,
    term_digest: str,
    findings: list[HeldOutLeakFinding],
    nested_depth: int = 0,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_location = f"{location}.<key>"
            _scan_string(
                str(key),
                location=key_location,
                term=term,
                term_digest=term_digest,
                findings=findings,
                nested_depth=nested_depth,
            )
            _scan_value(
                child,
                location=f"{location}.{key}",
                term=term,
                term_digest=term_digest,
                findings=findings,
                nested_depth=nested_depth,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_value(
                child,
                location=f"{location}[{index}]",
                term=term,
                term_digest=term_digest,
                findings=findings,
                nested_depth=nested_depth,
            )
    elif value is not None:
        _scan_string(
            str(value),
            location=location,
            term=term,
            term_digest=term_digest,
            findings=findings,
            nested_depth=nested_depth,
        )


def scan_held_out_terms(
    value: Any,
    *,
    sensitive_terms: Sequence[str],
    location: str = "$",
) -> tuple[HeldOutLeakFinding, ...]:
    """Scan any runtime-only probe input with the canonical held-out detector."""
    canonical_terms = tuple(
        sorted(
            {
                term.strip()
                for term in sensitive_terms
                if isinstance(term, str) and term.strip()
            }
        )
    )
    findings: list[HeldOutLeakFinding] = []
    for term in canonical_terms:
        _scan_value(
            value,
            location=location,
            term=term,
            term_digest=sha256_json({"term": term}),
            findings=findings,
        )
    return tuple(
        sorted(
            set(findings),
            key=lambda item: (item.location, item.variant, item.term_digest),
        )
    )


def build_held_out_redaction_report(
    evidence: Mapping[str, Any],
    *,
    decision: HeldOutDecision,
    sensitive_terms: Sequence[str],
    authority: CertificationAuthority,
) -> HeldOutRedactionReport:
    claimed_evidence_digest = evidence.get("bundle_digest")
    unsigned_evidence = {
        key: value for key, value in evidence.items() if key != "bundle_digest"
    }
    if (
        not isinstance(claimed_evidence_digest, str)
        or claimed_evidence_digest != sha256_json(unsigned_evidence)
    ):
        raise HeldOutError("held-out scan requires a verified evidence bundle")
    canonical_terms = tuple(
        sorted(
            {
                term.strip()
                for term in sensitive_terms
                if isinstance(term, str) and term.strip()
            }
        )
    )
    if decision.status == "required" and decision.policy_digest is None:
        raise HeldOutError("required held-out decision has no policy digest")
    term_digests = tuple(sha256_json({"term": term}) for term in canonical_terms)
    canonical_findings = scan_held_out_terms(
        evidence,
        sensitive_terms=canonical_terms,
    )
    if canonical_findings:
        raise HeldOutLeakageError(canonical_findings)
    document: dict[str, Any] = {
        "schema_version": HELD_OUT_REDACTION_VERSION,
        "decision_digest": decision.decision_digest,
        "policy_digest": decision.policy_digest,
        "evidence_digest": claimed_evidence_digest,
        "terms_commitment_digest": sha256_json({"term_digests": term_digests}),
        "term_count": len(canonical_terms),
        "variants_checked": list(_VARIANTS),
        "findings": [],
        "signing_key_id": authority.key_id,
    }
    document["report_digest"] = sha256_json(document)
    document["signature"] = b64encode(
        authority.private_key.sign(document["report_digest"].encode("ascii"))
    ).decode("ascii")
    return HeldOutRedactionReport.model_validate(document)


def verify_held_out_redaction_report(
    report: HeldOutRedactionReport,
    *,
    decision: HeldOutDecision,
    evidence_digest: str,
    trusted_public_keys: Mapping[str, Ed25519PublicKey],
    sensitive_terms: Sequence[str] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    if report.findings:
        raise HeldOutError("held-out redaction report contains leakage findings")
    if (
        report.decision_digest != decision.decision_digest
        or report.policy_digest != decision.policy_digest
        or report.evidence_digest != evidence_digest
    ):
        raise HeldOutError("held-out redaction report binding mismatch")
    if decision.status == "required" and (
        sensitive_terms is None or evidence is None
    ):
        raise HeldOutError(
            "required held-out proof must be re-scanned from reviewed term material"
        )
    if sensitive_terms is not None:
        canonical_terms = tuple(
            sorted(
                {
                    term.strip()
                    for term in sensitive_terms
                    if isinstance(term, str) and term.strip()
                }
            )
        )
        term_digests = tuple(
            sha256_json({"term": term}) for term in canonical_terms
        )
        if report.terms_commitment_digest != sha256_json(
            {"term_digests": term_digests}
        ):
            raise HeldOutError("held-out term commitment does not match reviewed material")
        if evidence is not None:
            findings: list[HeldOutLeakFinding] = []
            for term, term_digest in zip(canonical_terms, term_digests, strict=True):
                _scan_value(
                    evidence,
                    location="$",
                    term=term,
                    term_digest=term_digest,
                    findings=findings,
                )
            if findings:
                raise HeldOutError("held-out evidence fails fresh leakage scan")
    public_key = trusted_public_keys.get(report.signing_key_id)
    if public_key is None:
        raise HeldOutError(
            f"held-out signing key {report.signing_key_id!r} is not trusted"
        )
    try:
        public_key.verify(
            b64decode(report.signature, validate=True),
            report.report_digest.encode("ascii"),
        )
    except (InvalidSignature, ValueError) as exc:
        raise HeldOutError("held-out redaction signature is invalid") from exc


def load_held_out_redaction_report(path: Path) -> HeldOutRedactionReport:
    source = path.resolve()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HeldOutError(
                    f"held-out redaction report repeats JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
        return HeldOutRedactionReport.model_validate(document)
    except HeldOutError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HeldOutError(
            f"cannot load held-out redaction report {source}: {exc}"
        ) from exc
