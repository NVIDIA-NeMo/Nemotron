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

"""BFCL-owned, signed source-adapter certification."""

from __future__ import annotations

import json
import re
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    AdapterDescriptor,
    CleanupKind,
    ProbeSafetyKind,
)

if TYPE_CHECKING:
    from nemotron.steps.byob.runtime.source_adapters.evidence import CertificationReference


CERTIFICATION_PROFILE_VERSION: Literal["bfcl-adapter-certification-profile-v1"] = (
    "bfcl-adapter-certification-profile-v1"
)
CERTIFICATION_REPORT_VERSION: Literal["bfcl-adapter-certification-report-v1"] = "bfcl-adapter-certification-report-v1"
CERTIFICATION_ISSUER: Literal["bfcl-source-adapter-verifier-v1"] = "bfcl-source-adapter-verifier-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REASON = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_KEY_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
PROBE_OUTCOME_VERSION: Literal["bfcl-adapter-probe-outcome-v1"] = "bfcl-adapter-probe-outcome-v1"
PROBE_INPUT_BINDING_VERSION: Literal["bfcl-adapter-probe-input-v1"] = "bfcl-adapter-probe-input-v1"


class CertificationError(ValueError):
    """Raised when certification evidence cannot support a trusted tier."""

    def __init__(self, message: str, *, code: str = "certification_invalid") -> None:
        super().__init__(message)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class CertificationAuthority:
    """A BFCL-controlled signing identity; adapters receive no private key."""

    key_id: str
    private_key: Ed25519PrivateKey = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _KEY_ID.fullmatch(self.key_id):
            raise CertificationError("certification key_id must be a safe identifier")

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()


def load_certification_authority(
    path: Path,
    *,
    key_id: str,
    password: bytes | None = None,
) -> CertificationAuthority:
    try:
        key = load_pem_private_key(path.resolve().read_bytes(), password=password)
    except (OSError, ValueError, TypeError) as exc:
        raise CertificationError(f"cannot load certification private key {path.resolve()}: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise CertificationError("certification private key must be Ed25519")
    return CertificationAuthority(key_id=key_id, private_key=key)


def load_trusted_certification_key(
    path: Path,
    *,
    key_id: str,
) -> dict[str, Ed25519PublicKey]:
    try:
        key = load_pem_public_key(path.resolve().read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise CertificationError(f"cannot load certification public key {path.resolve()}: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise CertificationError("certification public key must be Ed25519")
    if not _KEY_ID.fullmatch(key_id):
        raise CertificationError("certification key_id must be a safe identifier")
    return {key_id: key}


class AdapterTier(str, Enum):
    NONE = "none"
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"


_TIER_ORDER = {
    AdapterTier.NONE: 0,
    AdapterTier.A0: 1,
    AdapterTier.A1: 2,
    AdapterTier.A2: 3,
}


class CertificationProbe(str, Enum):
    IDENTITY_INTEGRITY = "identity_integrity"
    CATALOG_INTEGRITY = "catalog_integrity"
    EXECUTABLE_OBSERVATION = "executable_observation"
    STRUCTURED_ERROR_SHAPE = "structured_error_shape"
    RESET_DETERMINISM = "reset_determinism"
    EPISODE_ISOLATION = "episode_isolation"
    CONFIRMATION_SAFETY = "confirmation_safety"
    TIMEOUT_CLEANUP = "timeout_cleanup"
    MUTATION_DECLARATION = "mutation_declaration"
    RESULT_SHAPE_COVERAGE = "result_shape_coverage"


class CertificationRefusalCode(str, Enum):
    """Stable machine-readable reasons emitted by BFCL certification."""

    ADAPTER_UNDER_CERTIFIED = "adapter_under_certified"
    APPLICABILITY_MISMATCH = "applicability_mismatch"
    ATTESTATION_MISMATCH = "attestation_mismatch"
    CATALOG_MISMATCH = "catalog_mismatch"
    CLEANUP_FAILED = "cleanup_failed"
    CROSS_ORIGIN_REDIRECT = "cross_origin_redirect"
    DEPENDENCY_LOCK_INVALID = "dependency_lock_invalid"
    DEPENDENCY_LOCK_MISSING = "dependency_lock_missing"
    DYNAMIC_IMPORT = "dynamic_import"
    EPISODE_STATE_LEAKAGE = "episode_state_leakage"
    FIXTURE_METADATA_INVALID = "fixture_metadata_invalid"
    IDENTITY_DRIFT = "identity_drift"
    IMPORT_PATH_AMBIGUOUS = "import_path_ambiguous"
    MUTATION_DECLARATION_MISMATCH = "mutation_declaration_mismatch"
    NAMESPACE_PACKAGE_AMBIGUOUS = "namespace_package_ambiguous"
    PROBE_EVIDENCE_INVALID = "probe_evidence_invalid"
    PROBE_FAILED = "probe_failed"
    PROBE_MISSING = "probe_missing"
    PROFILE_MISMATCH = "profile_mismatch"
    PROBE_TIMEOUT = "probe_timeout"
    PROBE_UNSAFE = "probe_unsafe"
    RESET_NONDETERMINISTIC = "reset_nondeterministic"
    RESPONSE_TOO_LARGE = "response_too_large"
    RESULT_SHAPE_INCOMPLETE = "result_shape_incomplete"
    REVIEWED_SCHEMA_INVALID = "reviewed_schema_invalid"
    REVIEWED_SCHEMA_MISSING = "reviewed_schema_missing"
    REVIEWED_SCHEMA_TOO_LARGE = "reviewed_schema_too_large"
    SOURCE_ENCODING_INVALID = "source_encoding_invalid"
    SOURCE_PACKAGE_INVALID = "source_package_invalid"
    SOURCE_PATH_ESCAPE = "source_path_escape"
    SOURCE_SYNTAX_INVALID = "source_syntax_invalid"
    STRUCTURED_ERROR_MISMATCH = "structured_error_mismatch"
    UNDECLARED_IMPORT = "undeclared_import"
    UNKNOWN_COMMIT_STATE = "unknown_commit_state"
    UNSUPPORTED_AUTH = "unsupported_auth"


class ProbeExecutionPolicy(_StrictModel):
    """BFCL-owned bounds for executing and judging one generic probe."""

    executor: Literal["bfcl"]
    evidence_issuer: Literal["bfcl-source-adapter-verifier-v1"]
    input_binding: Literal["bfcl-adapter-probe-input-v1"]
    outcome_schema: Literal["bfcl-adapter-probe-outcome-v1"]
    safety: ProbeSafetyKind
    max_calls: StrictInt
    timeout_s: StrictFloat
    cleanup: CleanupKind
    cleanup_timeout_s: StrictFloat

    @field_validator("max_calls")
    @classmethod
    def _positive_calls(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("probe max_calls must be positive")
        return value

    @field_validator("timeout_s", "cleanup_timeout_s")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("probe timeouts must be positive")
        return value


_PROBE_ORDER = tuple(CertificationProbe)
_PROBE_TIER = {
    CertificationProbe.IDENTITY_INTEGRITY: AdapterTier.A0,
    CertificationProbe.CATALOG_INTEGRITY: AdapterTier.A0,
    CertificationProbe.EXECUTABLE_OBSERVATION: AdapterTier.A1,
    CertificationProbe.STRUCTURED_ERROR_SHAPE: AdapterTier.A1,
    CertificationProbe.RESET_DETERMINISM: AdapterTier.A2,
    CertificationProbe.EPISODE_ISOLATION: AdapterTier.A2,
    CertificationProbe.CONFIRMATION_SAFETY: AdapterTier.A2,
    CertificationProbe.TIMEOUT_CLEANUP: AdapterTier.A2,
    CertificationProbe.MUTATION_DECLARATION: AdapterTier.A2,
    CertificationProbe.RESULT_SHAPE_COVERAGE: AdapterTier.A2,
}


class ProbeRequirement(_StrictModel):
    probe: CertificationProbe
    requirement: Literal["required", "conditional"]
    execution: ProbeExecutionPolicy
    allowed_failure_reasons: tuple[CertificationRefusalCode, ...]
    allowed_not_applicable_reasons: tuple[StrictStr, ...] = ()

    @field_validator("allowed_failure_reasons")
    @classmethod
    def _canonical_failures(
        cls,
        value: tuple[CertificationRefusalCode, ...],
    ) -> tuple[CertificationRefusalCode, ...]:
        if not value:
            raise ValueError("every probe must declare stable failure reasons")
        if len(value) != len(set(value)):
            raise ValueError("probe failure reasons must be unique")
        if tuple(sorted(value, key=lambda item: item.value)) != value:
            raise ValueError("probe failure reasons must be sorted")
        return value

    @field_validator("allowed_not_applicable_reasons")
    @classmethod
    def _canonical_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for reason in value:
            if not _REASON.fullmatch(reason):
                raise ValueError("not-applicable reasons must be safe machine-readable codes")
        if len(value) != len(set(value)):
            raise ValueError("not-applicable reasons must be unique")
        if tuple(sorted(value)) != value:
            raise ValueError("not-applicable reasons must be sorted")
        return value

    @model_validator(mode="after")
    def _requirement_contract(self) -> ProbeRequirement:
        if self.requirement == "required" and self.allowed_not_applicable_reasons:
            raise ValueError("required probes cannot allow not_applicable")
        if self.requirement == "conditional" and not self.allowed_not_applicable_reasons:
            raise ValueError("conditional probes require allowed not-applicable reasons")
        return self


class CertificationProfile(_StrictModel):
    schema_version: Literal["bfcl-adapter-certification-profile-v1"]
    profile_id: StrictStr
    owner: Literal["bfcl"]
    adapter_kinds: tuple[StrictStr, ...]
    max_total_calls: StrictInt
    max_wall_time_s: StrictFloat
    probes: tuple[ProbeRequirement, ...]

    @field_validator("profile_id")
    @classmethod
    def _profile_id(cls, value: str) -> str:
        if not _REASON.fullmatch(value):
            raise ValueError("profile_id must be a safe lowercase identifier")
        return value

    @field_validator("adapter_kinds")
    @classmethod
    def _adapter_kinds(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a certification profile must select adapter kinds")
        if any(not _REASON.fullmatch(kind) for kind in value):
            raise ValueError("adapter kinds must be safe lowercase identifiers")
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("adapter kinds must be unique and sorted")
        return value

    @field_validator("probes")
    @classmethod
    def _complete_probe_set(
        cls,
        value: tuple[ProbeRequirement, ...],
    ) -> tuple[ProbeRequirement, ...]:
        names = tuple(item.probe for item in value)
        if names != _PROBE_ORDER:
            raise ValueError("certification profile must contain every generic probe in canonical order")
        return value

    @field_validator("max_total_calls")
    @classmethod
    def _positive_total_calls(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("profile max_total_calls must be positive")
        return value

    @field_validator("max_wall_time_s")
    @classmethod
    def _positive_wall_time(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("profile max_wall_time_s must be positive")
        return value

    @model_validator(mode="after")
    def _execution_budget(self) -> CertificationProfile:
        if sum(item.execution.max_calls for item in self.probes) > self.max_total_calls:
            raise ValueError("per-probe call budgets exceed profile max_total_calls")
        if sum(item.execution.timeout_s for item in self.probes) > self.max_wall_time_s:
            raise ValueError("per-probe timeouts exceed profile max_wall_time_s")
        return self

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class ProbeOutcome(_StrictModel):
    probe: CertificationProbe
    status: Literal["pass", "fail", "not_applicable"]
    input_digest: StrictStr
    evidence_digest: StrictStr | None = None
    evidence: Any | None = None
    reason: StrictStr | None = None

    @field_validator("input_digest", "evidence_digest")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST.fullmatch(value):
            raise ValueError("probe digests must be lowercase SHA-256 values")
        return value

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str | None) -> str | None:
        if value is not None and not _REASON.fullmatch(value):
            raise ValueError("probe reasons must be safe machine-readable codes")
        return value

    @model_validator(mode="after")
    def _outcome_contract(self) -> ProbeOutcome:
        if self.evidence is not None:
            try:
                observed = sha256_json(self.evidence)
            except (TypeError, ValueError) as exc:
                raise ValueError("probe evidence must be canonical JSON") from exc
            if self.evidence_digest != observed:
                raise ValueError("probe evidence digest mismatch")
        if self.status == "pass":
            if self.evidence_digest is None or self.evidence is None or self.reason is not None:
                raise ValueError("passing probes require digest-bound evidence and cannot carry a reason")
        elif self.reason is None:
            raise ValueError("failed and not-applicable probes require a reason")
        if self.status == "not_applicable" and self.evidence is None:
            raise ValueError("not-applicable probes require applicability evidence")
        if self.evidence is None and self.evidence_digest is not None:
            raise ValueError("probe evidence_digest cannot exist without evidence")
        return self


class AdapterProbeObservation(_StrictModel):
    """Untrusted raw observation; deliberately carries no digest, tier, or issuer."""

    probe: CertificationProbe
    status: Literal["pass", "fail", "not_applicable"]
    evidence: Any | None = None
    reason: StrictStr | None = None

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str | None) -> str | None:
        if value is not None and not _REASON.fullmatch(value):
            raise ValueError("observation reasons must be safe machine-readable codes")
        return value

    @model_validator(mode="after")
    def _observation_contract(self) -> AdapterProbeObservation:
        if self.evidence is not None:
            try:
                sha256_json(self.evidence)
            except (TypeError, ValueError) as exc:
                raise ValueError("observation evidence must be canonical JSON") from exc
        if self.status == "pass" and (self.evidence is None or self.reason is not None):
            raise ValueError("passing observations require evidence and cannot carry a reason")
        if self.status != "pass" and self.reason is None:
            raise ValueError("non-passing observations require a reason")
        if self.status == "not_applicable" and self.evidence is None:
            raise ValueError("not-applicable observations require applicability evidence")
        return self


class ProbeExecutionRecord(_StrictModel):
    """BFCL-measured execution facts paired with one untrusted observation."""

    observation: AdapterProbeObservation
    observed_calls: StrictInt
    elapsed_s: StrictFloat
    cleanup_status: Literal["passed", "failed", "not_required"]

    @field_validator("observed_calls")
    @classmethod
    def _nonnegative_calls(cls, value: int) -> int:
        if value < 0:
            raise ValueError("observed_calls must be non-negative")
        return value

    @field_validator("elapsed_s")
    @classmethod
    def _nonnegative_elapsed(cls, value: float) -> float:
        if value < 0:
            raise ValueError("elapsed_s must be non-negative")
        return value


class AdapterCertificationReport(_StrictModel):
    schema_version: Literal["bfcl-adapter-certification-report-v1"]
    issuer: Literal["bfcl-source-adapter-verifier-v1"]
    profile_id: StrictStr
    profile_digest: StrictStr
    descriptor_digest: StrictStr
    source_identity_digest: StrictStr
    adapter_kind: StrictStr
    outcomes: tuple[ProbeOutcome, ...]
    attained_tier: AdapterTier
    signing_key_id: StrictStr
    report_digest: StrictStr
    signature: StrictStr

    @field_validator(
        "profile_digest",
        "descriptor_digest",
        "source_identity_digest",
        "report_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("certification report digests must be lowercase SHA-256 values")
        return value

    @field_validator("outcomes")
    @classmethod
    def _canonical_outcomes(
        cls,
        value: tuple[ProbeOutcome, ...],
    ) -> tuple[ProbeOutcome, ...]:
        if tuple(item.probe for item in value) != _PROBE_ORDER:
            raise ValueError("certification report must contain every generic probe in canonical order")
        return value

    @field_validator("signing_key_id")
    @classmethod
    def _key_id(cls, value: str) -> str:
        if not _KEY_ID.fullmatch(value):
            raise ValueError("certification signing_key_id must be a safe identifier")
        return value

    @field_validator("signature")
    @classmethod
    def _signature(cls, value: str) -> str:
        try:
            decoded = b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("certification signature must be canonical base64") from exc
        if len(decoded) != 64 or b64encode(decoded).decode("ascii") != value:
            raise ValueError("certification signature must encode 64 bytes")
        return value

    @model_validator(mode="after")
    def _verify_report_digest(self) -> AdapterCertificationReport:
        unsigned = self.model_dump(
            mode="json",
            exclude={"report_digest", "signature"},
        )
        if self.report_digest != sha256_json(unsigned):
            raise ValueError("adapter certification report digest mismatch")
        return self


def _satisfies(outcome: ProbeOutcome, requirement: ProbeRequirement) -> bool:
    if outcome.status == "pass":
        return True
    return (
        requirement.requirement == "conditional"
        and outcome.status == "not_applicable"
        and outcome.reason in requirement.allowed_not_applicable_reasons
    )


def _validate_applicability(
    profile: CertificationProfile,
    outcomes: Sequence[ProbeOutcome],
) -> None:
    requirements = {item.probe: item for item in profile.probes}
    for outcome in outcomes:
        requirement = requirements[outcome.probe]
        if outcome.status == "fail":
            if outcome.reason is None:
                raise CertificationError(f"probe {outcome.probe.value!r} has no failure reason")
            try:
                refusal = CertificationRefusalCode(outcome.reason)
            except ValueError as exc:
                raise CertificationError(
                    f"probe {outcome.probe.value!r} uses an unknown failure reason {outcome.reason!r}",
                    code=CertificationRefusalCode.PROBE_EVIDENCE_INVALID.value,
                ) from exc
            if refusal not in requirement.allowed_failure_reasons:
                raise CertificationError(
                    f"probe {outcome.probe.value!r} uses a failure reason outside its profile: {outcome.reason!r}",
                    code=CertificationRefusalCode.PROFILE_MISMATCH.value,
                )
        if outcome.status != "not_applicable":
            continue
        if (
            requirement.requirement != "conditional"
            or outcome.reason not in requirement.allowed_not_applicable_reasons
        ):
            raise CertificationError(
                f"probe {outcome.probe.value!r} uses an unapproved not_applicable reason {outcome.reason!r}",
                code=CertificationRefusalCode.APPLICABILITY_MISMATCH.value,
            )


def derive_attained_tier(
    profile: CertificationProfile,
    outcomes: Sequence[ProbeOutcome],
) -> AdapterTier:
    """Return the highest tier fully established by the supplied outcomes."""

    if tuple(item.probe for item in outcomes) != _PROBE_ORDER:
        raise CertificationError("probe outcomes must contain every generic probe in canonical order")
    _validate_applicability(profile, outcomes)
    by_probe = {item.probe: item for item in outcomes}
    requirements = {item.probe: item for item in profile.probes}
    attained = AdapterTier.NONE
    for tier in (AdapterTier.A0, AdapterTier.A1, AdapterTier.A2):
        required = (probe for probe in _PROBE_ORDER if _TIER_ORDER[_PROBE_TIER[probe]] <= _TIER_ORDER[tier])
        if not all(_satisfies(by_probe[probe], requirements[probe]) for probe in required):
            break
        attained = tier
    return attained


def project_probe_executions(
    profile: CertificationProfile,
    records: Sequence[ProbeExecutionRecord],
    *,
    input_digest: str,
) -> tuple[ProbeOutcome, ...]:
    """Enforce profile budgets and create BFCL-owned, digest-bound outcomes."""
    if not _DIGEST.fullmatch(input_digest):
        raise CertificationError("probe input digest must be a lowercase SHA-256 value")
    indexed: dict[CertificationProbe, ProbeExecutionRecord] = {}
    for execution in records:
        probe = execution.observation.probe
        if probe in indexed:
            raise CertificationError(f"probe execution records repeat probe {probe.value!r}")
        indexed[probe] = execution
    requirements = {item.probe: item for item in profile.probes}
    outcomes: list[ProbeOutcome] = []
    for probe in _PROBE_ORDER:
        record = indexed.get(probe)
        if record is None:
            outcomes.append(
                ProbeOutcome(
                    probe=probe,
                    status="fail",
                    input_digest=input_digest,
                    reason=CertificationRefusalCode.PROBE_MISSING.value,
                )
            )
            continue
        requirement = requirements[probe]
        policy = requirement.execution
        evidence = {
            "observation": record.observation.model_dump(mode="json"),
            "execution": {
                "observed_calls": record.observed_calls,
                "cleanup_status": record.cleanup_status,
                # Probe runners enforce operation deadlines at the call boundary. Raw
                # process elapsed time is host-load diagnostics, not deterministic
                # correctness evidence and therefore is not sealed into the report.
                "bounded_by_policy_timeout_s": policy.timeout_s,
            },
        }
        evidence_digest = sha256_json(evidence)
        refusal: CertificationRefusalCode | None = None
        if record.observed_calls > policy.max_calls:
            refusal = CertificationRefusalCode.PROBE_UNSAFE
        elif (policy.cleanup is CleanupKind.NONE and record.cleanup_status == "failed") or (
            policy.cleanup is not CleanupKind.NONE and record.cleanup_status != "passed"
        ):
            refusal = CertificationRefusalCode.CLEANUP_FAILED
        if refusal is not None:
            outcomes.append(
                ProbeOutcome(
                    probe=probe,
                    status="fail",
                    input_digest=input_digest,
                    evidence_digest=evidence_digest,
                    evidence=evidence,
                    reason=refusal.value,
                )
            )
            continue
        outcomes.append(
            ProbeOutcome(
                probe=probe,
                status=record.observation.status,
                input_digest=input_digest,
                evidence_digest=evidence_digest,
                evidence=evidence,
                reason=record.observation.reason,
            )
        )
    canonical = tuple(outcomes)
    _validate_applicability(profile, canonical)
    return canonical


def certification_input_digest(
    descriptor: AdapterDescriptor,
    *,
    source_identity_digest: str,
    profile: CertificationProfile,
    execution_inputs_digest: str | None = None,
) -> str:
    """Bind every probe to the exact descriptor, identity, and profile under test."""

    if execution_inputs_digest is not None and not _DIGEST.fullmatch(execution_inputs_digest):
        raise CertificationError("execution_inputs_digest must be a lowercase SHA-256 value")
    return sha256_json(
        {
            "descriptor_digest": sha256_json(descriptor.model_dump(mode="json")),
            "source_identity_digest": source_identity_digest,
            "profile_digest": profile.digest,
            "execution_inputs_digest": execution_inputs_digest,
        }
    )


def _descriptor_tier_ceiling(descriptor: AdapterDescriptor) -> AdapterTier:
    capabilities = set(descriptor.capabilities)
    if (
        not {
            AdapterCapability.DESCRIBE_TOOLS,
            AdapterCapability.PIN_IDENTITY,
        }
        <= capabilities
    ):
        return AdapterTier.NONE
    ceiling = AdapterTier.A0
    if AdapterCapability.OBSERVE in capabilities and descriptor.probe_safety.kind in {
        ProbeSafetyKind.READ_ONLY,
        ProbeSafetyKind.RESET_ISOLATED,
    }:
        ceiling = AdapterTier.A1
    if (
        {
            AdapterCapability.DESCRIBE_STATE,
            AdapterCapability.GET_STATE,
            AdapterCapability.OBSERVE,
            AdapterCapability.RESET_STATE,
        }
        <= capabilities
        and descriptor.probe_safety.kind is ProbeSafetyKind.RESET_ISOLATED
        and descriptor.cleanup.kind is not CleanupKind.NONE
    ):
        ceiling = AdapterTier.A2
    return ceiling


def build_certification_report(
    descriptor: AdapterDescriptor,
    *,
    source_identity_digest: str,
    profile: CertificationProfile,
    outcomes: Sequence[ProbeOutcome],
    authority: CertificationAuthority,
    execution_inputs_digest: str | None = None,
) -> AdapterCertificationReport:
    """Create a deterministic BFCL-issued report from independently supplied probes."""

    if descriptor.kind not in profile.adapter_kinds:
        raise CertificationError(
            f"profile {profile.profile_id!r} does not allow adapter kind {descriptor.kind!r}",
            code=CertificationRefusalCode.PROFILE_MISMATCH.value,
        )
    if not _DIGEST.fullmatch(source_identity_digest):
        raise CertificationError("source_identity_digest must be a lowercase SHA-256 value")
    if execution_inputs_digest is not None and not _DIGEST.fullmatch(execution_inputs_digest):
        raise CertificationError("execution_inputs_digest must be a lowercase SHA-256 value")
    canonical_outcomes = tuple(outcomes)
    expected_input = certification_input_digest(
        descriptor,
        source_identity_digest=source_identity_digest,
        profile=profile,
        execution_inputs_digest=execution_inputs_digest,
    )
    if any(item.input_digest != expected_input for item in canonical_outcomes):
        raise CertificationError("probe outcomes do not match the certified descriptor, identity, and profile")
    attained = derive_attained_tier(profile, canonical_outcomes)
    ceiling = _descriptor_tier_ceiling(descriptor)
    if _TIER_ORDER[attained] > _TIER_ORDER[ceiling]:
        raise CertificationError(
            f"probe outcomes attain {attained.value}, but adapter declarations permit at most {ceiling.value}",
            code=CertificationRefusalCode.PROBE_UNSAFE.value,
        )
    document: dict[str, Any] = {
        "schema_version": CERTIFICATION_REPORT_VERSION,
        "issuer": CERTIFICATION_ISSUER,
        "profile_id": profile.profile_id,
        "profile_digest": profile.digest,
        "descriptor_digest": sha256_json(descriptor.model_dump(mode="json")),
        "source_identity_digest": source_identity_digest,
        "adapter_kind": descriptor.kind,
        "outcomes": [item.model_dump(mode="json") for item in canonical_outcomes],
        "attained_tier": attained.value,
        "signing_key_id": authority.key_id,
    }
    document["report_digest"] = sha256_json(document)
    document["signature"] = b64encode(
        authority.private_key.sign(str(document["report_digest"]).encode("ascii"))
    ).decode("ascii")
    return AdapterCertificationReport.model_validate(document)


def load_certification_report(path: Path) -> AdapterCertificationReport:
    """Load a BFCL report while rejecting duplicate keys and schema drift."""

    source = path.resolve()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CertificationError(f"adapter certification report repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
        return AdapterCertificationReport.model_validate(document)
    except CertificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CertificationError(f"cannot load adapter certification report {source}: {exc}") from exc


def verify_certification_report(
    report: AdapterCertificationReport,
    *,
    descriptor: AdapterDescriptor,
    source_identity_digest: str,
    profile: CertificationProfile,
    required_tier: AdapterTier,
    trusted_public_keys: Mapping[str, Ed25519PublicKey],
    execution_inputs_digest: str | None = None,
) -> None:
    """Rebind a report to trusted inputs and enforce the requested certification tier."""

    expected = {
        "profile_id": profile.profile_id,
        "profile_digest": profile.digest,
        "descriptor_digest": sha256_json(descriptor.model_dump(mode="json")),
        "source_identity_digest": source_identity_digest,
        "adapter_kind": descriptor.kind,
    }
    mismatches = [name for name, value in expected.items() if getattr(report, name) != value]
    derived = derive_attained_tier(profile, report.outcomes)
    ceiling = _descriptor_tier_ceiling(descriptor)
    if _TIER_ORDER[derived] > _TIER_ORDER[ceiling]:
        mismatches.append("descriptor_tier_ceiling")
    expected_input = certification_input_digest(
        descriptor,
        source_identity_digest=source_identity_digest,
        profile=profile,
        execution_inputs_digest=execution_inputs_digest,
    )
    if any(item.input_digest != expected_input for item in report.outcomes):
        mismatches.append("probe_input_digest")
    if report.attained_tier != derived:
        mismatches.append("attained_tier")
    if mismatches:
        raise CertificationError(
            "adapter certification report does not match trusted inputs: " + ", ".join(sorted(mismatches))
        )
    public_key = trusted_public_keys.get(report.signing_key_id)
    if public_key is None:
        raise CertificationError(f"certification signing key {report.signing_key_id!r} is not trusted")
    try:
        public_key.verify(
            b64decode(report.signature, validate=True),
            report.report_digest.encode("ascii"),
        )
    except (InvalidSignature, ValueError) as exc:
        raise CertificationError("adapter certification signature is invalid") from exc
    if _TIER_ORDER[derived] < _TIER_ORDER[required_tier]:
        raise CertificationError(
            f"adapter attained {derived.value}, below required {required_tier.value}",
            code=CertificationRefusalCode.ADAPTER_UNDER_CERTIFIED.value,
        )


def certification_reference(
    report: AdapterCertificationReport,
) -> CertificationReference:
    """Build the v2 evidence reference for a report that attained a usable tier."""

    if report.attained_tier is AdapterTier.NONE:
        raise CertificationError("an uncertified report cannot be referenced as evidence")
    # Local import keeps the evidence schema dependent on the small contract module,
    # rather than creating an import cycle between the two persisted formats.
    from nemotron.steps.byob.runtime.source_adapters.evidence import (
        CERTIFICATION_REFERENCE_VERSION,
        CertificationReference,
    )

    return CertificationReference(
        reference_version=CERTIFICATION_REFERENCE_VERSION,
        report_schema_version=report.schema_version,
        report_digest=report.report_digest,
        descriptor_digest=report.descriptor_digest,
        issuer=report.issuer,
        profile_id=report.profile_id,
        attained_tier=report.attained_tier.value,
    )


MCP_PROBE_MAPPING: Mapping[CertificationProbe, tuple[str, ...]] = {
    CertificationProbe.IDENTITY_INTEGRITY: ("P1",),
    CertificationProbe.CATALOG_INTEGRITY: ("P2", "P3"),
    CertificationProbe.EXECUTABLE_OBSERVATION: ("P4",),
    CertificationProbe.STRUCTURED_ERROR_SHAPE: ("P7",),
    CertificationProbe.RESET_DETERMINISM: ("P5",),
    CertificationProbe.EPISODE_ISOLATION: ("P6",),
    CertificationProbe.CONFIRMATION_SAFETY: ("P8",),
    CertificationProbe.TIMEOUT_CLEANUP: ("P9",),
    CertificationProbe.MUTATION_DECLARATION: ("P10",),
    CertificationProbe.RESULT_SHAPE_COVERAGE: ("P11",),
}
_MCP_NOT_APPLICABLE = {
    CertificationProbe.STRUCTURED_ERROR_SHAPE: "no_structured_error_case",
    CertificationProbe.CONFIRMATION_SAFETY: "no_confirmation_tools",
}
_MCP_NOT_APPLICABLE_SOURCE_REASON = {
    CertificationProbe.STRUCTURED_ERROR_SHAPE: ("the pack declares no structured-error validation case"),
    CertificationProbe.CONFIRMATION_SAFETY: ("the pack declares no confirmation-gated tool"),
}


_FAILURE_REASONS: Mapping[CertificationProbe, tuple[CertificationRefusalCode, ...]] = MappingProxyType(
    {
        CertificationProbe.IDENTITY_INTEGRITY: (
            CertificationRefusalCode.IDENTITY_DRIFT,
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
        ),
        CertificationProbe.CATALOG_INTEGRITY: (
            CertificationRefusalCode.CATALOG_MISMATCH,
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.REVIEWED_SCHEMA_MISSING,
        ),
        CertificationProbe.EXECUTABLE_OBSERVATION: (
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.PROBE_TIMEOUT,
            CertificationRefusalCode.PROBE_UNSAFE,
            CertificationRefusalCode.UNKNOWN_COMMIT_STATE,
        ),
        CertificationProbe.STRUCTURED_ERROR_SHAPE: (
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.STRUCTURED_ERROR_MISMATCH,
        ),
        CertificationProbe.RESET_DETERMINISM: (
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.RESET_NONDETERMINISTIC,
        ),
        CertificationProbe.EPISODE_ISOLATION: (
            CertificationRefusalCode.EPISODE_STATE_LEAKAGE,
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
        ),
        CertificationProbe.CONFIRMATION_SAFETY: (
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.PROBE_UNSAFE,
        ),
        CertificationProbe.TIMEOUT_CLEANUP: (
            CertificationRefusalCode.CLEANUP_FAILED,
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.PROBE_TIMEOUT,
            CertificationRefusalCode.UNKNOWN_COMMIT_STATE,
        ),
        CertificationProbe.MUTATION_DECLARATION: (
            CertificationRefusalCode.MUTATION_DECLARATION_MISMATCH,
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
        ),
        CertificationProbe.RESULT_SHAPE_COVERAGE: (
            CertificationRefusalCode.PROBE_EVIDENCE_INVALID,
            CertificationRefusalCode.PROBE_FAILED,
            CertificationRefusalCode.PROBE_MISSING,
            CertificationRefusalCode.RESULT_SHAPE_INCOMPLETE,
        ),
    }
)
_EXECUTION_FAILURE_REASONS = frozenset(
    {
        CertificationRefusalCode.CLEANUP_FAILED,
        CertificationRefusalCode.PROBE_TIMEOUT,
        CertificationRefusalCode.PROBE_UNSAFE,
    }
)


def _execution_policy(
    probe: CertificationProbe,
    *,
    cleanup: CleanupKind,
    a0_cleanup: CleanupKind | None,
    timeout_s: float,
    max_call_overrides: Mapping[CertificationProbe, int],
    timeout_overrides: Mapping[CertificationProbe, float],
) -> ProbeExecutionPolicy:
    tier = _PROBE_TIER[probe]
    selected_cleanup = a0_cleanup if tier is AdapterTier.A0 and a0_cleanup else cleanup
    safety = {
        AdapterTier.A0: ProbeSafetyKind.IDENTITY_ONLY,
        AdapterTier.A1: ProbeSafetyKind.READ_ONLY,
        AdapterTier.A2: ProbeSafetyKind.RESET_ISOLATED,
    }[tier]
    max_calls = max_call_overrides.get(
        probe,
        {
            CertificationProbe.CATALOG_INTEGRITY: 2,
            CertificationProbe.EXECUTABLE_OBSERVATION: 2,
            CertificationProbe.STRUCTURED_ERROR_SHAPE: 2,
            CertificationProbe.RESET_DETERMINISM: 2,
            CertificationProbe.EPISODE_ISOLATION: 2,
            CertificationProbe.CONFIRMATION_SAFETY: 2,
            CertificationProbe.TIMEOUT_CLEANUP: 2,
            CertificationProbe.MUTATION_DECLARATION: 2,
            CertificationProbe.RESULT_SHAPE_COVERAGE: 4,
        }.get(probe, 1),
    )
    return ProbeExecutionPolicy(
        executor="bfcl",
        evidence_issuer=CERTIFICATION_ISSUER,
        input_binding=PROBE_INPUT_BINDING_VERSION,
        outcome_schema=PROBE_OUTCOME_VERSION,
        safety=safety,
        max_calls=max_calls,
        # A probe that opens one isolated episode per reviewed case costs as much wall
        # time as its own call budget allows, so those probes carry their own deadline.
        # Cleanup stays on the base deadline: it is one operation however many ran.
        timeout_s=timeout_overrides.get(probe, timeout_s),
        cleanup=selected_cleanup,
        cleanup_timeout_s=timeout_s,
    )


def _reference_profile(
    *,
    profile_id: str,
    adapter_kind: str,
    cleanup: CleanupKind,
    timeout_s: float,
    max_wall_time_s: float,
    max_total_calls: int = 24,
    a0_cleanup: CleanupKind | None = None,
    max_call_overrides: Mapping[CertificationProbe, int] | None = None,
    timeout_overrides: Mapping[CertificationProbe, float] | None = None,
) -> CertificationProfile:
    return CertificationProfile(
        schema_version=CERTIFICATION_PROFILE_VERSION,
        profile_id=profile_id,
        owner="bfcl",
        adapter_kinds=(adapter_kind,),
        max_total_calls=max_total_calls,
        max_wall_time_s=max_wall_time_s,
        probes=tuple(
            ProbeRequirement(
                probe=probe,
                execution=_execution_policy(
                    probe,
                    cleanup=cleanup,
                    a0_cleanup=a0_cleanup,
                    timeout_s=timeout_s,
                    max_call_overrides=max_call_overrides or {},
                    timeout_overrides=timeout_overrides or {},
                ),
                requirement=("conditional" if probe in _MCP_NOT_APPLICABLE else "required"),
                allowed_failure_reasons=tuple(
                    sorted(
                        set(_FAILURE_REASONS[probe]) | _EXECUTION_FAILURE_REASONS,
                        key=lambda item: item.value,
                    )
                ),
                allowed_not_applicable_reasons=((_MCP_NOT_APPLICABLE[probe],) if probe in _MCP_NOT_APPLICABLE else ()),
            )
            for probe in _PROBE_ORDER
        ),
    )


_MCP_REFERENCE_PROFILE = _reference_profile(
    profile_id="mcp-mode-a-v1",
    adapter_kind="mcp_mode_a",
    cleanup=CleanupKind.SESSION,
    # A probe against Mode A crosses two hops, the gateway and the MCP server behind it,
    # and every episode pays a session open on both. The old ten-second, twenty-four-call
    # budget only ever covered discovery, so it decided this transport's tier by
    # arithmetic. These budgets are the endpoint ones, because the cost is the same shape.
    timeout_s=60.0,
    max_wall_time_s=600.0,
    max_total_calls=128,
    max_call_overrides={
        CertificationProbe.IDENTITY_INTEGRITY: 2,
        CertificationProbe.EXECUTABLE_OBSERVATION: 16,
        CertificationProbe.STRUCTURED_ERROR_SHAPE: 8,
        CertificationProbe.RESET_DETERMINISM: 32,
        CertificationProbe.EPISODE_ISOLATION: 16,
        CertificationProbe.CONFIRMATION_SAFETY: 16,
        CertificationProbe.MUTATION_DECLARATION: 16,
        CertificationProbe.RESULT_SHAPE_COVERAGE: 16,
    },
)
_LOCAL_PYTHON_REFERENCE_PROFILE = _reference_profile(
    profile_id="local-python-v1",
    adapter_kind="local_python",
    cleanup=CleanupKind.PROCESS,
    a0_cleanup=CleanupKind.NONE,
    timeout_s=10.0,
    # Ten seconds still bounds every probe that reads identity or reuses one episode.
    # The two probes below open a fresh spawned interpreter per reviewed case, so their
    # cost follows the reviewed surface: a nine-tool source needs nine observation
    # episodes, and ten seconds would fail it for the size of its catalogue rather than
    # for misbehaving. Each is allowed four seconds per call its own budget admits.
    max_wall_time_s=180.0,
    max_total_calls=128,
    timeout_overrides={
        CertificationProbe.EXECUTABLE_OBSERVATION: 64.0,
        CertificationProbe.STRUCTURED_ERROR_SHAPE: 32.0,
    },
    max_call_overrides={
        CertificationProbe.EXECUTABLE_OBSERVATION: 16,
        CertificationProbe.STRUCTURED_ERROR_SHAPE: 8,
        CertificationProbe.RESET_DETERMINISM: 32,
        CertificationProbe.EPISODE_ISOLATION: 16,
        CertificationProbe.CONFIRMATION_SAFETY: 16,
        CertificationProbe.MUTATION_DECLARATION: 16,
        CertificationProbe.RESULT_SHAPE_COVERAGE: 16,
    },
)
_HTTP_PACKAGE_REFERENCE_PROFILE = _reference_profile(
    profile_id="http-package-v1",
    adapter_kind="http_package",
    cleanup=CleanupKind.SESSION,
    # One probe is several episodes, and one episode against an endpoint costs a worker
    # start and a fresh session. Fifteen seconds bounded a probe that only read identity;
    # it would now fail an honest source for being remote rather than for misbehaving.
    # The deadline that still means something is the per-call one the runner enforces.
    timeout_s=60.0,
    max_wall_time_s=600.0,
    max_total_calls=128,
    # An endpoint is probed by calling it, and one call per tool per probe is the floor
    # for that. The earlier budget of two calls could only cover identity and catalog,
    # which capped this transport at A0 by arithmetic rather than by evidence.
    max_call_overrides={
        CertificationProbe.IDENTITY_INTEGRITY: 2,
        CertificationProbe.EXECUTABLE_OBSERVATION: 16,
        CertificationProbe.STRUCTURED_ERROR_SHAPE: 8,
        CertificationProbe.RESET_DETERMINISM: 32,
        CertificationProbe.EPISODE_ISOLATION: 16,
        CertificationProbe.CONFIRMATION_SAFETY: 16,
        CertificationProbe.MUTATION_DECLARATION: 16,
        CertificationProbe.RESULT_SHAPE_COVERAGE: 16,
    },
)
PUBLISHED_CERTIFICATION_PROFILES: Mapping[str, CertificationProfile] = MappingProxyType(
    {
        "http_package": _HTTP_PACKAGE_REFERENCE_PROFILE,
        "local_python": _LOCAL_PYTHON_REFERENCE_PROFILE,
        "mcp_mode_a": _MCP_REFERENCE_PROFILE,
    }
)


def certification_profile_for(adapter_kind: str) -> CertificationProfile:
    """Resolve a built-in profile without importing or executing adapter code."""
    profile = PUBLISHED_CERTIFICATION_PROFILES.get(adapter_kind)
    if profile is None:
        raise CertificationError(
            f"no published certification profile for adapter kind {adapter_kind!r}",
            code=CertificationRefusalCode.PROFILE_MISMATCH.value,
        )
    if profile.adapter_kinds != (adapter_kind,):
        raise CertificationError(
            "published certification profile registry does not match its adapter kind",
            code=CertificationRefusalCode.PROFILE_MISMATCH.value,
        )
    return profile


def certification_profile_by_id(profile_id: str) -> CertificationProfile:
    """Resolve one published profile by its persisted identity."""
    matches = tuple(
        profile for profile in PUBLISHED_CERTIFICATION_PROFILES.values() if profile.profile_id == profile_id
    )
    if len(matches) != 1:
        raise CertificationError(
            f"no unique published certification profile for id {profile_id!r}",
            code=CertificationRefusalCode.PROFILE_MISMATCH.value,
        )
    return matches[0]


def mcp_reference_profile() -> CertificationProfile:
    return certification_profile_for("mcp_mode_a")


def local_python_reference_profile() -> CertificationProfile:
    return certification_profile_for("local_python")


def http_package_reference_profile() -> CertificationProfile:
    return certification_profile_for("http_package")


def project_mcp_probe_report(
    document: Mapping[str, Any],
    *,
    input_digest: str,
    structured_error_applicable: bool,
    confirmation_applicable: bool,
) -> tuple[ProbeOutcome, ...]:
    """Project strict P1–P11 results onto generic certification probes."""

    if set(document) != {"probes"} or not isinstance(document.get("probes"), list):
        raise CertificationError("MCP probe report must contain exactly a probes array")
    raw = document["probes"]
    indexed: dict[str, Mapping[str, Any]] = {}
    identifiers: list[str] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != {
            "id",
            "requirement",
            "status",
            "reason",
        }:
            raise CertificationError("MCP probe entries have an invalid field set")
        if not isinstance(entry.get("id"), str):
            raise CertificationError("MCP probe entries must have string ids")
        identifier = str(entry["id"])
        if identifier in indexed:
            raise CertificationError(f"MCP probe report repeats {identifier}")
        if identifier not in {f"P{index}" for index in range(1, 12)}:
            raise CertificationError(f"MCP probe report contains unknown probe {identifier}")
        expected_requirement = "conditional" if identifier in {"P7", "P8"} else "required"
        if entry.get("requirement") != expected_requirement:
            raise CertificationError(f"MCP probe {identifier} has an invalid requirement")
        if entry.get("status") not in {
            "pass",
            "fail",
            "skipped",
            "not_applicable",
        }:
            raise CertificationError(f"MCP probe {identifier} has an invalid status")
        reason = entry.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise CertificationError(f"MCP probe {identifier} reason must be null or non-empty")
        status = entry["status"]
        if status == "pass" and reason is not None:
            raise CertificationError(f"passing MCP probe {identifier} cannot carry a reason")
        if status != "pass" and reason is None:
            raise CertificationError(f"non-passing MCP probe {identifier} requires a reason")
        if status == "not_applicable" and expected_requirement != "conditional":
            raise CertificationError(f"required MCP probe {identifier} cannot be not_applicable")
        identifiers.append(identifier)
        indexed[identifier] = entry
    expected_order = [f"P{index}" for index in range(1, 12) if f"P{index}" in indexed]
    if identifiers != expected_order:
        raise CertificationError("MCP probes must appear in P1 through P11 order")

    outcomes: list[ProbeOutcome] = []
    applicability = {
        CertificationProbe.STRUCTURED_ERROR_SHAPE: structured_error_applicable,
        CertificationProbe.CONFIRMATION_SAFETY: confirmation_applicable,
    }
    for probe in _PROBE_ORDER:
        mapped_ids = MCP_PROBE_MAPPING[probe]
        entries = [indexed.get(identifier) for identifier in mapped_ids]
        evidence_document = [dict(entry) for entry in entries if entry is not None]
        evidence_digest = sha256_json(evidence_document)
        if any(entry is None for entry in entries):
            outcomes.append(
                ProbeOutcome(
                    probe=probe,
                    status="fail",
                    input_digest=input_digest,
                    reason=CertificationRefusalCode.PROBE_MISSING.value,
                )
            )
            continue
        statuses = [entry.get("status") for entry in entries if entry is not None]
        if probe in applicability:
            applicable = applicability[probe]
            source_reason = next(entry.get("reason") for entry in entries if entry is not None)
            if not applicable and (
                statuses != ["not_applicable"] or source_reason != _MCP_NOT_APPLICABLE_SOURCE_REASON[probe]
            ):
                raise CertificationError(
                    f"MCP probe {mapped_ids[0]} contradicts BFCL-derived applicability",
                    code=CertificationRefusalCode.APPLICABILITY_MISMATCH.value,
                )
            if applicable and "not_applicable" in statuses:
                raise CertificationError(
                    f"MCP probe {mapped_ids[0]} cannot be not_applicable",
                    code=CertificationRefusalCode.APPLICABILITY_MISMATCH.value,
                )
        if all(status == "pass" for status in statuses):
            outcomes.append(
                ProbeOutcome(
                    probe=probe,
                    status="pass",
                    input_digest=input_digest,
                    evidence_digest=evidence_digest,
                    evidence=evidence_document,
                )
            )
            continue
        allowed_reason = _MCP_NOT_APPLICABLE.get(probe)
        if allowed_reason is not None and all(status == "not_applicable" for status in statuses):
            outcomes.append(
                ProbeOutcome(
                    probe=probe,
                    status="not_applicable",
                    input_digest=input_digest,
                    evidence_digest=evidence_digest,
                    evidence=evidence_document,
                    reason=allowed_reason,
                )
            )
            continue
        outcomes.append(
            ProbeOutcome(
                probe=probe,
                status="fail",
                input_digest=input_digest,
                evidence_digest=evidence_digest,
                evidence=evidence_document,
                reason=CertificationRefusalCode.PROBE_FAILED.value,
            )
        )
    return tuple(outcomes)
