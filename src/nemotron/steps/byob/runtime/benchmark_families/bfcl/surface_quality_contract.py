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

"""Versioned, truth-agnostic contract for BFCL surface-quality checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator

SURFACE_QUALITY_CONTRACT_VERSION = "1.1"
SURFACE_QUALITY_CHECKS = (
    "surface_shape",
    "semantic_preservation",
    "leakage",
    "language_locale",
    "fluency_naturalness",
    "clarity_coherence",
)
DETERMINISTIC_SURFACE_CHECKS = frozenset(SURFACE_QUALITY_CHECKS[:3])
JUDGED_SURFACE_CHECKS = frozenset(SURFACE_QUALITY_CHECKS[3:])
SURFACE_CHECK_OWNER = {
    check: "python" if check in DETERMINISTIC_SURFACE_CHECKS else "surface_judge" for check in SURFACE_QUALITY_CHECKS
}
SURFACE_QUALITY_REASON_CODES = {
    "surface_shape": frozenset(
        {
            "empty_user_turn",
            "unchanged_surface",
            "user_turn_count_changed",
        }
    ),
    "semantic_preservation": frozenset(
        {
            "must_omit",
            "must_preserve",
            "novel_literal",
        }
    ),
    "leakage": frozenset(
        {
            "expected_result_leakage",
            "forbidden_mention",
            "tool_name_leakage",
        }
    ),
    "language_locale": frozenset(
        {
            "invalid_locale_format",
            "mixed_language",
            "wrong_language",
        }
    ),
    "fluency_naturalness": frozenset(
        {
            "grammar_error",
            "model_artifact",
            "repetition",
            "unnatural_wording",
        }
    ),
    "clarity_coherence": frozenset(
        {
            "ambiguous_reference",
            "contradictory_turn",
            "incoherent_turn",
        }
    ),
}
SURFACE_QUALITY_ERROR_CODES = frozenset(
    {
        "invalid_response",
        "judge_error",
        "missing_response",
        "model_contract",
    }
)
INAPPLICABLE_FAILURES_BY_TURN_POLICY = {
    # Ambiguity is the intended capability under test for clarify-only tasks.
    # Stage 10 may still reject contradiction or incoherence in the wording.
    "clarify_only": frozenset({("clarity_coherence", "ambiguous_reference")}),
}

SurfaceQualityCheckName = Literal[
    "surface_shape",
    "semantic_preservation",
    "leakage",
    "language_locale",
    "fluency_naturalness",
    "clarity_coherence",
]
SurfaceQualitySource = Literal["python", "surface_judge"]
SurfaceQualityStatus = Literal[
    "passed",
    "failed",
    "not_applicable",
    "not_run",
    "error",
]


class SurfaceQualityCheckResult(BaseModel):
    """One normalized check result written by Python or the surface judge."""

    model_config = ConfigDict(extra="forbid")

    check: SurfaceQualityCheckName
    status: SurfaceQualityStatus
    source: SurfaceQualitySource
    reason_code: str | None = None
    evidence: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> SurfaceQualityCheckResult:
        expected_source = SURFACE_CHECK_OWNER[self.check]
        if self.source != expected_source:
            raise ValueError(f"{self.check} must be produced by {expected_source}, got {self.source}")
        if self.status in {"not_run", "error"} and self.check not in JUDGED_SURFACE_CHECKS:
            raise ValueError(f"{self.check} is deterministic and cannot be {self.status}")
        if self.status == "not_applicable":
            if self.check not in JUDGED_SURFACE_CHECKS:
                raise ValueError(f"{self.check} is deterministic and cannot be not_applicable")
            if (
                not isinstance(self.reason_code, str)
                or self.reason_code not in SURFACE_QUALITY_REASON_CODES[self.check]
            ):
                raise ValueError("a not_applicable surface-quality check requires a valid dimension reason_code")
            if self.evidence is not None:
                raise ValueError("a not_applicable surface-quality check cannot carry evidence")
            return self
        if self.status in {"passed", "not_run"}:
            if self.reason_code is not None or self.evidence is not None:
                raise ValueError(f"a {self.status} surface-quality check cannot carry failure detail")
            return self
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError(f"a {self.status} surface-quality check requires reason_code")
        if self.status == "error":
            if self.reason_code not in SURFACE_QUALITY_ERROR_CODES:
                raise ValueError(f"{self.reason_code!r} is not a valid surface-quality error code")
            if self.evidence is not None:
                raise ValueError("a surface-quality error cannot carry free-text evidence")
            return self
        if self.reason_code not in SURFACE_QUALITY_REASON_CODES[self.check]:
            raise ValueError(f"{self.reason_code!r} is not valid for surface-quality check {self.check!r}")
        if self.source == "surface_judge" and self.evidence is not None:
            raise ValueError("surface-judge checks cannot carry free-text evidence")
        if self.evidence is not None and not self.evidence.strip():
            raise ValueError("surface-quality evidence must be non-empty when present")
        return self


class SurfaceJudgeDimensionResult(BaseModel):
    """Surface-only judgement for one model-owned quality dimension.

    The model may only return a pass/fail verdict and a controlled reason code.
    Free-text evidence is forbidden so a judge cannot smuggle tool-correctness
    claims into the stored result.
    """

    model_config = ConfigDict(extra="forbid")

    passed: StrictBool
    reason_code: str | None = None


class SurfaceJudgeResult(BaseModel):
    """Exact structured response allowed from the optional surface judge."""

    model_config = ConfigDict(extra="forbid")

    language_locale: SurfaceJudgeDimensionResult
    fluency_naturalness: SurfaceJudgeDimensionResult
    clarity_coherence: SurfaceJudgeDimensionResult

    @model_validator(mode="after")
    def validate_dimensions(self) -> SurfaceJudgeResult:
        for check in JUDGED_SURFACE_CHECKS:
            result = getattr(self, check)
            if result.passed:
                if result.reason_code is not None:
                    raise ValueError(f"a passing {check} judgement cannot carry failure detail")
                continue
            if not isinstance(result.reason_code, str) or not result.reason_code.strip():
                raise ValueError(f"a failing {check} judgement requires reason_code")
            if result.reason_code not in SURFACE_QUALITY_REASON_CODES[check]:
                raise ValueError(f"{result.reason_code!r} is not valid for surface-quality check {check!r}")
        return self

    def check_results(self) -> list[SurfaceQualityCheckResult]:
        """Project the model response into the common six-check result shape."""
        return [
            SurfaceQualityCheckResult(
                check=check,
                status="passed" if getattr(self, check).passed else "failed",
                source="surface_judge",
                reason_code=getattr(self, check).reason_code,
            )
            for check in SURFACE_QUALITY_CHECKS
            if check in JUDGED_SURFACE_CHECKS
        ]


def not_run_judge_checks() -> list[SurfaceQualityCheckResult]:
    """Record the three judged checks as skipped when no judge ran."""
    return [
        SurfaceQualityCheckResult(
            check=check,
            status="not_run",
            source="surface_judge",
        )
        for check in SURFACE_QUALITY_CHECKS
        if check in JUDGED_SURFACE_CHECKS
    ]


def judge_error_checks(reason_code: str) -> list[SurfaceQualityCheckResult]:
    """Record the three judged checks as infrastructure failures."""
    return [
        SurfaceQualityCheckResult(
            check=check,
            status="error",
            source="surface_judge",
            reason_code=reason_code,
        )
        for check in SURFACE_QUALITY_CHECKS
        if check in JUDGED_SURFACE_CHECKS
    ]


def validate_complete_check_set(
    values: Sequence[SurfaceQualityCheckResult | dict[str, Any]],
    *,
    turn_policy: str,
) -> list[SurfaceQualityCheckResult]:
    """Validate one result per check, including task-policy applicability."""
    results = [
        value if isinstance(value, SurfaceQualityCheckResult) else SurfaceQualityCheckResult.model_validate(value)
        for value in values
    ]
    by_check: dict[str, SurfaceQualityCheckResult] = {}
    for result in results:
        if result.check in by_check:
            raise ValueError(f"duplicate surface-quality check {result.check!r}")
        by_check[result.check] = result
    missing = [check for check in SURFACE_QUALITY_CHECKS if check not in by_check]
    if missing:
        raise ValueError("surface-quality result is missing checks: " + ", ".join(missing))
    ordered = [by_check[check] for check in SURFACE_QUALITY_CHECKS]
    inapplicable = INAPPLICABLE_FAILURES_BY_TURN_POLICY.get(
        turn_policy,
        frozenset(),
    )
    invalid = [
        (result.check, result.reason_code)
        for result in ordered
        if result.status == "failed" and (result.check, result.reason_code) in inapplicable
    ]
    if invalid:
        detail = ", ".join(f"{check}:{reason}" for check, reason in invalid)
        raise ValueError(f"surface-quality failures are inapplicable to turn_policy {turn_policy!r}: {detail}")
    invalid_not_applicable = [
        (result.check, result.reason_code)
        for result in ordered
        if result.status == "not_applicable" and (result.check, result.reason_code) not in inapplicable
    ]
    if invalid_not_applicable:
        detail = ", ".join(f"{check}:{reason}" for check, reason in invalid_not_applicable)
        raise ValueError(
            f"surface-quality checks are not applicable only when allowed by turn_policy {turn_policy!r}: {detail}"
        )
    return ordered
