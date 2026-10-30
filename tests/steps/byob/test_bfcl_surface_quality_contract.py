from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_quality_contract import (
    DETERMINISTIC_SURFACE_CHECKS,
    JUDGED_SURFACE_CHECKS,
    SURFACE_CHECK_OWNER,
    SURFACE_QUALITY_CHECKS,
    SurfaceJudgeResult,
    SurfaceQualityCheckResult,
    judge_error_checks,
    not_run_judge_checks,
    validate_complete_check_set,
)


def _judge_payload() -> dict:
    return {
        "language_locale": {"passed": True},
        "fluency_naturalness": {
            "passed": False,
            "reason_code": "unnatural_wording",
        },
        "clarity_coherence": {"passed": True},
    }


def _deterministic_passes() -> list[SurfaceQualityCheckResult]:
    return [
        SurfaceQualityCheckResult(check=check, status="passed", source="python")
        for check in SURFACE_QUALITY_CHECKS
        if check in DETERMINISTIC_SURFACE_CHECKS
    ]


def test_six_check_contract_has_disjoint_fixed_ownership() -> None:
    assert len(SURFACE_QUALITY_CHECKS) == 6
    assert DETERMINISTIC_SURFACE_CHECKS.isdisjoint(JUDGED_SURFACE_CHECKS)
    assert DETERMINISTIC_SURFACE_CHECKS | JUDGED_SURFACE_CHECKS == set(SURFACE_QUALITY_CHECKS)
    assert {check for check, owner in SURFACE_CHECK_OWNER.items() if owner == "python"} == DETERMINISTIC_SURFACE_CHECKS
    assert {check for check, owner in SURFACE_CHECK_OWNER.items() if owner == "surface_judge"} == JUDGED_SURFACE_CHECKS


def test_check_result_enforces_owner_and_failure_shape() -> None:
    result = SurfaceQualityCheckResult(
        check="semantic_preservation",
        status="failed",
        source="python",
        reason_code="must_preserve",
        evidence="account_id",
    )
    assert result.status == "failed"
    assert result.reason_code == "must_preserve"

    with pytest.raises(ValidationError, match="must be produced by python"):
        SurfaceQualityCheckResult(
            check="semantic_preservation",
            status="failed",
            source="surface_judge",
            reason_code="must_preserve",
        )
    with pytest.raises(ValidationError, match="cannot carry failure detail"):
        SurfaceQualityCheckResult(
            check="language_locale",
            status="passed",
            source="surface_judge",
            reason_code="wrong_language",
        )
    with pytest.raises(ValidationError, match="not valid"):
        SurfaceQualityCheckResult(
            check="leakage",
            status="failed",
            source="python",
            reason_code="grammar_error",
        )


def test_disabled_judge_records_not_run_instead_of_inventing_a_pass() -> None:
    complete = validate_complete_check_set(
        [*_deterministic_passes(), *not_run_judge_checks()],
        turn_policy="single_turn",
    )

    judged = [result for result in complete if result.check in JUDGED_SURFACE_CHECKS]
    assert all(result.status == "not_run" for result in judged)
    assert all("passed" not in result.model_dump() for result in judged)

    with pytest.raises(ValidationError, match="cannot be not_run"):
        SurfaceQualityCheckResult(
            check="leakage",
            status="not_run",
            source="python",
        )


def test_judge_errors_are_distinct_from_quality_failures() -> None:
    complete = validate_complete_check_set(
        [*_deterministic_passes(), *judge_error_checks("missing_response")],
        turn_policy="single_turn",
    )
    judged = [result for result in complete if result.check in JUDGED_SURFACE_CHECKS]
    assert {result.status for result in judged} == {"error"}
    assert {result.reason_code for result in judged} == {"missing_response"}
    assert all("passed" not in result.model_dump() for result in judged)

    with pytest.raises(ValidationError, match="not a valid surface-quality error code"):
        SurfaceQualityCheckResult(
            check="fluency_naturalness",
            status="error",
            source="surface_judge",
            reason_code="unnatural_wording",
        )


def test_clarity_does_not_treat_missing_information_as_a_quality_failure() -> None:
    checks = [
        *_deterministic_passes(),
        SurfaceQualityCheckResult(
            check="language_locale",
            status="passed",
            source="surface_judge",
        ),
        SurfaceQualityCheckResult(
            check="fluency_naturalness",
            status="passed",
            source="surface_judge",
        ),
        SurfaceQualityCheckResult(
            check="clarity_coherence",
            status="failed",
            source="surface_judge",
            reason_code="ambiguous_reference",
        ),
    ]

    with pytest.raises(ValueError, match="inapplicable.*clarify_only"):
        validate_complete_check_set(checks, turn_policy="clarify_only")
    assert validate_complete_check_set(checks, turn_policy="single_turn")[-1].status == "failed"


def test_not_applicable_is_judge_only_and_preserves_the_reason() -> None:
    result = SurfaceQualityCheckResult(
        check="clarity_coherence",
        status="not_applicable",
        source="surface_judge",
        reason_code="ambiguous_reference",
    )
    assert result.reason_code == "ambiguous_reference"
    judged = SurfaceJudgeResult.model_validate(_judge_payload()).check_results()
    complete = [*_deterministic_passes(), *[result if item.check == result.check else item for item in judged]]
    assert validate_complete_check_set(complete, turn_policy="clarify_only")[-1] == result
    with pytest.raises(ValueError, match="only when allowed by turn_policy"):
        validate_complete_check_set(complete, turn_policy="single_turn")

    with pytest.raises(ValidationError, match="deterministic"):
        SurfaceQualityCheckResult(
            check="surface_shape",
            status="not_applicable",
            source="python",
            reason_code="empty_user_turn",
        )
    with pytest.raises(ValidationError, match="valid dimension reason_code"):
        SurfaceQualityCheckResult(
            check="clarity_coherence",
            status="not_applicable",
            source="surface_judge",
            reason_code="wrong_language",
        )


def test_surface_judge_response_is_exactly_surface_only() -> None:
    response = SurfaceJudgeResult.model_validate(_judge_payload())
    checks = response.check_results()

    assert [result.check for result in checks] == [
        "language_locale",
        "fluency_naturalness",
        "clarity_coherence",
    ]
    assert all(result.source == "surface_judge" for result in checks)
    assert all(result.evidence is None for result in checks)
    assert [result.status for result in checks] == ["passed", "failed", "passed"]

    invalid = _judge_payload()
    invalid["tool_correctness"] = {"passed": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SurfaceJudgeResult.model_validate(invalid)

    invalid = _judge_payload()
    invalid["clarity_coherence"] = {"passed": False}
    with pytest.raises(ValidationError, match="requires reason_code"):
        SurfaceJudgeResult.model_validate(invalid)

    invalid = _judge_payload()
    invalid["fluency_naturalness"]["evidence"] = "The correct tool should be lookup_book."
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SurfaceJudgeResult.model_validate(invalid)


@pytest.mark.parametrize("coerced", ["true", "false", "yes", 1, 0])
def test_surface_judge_rejects_coercive_boolean_values(coerced: object) -> None:
    payload = _judge_payload()
    payload["language_locale"]["passed"] = coerced

    with pytest.raises(ValidationError):
        SurfaceJudgeResult.model_validate(payload)


def test_complete_check_set_requires_each_check_once_in_canonical_order() -> None:
    judge_checks = SurfaceJudgeResult.model_validate(_judge_payload()).check_results()
    deterministic = _deterministic_passes()

    complete = validate_complete_check_set(
        [*judge_checks, *deterministic],
        turn_policy="single_turn",
    )
    assert [result.check for result in complete] == list(SURFACE_QUALITY_CHECKS)

    with pytest.raises(ValueError, match="missing checks"):
        validate_complete_check_set(deterministic, turn_policy="single_turn")
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_check_set(
            [*complete, complete[0]],
            turn_policy="single_turn",
        )
