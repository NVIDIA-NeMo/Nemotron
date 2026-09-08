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

"""Decide which rows each candidate may be scored on.

A verified source also says which models read its rows. This module compares
those models against the candidates and turns the answer into an authorization:
:class:`...contamination_contract.EligibleEvalPlan`.

The decision has three moving parts, and the order they are applied in is the
contract.

*Compare, then classify.* Every candidate is compared against every exposure.
A ``match`` is a violation. A ``different`` is silent. An ``unknown`` — the two
identities cannot be told apart — is recorded as evidence and never treated as
either, because guessing in one direction clears a contaminated candidate and
guessing in the other destroys a valid benchmark.

*Apply the policy, which only ever narrows.* ``fail_run`` refuses the run;
``exclude_row`` removes exactly the rows the exposure covers. Unresolved
evidence does neither: it does not shrink a task set on suspicion, and it does
not abort a debug run. What it always does is block publication — and when the
operator asked for a publishable run, the refusal happens here rather than
producing a number that cannot be published.

*Intersect last.* Under ``common_intersection`` every candidate is scored on the
rows all of them may answer, so the comparison is comparable by construction.
Under ``per_candidate`` each keeps its own set, which is why the evaluation
configuration contract refuses to call such a run publishable.

Nothing here contacts a model or reads the benchmark again: the gate runs on the
verified handle alone, so it cannot disagree with what was verified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    CONTAMINATION_CONTRACT_VERSION,
    CONTAMINATION_FAILURE_FILE,
    CONTAMINATION_REPORT_FILE,
    CandidateEligibility,
    CommonEvaluationTaskSet,
    ContaminationCollision,
    ContaminationReport,
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_errors import (
    CandidateContaminationError,
    ContaminationError,
    ContaminationPlanDriftError,
    EmptyEvaluationTaskSetError,
    TaskSetConsistencyError,
    UnresolvedContaminationError,
    describe_contamination_error,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    ModelIdentityClaim,
    VerifiedModelExposure,
    candidate_identity_claim,
    compare_model_identity,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import BfclEvalConfig, EvalCandidate
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    SourceCheck,
    VerifiedEvalSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_verification import (
    assert_source_unchanged,
    write_eval_artifact,
)

# Why a collision was recorded, in the report's own words. These are the only
# two ways a candidate can fail to be separated from a model that read the rows.
_MATCH_REASON = "the candidate and the exposed model resolve to the same weights"
_UNKNOWN_REASON = "the candidate and the exposed model cannot be told apart from what either side pinned"


def evaluate_contamination(config: BfclEvalConfig, source: VerifiedEvalSource) -> EligibleEvalPlan:
    """Gate the candidates against the source's exposures and plan the run.

    Raises rather than returns a degraded plan whenever continuing would produce
    a number that misrepresents what it measured: a candidate that helped write
    the tasks under ``fail_run``, an unresolvable identity in a run that asked to
    be published, or a task set that contamination emptied out.
    """
    if source.eval_config_hash != config.eval_config_hash:
        raise ContaminationPlanDriftError(
            "eval_config",
            "is not the config the source was verified against",
            actual=config.eval_config_hash,
            expected=source.eval_config_hash,
            recovery="verify the source with this config before gating candidates against it",
        )
    checks: list[SourceCheck] = []
    exposures = source.exposures
    findings = tuple(
        _examine_candidate(config, source, candidate, exposures)
        for candidate in sorted(config.candidates, key=lambda item: item.alias)
    )
    _refuse_violations(config, source, findings)
    eligibility = tuple(_eligibility(finding) for finding in findings)
    common = _common_task_set(config, source, eligibility)
    reasons = _non_publication_reasons(config, eligibility)
    checks.append(
        SourceCheck(
            name="exposure_inventory",
            detail=(
                f"{len(exposures)} exposure(s) checked against {len(eligibility)} candidate(s): "
                + (
                    ", ".join(
                        f"{exposure.role} as {exposure.display_name} over {len(exposure.task_ids)} row(s)"
                        for exposure in exposures
                    )
                    or "no model read the published rows"
                )
            ),
        )
    )
    checks.append(
        SourceCheck(
            name="comparable_task_set",
            detail=(
                f"{common.task_count} of {source.task_index.task_count} published row(s) are answerable by "
                f"every candidate under {config.contamination.comparison_set}, as {common.task_ids_hash}"
            ),
        )
    )
    try:
        return EligibleEvalPlan(
            eval_config_hash=config.eval_config_hash,
            scoring_policy_hash=config.scoring.scoring_policy_hash,
            source_verification_identity=source.verification_identity,
            source_run_id=source.source_run_id,
            source_task_ids_hash=source.task_index.task_ids_hash,
            enforce=config.contamination.enforce,
            on_violation=config.contamination.on_violation,
            comparison_set=config.contamination.comparison_set,
            exposures=exposures,
            candidates=eligibility,
            common=common,
            publication_allowed=not reasons,
            non_publication_reasons=reasons,
            checks=tuple(checks),
        )
    except ValidationError as exc:
        raise TaskSetConsistencyError(
            "contamination.plan",
            f"the gated candidates do not form a comparable plan: {exc.errors()[0].get('msg', 'invalid')}",
            expected="one task set per candidate, all carrying the common set in publication order",
            recovery="re-verify the source and re-run the gate; a plan whose own task sets disagree cannot "
            "authorize a comparison",
        ) from exc


@dataclass(frozen=True)
class _CandidateFinding:
    """One candidate compared against every exposure, before any refusal.

    Every candidate is examined before the first one is refused, so an operator
    evaluating four models learns which of them are contaminated in one run
    instead of one per run.
    """

    candidate: EvalCandidate
    claim: ModelIdentityClaim
    collisions: tuple[ContaminationCollision, ...]
    excluded_task_ids: tuple[str, ...]
    eligible_task_ids: tuple[str, ...]
    published_task_count: int

    @property
    def alias(self) -> str:
        return self.candidate.alias

    @property
    def definite(self) -> tuple[ContaminationCollision, ...]:
        return tuple(collision for collision in self.collisions if collision.verdict == "match")

    @property
    def unresolved(self) -> tuple[ContaminationCollision, ...]:
        return tuple(collision for collision in self.collisions if collision.verdict == "unknown")

    def read_everything(self) -> bool:
        return not self.eligible_task_ids

    def describe_exposure(self) -> str:
        return f"{self.alias} " + ", ".join(
            f"as the {collision.role} model over {len(collision.task_ids)} of the "
            f"{self.published_task_count} published row(s)"
            for collision in self.definite
        )

    def describe_unresolved(self) -> str:
        return f"{self.alias} from the " + ", ".join(
            f"{collision.role} model {collision.exposed_model}" for collision in self.unresolved
        )


def _examine_candidate(
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    candidate: EvalCandidate,
    exposures: Sequence[VerifiedModelExposure],
) -> _CandidateFinding:
    """Compare one candidate against every exposure and apply the policy.

    This decides what the candidate may answer but refuses nothing: whether a
    collision ends the run is a question about the whole comparison, answered
    once every candidate has been examined.
    """
    claim = candidate_identity_claim(candidate)
    collisions = tuple(collision for exposure in exposures if (collision := _collision(claim, exposure)) is not None)
    definite = tuple(collision for collision in collisions if collision.verdict == "match")
    if config.contamination.enforce and definite and config.contamination.on_violation == "exclude_row":
        excluded = _ordered(
            source.task_index.task_ids,
            {task_id for collision in definite for task_id in collision.task_ids},
        )
    else:
        # Under fail_run a collision ends the run rather than narrowing it, and
        # with enforcement off it is recorded without acting on it. Unresolved
        # evidence never excludes a row: suspicion may not shrink a task set.
        excluded = ()
    excluded_set = set(excluded)
    return _CandidateFinding(
        candidate=candidate,
        claim=claim,
        collisions=collisions,
        excluded_task_ids=excluded,
        eligible_task_ids=tuple(task_id for task_id in source.task_index.task_ids if task_id not in excluded_set),
        published_task_count=source.task_index.task_count,
    )


def _refuse_violations(
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    findings: Sequence[_CandidateFinding],
) -> None:
    """Refuse the run once, naming every candidate that caused it.

    The three refusals are ordered by how much they settle. A proven collision
    under ``fail_run`` ends the run whatever else is true. An unresolved identity
    only ends a run that asked to be published. An emptied task set is last,
    because it is a consequence of exclusions the earlier checks let through.
    """
    if config.contamination.enforce and config.contamination.on_violation == "fail_run":
        if exposed := [finding for finding in findings if finding.definite]:
            raise CandidateContaminationError(
                _subject(exposed),
                "already read the rows they would be scored on — "
                + "; ".join(finding.describe_exposure() for finding in exposed)
                + " — so the score would measure recall of their own output",
                actual=exposed[0].definite[0].task_ids_hash,
                expected="candidates that did not read these rows while they were being built",
                recovery="evaluate different candidates, regenerate the benchmark with a different "
                "profile/paraphrase/judge model, or set contamination.on_violation to exclude_row and accept "
                "a score over the remaining rows",
            )
    if config.contamination.enforce and config.publication.requested:
        if ambiguous := [finding for finding in findings if finding.unresolved]:
            raise UnresolvedContaminationError(
                _subject(ambiguous),
                "cannot be told apart — "
                + "; ".join(finding.describe_unresolved() for finding in ambiguous)
                + " — so a published score could not claim they had not seen these rows",
                expected="a candidate and an exposed model that pin enough identity to be compared: a weights "
                "digest on both sides, an immutable revision on both sides, or provably different model names",
                recovery="pin weights_digest on the candidate and record the same identity for the generation role, "
                "or set publication.requested to false and treat this run as debug-only",
            )
    if emptied := [finding for finding in findings if finding.read_everything()]:
        raise EmptyEvaluationTaskSetError(
            _subject(emptied),
            f"read every one of {source.task_index.task_count} published row(s) while they were being built, "
            "so excluding what they saw leaves nothing to score",
            actual=source.task_index.task_ids_hash,
            expected="at least one published row the candidate did not help produce",
            recovery="evaluate these candidates against a benchmark they did not build, or regenerate the "
            "benchmark with a different surface model",
        )


def _subject(findings: Sequence[_CandidateFinding]) -> str:
    return f"candidates[{', '.join(finding.alias for finding in findings)}]"


def _eligibility(finding: _CandidateFinding) -> CandidateEligibility:
    try:
        return CandidateEligibility(
            alias=finding.alias,
            identity=finding.claim,
            canonical_model_identity=finding.candidate.canonical_model_identity,
            collisions=finding.collisions,
            eligible_task_ids=finding.eligible_task_ids,
            excluded_task_ids=finding.excluded_task_ids,
        )
    except ValidationError as exc:
        raise TaskSetConsistencyError(
            f"candidates[{finding.alias}]",
            f"does not have a usable task set: {exc.errors()[0].get('msg', 'invalid')}",
            expected="disjoint eligible and excluded task sets, both drawn from the published rows",
            recovery="re-verify the source and re-run the gate",
        ) from exc


def _collision(
    claim: ModelIdentityClaim,
    exposure: VerifiedModelExposure,
) -> ContaminationCollision | None:
    """Record a collision unless the candidate is provably a different model."""
    verdict = compare_model_identity(claim, exposure.identity)
    if verdict == "different":
        return None
    return ContaminationCollision(
        role=exposure.role,
        scope=exposure.scope,
        verdict=verdict,
        exposed_model=exposure.display_name,
        reason=_MATCH_REASON if verdict == "match" else _UNKNOWN_REASON,
        task_ids=exposure.task_ids,
    )


def _common_task_set(
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    eligibility: Sequence[CandidateEligibility],
) -> CommonEvaluationTaskSet:
    """Intersect the candidates' eligible sets, keeping publication order."""
    shared = set(source.task_index.task_ids)
    for candidate in eligibility:
        shared &= set(candidate.eligible_task_ids)
    task_ids = _ordered(source.task_index.task_ids, shared)
    if not task_ids:
        raise EmptyEvaluationTaskSetError(
            "contamination.comparison_set",
            "no published row is answerable by every candidate once each one's exposures are excluded",
            actual=source.task_index.task_ids_hash,
            expected="at least one row none of the candidates helped produce",
            recovery="evaluate the candidates separately with contamination.comparison_set per_candidate and "
            "publication.requested false, or regenerate the benchmark with surface models that are not under "
            "evaluation",
        )
    try:
        return CommonEvaluationTaskSet(
            comparison_set=config.contamination.comparison_set,
            task_ids=task_ids,
            candidate_aliases=tuple(candidate.alias for candidate in eligibility),
        )
    except ValidationError as exc:
        raise TaskSetConsistencyError(
            "contamination.comparison_set",
            f"the candidates cannot be compared: {exc.errors()[0].get('msg', 'invalid')}",
            expected="a non-empty shared task set over uniquely named candidates",
            recovery="re-run the gate against a verified source",
        ) from exc


def _non_publication_reasons(
    config: BfclEvalConfig,
    eligibility: Sequence[CandidateEligibility],
) -> tuple[str, ...]:
    """Every reason this plan's results may not be published, config first.

    The config's own reasons come first and unchanged, so a run that was already
    debug-only does not have its cause replaced by a contamination finding.
    """
    reasons = list(config.non_publication_reasons)
    for candidate in eligibility:
        if candidate.unresolved:
            reasons.append(f"contamination.unresolved:{candidate.alias}")
        if candidate.excluded_task_ids:
            # Rows were dropped for a definite collision under exclude_row. The
            # remaining score is honest but covers less than the benchmark, so it
            # is not the published number for this benchmark. Reported instead of
            # the bare collision, because it is the narrower, actionable fact.
            reasons.append(f"contamination.excluded_rows:{candidate.alias}")
        elif candidate.exposed:
            reasons.append(f"contamination.exposed:{candidate.alias}")
    return tuple(reasons)


def _ordered(publication_order: Sequence[str], selected: set[str]) -> tuple[str, ...]:
    return tuple(task_id for task_id in publication_order if task_id in selected)


def contamination_report(
    plan: EligibleEvalPlan,
    *,
    decided_at: datetime | None = None,
) -> ContaminationReport:
    """Wrap a plan into the artifact a score can cite."""
    moment = decided_at or datetime.now(timezone.utc)
    return ContaminationReport(decided_at=moment.isoformat(), plan=plan)


def write_contamination_report(
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    *,
    decided_at: datetime | None = None,
) -> tuple[Path, str]:
    """Write the passing gate decision into the eval output tree, atomically."""
    report = contamination_report(plan, decided_at=decided_at)
    return write_eval_artifact(
        config,
        CONTAMINATION_REPORT_FILE,
        report.as_document(),
        supersedes=CONTAMINATION_FAILURE_FILE,
    )


def write_contamination_failure(config: BfclEvalConfig, error: Exception) -> tuple[Path, str]:
    """Record why the gate refused, under a name no reader can mistake for a pass."""
    document: dict[str, Any] = {
        "schema_version": CONTAMINATION_CONTRACT_VERSION,
        "status": "failed",
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
        "eval_config_hash": config.eval_config_hash,
        "source_run_id": config.source.run_id,
        "error": (
            error.as_report()
            if isinstance(error, ContaminationError)
            else {"code": "eval_contamination_invalid", "problem": type(error).__name__}
        ),
    }
    return write_eval_artifact(
        config,
        CONTAMINATION_FAILURE_FILE,
        document,
        supersedes=CONTAMINATION_REPORT_FILE,
    )


def assert_plan_unchanged(
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
) -> None:
    """Re-check the authorization immediately before the first request is paid for.

    Source verification installed the pin, and it runs first: a plan is
    only meaningful about a benchmark that has not moved. What this adds is the
    decision itself — the gate is re-run and the identity compared, so a plan
    that was widened between authorization and execution (a candidate added, an
    exclusion dropped, a policy relaxed) cannot be the plan a runner acts on.

    The re-derivation is not a second opinion about contamination. The config and
    the source are both pinned by the checks above, and the gate is a pure
    function of the two, so a legitimately produced plan re-derives to itself.
    What it catches is a plan that this config and this source never produced:
    one assembled by hand, or carried over from another run. That is why a
    refusal from the re-run is reported as drift rather than passed through — at
    this point the operator's problem is the plan they are holding, not the
    collision it fails to describe.
    """
    assert_source_unchanged(source)
    if plan.eval_config_hash != config.eval_config_hash:
        raise ContaminationPlanDriftError(
            "contamination.plan.eval_config_hash",
            "was authorized under a different eval config",
            actual=config.eval_config_hash,
            expected=plan.eval_config_hash,
            recovery="re-run the contamination gate with the config the run will execute",
        )
    if plan.source_verification_identity != source.verification_identity:
        raise ContaminationPlanDriftError(
            "contamination.plan.source_verification_identity",
            "was authorized against a different verified source",
            actual=source.verification_identity,
            expected=plan.source_verification_identity,
            recovery="verify the source and re-run the contamination gate together",
        )
    try:
        current = evaluate_contamination(config, source)
    except ContaminationPlanDriftError:
        raise
    except ContaminationError as exc:
        raise ContaminationPlanDriftError(
            "contamination.plan",
            f"was never a decision this config and this source could authorize: {exc.problem}",
            actual=describe_contamination_error(exc),
            expected="a plan produced by evaluate_contamination for this config and this verified source",
            recovery="gate the candidates again and run under the plan the gate returns; a plan the gate would "
            "not produce is not an authorization",
        ) from exc
    if current.plan_identity != plan.plan_identity:
        raise ContaminationPlanDriftError(
            "contamination.plan",
            "no longer resolves to the decision this run was authorized with",
            actual=current.plan_identity,
            expected=plan.plan_identity,
            recovery="stop the run and gate again; scoring under a plan that changed would report one number "
            "for two different task sets",
        )
