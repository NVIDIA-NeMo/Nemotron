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

"""Run-level aggregation of authorized trace-only task scores.

A trace score is a set of per-task gate verdicts, so a run-level number over them
is a rate of tasks rather than a rate of calls. That is why the metric names here
are not the executable ones: ``argument_accuracy`` counts calls whose arguments
matched, while ``arguments_pass_rate`` counts tasks whose argument gate passed,
and reporting one under the other's name would make two incomparable numbers look
like the same measurement.

A gate that did not apply to a task is left out of that metric's denominator
rather than counted as a pass, exactly as the per-task contract records it. A
metric no task could apply is reported as N/A with a stable reason instead of a
vacuous zero or one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    METRIC_NOT_APPLICABLE_CODES,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_contract import (
    SCORING_GATES,
    TraceTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_errors import (
    TraceAggregationError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    NonNegativeInt,
    thaw_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

TRACE_AGGREGATION_CONTRACT_VERSION: Final = "1.0"

# One rate per gate the trace scoring contract defines, in that contract's order,
# followed by the task verdict the gates derive. Deriving the tuple from
# ``SCORING_GATES`` is what keeps a gate from being added to scoring without a
# published metric, which would silently drop it from every run-level report.
TRACE_METRIC_TAXONOMY: Final = tuple(
    f"{gate}_pass_rate" for gate in SCORING_GATES
) + ("task_success_rate",)
TraceMetricName = Literal[
    "tool_selection_pass_rate",
    "arguments_pass_rate",
    "schema_valid_pass_rate",
    "call_grouping_pass_rate",
    "call_ordering_pass_rate",
    "text_turn_pass_rate",
    "trace_completion_pass_rate",
    "task_success_rate",
]


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TraceMetricResult(_Frozen):
    """One trace metric with the task counts it was taken over."""

    metric: TraceMetricName
    numerator: NonNegativeInt
    denominator: NonNegativeInt
    value: StrictFloat | None
    not_applicable_reason: StrictStr | None = None

    @model_validator(mode="after")
    def _coherent(self) -> TraceMetricResult:
        if self.numerator > self.denominator:
            raise ValueError("a metric numerator cannot exceed its denominator")
        if self.denominator == 0:
            if self.value is not None or not self.not_applicable_reason:
                raise ValueError("a zero-denominator metric is N/A with a stable reason")
            if self.not_applicable_reason not in METRIC_NOT_APPLICABLE_CODES:
                raise ValueError("a metric N/A reason belongs to the registered taxonomy")
        else:
            if self.not_applicable_reason is not None:
                raise ValueError("an applicable metric has no N/A reason")
            expected = self.numerator / self.denominator
            if self.value is None or abs(self.value - expected) > 1e-12:
                raise ValueError("metric value equals numerator divided by denominator")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_payload(self) -> dict[str, Any]:
        """The counts a metric is made of, without the quotient they imply."""
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "value"
        }


class TraceCandidateScore(_Frozen):
    """One candidate's complete trace-only aggregate over its authorized tasks."""

    schema_version: Literal["1.0"] = TRACE_AGGREGATION_CONTRACT_VERSION
    # What these numbers are allowed to claim. Oracle replay and pack assertions
    # add gates a trace score never computed, so this aggregate never stands in
    # for an executable one even when both cover the same task set.
    scope: Literal["trace"] = "trace"
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    scoring_policy_hash: ContentHash
    scoring_contract_hash: ContentHash
    source_verification_identity: ContentHash
    task_ids: tuple[StrictStr, ...]
    task_ids_hash: ContentHash
    task_score_hashes: tuple[ContentHash, ...]
    task_count: NonNegativeInt
    successful_tasks: NonNegativeInt
    non_candidate_stops: NonNegativeInt
    metrics: tuple[TraceMetricResult, ...]

    @model_validator(mode="after")
    def _coherent(self) -> TraceCandidateScore:
        if not self.candidate_alias.strip() or not self.canonical_model_identity.strip():
            raise ValueError("an aggregate score identifies one candidate")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("an aggregate score covers a non-empty unique task set")
        if self.task_count != len(self.task_ids):
            raise ValueError("task_count equals the ordered task set")
        if len(self.task_score_hashes) != self.task_count:
            raise ValueError("every aggregate task cites one task score hash")
        if self.successful_tasks > self.task_count:
            raise ValueError("successful_tasks cannot exceed task_count")
        if self.non_candidate_stops > self.task_count:
            raise ValueError("non_candidate_stops cannot exceed task_count")
        if self.task_ids_hash != _sha256_json(list(self.task_ids)):
            raise ValueError("task_ids_hash identifies the ordered aggregate task set")
        if tuple(metric.metric for metric in self.metrics) != TRACE_METRIC_TAXONOMY:
            raise ValueError("an aggregate reports every metric in taxonomy order")
        task_success = self.metric("task_success_rate")
        if (
            task_success.numerator != self.successful_tasks
            or task_success.denominator != self.task_count
        ):
            raise ValueError("task_success_rate aggregates the candidate task verdicts")
        return self

    def metric(self, name: str) -> TraceMetricResult:
        for metric in self.metrics:
            if metric.metric == name:
                return metric
        raise KeyError(name)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "candidate_alias": self.candidate_alias,
            "canonical_model_identity": self.canonical_model_identity,
            "plan_identity": self.plan_identity,
            "eval_config_hash": self.eval_config_hash,
            "scoring_policy_hash": self.scoring_policy_hash,
            "scoring_contract_hash": self.scoring_contract_hash,
            "source_verification_identity": self.source_verification_identity,
            "task_ids": list(self.task_ids),
            "task_ids_hash": self.task_ids_hash,
            "task_score_hashes": list(self.task_score_hashes),
            "task_count": self.task_count,
            "successful_tasks": self.successful_tasks,
            "non_candidate_stops": self.non_candidate_stops,
            "metrics": [metric.semantic_payload() for metric in self.metrics],
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "metrics"
        }
        payload["metrics"] = [metric.identity_payload() for metric in self.metrics]
        return payload

    @property
    def aggregate_hash(self) -> str:
        return _sha256_json(self.identity_payload())

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "aggregate_hash": self.aggregate_hash}


def _policy_hash(score: TraceTaskScore) -> str:
    """The content identity of the policy a trace score was taken under.

    A trace score carries the policy itself rather than its hash, because the
    hash is derivable and a score that stored both could disagree with itself.
    """
    return _sha256_json(thaw_json(score.scoring_policy))


def _gate_metrics(scores: Sequence[TraceTaskScore]) -> list[TraceMetricResult]:
    metrics: list[TraceMetricResult] = []
    for gate in SCORING_GATES:
        applicable = tuple(
            result for result in (score.gate(gate) for score in scores) if result.applies
        )
        denominator = len(applicable)
        numerator = sum(result.outcome == "passed" for result in applicable)
        metrics.append(
            TraceMetricResult(
                metric=f"{gate}_pass_rate",  # type: ignore[arg-type]
                numerator=numerator,
                denominator=denominator,
                value=numerator / denominator if denominator else None,
                not_applicable_reason=(
                    None if denominator else "metric.no_applicable_task"
                ),
            )
        )
    return metrics


def aggregate_trace_scores(
    *,
    scores: Sequence[TraceTaskScore],
    plan: EligibleEvalPlan,
    candidate_alias: str,
) -> TraceCandidateScore:
    """Roll one candidate's complete, ordered authorization boundary into one score."""
    ordered = tuple(scores)
    expected_task_ids = plan.evaluation_task_ids(candidate_alias)
    actual_task_ids = tuple(score.task_id for score in ordered)
    if actual_task_ids != expected_task_ids:
        raise TraceAggregationError(
            f"candidates[{candidate_alias}].task_scores",
            "do not cover the authorized task set in publication order",
            actual=actual_task_ids,
            expected=str(expected_task_ids),
            recovery="score every authorized task exactly once and aggregate in plan order",
        )
    if not ordered:
        raise TraceAggregationError(
            f"candidates[{candidate_alias}].task_scores",
            "is empty",
            expected="at least one authorized task score",
            recovery="evaluate the candidate's authorized task set before aggregation",
        )
    candidate = plan.candidate(candidate_alias)
    expected = {
        "candidate_alias": candidate_alias,
        "canonical_model_identity": candidate.canonical_model_identity,
        "plan_identity": plan.plan_identity,
        "eval_config_hash": plan.eval_config_hash,
        "scoring_policy_hash": plan.scoring_policy_hash,
        "source_verification_identity": plan.source_verification_identity,
    }
    for index, score in enumerate(ordered):
        actual = {
            "candidate_alias": score.candidate_alias,
            "canonical_model_identity": score.canonical_model_identity,
            "plan_identity": score.plan_identity,
            "eval_config_hash": score.eval_config_hash,
            "scoring_policy_hash": _policy_hash(score),
            "source_verification_identity": score.source_verification_identity,
        }
        if actual != expected:
            raise TraceAggregationError(
                f"task_scores[{index}]",
                "belongs to a different candidate or authorization boundary",
                actual=actual,
                expected=str(expected),
                recovery="aggregate only scores produced under this exact plan",
            )
    scoring_contracts = {score.scoring_contract_hash for score in ordered}
    if len(scoring_contracts) != 1:
        raise TraceAggregationError(
            f"candidates[{candidate_alias}].task_scores",
            "mix scoring-contract identities",
            actual=sorted(scoring_contracts),
            expected="one scoring contract",
            recovery="split scores by the scoring contract they were taken under",
        )

    metrics = _gate_metrics(ordered)
    successful = sum(score.task_success for score in ordered)
    metrics.append(
        TraceMetricResult(
            metric="task_success_rate",
            numerator=successful,
            denominator=len(ordered),
            value=successful / len(ordered),
        )
    )
    return TraceCandidateScore(
        candidate_alias=candidate_alias,
        canonical_model_identity=candidate.canonical_model_identity,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        scoring_policy_hash=plan.scoring_policy_hash,
        scoring_contract_hash=next(iter(scoring_contracts)),
        source_verification_identity=plan.source_verification_identity,
        task_ids=actual_task_ids,
        task_ids_hash=_sha256_json(list(actual_task_ids)),
        task_score_hashes=tuple(score.score_hash for score in ordered),
        task_count=len(ordered),
        successful_tasks=successful,
        non_candidate_stops=sum(score.non_candidate_stop for score in ordered),
        metrics=tuple(metrics),
    )


__all__ = [
    "TRACE_AGGREGATION_CONTRACT_VERSION",
    "TRACE_METRIC_TAXONOMY",
    "TraceCandidateScore",
    "TraceMetricName",
    "TraceMetricResult",
    "aggregate_trace_scores",
]
