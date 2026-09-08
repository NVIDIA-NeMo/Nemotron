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

"""Run-level aggregation of authorized executable task scores."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_contract import (
    EXECUTABLE_METRIC_TAXONOMY,
    ExecutableMetricResult,
    ExecutableTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_errors import (
    ExecutableAggregationError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    NonNegativeInt,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

EXECUTABLE_AGGREGATION_CONTRACT_VERSION: Final = "1.1"


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class ExecutableCandidateScore(BaseModel):
    """One candidate's complete aggregate over its authorized task set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.1"] = EXECUTABLE_AGGREGATION_CONTRACT_VERSION
    # What these numbers measured. A trace-only aggregate reports a different
    # metric taxonomy over the same tasks, so a reader that could not tell the two
    # apart would compare numbers that answer different questions.
    scope: Literal["trace_and_executable"] = "trace_and_executable"
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    scoring_policy_hash: ContentHash
    scoring_contract_hash: ContentHash
    source_verification_identity: ContentHash
    oracle_verification_identity: ContentHash
    task_ids: tuple[StrictStr, ...]
    task_ids_hash: ContentHash
    task_score_hashes: tuple[ContentHash, ...]
    task_count: NonNegativeInt
    successful_tasks: NonNegativeInt
    non_candidate_stops: NonNegativeInt
    metrics: tuple[ExecutableMetricResult, ...]

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableCandidateScore:
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
        if tuple(metric.metric for metric in self.metrics) != EXECUTABLE_METRIC_TAXONOMY:
            raise ValueError("an aggregate reports every metric in taxonomy order")
        task_success = self.metric("task_success_rate")
        if (
            task_success.numerator != self.successful_tasks
            or task_success.denominator != self.task_count
        ):
            raise ValueError("task_success_rate aggregates the candidate task verdicts")
        return self

    def metric(self, name: str) -> ExecutableMetricResult:
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
            "oracle_verification_identity": self.oracle_verification_identity,
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


def aggregate_executable_scores(
    *,
    scores: Sequence[ExecutableTaskScore],
    plan: EligibleEvalPlan,
    candidate_alias: str,
) -> ExecutableCandidateScore:
    """Sum metric counts over one complete, ordered authorization boundary."""
    ordered = tuple(scores)
    expected_task_ids = plan.evaluation_task_ids(candidate_alias)
    actual_task_ids = tuple(score.task_id for score in ordered)
    if actual_task_ids != expected_task_ids:
        raise ExecutableAggregationError(
            f"candidates[{candidate_alias}].task_scores",
            "do not cover the authorized task set in publication order",
            actual=actual_task_ids,
            expected=str(expected_task_ids),
            recovery="score every authorized task exactly once and aggregate in plan order",
        )
    if not ordered:
        raise ExecutableAggregationError(
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
            "scoring_policy_hash": score.scoring_policy_hash,
            "source_verification_identity": score.source_verification_identity,
        }
        if actual != expected:
            raise ExecutableAggregationError(
                f"task_scores[{index}]",
                "belongs to a different candidate or authorization boundary",
                actual=actual,
                expected=str(expected),
                recovery="aggregate only scores produced under this exact plan",
            )
    oracle_identities = {score.oracle_verification_identity for score in ordered}
    scoring_contracts = {score.scoring_contract_hash for score in ordered}
    if len(oracle_identities) != 1 or len(scoring_contracts) != 1:
        raise ExecutableAggregationError(
            f"candidates[{candidate_alias}].task_scores",
            "mix oracle or scoring-contract identities",
            actual={
                "oracle_verification_identities": sorted(oracle_identities),
                "scoring_contract_hashes": sorted(scoring_contracts),
            },
            expected="one verified oracle and one scoring contract",
            recovery="split scores by source oracle and scoring contract",
        )

    aggregate_metrics: list[ExecutableMetricResult] = []
    for name in EXECUTABLE_METRIC_TAXONOMY:
        contributions = [score.metric(name) for score in ordered]
        numerator = sum(metric.numerator for metric in contributions)
        denominator = sum(metric.denominator for metric in contributions)
        reasons = sorted(
            {
                metric.not_applicable_reason
                for metric in contributions
                if metric.not_applicable_reason is not None
            }
        )
        aggregate_metrics.append(
            ExecutableMetricResult(
                metric=name,
                numerator=numerator,
                denominator=denominator,
                value=numerator / denominator if denominator else None,
                not_applicable_reason=(
                    None
                    if denominator
                    else reasons[0]
                    if len(reasons) == 1
                    else "metric.no_applicable_task"
                ),
            )
        )

    return ExecutableCandidateScore(
        candidate_alias=candidate_alias,
        canonical_model_identity=candidate.canonical_model_identity,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        scoring_policy_hash=plan.scoring_policy_hash,
        scoring_contract_hash=ordered[0].scoring_contract_hash,
        source_verification_identity=plan.source_verification_identity,
        oracle_verification_identity=next(iter(oracle_identities)),
        task_ids=actual_task_ids,
        task_ids_hash=_sha256_json(list(actual_task_ids)),
        task_score_hashes=tuple(score.score_hash for score in ordered),
        task_count=len(ordered),
        successful_tasks=sum(score.task_success for score in ordered),
        non_candidate_stops=sum(score.non_candidate_stop for score in ordered),
        metrics=tuple(aggregate_metrics),
    )


__all__ = [
    "EXECUTABLE_AGGREGATION_CONTRACT_VERSION",
    "ExecutableCandidateScore",
    "aggregate_executable_scores",
]
