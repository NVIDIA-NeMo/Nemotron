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

"""Deterministic executable score evidence for one candidate and task."""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictStr,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    EXECUTABLE_NON_CANDIDATE_STOPS,
    METRIC_NOT_APPLICABLE_CODES,
    episode_attribution,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    AssertionCategory,
    AssertionStatus,
    DependencyResolutionStatus,
    ExecutableEpisodeStatus,
    StateCommitStatus,
    ToolExecutionStatus,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_contract import (
    ScoredTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    FrozenDict,
    NonNegativeInt,
    freeze_json,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

EXECUTABLE_SCORING_CONTRACT_VERSION: Final = "1.3"
EXECUTABLE_SCORING_GATES: Final = (
    "tool_selection",
    "arguments",
    "schema_valid",
    "call_grouping",
    "call_ordering",
    "text_turn",
    "trace_completion",
    "oracle_execution",
    "dependency_resolution",
    "commit_state_known",
    "assertions",
    "executable_completion",
)

ExecutableScoringGate = Literal[
    "tool_selection",
    "arguments",
    "schema_valid",
    "call_grouping",
    "call_ordering",
    "text_turn",
    "trace_completion",
    "oracle_execution",
    "dependency_resolution",
    "commit_state_known",
    "assertions",
    "executable_completion",
]
ExecutableGateOutcome = Literal["passed", "failed", "not_applicable"]
GateFailureClass = Literal["none", "candidate", "infrastructure", "evidence"]
EXECUTABLE_METRIC_TAXONOMY: Final = (
    "schema_valid_rate",
    "tool_name_accuracy",
    "argument_accuracy",
    "call_group_accuracy",
    "call_order_accuracy",
    "required_call_subset_accuracy",
    "milestone_accuracy",
    "turn_success_rate",
    "tool_execution_success_rate",
    "assertion_success_rate",
    "state_match_rate",
    "path_success_rate",
    "final_answer_success_rate",
    "task_success_rate",
)
ExecutableMetricName = Literal[
    "schema_valid_rate",
    "tool_name_accuracy",
    "argument_accuracy",
    "call_group_accuracy",
    "call_order_accuracy",
    "required_call_subset_accuracy",
    "milestone_accuracy",
    "turn_success_rate",
    "tool_execution_success_rate",
    "assertion_success_rate",
    "state_match_rate",
    "path_success_rate",
    "final_answer_success_rate",
    "task_success_rate",
]


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class ExecutableGateResult(_Frozen):
    """One executable scoring dimension and its stable failure classification."""

    gate: ExecutableScoringGate
    outcome: ExecutableGateOutcome
    failure_class: GateFailureClass = "none"
    reason_code: StrictStr
    detail: StrictStr
    turn_index: NonNegativeInt | None = None
    execution_index: NonNegativeInt | None = None
    assertion_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableGateResult:
        if not self.reason_code.strip():
            raise ValueError("an executable gate carries a stable reason code")
        coordinates = (self.turn_index, self.execution_index, self.assertion_index)
        if self.outcome == "failed":
            if self.failure_class == "none":
                raise ValueError("a failed executable gate classifies its failure")
        elif self.failure_class != "none" or any(value is not None for value in coordinates):
            raise ValueError("only a failed executable gate carries failure metadata")
        return self

    @property
    def counts_against_all_gates(self) -> bool:
        return self.outcome == "failed"

    @property
    def blocks_assertions_only(self) -> bool:
        return self.outcome == "failed" and self.failure_class in {
            "infrastructure",
            "evidence",
        }

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "turn_index": self.turn_index,
            "execution_index": self.execution_index,
            "assertion_index": self.assertion_index,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "detail"
        }


class ScoredExecution(_Frozen):
    """One canonical live tool outcome as consumed by executable scoring."""

    execution_index: NonNegativeInt
    turn_index: NonNegativeInt
    function_name: StrictStr | None = None
    status: ToolExecutionStatus
    state_commit: StateCommitStatus
    result_hash: ContentHash | None = None
    malformed_result_hash: ContentHash | None = None
    state_before_hash: ContentHash | None = None
    state_after_hash: ContentHash | None = None
    attempted: StrictBool
    oracle_succeeded: StrictBool
    mutates: StrictBool
    reason_code: StrictStr
    detail: StrictStr

    @model_validator(mode="after")
    def _coherent(self) -> ScoredExecution:
        if not self.reason_code.strip():
            raise ValueError("a scored execution carries a stable reason code")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "detail"
        }


class ScoredAssertion(_Frozen):
    """One required pack assertion verdict, without re-running pack code."""

    assertion_index: NonNegativeInt
    name: StrictStr
    category: AssertionCategory
    status: AssertionStatus
    reason_code: StrictStr
    detail: StrictStr

    @model_validator(mode="after")
    def _coherent(self) -> ScoredAssertion:
        if not self.name.strip() or not self.reason_code.strip():
            raise ValueError("a scored assertion has a name and stable reason code")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "detail"
        }


class ScoredDependency(_Frozen):
    """One live-result dependency outcome consumed by executable scoring."""

    dependency_index: NonNegativeInt
    consumer_call_index: NonNegativeInt
    consumer_turn_index: NonNegativeInt
    producer_call_index: NonNegativeInt
    producer_execution_index: NonNegativeInt | None = None
    status: DependencyResolutionStatus
    resolved_value_hash: ContentHash | None = None
    reason_code: StrictStr
    detail: StrictStr

    def semantic_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "detail"
        }


class ExecutableMetricResult(_Frozen):
    """One per-task metric contribution with an explicit denominator policy."""

    metric: ExecutableMetricName
    numerator: NonNegativeInt
    denominator: NonNegativeInt
    value: StrictFloat | None
    not_applicable_reason: StrictStr | None = None

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableMetricResult:
        if self.numerator > self.denominator:
            raise ValueError("a metric numerator cannot exceed its denominator")
        if self.denominator == 0:
            if self.value is not None or not self.not_applicable_reason:
                raise ValueError(
                    "a zero-denominator metric is N/A with a stable reason"
                )
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
        """The counts a metric is made of, without the quotient they imply.

        ``value`` is numerator over denominator, so hashing it would make a
        float representation part of a score's identity for no added evidence.
        """
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "value"
        }


class ExecutableTaskScore(_Frozen):
    """One score derived only from an authorized executable episode."""

    schema_version: Literal["1.3"] = EXECUTABLE_SCORING_CONTRACT_VERSION
    scope: Literal["trace_and_executable"] = "trace_and_executable"
    task_id: StrictStr
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    scoring_policy_hash: ContentHash
    source_verification_identity: ContentHash
    oracle_verification_identity: ContentHash
    script_hash: ContentHash
    task_spec_hash: ContentHash
    episode_hash: ContentHash
    scoring_contract_hash: ContentHash
    scoring_policy: FrozenDict
    episode_status: ExecutableEpisodeStatus
    non_candidate_stop: StrictBool
    expected_calls: NonNegativeInt
    observed_calls: NonNegativeInt
    attempted_calls: NonNegativeInt
    successful_executions: NonNegativeInt
    expected_dependencies: NonNegativeInt
    required_assertions: NonNegativeInt
    turns: tuple[ScoredTurn, ...] = ()
    executions: tuple[ScoredExecution, ...] = ()
    dependencies: tuple[ScoredDependency, ...] = ()
    assertions: tuple[ScoredAssertion, ...] = ()
    gates: tuple[ExecutableGateResult, ...]
    metrics: tuple[ExecutableMetricResult, ...]
    task_success: StrictBool
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_policy(cls, value: Any) -> Any:
        if isinstance(value, dict) and "scoring_policy" in value:
            value = dict(value)
            validate_json_value(value["scoring_policy"], label="executable scoring policy")
            value["scoring_policy"] = freeze_json(value["scoring_policy"])
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableTaskScore:
        if tuple(gate.gate for gate in self.gates) != EXECUTABLE_SCORING_GATES:
            raise ValueError(
                "an executable score reports every gate exactly once in contract order"
            )
        if tuple(metric.metric for metric in self.metrics) != EXECUTABLE_METRIC_TAXONOMY:
            raise ValueError(
                "an executable score reports every metric exactly once in taxonomy order"
            )
        if [item.execution_index for item in self.executions] != list(
            range(len(self.executions))
        ):
            raise ValueError("scored executions are contiguous and zero-based")
        if [item.assertion_index for item in self.assertions] != list(
            range(len(self.assertions))
        ):
            raise ValueError("scored assertions are contiguous and zero-based")
        if [item.dependency_index for item in self.dependencies] != list(
            range(len(self.dependencies))
        ):
            raise ValueError("scored dependencies are a contiguous declaration prefix")
        if [turn.turn_index for turn in self.turns] != list(range(len(self.turns))):
            raise ValueError("scored executable turns are contiguous and zero-based")
        policy = thaw_json(self.scoring_policy)
        if self.scoring_policy_hash != _sha256_json(policy):
            raise ValueError("scoring_policy_hash identifies the pinned scoring policy")
        if self.observed_calls != sum(len(turn.calls) for turn in self.turns):
            raise ValueError("observed_calls counts calls retained by scored turns")
        if self.attempted_calls != sum(item.attempted for item in self.executions):
            raise ValueError("attempted_calls counts executions that reached the oracle")
        if self.successful_executions != sum(
            item.oracle_succeeded for item in self.executions
        ):
            raise ValueError("successful_executions counts canonical oracle results")
        if self.required_assertions < len(self.assertions):
            raise ValueError("assertion evidence cannot exceed required_assertions")
        if self.expected_dependencies < len(self.dependencies):
            raise ValueError("dependency evidence cannot exceed expected_dependencies")
        completion = self.gate("executable_completion")
        if self.episode_status == "completed":
            if completion.outcome != "passed":
                raise ValueError(
                    "a completed episode has a passed executable_completion gate"
                )
        else:
            attribution = episode_attribution(self.episode_status, executable=True)
            if (
                completion.outcome != "failed"
                or completion.failure_class != attribution
            ):
                raise ValueError(
                    "an incomplete episode has an attributed executable_completion failure"
                )
        mode = policy.get("task_success") if isinstance(policy, dict) else None
        if mode == "all_applicable_gates":
            expected = all(not gate.counts_against_all_gates for gate in self.gates)
        elif mode == "assertions_only":
            assertion_gate = next(
                gate for gate in self.gates if gate.gate == "assertions"
            )
            expected = assertion_gate.outcome == "passed" and not any(
                gate.blocks_assertions_only for gate in self.gates
            )
        else:
            raise ValueError("scoring_policy carries a supported task_success mode")
        if self.task_success != expected:
            raise ValueError("task_success must be derived from the pinned gate policy")
        task_success_metric = self.metric("task_success_rate")
        if (
            task_success_metric.numerator != int(self.task_success)
            or task_success_metric.denominator != 1
        ):
            raise ValueError("task_success_rate records the derived task verdict")
        expected_non_candidate_stop = (
            self.episode_status in EXECUTABLE_NON_CANDIDATE_STOPS
            or any(
                gate.outcome == "failed" and gate.failure_class == "infrastructure"
                for gate in self.gates
            )
        )
        if self.non_candidate_stop != expected_non_candidate_stop:
            raise ValueError(
                "non_candidate_stop identifies an infrastructure terminal or gate failure"
            )
        return self

    @property
    def failed_gates(self) -> tuple[ExecutableScoringGate, ...]:
        return tuple(gate.gate for gate in self.gates if gate.outcome == "failed")

    def gate(self, name: ExecutableScoringGate) -> ExecutableGateResult:
        for result in self.gates:
            if result.gate == name:
                return result
        raise KeyError(name)

    def metric(self, name: ExecutableMetricName) -> ExecutableMetricResult:
        for result in self.metrics:
            if result.metric == name:
                return result
        raise KeyError(name)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "task_id": self.task_id,
            "candidate_alias": self.candidate_alias,
            "canonical_model_identity": self.canonical_model_identity,
            "plan_identity": self.plan_identity,
            "eval_config_hash": self.eval_config_hash,
            "scoring_policy_hash": self.scoring_policy_hash,
            "source_verification_identity": self.source_verification_identity,
            "oracle_verification_identity": self.oracle_verification_identity,
            "script_hash": self.script_hash,
            "task_spec_hash": self.task_spec_hash,
            "episode_hash": self.episode_hash,
            "scoring_contract_hash": self.scoring_contract_hash,
            "scoring_policy": thaw_json(self.scoring_policy),
            "episode_status": self.episode_status,
            "non_candidate_stop": self.non_candidate_stop,
            "expected_calls": self.expected_calls,
            "observed_calls": self.observed_calls,
            "attempted_calls": self.attempted_calls,
            "successful_executions": self.successful_executions,
            "expected_dependencies": self.expected_dependencies,
            "required_assertions": self.required_assertions,
            "turns": [turn.semantic_payload() for turn in self.turns],
            "executions": [item.semantic_payload() for item in self.executions],
            "dependencies": [
                item.semantic_payload() for item in self.dependencies
            ],
            "assertions": [item.semantic_payload() for item in self.assertions],
            "gates": [gate.semantic_payload() for gate in self.gates],
            "metrics": [metric.semantic_payload() for metric in self.metrics],
            "task_success": self.task_success,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.semantic_payload().items()
            if key
            not in {
                "detail",
                "turns",
                "executions",
                "dependencies",
                "assertions",
                "gates",
                "metrics",
            }
        }
        payload["turns"] = [turn.identity_payload() for turn in self.turns]
        payload["executions"] = [
            item.identity_payload() for item in self.executions
        ]
        payload["dependencies"] = [
            item.identity_payload() for item in self.dependencies
        ]
        payload["assertions"] = [
            item.identity_payload() for item in self.assertions
        ]
        payload["gates"] = [gate.identity_payload() for gate in self.gates]
        payload["metrics"] = [metric.identity_payload() for metric in self.metrics]
        return payload

    @property
    def score_hash(self) -> str:
        return _sha256_json(self.identity_payload())

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "score_hash": self.score_hash}


__all__ = [
    "EXECUTABLE_SCORING_CONTRACT_VERSION",
    "EXECUTABLE_SCORING_GATES",
    "EXECUTABLE_METRIC_TAXONOMY",
    "ExecutableGateOutcome",
    "ExecutableGateResult",
    "ExecutableMetricName",
    "ExecutableMetricResult",
    "ExecutableScoringGate",
    "ExecutableTaskScore",
    "GateFailureClass",
    "ScoredAssertion",
    "ScoredDependency",
    "ScoredExecution",
]
