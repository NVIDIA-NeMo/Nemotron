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

"""Pure deterministic scoring for one authorized executable episode."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    ExecutableEpisode,
    ExecutedToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_dependencies import (
    resolved_script_from_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_contract import (
    EXECUTABLE_SCORING_GATES,
    ExecutableGateResult,
    ExecutableMetricName,
    ExecutableMetricResult,
    ExecutableScoringGate,
    ExecutableTaskScore,
    GateFailureClass,
    ScoredAssertion,
    ScoredDependency,
    ScoredExecution,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_errors import (
    ExecutableEvidenceError,
    ExecutableScoringPolicyError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_trace_parser import (
    EXECUTABLE_NON_CANDIDATE_STOPS,
    parse_executable_trace,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    EvalScoringConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scorer import (
    score_normalized_trace,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_contract import (
    GateResult,
    ScoredTurn,
)

_ORACLE_SUCCESS = frozenset({"completed", "business_rejection"})
_ASSERTIONS_MAY_BE_INCOMPLETE = frozenset(
    {
        "oracle_reset_failed",
        "oracle_call_failed",
        "oracle_timeout",
        "oracle_result_malformed",
        "oracle_state_failed",
        "unknown_commit_state",
        "dependency_resolution_failed",
        "assertion_infrastructure_failed",
    }
)


def score_executable_episode(
    *,
    episode: ExecutableEpisode,
    task: ExecutableTaskSpec,
    scoring: EvalScoringConfig,
    plan: EligibleEvalPlan,
) -> ExecutableTaskScore:
    """Score canonical evidence without contacting any external component."""

    _refuse_unsupported_policy(task, scoring=scoring)
    _authorize_score(episode, task, scoring=scoring, plan=plan)
    try:
        resolved_script = resolved_script_from_episode(task=task, episode=episode)
    except ValueError as exc:
        _mismatch(
            "episode.dependencies",
            str(exc),
            "dependency evidence derived from the task's paired live results",
            recovery="re-drive the dependent task and retain its canonical evidence",
        )
    trace = parse_executable_trace(episode, task)
    trace_score = score_normalized_trace(
        trace=trace,
        script=resolved_script,
        scoring=scoring,
        completion_detail=episode.detail,
    )
    executions = tuple(_score_execution(item, task=task) for item in episode.executions)
    assertions = tuple(
        ScoredAssertion(
            assertion_index=item.assertion_index,
            name=item.name,
            category=item.category,
            status=item.status,
            reason_code=item.reason_code,
            detail=item.detail,
        )
        for item in episode.assertions
    )
    dependencies = tuple(
        ScoredDependency(
            dependency_index=item.dependency_index,
            consumer_call_index=item.consumer_call_index,
            consumer_turn_index=item.consumer_turn_index,
            producer_call_index=item.producer_call_index,
            producer_execution_index=item.producer_execution_index,
            status=item.status,
            resolved_value_hash=item.resolved_value_hash,
            reason_code=item.reason_code,
            detail=item.detail,
        )
        for item in episode.dependencies
    )
    gates_by_name: dict[str, ExecutableGateResult] = {
        gate.gate: _trace_gate(gate) for gate in trace_score.gates
    }
    gates_by_name.update(
        {
            "oracle_execution": _oracle_execution_gate(executions),
            "dependency_resolution": _dependency_gate(
                dependencies,
                required=len(task.dependencies),
            ),
            "commit_state_known": _commit_gate(executions),
            "assertions": _assertion_gate(
                assertions,
                required=task.success_assertions,
                episode_status=episode.status,
                final_state_known=episode.final_state_hash is not None,
            ),
            "executable_completion": _completion_gate(episode),
        }
    )
    gates = tuple(gates_by_name[name] for name in EXECUTABLE_SCORING_GATES)
    success = _task_success(gates, mode=scoring.task_success)
    metrics = _metric_results(
        task=task,
        turns=trace_score.turns,
        executions=executions,
        assertions=assertions,
        gates=gates,
        task_success=success,
    )
    failed = tuple(gate.gate for gate in gates if gate.outcome == "failed")
    return ExecutableTaskScore(
        task_id=episode.task_id,
        candidate_alias=episode.candidate_alias,
        canonical_model_identity=episode.canonical_model_identity,
        plan_identity=episode.plan_identity,
        eval_config_hash=episode.eval_config_hash,
        scoring_policy_hash=task.scoring_policy_hash,
        source_verification_identity=episode.source_verification_identity,
        oracle_verification_identity=episode.oracle_verification_identity,
        script_hash=episode.script_hash,
        task_spec_hash=episode.task_spec_hash,
        episode_hash=episode.episode_hash,
        scoring_contract_hash=scoring.contract.content_hash,
        scoring_policy=scoring.semantic_payload(),
        episode_status=episode.status,
        # Both sources are stated because they answer different questions: the
        # status covers a terminal the taxonomy attributes away from the model,
        # and a gate covers infrastructure that broke inside a finished episode.
        non_candidate_stop=(
            episode.status in EXECUTABLE_NON_CANDIDATE_STOPS
            or any(
                gate.outcome == "failed" and gate.failure_class == "infrastructure"
                for gate in gates
            )
        ),
        expected_calls=task.script.expected_call_count,
        observed_calls=trace.observed_calls,
        attempted_calls=sum(item.attempted for item in executions),
        successful_executions=sum(item.oracle_succeeded for item in executions),
        expected_dependencies=len(task.dependencies),
        required_assertions=len(task.success_assertions),
        turns=trace_score.turns,
        executions=executions,
        dependencies=dependencies,
        assertions=assertions,
        gates=gates,
        metrics=metrics,
        task_success=success,
        detail=(
            "every gate required by the executable scoring policy passed"
            if success
            else f"failed gate(s): {', '.join(failed)}"
        ),
    )


def _metric(
    name: ExecutableMetricName,
    numerator: int,
    denominator: int,
    *,
    na_reason: str | None = None,
) -> ExecutableMetricResult:
    return ExecutableMetricResult(
        metric=name,
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
        not_applicable_reason=(na_reason if denominator == 0 else None),
    )


def _gate_metric(
    name: ExecutableMetricName,
    gate: ExecutableGateResult,
) -> ExecutableMetricResult:
    if gate.outcome == "not_applicable":
        return _metric(name, 0, 0, na_reason="metric.gate_not_applicable")
    return _metric(name, int(gate.outcome == "passed"), 1)


def _assertion_metric(
    name: ExecutableMetricName,
    assertions: Sequence[ScoredAssertion],
    *,
    declared_names: Sequence[str],
    na_reason: str,
) -> ExecutableMetricResult:
    if not declared_names:
        return _metric(name, 0, 0, na_reason=na_reason)
    by_name = {item.name: item for item in assertions}
    # A declared assertion with no verdict was never run, which only happens when
    # an infrastructure stop cut the episode short. Counting it as a candidate
    # failure would charge a broken oracle to the model, so the metric reports no
    # denominator at all rather than a number the evidence cannot support. An
    # assertion that ran and raised is the same missing evidence: the verdict
    # names a broken assertion, not a wrong answer, so it cannot be a denominator
    # the candidate is measured against either.
    if any(
        declared not in by_name or by_name[declared].status == "infrastructure_error"
        for declared in declared_names
    ):
        return _metric(name, 0, 0, na_reason="metric.assertion_evidence_incomplete")
    applicable = [
        by_name[declared]
        for declared in declared_names
        if by_name[declared].status != "not_applicable"
    ]
    if not applicable:
        return _metric(name, 0, 0, na_reason="metric.all_assertions_not_applicable")
    return _metric(
        name,
        sum(item.status == "passed" for item in applicable),
        len(applicable),
    )


def _metric_results(
    *,
    task: ExecutableTaskSpec,
    turns: Sequence[ScoredTurn],
    executions: Sequence[ScoredExecution],
    assertions: Sequence[ScoredAssertion],
    gates: Sequence[ExecutableGateResult],
    task_success: bool,
) -> tuple[ExecutableMetricResult, ...]:
    gate = {item.gate: item for item in gates}
    calls = [call for turn in turns for call in turn.calls]
    expected_calls = task.script.expected_call_count
    expected_text_turns = sum(
        turn.expected_assistant_content is not None for turn in task.script.turns
    )
    attempted = [item for item in executions if item.attempted]
    declared_by_category = {
        category: tuple(
            spec.name for spec in task.assertion_specs if spec.category == category
        )
        for category in ("state", "path", "final_answer")
    }
    path_gates = (
        "tool_selection",
        "arguments",
        "schema_valid",
        "call_grouping",
        "call_ordering",
        "text_turn",
        "trace_completion",
        "oracle_execution",
        "dependency_resolution",
        "executable_completion",
    )
    path_assertions = declared_by_category["path"]
    path_verdicts = {item.name: item for item in assertions}
    path_evidence_incomplete = any(
        gate[name].failure_class == "infrastructure" for name in path_gates
    ) or any(name not in path_verdicts for name in path_assertions)
    path_passed = all(
        gate[name].outcome != "failed" for name in path_gates
    ) and all(
        path_verdicts[name].status in {"passed", "not_applicable"}
        for name in path_assertions
    )
    return (
        _gate_metric("schema_valid_rate", gate["schema_valid"]),
        _metric(
            "tool_name_accuracy",
            sum(call.name_matched for call in calls),
            expected_calls,
            na_reason="metric.no_expected_call",
        ),
        _metric(
            "argument_accuracy",
            sum(call.arguments_matched for call in calls),
            expected_calls,
            na_reason="metric.no_expected_call",
        ),
        _gate_metric("call_group_accuracy", gate["call_grouping"]),
        _gate_metric("call_order_accuracy", gate["call_ordering"]),
        _gate_metric("required_call_subset_accuracy", gate["tool_selection"]),
        _metric(
            "milestone_accuracy",
            sum(
                turn.kind == "text" and turn.text_matched is True for turn in turns
            ),
            expected_text_turns,
            na_reason="metric.no_text_milestone",
        ),
        _metric(
            "turn_success_rate",
            sum(turn.advanced for turn in turns),
            len(turns),
            na_reason="metric.no_evaluated_turn",
        ),
        _metric(
            "tool_execution_success_rate",
            sum(item.oracle_succeeded for item in attempted),
            len(attempted),
            na_reason="metric.no_attempted_call",
        ),
        _assertion_metric(
            "assertion_success_rate",
            assertions,
            declared_names=task.success_assertions,
            na_reason="metric.no_declared_assertion",
        ),
        _assertion_metric(
            "state_match_rate",
            assertions,
            declared_names=declared_by_category["state"],
            na_reason="metric.no_state_assertion",
        ),
        _metric(
            "path_success_rate",
            0 if path_evidence_incomplete else int(path_passed),
            0 if path_evidence_incomplete else 1,
            na_reason="metric.path_evidence_incomplete",
        ),
        _assertion_metric(
            "final_answer_success_rate",
            assertions,
            declared_names=declared_by_category["final_answer"],
            na_reason="metric.no_final_answer_assertion",
        ),
        _metric("task_success_rate", int(task_success), 1),
    )


def _mismatch(
    subject: str,
    actual: Any,
    expected: Any,
    *,
    recovery: str,
) -> NoReturn:
    raise ExecutableEvidenceError(
        subject,
        "does not match the authorized executable evidence",
        actual=actual,
        expected=str(expected),
        recovery=recovery,
    )


def _authorize_score(
    episode: ExecutableEpisode,
    task: ExecutableTaskSpec,
    *,
    scoring: EvalScoringConfig,
    plan: EligibleEvalPlan,
) -> None:
    """Reject cross-boundary identity drift instead of grading mismatched evidence."""

    checks = (
        ("episode.plan_identity", episode.plan_identity, plan.plan_identity),
        ("task.plan_identity", task.plan_identity, plan.plan_identity),
        (
            "episode.source_verification_identity",
            episode.source_verification_identity,
            plan.source_verification_identity,
        ),
        (
            "task.source_verification_identity",
            task.source_verification_identity,
            plan.source_verification_identity,
        ),
        ("episode.eval_config_hash", episode.eval_config_hash, plan.eval_config_hash),
        ("task.eval_config_hash", task.eval_config_hash, plan.eval_config_hash),
        (
            "task.scoring_policy_hash",
            task.scoring_policy_hash,
            plan.scoring_policy_hash,
        ),
        (
            "scoring.scoring_policy_hash",
            scoring.scoring_policy_hash,
            plan.scoring_policy_hash,
        ),
        ("episode.task_id", episode.task_id, task.task_id),
        ("episode.candidate_alias", episode.candidate_alias, task.candidate_alias),
        (
            "episode.canonical_model_identity",
            episode.canonical_model_identity,
            task.canonical_model_identity,
        ),
        (
            "episode.oracle_verification_identity",
            episode.oracle_verification_identity,
            task.oracle_verification_identity,
        ),
        ("episode.script_hash", episode.script_hash, task.script.script_hash),
        ("episode.task_spec_hash", episode.task_spec_hash, task.task_spec_hash),
        (
            "task.script.source_verification_identity",
            task.script.source_verification_identity,
            task.source_verification_identity,
        ),
    )
    for subject, actual, expected in checks:
        if actual != expected:
            _mismatch(
                subject,
                actual,
                expected,
                recovery="score the episode with the exact task, plan, and policy used to drive it",
            )
    try:
        candidate = plan.candidate(episode.candidate_alias)
    except KeyError:
        _mismatch(
            "episode.candidate_alias",
            episode.candidate_alias,
            f"one of {list(plan.candidate_aliases)}",
            recovery="score only candidates authorized by the contamination plan",
        )
    if candidate.canonical_model_identity != episode.canonical_model_identity:
        _mismatch(
            "episode.canonical_model_identity",
            episode.canonical_model_identity,
            candidate.canonical_model_identity,
            recovery="re-run authorization when an alias points at different weights",
        )
    if episode.task_id not in plan.evaluation_task_ids(episode.candidate_alias):
        _mismatch(
            "episode.task_id",
            episode.task_id,
            "one of the candidate's authorized task ids",
            recovery="score only tasks admitted by the contamination plan",
        )
    observed_assertions = tuple(item.name for item in episode.assertions)
    expected_prefix = task.success_assertions[: len(observed_assertions)]
    if observed_assertions != expected_prefix or (
        len(observed_assertions) != len(task.success_assertions)
        and episode.status not in _ASSERTIONS_MAY_BE_INCOMPLETE
        and episode.final_state_hash is not None
    ):
        _mismatch(
            "episode.assertions",
            observed_assertions,
            task.success_assertions,
            recovery="retain every required assertion exactly once and in declared order",
        )
    _validate_commit_policy(episode.executions, task=task)


def _validate_commit_policy(
    executions: Sequence[ExecutedToolCall],
    *,
    task: ExecutableTaskSpec,
) -> None:
    for item in executions:
        policy = task.tool_policy(item.function_name or "")
        mutates = policy.mutates if policy is not None else False
        if item.attempted and policy is None:
            _mismatch(
                f"episode.executions[{item.execution_index}].function_name",
                item.function_name,
                "a tool policy declared by the executable task",
                recovery="score evidence only against the task that exposed its tools",
            )
        allowed = (
            {"committed", "not_committed", "unknown"}
            if mutates and item.attempted
            else {"not_started"}
            if not item.attempted
            else {"not_applicable"}
        )
        if item.state_commit not in allowed:
            _mismatch(
                f"episode.executions[{item.execution_index}].state_commit",
                item.state_commit,
                sorted(allowed),
                recovery="re-drive the task with mutation policy bound to the executed tool",
            )


def _refuse_unsupported_policy(
    task: ExecutableTaskSpec,
    *,
    scoring: EvalScoringConfig,
) -> None:
    if scoring.allow_llm_repair:
        raise ExecutableScoringPolicyError(
            "scoring.allow_llm_repair",
            "would make the score depend on an unrecorded repairer",
            actual=True,
            expected="false",
            recovery="disable LLM repair and score the canonical candidate call evidence",
        )
    if scoring.task_success == "assertions_only" and not task.success_assertions:
        raise ExecutableScoringPolicyError(
            "scoring.task_success",
            "requests assertions-only success for a task with no required assertion",
            actual=task.success_assertions,
            expected="at least one required pack assertion",
            recovery="use all_applicable_gates or declare the executable success assertions",
        )


def _score_execution(
    item: ExecutedToolCall,
    *,
    task: ExecutableTaskSpec,
) -> ScoredExecution:
    policy = task.tool_policy(item.function_name or "")
    return ScoredExecution(
        execution_index=item.execution_index,
        turn_index=item.turn_index,
        function_name=item.function_name,
        status=item.status,
        state_commit=item.state_commit,
        result_hash=item.result_hash,
        malformed_result_hash=item.malformed_result_hash,
        state_before_hash=item.state_before_hash,
        state_after_hash=item.state_after_hash,
        attempted=item.attempted,
        oracle_succeeded=item.status in _ORACLE_SUCCESS,
        mutates=policy.mutates if policy is not None else False,
        reason_code=item.reason_code,
        detail=item.detail,
    )


def _passed(gate: ExecutableScoringGate, reason: str, detail: str) -> ExecutableGateResult:
    return ExecutableGateResult(
        gate=gate,
        outcome="passed",
        reason_code=reason,
        detail=detail,
    )


def _skipped(gate: ExecutableScoringGate, detail: str) -> ExecutableGateResult:
    return ExecutableGateResult(
        gate=gate,
        outcome="not_applicable",
        reason_code=f"executable.{gate}.not_applicable",
        detail=detail,
    )


def _failed(
    gate: ExecutableScoringGate,
    *,
    failure_class: GateFailureClass,
    reason: str,
    detail: str,
    turn_index: int | None = None,
    execution_index: int | None = None,
    assertion_index: int | None = None,
) -> ExecutableGateResult:
    return ExecutableGateResult(
        gate=gate,
        outcome="failed",
        failure_class=failure_class,
        reason_code=reason,
        detail=detail,
        turn_index=turn_index,
        execution_index=execution_index,
        assertion_index=assertion_index,
    )


def _trace_gate(gate: GateResult) -> ExecutableGateResult:
    """Lift one shared trace gate into the executable gate set, verdict intact.

    The attribution is carried over rather than recomputed. The trace layer
    already knows whether the episode stopped for a reason the candidate did not
    choose — it read the terminal status through the executable attribution map to
    normalize the evidence — and deriving it a second time here would let the two
    layers disagree about the same failure.
    """
    if gate.outcome == "passed":
        return _passed(gate.gate, gate.reason_code, gate.detail)
    if gate.outcome == "not_applicable":
        return _skipped(gate.gate, gate.detail)
    return _failed(
        gate.gate,
        failure_class=gate.failure_class,
        reason=gate.reason_code,
        detail=gate.detail,
        turn_index=gate.turn_index,
    )


def _oracle_execution_gate(
    executions: Sequence[ScoredExecution],
) -> ExecutableGateResult:
    if not executions:
        return _skipped("oracle_execution", "the candidate proposed no tool call")
    for item in executions:
        if item.status not in _ORACLE_SUCCESS:
            infrastructure = item.status not in {"not_executed"}
            return _failed(
                "oracle_execution",
                failure_class="infrastructure" if infrastructure else "candidate",
                reason=(
                    "executable.call_not_executed"
                    if item.status == "not_executed"
                    else f"executable.oracle_{item.status}"
                ),
                detail=item.detail,
                turn_index=item.turn_index,
                execution_index=item.execution_index,
            )
    return _passed(
        "oracle_execution",
        "executable.oracle_execution_passed",
        "every attempted call produced a canonical result or business rejection",
    )


def _commit_gate(executions: Sequence[ScoredExecution]) -> ExecutableGateResult:
    mutating = [item for item in executions if item.mutates]
    if not mutating:
        return _skipped("commit_state_known", "the task attempted no mutating tool")
    for item in mutating:
        if item.state_commit == "unknown":
            return _failed(
                "commit_state_known",
                failure_class="infrastructure",
                reason="executable.commit_state_unknown",
                detail="the mutating call's commit state could not be established",
                turn_index=item.turn_index,
                execution_index=item.execution_index,
            )
    return _passed(
        "commit_state_known",
        "executable.commit_state_known",
        "every mutating call has a determined commit verdict or never started",
    )


def _dependency_gate(
    dependencies: Sequence[ScoredDependency],
    *,
    required: int,
) -> ExecutableGateResult:
    if not required:
        return _skipped(
            "dependency_resolution",
            "the task declares no live-result dependency",
        )
    if not dependencies:
        return _skipped(
            "dependency_resolution",
            "the conversation stopped before reaching its first dependent call",
        )
    for item in dependencies:
        if item.status != "resolved":
            return _failed(
                "dependency_resolution",
                failure_class="infrastructure",
                reason=f"executable.{item.reason_code}",
                detail=item.detail,
                turn_index=item.consumer_turn_index,
                execution_index=item.producer_execution_index,
            )
    return _passed(
        "dependency_resolution",
        "executable.dependencies_resolved",
        "every reached dependent argument was derived from paired live evidence",
    )


def _assertion_gate(
    assertions: Sequence[ScoredAssertion],
    *,
    required: Sequence[str],
    episode_status: str,
    final_state_known: bool,
) -> ExecutableGateResult:
    if not required:
        return _skipped("assertions", "the task declares no success assertion")
    for item in assertions:
        if item.status not in {"passed", "not_applicable"}:
            infrastructure = item.status == "infrastructure_error"
            return _failed(
                "assertions",
                failure_class="infrastructure" if infrastructure else "candidate",
                reason=(
                    "executable.assertion_infrastructure_error"
                    if infrastructure
                    else "executable.assertion_failed"
                ),
                detail=item.detail,
                assertion_index=item.assertion_index,
            )
    if len(assertions) < len(required):
        if not final_state_known:
            return _failed(
                "assertions",
                failure_class="infrastructure",
                reason="executable.assertion_state_unavailable",
                detail=(
                    "the final oracle state was unavailable, so required "
                    "assertions could not run"
                ),
            )
        return _skipped(
            "assertions",
            f"episode status {episode_status} stopped before every assertion could run",
        )
    if all(item.status == "not_applicable" for item in assertions):
        return _skipped(
            "assertions",
            "every required assertion was explicitly not applicable",
        )
    return _passed(
        "assertions",
        "executable.assertions_passed",
        "every required pack assertion passed",
    )


def _completion_gate(episode: ExecutableEpisode) -> ExecutableGateResult:
    if episode.status == "completed":
        return _passed(
            "executable_completion",
            "executable.episode_completed",
            "the executable conversation reached its terminal boundary",
        )
    infrastructure = episode.status in EXECUTABLE_NON_CANDIDATE_STOPS
    return _failed(
        "executable_completion",
        failure_class="infrastructure" if infrastructure else "candidate",
        reason="executable.episode_incomplete",
        detail=episode.detail,
        turn_index=episode.observed[-1].turn_index if episode.observed else None,
    )


def _task_success(
    gates: Sequence[ExecutableGateResult],
    *,
    mode: str,
) -> bool:
    if mode == "all_applicable_gates":
        return all(gate.outcome != "failed" for gate in gates)
    assertion = next(gate for gate in gates if gate.gate == "assertions")
    return assertion.outcome == "passed" and not any(
        gate.blocks_assertions_only for gate in gates
    )


__all__ = ["score_executable_episode"]
