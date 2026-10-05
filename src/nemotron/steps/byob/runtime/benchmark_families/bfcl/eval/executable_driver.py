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

"""Drive one authorized candidate through one verified live oracle episode."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_comparison import (
    finish_reason_problem,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_matching import (
    ContinuationGate,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
    build_candidate_request,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateResponse,
    CandidateToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_driver import (
    turn_request_id,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    AssertionOutcome,
    DependencyResolution,
    ExecutableEpisode,
    ExecutableEpisodeStatus,
    ExecutableEvent,
    ExecutableEventKind,
    ExecutableTurn,
    ExecutedToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_dependencies import (
    resolve_turn_dependencies,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ExecutableAuthorizationError,
    OracleAssertionError,
    OracleCallError,
    OracleResetError,
    OracleSessionError,
    OracleStateError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.oracle_session import (
    OracleSession,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    EvalCandidate,
    EvalLimits,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedEvalSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_verification import (
    assert_source_unchanged,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_cache import (
    ToolTraceCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_contract import (
    ToolTraceRequest,
    build_tool_trace_request,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ExportedMessage,
    json_equal,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    declared_function,
    validate_function_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

_ASSERTION_STATUSES: Final = frozenset(
    {"passed", "failed", "not_applicable", "infrastructure_error"}
)
_FATAL_EXECUTION_STATUSES: Final = frozenset(
    {
        "unknown_commit_state",
        "oracle_timeout",
        "oracle_call_failed",
        "oracle_result_malformed",
        "dependency_resolution_failed",
    }
)
# A failure met while winding the session down explains nothing the episode has
# not already established, and a session that already broke is expected to break
# again on the way out. Cleanup never relabels evidence recorded before it.


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _LiveConversation:
    """A prompt firewall that has no operation for adding gold material."""

    __slots__ = ("_messages", "_provenance")

    def __init__(self, task: ExecutableTaskSpec) -> None:
        self._messages = [_wire_message(message) for message in task.script.seed_messages]
        self._provenance = ["seed"] * len(self._messages)

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(message) for message in self._messages)

    def audit(self) -> None:
        allowed = {
            "system": {"seed"},
            "user": {"seed", "scripted_user"},
            "assistant": {"candidate"},
            "tool": {"live_result"},
        }
        for index, (message, provenance) in enumerate(
            zip(self._messages, self._provenance, strict=True)
        ):
            role = str(message.get("role"))
            if provenance not in allowed.get(role, set()):
                raise RuntimeError(
                    f"model-facing message {index} has forbidden provenance {provenance!r}"
                )

    def append_candidate(self, response: CandidateResponse) -> None:
        message: dict[str, Any] = {"role": "assistant"}
        if response.assistant_content is not None:
            message["content"] = thaw_json(response.assistant_content)
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function_name,
                        "arguments": call.raw_arguments,
                    },
                }
                for call in response.tool_calls
            ]
        if len(message) == 1:
            raise ValueError("an empty candidate turn cannot enter a live conversation")
        self._messages.append(message)
        self._provenance.append("candidate")

    def append_live_results(self, results: Sequence[tuple[str, dict[str, Any]]]) -> None:
        for call_id, result in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": canonical_json(result),
                }
            )
            self._provenance.append("live_result")

    def append_user(self, message: ExportedMessage) -> None:
        if message.role != "user":
            raise ValueError("only a scripted user message may continue the live conversation")
        self._messages.append(_wire_message(message))
        self._provenance.append("scripted_user")


def _wire_message(message: ExportedMessage) -> dict[str, Any]:
    return {"role": message.role, "content": message.content}


class _Log:
    def __init__(self) -> None:
        self.turns: list[ExecutableTurn] = []
        self.executions: list[ExecutedToolCall] = []
        self.dependencies: list[DependencyResolution] = []
        self.assertions: list[AssertionOutcome] = []
        self.events: list[ExecutableEvent] = []

    def event(
        self,
        kind: ExecutableEventKind,
        reason_code: str,
        *,
        turn_index: int | None = None,
        execution_index: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.events.append(
            ExecutableEvent(
                index=len(self.events),
                kind=kind,
                turn_index=turn_index,
                execution_index=execution_index,
                reason_code=reason_code,
                detail=detail,
            )
        )


def _authorize(
    candidate: EvalCandidate,
    task: ExecutableTaskSpec,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
    oracle: OracleSession,
    gate: ContinuationGate,
) -> None:
    if (
        plan.source_verification_identity != source.verification_identity
        or plan.source_run_id != source.source_run_id
        or plan.source_task_ids_hash != source.task_index.task_ids_hash
        or plan.eval_config_hash != source.eval_config_hash
    ):
        raise ExecutableAuthorizationError(
            "eval.executable_plan",
            "was authorized against a different verified source",
            actual={
                "source_verification_identity": source.verification_identity,
                "source_run_id": source.source_run_id,
                "source_task_ids_hash": source.task_index.task_ids_hash,
                "eval_config_hash": source.eval_config_hash,
            },
            expected=(
                "the source identity, run, ordered task set, and eval config "
                "recorded by EligibleEvalPlan"
            ),
            recovery="re-run contamination analysis for this exact VerifiedEvalSource",
        )
    if not source.executable or source.oracle is None or not source.gold_eligible:
        raise ExecutableAuthorizationError(
            "eval.executable_source",
            "is not a gold-eligible source verified for executable evaluation",
            actual={
                "executable": source.executable,
                "gold_eligible": source.gold_eligible,
                "has_oracle": source.oracle is not None,
            },
            expected="a gold-eligible trace_and_executable VerifiedEvalSource",
            recovery="verify the publication and its executable oracle again",
        )
    expected = {
        "candidate_alias": candidate.alias,
        "canonical_model_identity": candidate.canonical_model_identity,
        "plan_identity": plan.plan_identity,
        "eval_config_hash": plan.eval_config_hash,
        "scoring_policy_hash": plan.scoring_policy_hash,
        "source_verification_identity": source.verification_identity,
        "source_content_hash": source.evaluation_benchmark.content_hash,
        "oracle_verification_identity": source.oracle.verification_identity,
    }
    actual = {
        "candidate_alias": task.candidate_alias,
        "canonical_model_identity": task.canonical_model_identity,
        "plan_identity": task.plan_identity,
        "eval_config_hash": task.eval_config_hash,
        "scoring_policy_hash": task.scoring_policy_hash,
        "source_verification_identity": task.source_verification_identity,
        "source_content_hash": task.source_content_hash,
        "oracle_verification_identity": task.oracle_verification_identity,
    }
    if actual != expected:
        raise ExecutableAuthorizationError(
            "eval.executable_task",
            "does not match the candidate, plan, source, and oracle being driven",
            actual=actual,
            expected=str(expected),
            recovery="rebuild the executable task from these exact authorized handles",
        )
    if task.task_id not in plan.evaluation_task_ids(candidate.alias):
        raise ExecutableAuthorizationError(
            f"candidates[{candidate.alias}]",
            "is not authorized to answer this executable task",
            actual=task.task_id,
            expected="one of plan.evaluation_task_ids(candidate.alias)",
            recovery="drive only contamination-eligible tasks",
        )
    if oracle.oracle_verification_identity != task.oracle_verification_identity:
        raise ExecutableAuthorizationError(
            "eval.oracle",
            "is not the verified resource bound into the task",
            actual=oracle.oracle_verification_identity,
            expected=task.oracle_verification_identity,
            recovery="open the oracle session from the same VerifiedEvalSource",
        )
    gate_hash = getattr(gate, "scoring_policy_hash", None)
    if gate_hash is not None and gate_hash != task.scoring_policy_hash:
        raise ExecutableAuthorizationError(
            "eval.continuation_gate",
            "uses a different scoring policy than the authorized task",
            actual=gate_hash,
            expected=task.scoring_policy_hash,
            recovery="construct the gate from the gated evaluation config",
        )


def _not_executed(
    call: CandidateToolCall,
    *,
    execution_index: int,
    turn_index: int,
    position: int,
    schema_valid: bool | None,
    schema_failures: Sequence[dict[str, Any]],
    reason_code: str,
    detail: str,
) -> ExecutedToolCall:
    return ExecutedToolCall(
        execution_index=execution_index,
        turn_index=turn_index,
        position_in_turn=position,
        provider_call_index=call.index,
        call_id=call.id,
        type=call.type,
        function_name=call.function_name,
        arguments_status=call.arguments_status,
        parsed_arguments=call.parsed_arguments,
        schema_valid=schema_valid,
        schema_failures=tuple(schema_failures),
        status="not_executed",
        state_commit="not_started",
        reason_code=reason_code,
        detail=detail,
    )


def _call_shape(
    call: CandidateToolCall,
    task: ExecutableTaskSpec,
) -> tuple[dict[str, Any] | None, bool | None, list[dict[str, Any]], str | None]:
    if call.type != "function" or not call.function_name or not call.id:
        return None, None, [], "tool_execution.invalid_call_shape"
    if call.arguments_status != "valid_object" or call.parsed_arguments is None:
        return None, None, [], "tool_execution.invalid_arguments"
    function = declared_function(task.script.tools, call.function_name)
    if function is None:
        return None, None, [], "tool_execution.undeclared_tool"
    arguments = thaw_json(call.parsed_arguments)
    failures = validate_function_arguments(function, arguments)
    if failures:
        return arguments, False, failures, "tool_execution.schema_invalid"
    return arguments, True, [], None


async def _state_before_mutation(
    oracle: OracleSession,
    *,
    mutates: bool,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not mutates:
        return True, None, None
    try:
        state = await oracle.get_state()
        return True, state, _sha256_json(state)
    except OracleStateError:
        # The call can still be attempted, but no result can prove whether it
        # committed without both sides of the state transition.
        return False, None, None


async def _commit_after_result(
    oracle: OracleSession,
    *,
    mutates: bool,
    before_known: bool,
    before_state: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if not mutates:
        return "not_applicable", None
    try:
        after_state = await oracle.get_state()
    except OracleStateError:
        return "unknown", None
    after_hash = _sha256_json(after_state)
    if not before_known or before_state is None:
        return "unknown", after_hash
    return (
        "not_committed" if json_equal(before_state, after_state) else "committed",
        after_hash,
    )


async def _execute_calls(
    response: CandidateResponse,
    *,
    turn_index: int,
    task: ExecutableTaskSpec,
    oracle: OracleSession,
    log: _Log,
    turn_authorized: bool,
) -> tuple[list[int], list[tuple[str, dict[str, Any]]], ExecutableEpisodeStatus | None, str | None]:
    indexes: list[int] = []
    live_results: list[tuple[str, dict[str, Any]]] = []
    terminal_status: ExecutableEpisodeStatus | None = None
    terminal_detail: str | None = None
    stop = False
    for position, call in enumerate(response.tool_calls):
        execution_index = len(log.executions)
        indexes.append(execution_index)
        arguments, schema_valid, failures, refusal = _call_shape(call, task)
        if stop or refusal is not None:
            reason = "tool_execution.prior_call_failed" if stop else refusal
            outcome = _not_executed(
                call,
                execution_index=execution_index,
                turn_index=turn_index,
                position=position,
                schema_valid=schema_valid,
                schema_failures=failures,
                reason_code=reason or "tool_execution.not_executed",
                detail=(
                    "a prior call in this completion ended the executable turn"
                    if stop
                    else f"the candidate call is not executable: {reason}"
                ),
            )
            log.executions.append(outcome)
            log.event(
                "tool_execution",
                outcome.reason_code,
                turn_index=turn_index,
                execution_index=execution_index,
                detail=outcome.detail,
            )
            continue
        assert arguments is not None and call.function_name is not None and call.id is not None
        policy = task.tool_policy(call.function_name)
        assert policy is not None
        # Only a call that asserts the pack's confirmation parameter commits the
        # protected mutation. A probe that leaves it unset is the very call a
        # confirmation template makes before the user has answered, so gating it
        # would refuse the published gold trace.
        confirm_parameter = policy.confirmation_parameter
        if (
            confirm_parameter is not None
            and arguments.get(confirm_parameter) is True
            and (turn_index not in task.confirmed_call_turns or not turn_authorized)
        ):
            outcome = _not_executed(
                call,
                execution_index=execution_index,
                turn_index=turn_index,
                position=position,
                schema_valid=True,
                schema_failures=(),
                reason_code="tool_execution.confirmation_not_earned",
                detail=(
                    "the candidate attempted a confirmation-protected call before "
                    "matching the call batch authorized by the user's confirmation"
                ),
            )
            log.executions.append(outcome)
            log.event(
                "tool_execution",
                outcome.reason_code,
                turn_index=turn_index,
                execution_index=execution_index,
                detail=outcome.detail,
            )
            terminal_status = "confirmation_not_earned"
            terminal_detail = outcome.detail
            stop = True
            continue
        before_known, before_state, state_before_hash = await _state_before_mutation(
            oracle,
            mutates=policy.mutates,
        )
        state_commit = "unknown" if policy.mutates else "not_applicable"
        state_after_hash: str | None = None
        try:
            raw_result = await oracle.call_tool(
                call.function_name,
                arguments,
                turn_index=turn_index,
            )
            state_commit, state_after_hash = await _commit_after_result(
                oracle,
                mutates=policy.mutates,
                before_known=before_known,
                before_state=before_state,
            )
            if not isinstance(raw_result, Mapping):
                result_type = {
                    type(None): "null",
                    bool: "bool",
                    int: "int",
                    float: "float",
                    str: "str",
                    list: "array",
                    tuple: "array",
                }.get(type(raw_result))
                if result_type is None:
                    raise TypeError(
                        f"oracle returned non-JSON {type(raw_result).__name__}"
                    )
                malformed_value = (
                    list(raw_result) if isinstance(raw_result, tuple) else raw_result
                )
                outcome = ExecutedToolCall(
                    execution_index=execution_index,
                    turn_index=turn_index,
                    position_in_turn=position,
                    provider_call_index=call.index,
                    call_id=call.id,
                    type=call.type,
                    function_name=call.function_name,
                    arguments_status=call.arguments_status,
                    parsed_arguments=call.parsed_arguments,
                    schema_valid=True,
                    status="malformed_result",
                    state_commit=state_commit,
                    state_before_hash=state_before_hash,
                    state_after_hash=state_after_hash,
                    malformed_result=malformed_value,
                    malformed_result_type=result_type,
                    malformed_result_hash=_sha256_json(malformed_value),
                    reason_code="tool_execution.malformed_result",
                    detail="the oracle returned valid JSON that was not an object",
                )
                terminal_status = (
                    "unknown_commit_state"
                    if state_commit == "unknown"
                    else "oracle_result_malformed"
                )
                terminal_detail = outcome.detail
                stop = True
            else:
                result = dict(raw_result)
                try:
                    validate_json_value(result, label="oracle tool result")
                except ValueError as exc:
                    raise TypeError("oracle returned a non-canonical JSON object") from exc
                error = result.get("error")
                business = isinstance(error, dict)
                if business:
                    code = error.get("code")
                    if not isinstance(code, str) or not code.strip():
                        raise TypeError("structured oracle error has no stable error.code")
                outcome = ExecutedToolCall(
                    execution_index=execution_index,
                    turn_index=turn_index,
                    position_in_turn=position,
                    provider_call_index=call.index,
                    call_id=call.id,
                    type=call.type,
                    function_name=call.function_name,
                    arguments_status=call.arguments_status,
                    parsed_arguments=call.parsed_arguments,
                    schema_valid=True,
                    status="business_rejection" if business else "completed",
                    state_commit=state_commit,
                    state_before_hash=state_before_hash,
                    state_after_hash=state_after_hash,
                    result=result,
                    result_hash=_sha256_json(result),
                    reason_code=(
                        "tool_execution.business_rejection"
                        if business
                        else "tool_execution.completed"
                    ),
                    detail=(
                        "the oracle returned a structured business rejection"
                        if business
                        else "the oracle returned a canonical JSON result"
                    ),
                )
                live_results.append((call.id, result))
                if state_commit == "unknown":
                    terminal_status = "unknown_commit_state"
                    terminal_detail = (
                        "the oracle returned a result but its state transition "
                        "could not be established"
                    )
                    stop = True
        except ValidationError:
            # An evidence record that violates the executable contract is a defect
            # in this runner, not oracle behavior. Relabelling it as a tool error
            # would publish a false account of what the oracle did.
            raise
        except TimeoutError as exc:
            outcome = ExecutedToolCall(
                execution_index=execution_index,
                turn_index=turn_index,
                position_in_turn=position,
                provider_call_index=call.index,
                call_id=call.id,
                type=call.type,
                function_name=call.function_name,
                arguments_status=call.arguments_status,
                parsed_arguments=call.parsed_arguments,
                schema_valid=True,
                status="timeout" if not policy.mutates else "unknown_commit_state",
                state_commit="not_applicable" if not policy.mutates else "unknown",
                state_before_hash=state_before_hash,
                reason_code=(
                    "tool_execution.timeout"
                    if not policy.mutates
                    else "tool_execution.unknown_commit_state"
                ),
                detail=f"the oracle call timed out as {type(exc).__name__}",
            )
            terminal_status = (
                "oracle_timeout" if not policy.mutates else "unknown_commit_state"
            )
            terminal_detail = outcome.detail
            stop = True
        except (OracleCallError, TypeError, ValueError) as exc:
            outcome = ExecutedToolCall(
                execution_index=execution_index,
                turn_index=turn_index,
                position_in_turn=position,
                provider_call_index=call.index,
                call_id=call.id,
                type=call.type,
                function_name=call.function_name,
                arguments_status=call.arguments_status,
                parsed_arguments=call.parsed_arguments,
                schema_valid=True,
                status=(
                    "unknown_commit_state"
                    if state_commit == "unknown"
                    else "tool_error"
                ),
                state_commit=state_commit,
                state_before_hash=state_before_hash,
                state_after_hash=state_after_hash,
                reason_code=(
                    "tool_execution.unknown_commit_state"
                    if state_commit == "unknown"
                    else "tool_execution.tool_error"
                ),
                detail=f"the oracle call failed as {type(exc).__name__}",
            )
            terminal_status = (
                "oracle_call_failed"
                if state_commit != "unknown"
                else "unknown_commit_state"
            )
            terminal_detail = outcome.detail
            stop = True
        log.executions.append(outcome)
        log.event(
            "tool_execution",
            outcome.reason_code,
            turn_index=turn_index,
            execution_index=execution_index,
            detail=outcome.detail,
        )
    return indexes, live_results, terminal_status, terminal_detail


async def run_executable_episode(
    *,
    candidate: EvalCandidate,
    limits: EvalLimits,
    client: NativeFunctionCallingClient,
    task: ExecutableTaskSpec,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
    oracle: OracleSession,
    gate: ContinuationGate,
    tool_trace_cache: ToolTraceCache | None = None,
) -> ExecutableEpisode:
    """Interleave candidate turns with live oracle results and return evidence.

    A cache hit replays a complete authorized episode. Individual calls are not
    memoized because a mutating result without its state transition cannot
    reproduce downstream calls, final state, or assertions.
    """

    conversation = _LiveConversation(task)
    log = _Log()
    log.event("seed", "episode.answer_free_seed", detail="answer-free opening")
    deadline = time.monotonic() + float(limits.episode_timeout_s)
    status: ExecutableEpisodeStatus = "completed"
    reason_code = "episode.completed"
    detail = "the live conversation reached its terminal milestone"
    final_state_hash: str | None = None
    session_ready = False
    pending_turn: int | None = None
    pending_executions: list[int] = []
    pending_user_hash: str | None = None
    producer_executions: dict[int, list[int]] = {}
    trace_request: ToolTraceRequest | None = None

    try:
        assert_source_unchanged(source)
        _authorize(candidate, task, source, plan, oracle, gate)
        if tool_trace_cache is not None:
            trace_request = build_tool_trace_request(
                candidate=candidate,
                task=task,
                source=source,
                plan=plan,
            )
            cached = tool_trace_cache.get(trace_request)
            if cached is not None:
                return cached
            tool_trace_cache.put_request(trace_request)
        try:
            await oracle.reset()
            session_ready = True
            log.event("oracle_reset", "oracle.reset.completed")
        except OracleResetError as exc:
            status = "oracle_reset_failed"
            reason_code = "episode.oracle_reset_failed"
            detail = str(exc)

        if session_ready:
            for scripted in task.script.turns:
                if scripted.turn_index >= limits.max_turns:
                    status = "max_turns_exceeded"
                    reason_code = "episode.max_turns_exceeded"
                    detail = f"limits.max_turns={limits.max_turns} ended the episode"
                    break
                if time.monotonic() >= deadline:
                    status = "episode_timeout"
                    reason_code = "episode.timeout"
                    detail = "the episode deadline elapsed before the next candidate turn"
                    break

                scripted, dependency_outcomes = resolve_turn_dependencies(
                    task=task,
                    turn=scripted,
                    executions=log.executions,
                    producer_executions=producer_executions,
                )
                log.dependencies.extend(dependency_outcomes)
                failed_dependency = next(
                    (
                        item
                        for item in dependency_outcomes
                        if item.status != "resolved"
                    ),
                    None,
                )
                if failed_dependency is not None:
                    status = "dependency_resolution_failed"
                    reason_code = f"episode.{failed_dependency.reason_code}"
                    detail = failed_dependency.detail
                    break

                # These messages become model-visible only when this request is
                # actually sent. Commit their release evidence at that boundary.
                if pending_turn is not None:
                    previous = log.turns[pending_turn]
                    log.turns[pending_turn] = previous.model_copy(
                        update={
                            "advanced": True,
                            "released_user_message_hash": pending_user_hash,
                            "reason_code": "turn.advanced",
                            "detail": "the next live candidate request admitted earned observations",
                        }
                    )
                    for index in pending_executions:
                        log.executions[index] = log.executions[index].model_copy(
                            update={"released_to_model": True}
                        )
                    if pending_executions:
                        log.event(
                            "tool_results",
                            "conversation.live_results_released",
                            turn_index=pending_turn,
                        )
                    if pending_user_hash is not None:
                        log.event(
                            "user_turn",
                            "conversation.scripted_user_released",
                            turn_index=pending_turn,
                        )
                    pending_turn = None
                    pending_executions = []
                    pending_user_hash = None

                conversation.audit()
                request = build_candidate_request(
                    candidate,
                    request_id=turn_request_id(
                        candidate.alias, task.task_id, scripted.turn_index
                    ),
                    task_id=task.task_id,
                    turn_index=scripted.turn_index,
                    messages=conversation.messages,
                    tools=[thaw_json(tool) for tool in task.script.tools],
                )
                outcome = await client.complete(request, deadline=deadline)
                if outcome.status != "completed" or outcome.response is None:
                    log.turns.append(
                        ExecutableTurn(
                            turn_index=scripted.turn_index,
                            request_hash=request.request_hash,
                            call_status=outcome.status,
                            advanced=False,
                            reason_code=f"turn.{outcome.status}",
                            detail=f"the candidate call ended as {outcome.status}",
                        )
                    )
                    status = (
                        "malformed_response"
                        if outcome.status == "malformed_response"
                        else "candidate_call_failed"
                    )
                    reason_code = f"episode.{status}"
                    detail = f"candidate turn {scripted.turn_index} ended as {outcome.status}"
                    break
                response = outcome.response
                log.event(
                    "candidate_turn",
                    "candidate.turn.completed",
                    turn_index=scripted.turn_index,
                    detail=f"finish_reason={response.finish_reason}",
                )
                if finish_reason_problem(response.finish_reason) is not None:
                    # A truncated or filtered completion is not an answer, so the
                    # gate is not asked and no proposed call is executed.
                    execution_indexes = [
                        len(log.executions) + position
                        for position in range(len(response.tool_calls))
                    ]
                    for position, call in enumerate(response.tool_calls):
                        invalid = _not_executed(
                            call,
                            execution_index=execution_indexes[position],
                            turn_index=scripted.turn_index,
                            position=position,
                            schema_valid=None,
                            schema_failures=(),
                            reason_code="tool_execution.incomplete_response",
                            detail="the provider marked this response incomplete",
                        )
                        log.executions.append(invalid)
                        log.event(
                            "tool_execution",
                            invalid.reason_code,
                            turn_index=scripted.turn_index,
                            execution_index=invalid.execution_index,
                        )
                    log.turns.append(
                        ExecutableTurn(
                            turn_index=scripted.turn_index,
                            request_hash=request.request_hash,
                            call_status=outcome.status,
                            response_hash=response.response_hash,
                            finish_reason=response.finish_reason,
                            assistant_content=response.assistant_content,
                            tool_calls=response.tool_calls,
                            tool_call_outcome_indexes=tuple(execution_indexes),
                            advanced=False,
                            reason_code="turn.incomplete_finish",
                            detail="the provider marked the completion incomplete",
                        )
                    )
                    status = "candidate_mismatch"
                    reason_code = "episode.incomplete_finish"
                    detail = "an incomplete provider finish cannot advance executable evaluation"
                    break

                call_ids = [call.id for call in response.tool_calls]
                unusable_ids = bool(response.tool_calls) and (
                    any(not call_id for call_id in call_ids)
                    or len(set(call_ids)) != len(call_ids)
                )
                match = gate.evaluate(scripted, response, script=task.script)
                if unusable_ids:
                    execution_indexes = []
                    for position, call in enumerate(response.tool_calls):
                        invalid = _not_executed(
                            call,
                            execution_index=len(log.executions),
                            turn_index=scripted.turn_index,
                            position=position,
                            schema_valid=None,
                            schema_failures=(),
                            reason_code="tool_execution.unusable_call_id",
                            detail="tool-call ids are missing or duplicated",
                        )
                        execution_indexes.append(invalid.execution_index)
                        log.executions.append(invalid)
                        log.event(
                            "tool_execution",
                            invalid.reason_code,
                            turn_index=scripted.turn_index,
                            execution_index=invalid.execution_index,
                        )
                    live_results: list[tuple[str, dict[str, Any]]] = []
                    oracle_status = None
                    oracle_detail = None
                else:
                    (
                        execution_indexes,
                        live_results,
                        oracle_status,
                        oracle_detail,
                    ) = await _execute_calls(
                        response,
                        turn_index=scripted.turn_index,
                        task=task,
                        oracle=oracle,
                        log=log,
                        turn_authorized=match.advanced,
                    )
                turn = ExecutableTurn(
                    turn_index=scripted.turn_index,
                    request_hash=request.request_hash,
                    call_status=outcome.status,
                    response_hash=response.response_hash,
                    finish_reason=response.finish_reason,
                    assistant_content=response.assistant_content,
                    tool_calls=response.tool_calls,
                    tool_call_outcome_indexes=tuple(execution_indexes),
                    paired_call_indexes=(
                        match.paired_call_indexes
                        if match.advanced and not unusable_ids
                        else ()
                    ),
                    advanced=False,
                    reason_code="turn.observed",
                    detail="the candidate turn and its live outcomes were recorded",
                )
                log.turns.append(turn)
                if match.advanced and not unusable_ids:
                    for expected_index, execution_index in zip(
                        match.paired_call_indexes,
                        execution_indexes,
                        strict=True,
                    ):
                        producer_executions.setdefault(expected_index, []).append(
                            execution_index
                        )
                if unusable_ids:
                    status = "unusable_tool_call_ids"
                    reason_code = "episode.unusable_tool_call_ids"
                    detail = "tool results cannot be addressed to missing or duplicate ids"
                    break
                if oracle_status is not None:
                    status = oracle_status
                    reason_code = f"episode.{oracle_status}"
                    detail = oracle_detail or "live oracle execution failed"
                    break
                if not match.advanced:
                    status = "candidate_mismatch"
                    reason_code = "episode.candidate_mismatch"
                    detail = match.detail
                    break

                conversation.append_candidate(response)
                if live_results:
                    conversation.append_live_results(live_results)
                user_hash = None
                if scripted.releases_user_message is not None:
                    conversation.append_user(scripted.releases_user_message)
                    user_hash = _sha256_json(
                        scripted.releases_user_message.model_dump(mode="json")
                    )
                if scripted.is_terminal:
                    log.turns[-1] = turn.model_copy(
                        update={
                            "advanced": True,
                            "reason_code": "turn.terminal",
                            "detail": "the candidate reached the terminal milestone",
                        }
                    )
                else:
                    pending_turn = scripted.turn_index
                    pending_executions = [
                        index
                        for index in execution_indexes
                        if log.executions[index].has_result
                    ]
                    pending_user_hash = user_hash

        if session_ready:
            try:
                state = await oracle.get_state()
                final_state_hash = _sha256_json(state)
                log.event(
                    "state_snapshot",
                    "oracle.final_state_recorded",
                    detail=f"state_hash={final_state_hash}",
                )
            except OracleStateError as exc:
                # Only an episode that otherwise ran to term is defined by its
                # final state. A conversation that already ended for its own
                # reason keeps that reason: the snapshot is taken during wind
                # down, so failing to take it cannot be the terminal verdict.
                if status == "completed":
                    status = "oracle_state_failed"
                    reason_code = "episode.oracle_state_failed"
                    detail = str(exc)
            if final_state_hash is not None and status not in _FATAL_EXECUTION_STATUSES:
                for name in task.success_assertions:
                    try:
                        verdict = await oracle.run_assertion(
                            name,
                            task=thaw_json(task.assertion_task),
                        )
                        # OracleSession is a protocol, so an adapter this driver
                        # did not write can answer with any shape. An unreadable
                        # verdict is an infrastructure fact about the assertion,
                        # not a reason to lose the whole episode's evidence.
                        assertion_status = (
                            verdict.get("status") if isinstance(verdict, Mapping) else None
                        )
                        if assertion_status not in _ASSERTION_STATUSES:
                            raise OracleAssertionError(
                                f"eval.oracle.assertion[{name}]",
                                "returned a verdict this contract cannot read",
                                actual=assertion_status,
                                expected=f"one of {sorted(_ASSERTION_STATUSES)}",
                                recovery=(
                                    "return passed, failed, not_applicable, or "
                                    "infrastructure_error"
                                ),
                            )
                        assertion = AssertionOutcome(
                            assertion_index=len(log.assertions),
                            name=name,
                            category=(
                                task.assertion_spec(name).category
                                if task.assertion_spec(name) is not None
                                else "unclassified"
                            ),
                            status=assertion_status,
                            reason_code=f"assertion.{assertion_status}",
                            detail=str(verdict.get("detail") or assertion_status),
                        )
                    except OracleAssertionError as exc:
                        assertion = AssertionOutcome(
                            assertion_index=len(log.assertions),
                            name=name,
                            category=(
                                task.assertion_spec(name).category
                                if task.assertion_spec(name) is not None
                                else "unclassified"
                            ),
                            status="infrastructure_error",
                            reason_code="assertion.infrastructure_error",
                            detail=str(exc),
                        )
                    log.assertions.append(assertion)
                    if assertion.status == "infrastructure_error":
                        status = "assertion_infrastructure_failed"
                        reason_code = "episode.assertion_infrastructure_failed"
                        detail = f"assertion {name} could not run"
                        break
                if task.success_assertions:
                    log.event(
                        "assertions",
                        "assertions.completed",
                        detail=f"{len(log.assertions)} assertion(s) recorded",
                    )
    finally:
        try:
            await oracle.close()
        except OracleSessionError as exc:
            if status == "completed":
                status = "oracle_session_failed"
                reason_code = "episode.oracle_session_failed"
                detail = str(exc)

    log.event(
        "terminal",
        reason_code,
        turn_index=log.turns[-1].turn_index if log.turns else None,
        detail=detail,
    )
    episode = ExecutableEpisode(
        candidate_alias=candidate.alias,
        canonical_model_identity=candidate.canonical_model_identity,
        task_id=task.task_id,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        source_verification_identity=source.verification_identity,
        oracle_verification_identity=task.oracle_verification_identity,
        script_hash=task.script.script_hash,
        task_spec_hash=task.task_spec_hash,
        status=status,
        reason_code=reason_code,
        detail=detail,
        assistant_turns=len(log.turns),
        observed=tuple(log.turns),
        executions=tuple(log.executions),
        dependencies=tuple(log.dependencies),
        final_state_hash=final_state_hash,
        assertions=tuple(log.assertions),
        events=tuple(log.events),
        replayed=False,
    )
    if tool_trace_cache is not None:
        if trace_request is None:  # pragma: no cover - authorization creates it.
            raise RuntimeError("tool-trace request was not initialized")
        tool_trace_cache.put_completion(trace_request, episode)
    return episode


__all__ = ["run_executable_episode"]
