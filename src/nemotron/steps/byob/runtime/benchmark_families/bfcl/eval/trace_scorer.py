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

"""Turn one recorded episode into one task score, and nothing else.

The scorer is a pure function of evidence and policy. It reads a
:class:`CandidateEpisode`, the :class:`ConversationScript` that produced it, and
the pinned :class:`EvalScoringConfig`; it contacts no provider, executes no tool,
reads no clock, and re-parses no provider bytes. Scoring the same episode twice
therefore produces the same ``score_hash``, which is what makes a published
number auditable after the endpoint it came from is gone.

Two design choices are worth stating, because they decide what a gate rate means.

*Coverage versus consistency.* ``tool_selection`` and ``arguments`` are coverage
gates: they ask whether the whole gold trace was requested, so gold calls in
turns the episode never reached count against them. ``call_grouping``,
``call_ordering``, and ``text_turn`` are consistency gates: they ask whether what
the candidate did do was well-formed, so they are measured only over turns that
were actually asked. Whether the conversation finished at all is its own gate, and
because that gate always applies, no unfinished episode can be a success.

*No repair, ever.* A call whose arguments never parsed stays unparsed, a call
naming an undeclared tool stays undeclared, and a config that asks for LLM repair
is refused rather than quietly ignored. A score that had been repaired would be a
property of the repairer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_comparison import (
    CallComparison,
    compare_group_size,
    compare_text_turn,
    compare_turn_order,
    finish_reason_problem,
    pair_turn_calls,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import EligibleEvalPlan
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    CandidateEpisode,
    ConversationScript,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalScoringConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_parser import (
    ParsedCall,
    ParsedTrace,
    ParsedTurn,
    parse_observed_trace,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_contract import (
    SCORING_GATES,
    GateResult,
    ScoredCall,
    ScoredTurn,
    ScoringGate,
    TraceGateFailureClass,
    TraceTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_errors import (
    TraceEvidenceError,
    TraceScoringPolicyError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import thaw_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    declared_function,
    validate_function_arguments,
)


@dataclass(frozen=True)
class NormalizedTraceScore:
    """Shared pure trace-gate result for either episode evidence contract."""

    turns: tuple[ScoredTurn, ...]
    gates: tuple[GateResult, ...]
    matched_calls: int


def score_trace_episode(
    *,
    episode: CandidateEpisode,
    script: ConversationScript,
    scoring: EvalScoringConfig,
    plan: EligibleEvalPlan,
) -> TraceTaskScore:
    """Score one candidate's episode of one task against the gold trace."""
    _refuse_unsupported_policy(scoring)
    _authorize_score(episode, script, scoring=scoring, plan=plan)
    trace = parse_observed_trace(episode, script)
    normalized = score_normalized_trace(
        trace=trace,
        script=script,
        scoring=scoring,
        completion_detail=episode.detail,
    )

    failed = tuple(gate.gate for gate in normalized.gates if gate.counts_against)
    return TraceTaskScore(
        task_id=episode.task_id,
        candidate_alias=episode.candidate_alias,
        canonical_model_identity=episode.canonical_model_identity,
        plan_identity=episode.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        source_verification_identity=episode.source_verification_identity,
        script_hash=episode.script_hash,
        episode_hash=episode.episode_hash,
        scoring_contract_hash=scoring.contract.content_hash,
        scoring_policy=scoring.semantic_payload(),
        episode_status=episode.status,
        non_candidate_stop=trace.non_candidate_stop,
        expected_calls=script.expected_call_count,
        observed_calls=trace.observed_calls,
        matched_calls=normalized.matched_calls,
        turns=normalized.turns,
        gates=normalized.gates,
        task_success=not failed,
        detail=("every applicable gate passed" if not failed else f"failed gate(s): {', '.join(failed)}"),
    )


def score_normalized_trace(
    *,
    trace: ParsedTrace,
    script: ConversationScript,
    scoring: EvalScoringConfig,
    completion_detail: str,
) -> NormalizedTraceScore:
    """Run the shared trace gates over already-canonical normalized evidence."""

    turns: list[ScoredTurn] = []
    names_so_far: list[str | None] = []
    for parsed in trace.turns:
        names_so_far.extend(call.function_name for call in parsed.calls)
        turns.append(
            _score_turn(
                parsed,
                script.turn(parsed.turn_index),
                script=script,
                scoring=scoring,
                names_so_far=tuple(names_so_far),
            )
        )
    scored = tuple(turns)
    gates = _score_gates(
        script,
        trace,
        scored,
        scoring=scoring,
        completion_detail=completion_detail,
    )
    return NormalizedTraceScore(
        turns=scored,
        gates=gates,
        matched_calls=len(_matched_gold(scored)),
    )


def _authorize_score(
    episode: CandidateEpisode,
    script: ConversationScript,
    *,
    scoring: EvalScoringConfig,
    plan: EligibleEvalPlan,
) -> None:
    """Bind the score to the authorization and policy that produced the episode."""
    if episode.plan_identity != plan.plan_identity:
        raise TraceEvidenceError(
            "eval.episode.plan_identity",
            "does not identify the authorization plan supplied for scoring",
            actual=episode.plan_identity,
            expected=plan.plan_identity,
            recovery="score the episode with the EligibleEvalPlan the driver used",
        )
    if script.source_verification_identity != plan.source_verification_identity:
        raise TraceEvidenceError(
            "eval.script.source_verification_identity",
            "does not come from the benchmark this plan authorizes",
            actual=script.source_verification_identity,
            expected=plan.source_verification_identity,
            recovery="score source-bound scripts only with the plan gated against that source",
        )
    if scoring.scoring_policy_hash != plan.scoring_policy_hash:
        raise TraceScoringPolicyError(
            "scoring",
            "is not the scoring policy the authorization plan was created under",
            actual=scoring.scoring_policy_hash,
            expected=plan.scoring_policy_hash,
            recovery="use the scoring section of the eval config that produced the plan",
        )
    try:
        candidate = plan.candidate(episode.candidate_alias)
    except KeyError as exc:
        raise TraceEvidenceError(
            "eval.episode.candidate_alias",
            "is not a candidate this plan authorizes",
            actual=episode.candidate_alias,
            expected=f"one of {list(plan.candidate_aliases)}",
            recovery="score only episodes produced under the supplied plan",
        ) from exc
    if candidate.canonical_model_identity != episode.canonical_model_identity:
        raise TraceEvidenceError(
            "eval.episode.canonical_model_identity",
            "does not name the weights this plan authorized for the candidate alias",
            actual=episode.canonical_model_identity,
            expected=candidate.canonical_model_identity,
            recovery="re-run authorization when an alias points at different weights",
        )
    if episode.task_id not in plan.evaluation_task_ids(episode.candidate_alias):
        raise TraceEvidenceError(
            "eval.episode.task_id",
            "is not a task this plan authorizes for the candidate",
            actual=episode.task_id,
            expected="one of the candidate's authorized task ids",
            recovery="score only the task set returned by the contamination gate",
        )


def _refuse_unsupported_policy(scoring: EvalScoringConfig) -> None:
    """Refuse a policy this scorer cannot honour, rather than approximating it."""
    if scoring.allow_llm_repair:
        raise TraceScoringPolicyError(
            "scoring.allow_llm_repair",
            "asks for candidate output to be repaired before it is scored, which this scorer will not do",
            actual=True,
            expected="false",
            recovery=(
                "set scoring.allow_llm_repair to false; a repaired call measures the repairer, "
                "and scoring it un-repaired would not be the number the config asked for"
            ),
        )
    if scoring.task_success == "assertions_only":
        raise TraceScoringPolicyError(
            "scoring.task_success",
            "asks for a verdict from pack assertions, which a trace score has no oracle to evaluate",
            actual=scoring.task_success,
            expected="all_applicable_gates",
            recovery=(
                "score with task_success: all_applicable_gates, or run executable evaluation, "
                "which is the mode that replays the pack's assertions"
            ),
        )


def _score_turn(
    parsed: ParsedTurn,
    scripted: ScriptedTurn,
    *,
    script: ConversationScript,
    scoring: EvalScoringConfig,
    names_so_far: Sequence[str | None],
) -> ScoredTurn:
    if not scripted.expects_tool_calls:
        text = compare_text_turn(
            scripted,
            parsed.assistant_content,
            [call.function_name for call in parsed.calls],
            scoring=scoring,
        )
        return ScoredTurn(
            turn_index=parsed.turn_index,
            kind="text",
            call_status=parsed.call_status,
            advanced=parsed.advanced,
            finish_reason=parsed.finish_reason,
            calls=tuple(
                _scored_call(parsed.turn_index, position, call, script=script, scoring=scoring)
                for position, call in enumerate(parsed.calls)
            ),
            text_matched=text.matched,
            detail=text.detail,
        )

    # Paired as a set even on a row that orders its calls, because order is a gate
    # of its own. Pairing positionally would make a turn that called the right
    # tools in the wrong order fail tool selection too, and a report could no
    # longer tell "called the wrong tool" from "called them out of order".
    pairing = pair_turn_calls(
        scripted.calls,
        parsed.calls,
        tools=script.tools,
        scoring=scoring,
        scope="set",
    )
    gold_by_index = {call.call_index: call for call in scripted.calls}
    calls: list[ScoredCall] = []
    for position, call in enumerate(parsed.calls):
        gold_index = pairing.paired_gold_indexes[position]
        gold = gold_by_index[gold_index] if gold_index is not None else None
        calls.append(
            _scored_call(
                parsed.turn_index,
                position,
                call,
                script=script,
                scoring=scoring,
                gold_call_index=gold_index,
                gold_function_name=gold.function_name if gold is not None else None,
                comparison=pairing.comparisons[position],
            )
        )
    order_problem = compare_turn_order(
        scripted,
        parsed.calls,
        script=script,
        scoring=scoring,
        pairing=pairing,
        names_so_far=names_so_far,
    )
    ordered = None if not _orders_calls(script, scoring) else order_problem is None
    return ScoredTurn(
        turn_index=parsed.turn_index,
        kind="tool_calls",
        call_status=parsed.call_status,
        advanced=parsed.advanced,
        finish_reason=parsed.finish_reason,
        calls=tuple(calls),
        group_size_matched=compare_group_size(len(scripted.calls), len(parsed.calls)) is None,
        order_respected=ordered,
        order_detail=order_problem if ordered is False else None,
        detail=pairing.detail,
    )


def _scored_call(
    turn_index: int,
    position: int,
    call: ParsedCall,
    *,
    script: ConversationScript,
    scoring: EvalScoringConfig,
    gold_call_index: int | None = None,
    gold_function_name: str | None = None,
    comparison: CallComparison | None = None,
) -> ScoredCall:
    """Record one call: what it was, what it answered, and why that is or is not right.

    ``comparison`` is absent for a call made on a turn the trace answers in words.
    There is no gold call to hold it against, so it is recorded as a call the
    trace does not ask for and left to the selection and text gates.
    """
    matched_name = comparison is not None and comparison.name_matched
    unasked = f"the trace answers this request in words; the candidate called {call.function_name or '<unnamed>'}"
    return ScoredCall(
        turn_index=turn_index,
        position_in_turn=position,
        gold_call_index=gold_call_index if matched_name else None,
        gold_function_name=gold_function_name if matched_name else None,
        predicted_function_name=call.function_name,
        predicted_type=call.type,
        arguments_status=call.arguments_status,
        name_matched=matched_name,
        arguments_matched=comparison is not None and comparison.arguments_matched,
        diff=comparison.diff if comparison is not None else None,
        schema_failures=_schema_failures(call, script=script, scoring=scoring),
        detail=comparison.detail if comparison is not None else unasked,
    )


def _schema_failures(
    call: ParsedCall,
    *,
    script: ConversationScript,
    scoring: EvalScoringConfig,
) -> tuple[dict[str, Any], ...]:
    """Validate a predicted call against the schema the row declares for its tool.

    Declared defaults are deliberately *not* inserted first. Argument matching
    fills a default on whichever side omitted it so that spelling one out is
    neither rewarded nor punished, but a parameter the schema also marks required
    is one the caller has to pass: filling it in before validating would let
    default insertion launder a missing required argument into a pass.
    """
    if scoring.argument_matching != "schema_then_canonical":
        return ()
    function = declared_function(script.tools, call.function_name or "")
    if function is None:
        return ({"reason": "unknown_tool", "tool": call.function_name},)
    arguments = thaw_json(call.parsed_arguments) if call.parsed_arguments is not None else None
    return tuple(validate_function_arguments(function, arguments))


def _orders_calls(script: ConversationScript, scoring: EvalScoringConfig) -> bool:
    return scoring.respect_call_order and script.call_order != "any"


def _flat_calls(scored: Sequence[ScoredTurn]) -> Iterator[tuple[ScoredTurn, ScoredCall]]:
    for turn in scored:
        for call in turn.calls:
            yield turn, call


def _matched_gold(scored: Sequence[ScoredTurn]) -> set[int]:
    return {
        call.gold_call_index for _, call in _flat_calls(scored) if call.matched and call.gold_call_index is not None
    }


def _named_gold(scored: Sequence[ScoredTurn]) -> set[int]:
    return {
        call.gold_call_index
        for _, call in _flat_calls(scored)
        if call.name_matched and call.gold_call_index is not None
    }


def _gold_indexes(script: ConversationScript) -> tuple[int, ...]:
    return tuple(call.call_index for turn in script.turns for call in turn.calls)


def _turn_of_gold_call(script: ConversationScript, call_index: int) -> int | None:
    for turn in script.turns:
        if any(call.call_index == call_index for call in turn.calls):
            return turn.turn_index
    return None


def _passed(gate: ScoringGate, detail: str) -> GateResult:
    return GateResult(
        gate=gate,
        outcome="passed",
        reason_code=f"{gate}.matched",
        detail=detail,
    )


def _skipped(gate: ScoringGate, detail: str) -> GateResult:
    return GateResult(
        gate=gate,
        outcome="not_applicable",
        reason_code=f"{gate}.not_applicable",
        detail=detail,
    )


def _failed(
    gate: ScoringGate,
    detail: str,
    *,
    turn_index: int | None = None,
    reason_code: str | None = None,
    failure_class: TraceGateFailureClass = "candidate",
) -> GateResult:
    """Fail one gate, blaming the candidate unless the caller knows better.

    Every gate but completion compares what the candidate produced against what
    the trace asks for, so the candidate is the default subject. Completion is the
    one gate that can fail without the candidate having answered at all.
    """
    return GateResult(
        gate=gate,
        outcome="failed",
        failure_class=failure_class,
        reason_code=reason_code or f"{gate}.mismatch",
        detail=detail,
        turn_index=turn_index,
    )


def _score_gates(
    script: ConversationScript,
    trace: ParsedTrace,
    scored: Sequence[ScoredTurn],
    *,
    scoring: EvalScoringConfig,
    completion_detail: str,
) -> tuple[GateResult, ...]:
    results = {
        "tool_selection": _tool_selection_gate(script, scored),
        "arguments": _arguments_gate(script, scored),
        "schema_valid": _schema_gate(scored, scoring=scoring),
        "call_grouping": _grouping_gate(script, scored, scoring=scoring),
        "call_ordering": _ordering_gate(script, scored, scoring=scoring),
        "text_turn": _text_gate(script, scored),
        "trace_completion": _completion_gate(trace, completion_detail),
    }
    return tuple(_attributed(results[gate], trace) for gate in SCORING_GATES)


def _attributed(gate: GateResult, trace: ParsedTrace) -> GateResult:
    """Move a failure the candidate was never given the chance to avoid onto the run.

    Coverage is measured against the whole script, so a gold call the episode never
    got an answer for fails selection and arguments. That is the candidate's failure
    when the candidate is why the episode stopped, and the run's when it is not:
    blaming a model for a turn an unreachable endpoint or a spent budget prevented
    it from answering would make infrastructure look like weak tool use. The verdict
    itself is untouched — an unfinished task is failed either way.
    """
    if not trace.non_candidate_stop or not gate.counts_against:
        return gate
    if _answered(trace, gate.turn_index):
        return gate
    return gate.model_copy(update={"failure_class": "infrastructure"})


def _answered(trace: ParsedTrace, turn_index: int | None) -> bool:
    """Whether the candidate returned a usable answer for the turn a failure names.

    A turn the episode never sent and a turn whose request never came back are both
    turns the candidate did not answer, and neither can carry evidence of what it
    would have said.
    """
    if turn_index is None or turn_index >= len(trace.turns):
        return False
    return trace.turns[turn_index].answered


def _tool_selection_gate(script: ConversationScript, scored: Sequence[ScoredTurn]) -> GateResult:
    for turn, call in _flat_calls(scored):
        if not call.name_matched:
            return _failed("tool_selection", call.detail, turn_index=turn.turn_index)
    missing = tuple(sorted(set(_gold_indexes(script)) - _named_gold(scored)))
    if missing:
        return _failed(
            "tool_selection",
            f"the trace's call(s) {list(missing)} were never requested",
            turn_index=_turn_of_gold_call(script, missing[0]),
        )
    if not script.expected_call_count:
        return _passed("tool_selection", "the trace asks for no call, and the candidate made none")
    return _passed("tool_selection", "every gold call was requested, and nothing else was called")


def _arguments_gate(script: ConversationScript, scored: Sequence[ScoredTurn]) -> GateResult:
    if not script.expected_call_count:
        return _skipped("arguments", "the trace asks for no call, so there are no arguments to compare")
    for turn, call in _flat_calls(scored):
        if call.name_matched and not call.arguments_matched:
            return _failed("arguments", call.detail, turn_index=turn.turn_index)
    missing = tuple(sorted(set(_gold_indexes(script)) - _matched_gold(scored)))
    if missing:
        return _failed(
            "arguments",
            f"the trace's call(s) {list(missing)} were never matched",
            turn_index=_turn_of_gold_call(script, missing[0]),
        )
    return _passed("arguments", "every gold call's arguments matched the trace")


def _schema_gate(scored: Sequence[ScoredTurn], *, scoring: EvalScoringConfig) -> GateResult:
    if scoring.argument_matching != "schema_then_canonical":
        return _skipped("schema_valid", "scoring.argument_matching skips the schema step")
    calls = list(_flat_calls(scored))
    if not calls:
        return _skipped("schema_valid", "the candidate made no call to validate")
    for turn, call in calls:
        if call.schema_failures:
            reasons = ", ".join(sorted({str(failure.get("reason")) for failure in call.schema_failures}))
            return _failed(
                "schema_valid",
                f"the call to {call.predicted_function_name or '<unnamed>'} violates its declared schema: {reasons}",
                turn_index=turn.turn_index,
            )
    return _passed("schema_valid", "every call the candidate made satisfies its declared schema")


def _grouping_gate(
    script: ConversationScript,
    scored: Sequence[ScoredTurn],
    *,
    scoring: EvalScoringConfig,
) -> GateResult:
    if not scoring.respect_call_group:
        return _skipped("call_grouping", "scoring.respect_call_group does not hold calls to their gold group")
    if not any(turn.expects_tool_calls for turn in script.turns):
        return _skipped("call_grouping", "the trace groups no calls")
    for turn in scored:
        if turn.group_size_matched is False:
            return _failed("call_grouping", turn.detail, turn_index=turn.turn_index)
    return _passed("call_grouping", "every asked turn issued its gold call group")


def _ordering_gate(
    script: ConversationScript,
    scored: Sequence[ScoredTurn],
    *,
    scoring: EvalScoringConfig,
) -> GateResult:
    if not scoring.respect_call_order:
        return _skipped("call_ordering", "scoring.respect_call_order does not order calls")
    if script.call_order == "any":
        return _skipped("call_ordering", "the row declares call_order: any")
    if script.expected_call_count < 2:
        return _skipped("call_ordering", "a trace of fewer than two calls has no order to respect")
    for turn in scored:
        if turn.order_respected is False:
            return _failed(
                "call_ordering",
                turn.order_detail or turn.detail,
                turn_index=turn.turn_index,
            )
    return _passed("call_ordering", f"the calls that were made respect call_order: {script.call_order}")


def _text_gate(script: ConversationScript, scored: Sequence[ScoredTurn]) -> GateResult:
    if not any(not turn.expects_tool_calls for turn in script.turns):
        return _skipped("text_turn", "the trace answers every request with calls")
    for turn in scored:
        if turn.text_matched is False:
            return _failed("text_turn", turn.detail, turn_index=turn.turn_index)
    return _passed("text_turn", "every asked text turn answered as the trace does")


def _completion_gate(trace: ParsedTrace, completion_detail: str) -> GateResult:
    """Whether the conversation reached the end of the trace at all.

    This gate always applies, which is what makes an unfinished episode a failed
    task rather than a skipped one: silently dropping a timed-out or unreachable
    candidate would let a slow model score higher than a fast wrong one. What the
    attribution adds is who to send the failure to. An episode that stopped for a
    reason the candidate did not choose still fails the task, but it is a run to
    re-drive; a truncated or filtered answer is the candidate's own.
    """
    if trace.reached_the_end:
        for turn in trace.turns:
            problem = finish_reason_problem(turn.finish_reason)
            if problem is not None:
                # An episode that reached the end of its trace was not stopped by
                # the run, so a truncated answer inside it is the candidate's.
                return _failed(
                    "trace_completion",
                    problem,
                    turn_index=turn.turn_index,
                    reason_code="trace_completion.incomplete_finish_reason",
                )
        return _passed("trace_completion", "the conversation was replayed to the end of the trace")
    stopped_at = trace.unsent_turn_indexes[0] if trace.unsent_turn_indexes else len(trace.turns) - 1
    return _failed(
        "trace_completion",
        completion_detail,
        turn_index=max(stopped_at, 0),
        failure_class="infrastructure" if trace.non_candidate_stop else "candidate",
    )


__all__ = ["NormalizedTraceScore", "score_normalized_trace", "score_trace_episode"]
