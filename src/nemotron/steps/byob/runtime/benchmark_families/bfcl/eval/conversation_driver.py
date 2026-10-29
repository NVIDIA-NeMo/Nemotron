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

"""Drive one authorized task through one candidate, one turn at a time.

The driver is a state machine over a :class:`ConversationScript`. It sends the
answer-free prefix, asks the candidate for one assistant turn, decides — through
an injected :class:`ContinuationGate` — whether that turn earns the tool results
the benchmark recorded, injects the next scripted user request, and repeats until
the trace ends or the episode cannot faithfully continue.

Three things it deliberately does not do. It does not score:
:class:`CandidateEpisode` is evidence consumed by a separate scorer. It does not
execute a tool: the results it releases are the ones benchmark replay recorded,
so a candidate's answer cannot depend on a live fixture. Live execution is a
separate runtime mode. And it does not decide policy: turn budget, timeouts, and
matching flags all arrive from the pinned ``eval_config.yaml``.

Determinism is what makes an episode re-runnable. Everything the driver chooses is
a function of the script, the config, and the candidate's own bytes; nothing reads
a clock except the episode deadline, and that only ever ends an episode early.
"""

from __future__ import annotations

import time

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_matching import (
    ContinuationGate,
    TurnMatch,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
    build_candidate_request,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateCallOutcome,
    CandidateResponse,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import EligibleEvalPlan
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    CandidateEpisode,
    ConversationScript,
    EpisodeEvent,
    EpisodeStatus,
    EventKind,
    ObservedTurn,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_errors import (
    ConversationAuthorizationError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_projection import (
    ModelFacingConversation,
    released_results,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalCandidate, EvalLimits


def _failure_status(call_status: str) -> EpisodeStatus:
    """Separate the model's own failure from the transport's or the provider's."""
    return "malformed_response" if call_status == "malformed_response" else "candidate_call_failed"


def turn_request_id(candidate_alias: str, task_id: str, turn_index: int) -> str:
    """Name one turn of one task for one candidate.

    Two runs that reach an identical turn of an identical conversation share this
    id on purpose: the request hash covers the whole prompt, so a cache hit there
    is the same question asked twice, not a collision.
    """
    return f"{candidate_alias}:{task_id}:{turn_index}"


class _EpisodeLog:
    """Accumulates the ordered record of what the driver did."""

    __slots__ = ("_events", "_observed", "cache_hits", "released_tool_results", "released_user_turns")

    def __init__(self) -> None:
        self._events: list[EpisodeEvent] = []
        self._observed: list[ObservedTurn] = []
        self.released_user_turns = 0
        self.released_tool_results = 0
        self.cache_hits = 0

    def event(
        self,
        kind: EventKind,
        *,
        turn_index: int | None = None,
        messages_released: int = 0,
        detail: str | None = None,
    ) -> None:
        self._events.append(
            EpisodeEvent(
                index=len(self._events),
                kind=kind,
                turn_index=turn_index,
                messages_released=messages_released,
                detail=detail,
            )
        )

    def observe(self, turn: ObservedTurn) -> None:
        self._observed.append(turn)

    @property
    def events(self) -> tuple[EpisodeEvent, ...]:
        return tuple(self._events)

    @property
    def observed(self) -> tuple[ObservedTurn, ...]:
        return tuple(self._observed)


def _authorize(candidate: EvalCandidate, script: ConversationScript, plan: EligibleEvalPlan) -> None:
    if script.source_verification_identity != plan.source_verification_identity:
        raise ConversationAuthorizationError(
            "eval.script",
            "was built from a different verified benchmark than the plan authorizes",
            actual=script.source_verification_identity,
            expected=plan.source_verification_identity,
            recovery=(
                "build scripts from the rows of the VerifiedEvalSource the contamination gate "
                "ran against, or gate again against the source being read"
            ),
        )
    try:
        eligibility = plan.candidate(candidate.alias)
    except KeyError as exc:
        raise ConversationAuthorizationError(
            f"candidates[{candidate.alias}]",
            "is not a candidate the contamination gate authorized",
            actual=candidate.alias,
            expected=f"one of {list(plan.candidate_aliases)}",
            recovery="drive the EvalCandidate contract that was passed to evaluate_contamination",
        ) from exc
    if candidate.canonical_model_identity != eligibility.canonical_model_identity:
        raise ConversationAuthorizationError(
            f"candidates[{candidate.alias}].model_identity",
            "changed after the contamination gate authorized the alias",
            actual=candidate.canonical_model_identity,
            expected=eligibility.canonical_model_identity,
            recovery=(
                "stop the run and re-run the contamination gate for these exact weights; "
                "an alias does not authorize a different model"
            ),
        )
    authorized = plan.evaluation_task_ids(candidate.alias)
    if script.task_id not in authorized:
        raise ConversationAuthorizationError(
            f"candidates[{candidate.alias}]",
            "was asked a task the contamination gate did not authorize for it",
            actual=script.task_id,
            expected=f"one of the {len(authorized)} task(s) the plan assigns this candidate",
            recovery=(
                "drive only plan.evaluation_task_ids(alias); a task excluded for exposure "
                "must not be answered by the model that helped write it"
            ),
        )


def _observed_from(
    turn: ScriptedTurn,
    request_hash: str,
    outcome: CandidateCallOutcome,
    match: TurnMatch | None,
) -> ObservedTurn:
    response: CandidateResponse | None = outcome.response
    return ObservedTurn(
        turn_index=turn.turn_index,
        request_hash=request_hash,
        call_status=outcome.status,
        response_hash=response.response_hash if response is not None else None,
        finish_reason=response.finish_reason if response is not None else None,
        assistant_content=response.assistant_content if response is not None else None,
        tool_calls=response.tool_calls if response is not None else (),
        advanced=bool(match and match.advanced),
        detail=match.detail if match is not None else f"candidate call ended as {outcome.status}",
        paired_call_indexes=match.paired_call_indexes if match and match.advanced else (),
    )


async def run_candidate_episode(
    *,
    candidate: EvalCandidate,
    limits: EvalLimits,
    client: NativeFunctionCallingClient,
    script: ConversationScript,
    plan: EligibleEvalPlan,
    gate: ContinuationGate,
) -> CandidateEpisode:
    """Replay one task against one candidate and return what happened.

    Returns rather than raises for every outcome the candidate can cause: a model
    that calls the wrong tool, returns unparseable arguments, or cannot be reached
    is a result to be recorded. Only a driver-side violation — an unauthorized
    task, a row that is not a replayable conversation — raises, because those are
    bugs in the run rather than facts about the model.
    """
    _authorize(candidate, script, plan)
    gate_policy_hash = getattr(gate, "scoring_policy_hash", None)
    if gate_policy_hash is not None and gate_policy_hash != plan.scoring_policy_hash:
        raise ConversationAuthorizationError(
            "eval.continuation_gate.scoring_policy",
            "is not the scoring policy the authorization plan was created under",
            actual=gate_policy_hash,
            expected=plan.scoring_policy_hash,
            recovery="construct the continuation gate from the scoring section of the gated eval config",
        )
    conversation = ModelFacingConversation(script)
    log = _EpisodeLog()
    log.event("seed", messages_released=len(conversation), detail="answer-free opening")
    # One time base for the whole episode: the client compares the same deadline
    # against the same clock, so a turn cannot be refused for a budget the driver
    # thinks is still open.
    deadline = time.monotonic() + limits.episode_timeout_s
    status: EpisodeStatus = "completed"
    detail = "the conversation reached the end of the trace"

    for turn in script.turns:
        if turn.turn_index >= limits.max_turns:
            status, detail = (
                "max_turns_exceeded",
                f"the trace needs {len(script.turns)} assistant turns; limits.max_turns is {limits.max_turns}",
            )
            log.event("terminal", turn_index=turn.turn_index, detail=detail)
            break
        conversation.audit()
        request = build_candidate_request(
            candidate,
            request_id=turn_request_id(candidate.alias, script.task_id, turn.turn_index),
            task_id=script.task_id,
            turn_index=turn.turn_index,
            messages=conversation.messages,
            tools=[dict(tool) for tool in script.tools],
        )
        if time.monotonic() >= deadline:
            status, detail = (
                "episode_timeout",
                f"limits.episode_timeout_s ({limits.episode_timeout_s}) elapsed before "
                f"assistant turn {turn.turn_index} was sent",
            )
            log.event("terminal", turn_index=turn.turn_index, detail=detail)
            break

        outcome = await client.complete(request, deadline=deadline)
        log.cache_hits += int(outcome.replayed)
        if outcome.status != "completed" or outcome.response is None:
            log.observe(_observed_from(turn, request.request_hash, outcome, None))
            status = _failure_status(outcome.status)
            detail = f"the candidate call ended as {outcome.status} at assistant turn {turn.turn_index}"
            log.event("terminal", turn_index=turn.turn_index, detail=detail)
            break
        response = outcome.response
        log.event("candidate_turn", turn_index=turn.turn_index, detail=f"finish_reason={response.finish_reason}")

        match = gate.evaluate(turn, response, script=script)
        call_ids = [call.id for call in response.tool_calls]
        unusable = match.advanced and (
            any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids)
        )
        if unusable:
            match = TurnMatch(
                advanced=False,
                detail=(
                    "the provider returned missing or duplicate tool-call ids, "
                    "so recorded results cannot be addressed unambiguously"
                ),
            )
        log.observe(_observed_from(turn, request.request_hash, outcome, match))
        if not match.advanced:
            status = "unusable_tool_call_ids" if unusable else "candidate_mismatch"
            detail = f"assistant turn {turn.turn_index}: {match.detail}"
            log.event("terminal", turn_index=turn.turn_index, detail=detail)
            break

        conversation.append_candidate_turn(response)
        if turn.expects_tool_calls:
            released = released_results(turn, response, match.paired_call_indexes)
            conversation.append_tool_results(released)
            log.released_tool_results += len(released)
            log.event("tool_results", turn_index=turn.turn_index, messages_released=len(released))
        if turn.releases_user_message is not None:
            conversation.append_user_turn(turn.releases_user_message)
            log.released_user_turns += 1
            log.event("user_turn", turn_index=turn.turn_index, messages_released=1)
        if turn.is_terminal:
            log.event("terminal", turn_index=turn.turn_index, detail=detail)

    return CandidateEpisode(
        candidate_alias=candidate.alias,
        canonical_model_identity=candidate.canonical_model_identity,
        task_id=script.task_id,
        plan_identity=plan.plan_identity,
        source_verification_identity=script.source_verification_identity,
        script_hash=script.script_hash,
        status=status,
        detail=detail,
        assistant_turns=len(log.observed),
        released_user_turns=log.released_user_turns,
        released_tool_results=log.released_tool_results,
        observed=log.observed,
        events=log.events,
        # An episode is a replay only if it cost nothing: one paid call makes the
        # whole episode a fresh observation of the candidate.
        replayed=bool(log.observed) and log.cache_hits == len(log.observed),
    )


__all__ = ["run_candidate_episode", "turn_request_id"]
