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

"""Whether one assistant turn earns the tool results the benchmark recorded.

A replayed conversation has one recorded result per gold call, so continuing past
an assistant turn requires deciding *which gold call each predicted call is*. That
decision is the driver's, and it is transport rather than scoring: it selects the
observation the candidate sees next. The scoring component re-derives its verdict
from the recorded episode rather than trusting this gate's continuation decision.

Agreement between the two is structural rather than reviewed: every comparison
this gate makes comes from :mod:`call_comparison`, which the scorer reads as well.
What remains here is only the continuation question — is this turn's group
answerable at all, and does it satisfy the row's ordering policy — so a gate that
was stricter than the scorer would have to be written as a difference in this
file rather than as a drifting reimplementation.

The gate is deliberately an injected :class:`ContinuationGate` so a runtime can
substitute one, but the pinned publication semantics live here and are read from
:class:`EvalScoringConfig` rather than hard-coded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_comparison import (
    compare_group_size,
    compare_text_turn,
    compare_turn_order,
    finish_reason_problem,
    pair_turn_calls,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateResponse,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalScoringConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import NonNegativeInt


class TurnMatch(BaseModel):
    """Whether the turn advances, and if so which gold call answers which call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advanced: StrictBool
    detail: StrictStr
    paired_call_indexes: tuple[NonNegativeInt, ...] = ()


@runtime_checkable
class ContinuationGate(Protocol):
    """Decides whether a candidate turn may see the next recorded observation."""

    def evaluate(
        self,
        turn: ScriptedTurn,
        response: CandidateResponse,
        *,
        script: ConversationScript,
    ) -> TurnMatch: ...


class CanonicalCallMatchGate:
    """The pinned publication comparison, used only to decide continuation.

    Ordering within one assistant turn follows ``call_order``: a turn's calls are
    issued at once, so ``any`` accepts a permutation of the gold group, ``strict``
    requires every gold position, and ``prefix`` requires only the configured
    global prefix before matching the remainder as a set. Ordering *across* turns
    is not negotiable here — the driver walks the scripted turns in order, so a
    candidate that defers a group to a later turn ends the episode. That is a real
    restriction of trace replay, not a scoring rule: the recorded result for a
    group is only meaningful at the point the trace reached it.
    """

    def __init__(self, scoring: EvalScoringConfig) -> None:
        self._scoring = scoring

    @property
    def scoring_policy_hash(self) -> str:
        return self._scoring.scoring_policy_hash

    def evaluate(
        self,
        turn: ScriptedTurn,
        response: CandidateResponse,
        *,
        script: ConversationScript,
    ) -> TurnMatch:
        incomplete = finish_reason_problem(response.finish_reason)
        if incomplete is not None:
            return TurnMatch(advanced=False, detail=incomplete)
        predicted = response.tool_calls
        if not turn.expects_tool_calls:
            text = compare_text_turn(
                turn,
                response.assistant_content,
                [call.function_name for call in predicted],
                scoring=self._scoring,
            )
            return TurnMatch(advanced=text.matched, detail=text.detail)

        if not predicted:
            expected = ", ".join(call.function_name for call in turn.calls)
            return TurnMatch(
                advanced=False,
                detail=f"the candidate issued no tool call; the trace calls {expected}",
            )
        oversized = compare_group_size(len(turn.calls), len(predicted))
        if oversized is not None:
            return TurnMatch(advanced=False, detail=oversized)

        pairing = pair_turn_calls(
            turn.calls,
            predicted,
            tools=script.tools,
            scoring=self._scoring,
            scope="set",
        )
        if not pairing.matched:
            return TurnMatch(advanced=False, detail=pairing.detail)

        detail = pairing.detail
        names_so_far: list[str | None] = [
            call.function_name for prior in script.turns[: turn.turn_index] for call in prior.calls
        ]
        names_so_far.extend(call.function_name for call in predicted)
        problem = compare_turn_order(
            turn,
            predicted,
            script=script,
            scoring=self._scoring,
            pairing=pairing,
            names_so_far=names_so_far,
        )
        if problem is not None:
            return TurnMatch(advanced=False, detail=problem)
        if self._scoring.respect_call_order and script.call_order == "prefix":
            detail = "calls matched as a set and required-tool first appearances respect the prefix"
        return TurnMatch(advanced=True, detail=detail, paired_call_indexes=pairing.full_pairing)


__all__ = [
    "CanonicalCallMatchGate",
    "ContinuationGate",
    "TurnMatch",
]
