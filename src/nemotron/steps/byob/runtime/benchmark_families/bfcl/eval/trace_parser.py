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

"""Flatten a recorded episode into the shape a scorer can read, and nothing more.

The client parsed the provider's bytes once, under strict JSON, and the driver
recorded the result. Parsing them again here would give a second opinion about
what the model said, and the two could disagree — so this module never touches a
raw response. It reads :class:`CandidateEpisode`, lines it up against the
:class:`ConversationScript` that produced it, and hands the scorer a flat trace
with stable coordinates: turn index, position within the turn, and the argument
status the client already determined.

Nothing is repaired and nothing is filled in. A call with unparseable arguments
stays a call with unparseable arguments; a turn the episode never sent is listed
as never sent rather than invented as an empty turn. What the parser does insist
on is that the episode and the script are two halves of the same replay, because
a score derived from mismatched halves would silently grade the wrong task.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    ArgumentStatus,
    CallStatus,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    CandidateEpisode,
    ConversationScript,
    ObservedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    TRACE_NON_CANDIDATE_STOPS as NON_CANDIDATE_STOPS,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_errors import (
    TraceEvidenceError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    FrozenDict,
    NonNegativeInt,
    freeze_json,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class ParsedCall(_Frozen):
    """One recorded tool call, addressed by where it appeared in the trace."""

    turn_index: NonNegativeInt
    position_in_turn: NonNegativeInt
    # The index the provider itself gave the call, kept because a provider that
    # numbers its calls differently from their order is a fact about the response.
    provider_index: NonNegativeInt
    id: StrictStr | None = None
    type: StrictStr | None = None
    function_name: StrictStr | None = None
    arguments_status: ArgumentStatus
    # Named as the client named it, so this satisfies the same comparison
    # protocol the driver's freshly parsed calls do.
    parsed_arguments: FrozenDict | None = None


class ParsedTurn(_Frozen):
    """One assistant turn the episode actually asked for, as recorded."""

    turn_index: NonNegativeInt
    kind: Literal["tool_calls", "text"]
    call_status: CallStatus
    advanced: StrictBool
    finish_reason: StrictStr | None = None
    assistant_content: Any = None
    calls: tuple[ParsedCall, ...] = ()
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_content(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            value["assistant_content"] = freeze_json(value.get("assistant_content"))
        return value

    @property
    def answered(self) -> bool:
        """Whether the provider returned a usable envelope for this turn."""
        return self.call_status == "completed"


class ParsedTrace(_Frozen):
    """One episode, flattened and bound to the script it replayed."""

    task_id: StrictStr
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    plan_identity: ContentHash
    source_verification_identity: ContentHash
    script_hash: ContentHash
    episode_hash: ContentHash
    # Shared normalized views include both trace-only and executable terminal
    # statuses. Each source contract validates its own status before projection.
    status: StrictStr
    non_candidate_stop: StrictBool
    scripted_turns: NonNegativeInt
    turns: tuple[ParsedTurn, ...] = ()
    # Scripted turns the episode never reached. These carry gold calls the
    # candidate was never asked for, which is why coverage is measured against the
    # script rather than against the turns that happened to be sent.
    unsent_turn_indexes: tuple[NonNegativeInt, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> ParsedTrace:
        if [turn.turn_index for turn in self.turns] != list(range(len(self.turns))):
            raise ValueError("parsed turns must be contiguous and zero-based")
        if list(self.unsent_turn_indexes) != list(range(len(self.turns), self.scripted_turns)):
            raise ValueError("the unsent turns are exactly the scripted turns after the last one sent")
        return self

    @property
    def reached_the_end(self) -> bool:
        return self.status == "completed" and not self.unsent_turn_indexes

    @property
    def observed_calls(self) -> int:
        return sum(len(turn.calls) for turn in self.turns)


def parse_observed_trace(episode: CandidateEpisode, script: ConversationScript) -> ParsedTrace:
    """Line a recorded episode up against its script, refusing mismatched halves."""
    _same_replay(episode, script)
    turns = tuple(_parse_turn(observed, script) for observed in episode.observed)
    return ParsedTrace(
        task_id=episode.task_id,
        candidate_alias=episode.candidate_alias,
        canonical_model_identity=episode.canonical_model_identity,
        plan_identity=episode.plan_identity,
        source_verification_identity=episode.source_verification_identity,
        script_hash=episode.script_hash,
        episode_hash=episode.episode_hash,
        status=episode.status,
        non_candidate_stop=episode.status in NON_CANDIDATE_STOPS,
        scripted_turns=len(script.turns),
        turns=turns,
        unsent_turn_indexes=tuple(range(len(turns), len(script.turns))),
    )


def _same_replay(episode: CandidateEpisode, script: ConversationScript) -> None:
    if episode.task_id != script.task_id:
        raise TraceEvidenceError(
            "eval.episode.task_id",
            "was recorded for a different task than the script describes",
            actual=episode.task_id,
            expected=script.task_id,
            recovery="score each episode against the script the driver was given for that task",
        )
    if episode.script_hash != script.script_hash:
        raise TraceEvidenceError(
            "eval.episode.script_hash",
            "does not identify this script, so the episode answered a different conversation",
            actual=episode.script_hash,
            expected=script.script_hash,
            recovery=(
                "re-run the episode against the current script, or score it against the script "
                "it was driven with; a re-projected row is not the same question"
            ),
        )
    if episode.source_verification_identity != script.source_verification_identity:
        raise TraceEvidenceError(
            "eval.episode.source_verification_identity",
            "names a different verified benchmark than the script came from",
            actual=episode.source_verification_identity,
            expected=script.source_verification_identity,
            recovery=(
                "score the episode against the source-bound script it was driven with; "
                "do not restamp recorded evidence with another benchmark identity"
            ),
        )
    if len(episode.observed) > len(script.turns):
        raise TraceEvidenceError(
            "eval.episode.observed",
            "holds more assistant turns than the script has",
            actual=len(episode.observed),
            expected=f"at most {len(script.turns)} turns",
            recovery="re-drive the task; an episode cannot answer a turn the trace does not ask",
        )
    if episode.status == "completed" and len(episode.observed) != len(script.turns):
        raise TraceEvidenceError(
            "eval.episode.status",
            "claims completion before every scripted turn was observed",
            actual=len(episode.observed),
            expected=f"exactly {len(script.turns)} observed turns",
            recovery=(
                "retain the original incomplete terminal status, or re-drive the "
                "episode to the end of the script"
            ),
        )


def _parse_turn(observed: ObservedTurn, script: ConversationScript) -> ParsedTurn:
    scripted = script.turn(observed.turn_index)
    return ParsedTurn(
        turn_index=observed.turn_index,
        kind="tool_calls" if scripted.expects_tool_calls else "text",
        call_status=observed.call_status,
        finish_reason=observed.finish_reason,
        advanced=observed.advanced,
        assistant_content=observed.assistant_content,
        calls=tuple(
            ParsedCall(
                turn_index=observed.turn_index,
                position_in_turn=position,
                provider_index=call.index,
                id=call.id,
                type=call.type,
                function_name=call.function_name,
                arguments_status=call.arguments_status,
                parsed_arguments=call.parsed_arguments,
            )
            for position, call in enumerate(observed.tool_calls)
        ),
        detail=observed.detail,
    )


__all__ = [
    "NON_CANDIDATE_STOPS",
    "ParsedCall",
    "ParsedTrace",
    "ParsedTurn",
    "parse_observed_trace",
]
