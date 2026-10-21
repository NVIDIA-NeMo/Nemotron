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

"""What one evaluation episode asks, and what the candidate did with it.

A :class:`ConversationScript` is the deterministic half: the answer-free opening,
the tools, and — turn by turn — what the benchmark recorded that the model would
have to produce for the conversation to reach its next user request. A
:class:`CandidateEpisode` is the observed half: what the candidate actually
returned at each turn, whether the driver was able to continue, and why it
stopped.

Neither side scores anything. The script carries gold calls because a multi-turn
conversation cannot advance without knowing which recorded tool result answers
which call. Releasing that result is a transport decision; the scoring component,
not this contract, derives numbers from the comparison.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CallStatus,
    CandidateToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CallOrderPolicy,
    ContentHash,
    ExportedMessage,
    FrozenDict,
    NonNegativeInt,
    PositiveInt,
    freeze_json,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

CONVERSATION_CONTRACT_VERSION: Final = "1.0"

EpisodeStatus = Literal[
    "completed",
    "candidate_mismatch",
    "malformed_response",
    "candidate_call_failed",
    "unusable_tool_call_ids",
    "max_turns_exceeded",
    "episode_timeout",
]
EventKind = Literal["seed", "candidate_turn", "tool_results", "user_turn", "terminal"]


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class ScriptedCall(_Frozen):
    """One gold call and the tool result the benchmark recorded for it."""

    call_index: NonNegativeInt
    position_in_group: NonNegativeInt
    function_name: StrictStr
    arguments: FrozenDict
    # Canonical JSON text, exactly as the published tool message carries it. It is
    # released to the model only after the driver has paired it with a call the
    # candidate actually made.
    recorded_result: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_arguments(cls, value: Any) -> Any:
        if isinstance(value, dict) and "arguments" in value:
            value = dict(value)
            validate_json_value(value["arguments"], label="scripted call arguments")
            value["arguments"] = freeze_json(value["arguments"])
        return value

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "position_in_group": self.position_in_group,
            "function_name": self.function_name,
            "arguments": thaw_json(self.arguments),
            "recorded_result": self.recorded_result,
        }


class ScriptedTurn(_Frozen):
    """One assistant turn the benchmark's conversation reaches.

    ``releases_user_message`` is the next user request, and it hangs off the turn
    that must succeed before the user would have said it. A turn that ends the
    conversation releases nothing: there is no further request to answer.
    """

    turn_index: NonNegativeInt
    user_turn_index: NonNegativeInt
    call_group: NonNegativeInt | None = None
    calls: tuple[ScriptedCall, ...] = ()
    # The text benchmark replay recorded for a text-only assistant turn. Intermediate
    # turns must reproduce it before the next scripted observation is released;
    # terminal text remains a scoring observation rather than a continuation
    # condition.
    expected_assistant_content: StrictStr | None = None
    releases_user_message: ExportedMessage | None = None
    is_terminal: StrictBool = False

    @model_validator(mode="after")
    def _coherent(self) -> ScriptedTurn:
        if bool(self.calls) != (self.call_group is not None):
            raise ValueError("a turn carries a call group exactly when it issues calls")
        if bool(self.calls) == (self.expected_assistant_content is not None):
            raise ValueError("a turn expects exactly one of tool calls or assistant text")
        if [call.position_in_group for call in self.calls] != list(range(len(self.calls))):
            raise ValueError("a turn's calls must occupy positions 0..n-1 in order")
        if self.releases_user_message is not None:
            if self.releases_user_message.role != "user":
                raise ValueError("only a user message may be released to continue a conversation")
            if self.is_terminal:
                raise ValueError("a terminal turn has no further user request to release")
        return self

    @property
    def expects_tool_calls(self) -> bool:
        return bool(self.calls)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "user_turn_index": self.user_turn_index,
            "call_group": self.call_group,
            "calls": [call.semantic_payload() for call in self.calls],
            "expected_assistant_content": self.expected_assistant_content,
            "releases_user_message": (
                self.releases_user_message.model_dump(mode="json")
                if self.releases_user_message is not None
                else None
            ),
            "is_terminal": self.is_terminal,
        }


class ConversationScript(_Frozen):
    """One task's episode, projected into what may be sent and what must follow."""

    schema_version: Literal["1.0"] = CONVERSATION_CONTRACT_VERSION
    task_id: StrictStr
    # Which verified benchmark this row came out of. Carrying it here is what lets
    # a driver refuse a script built from a different publication than the one the
    # contamination gate authorized, without re-reading the parquet.
    source_verification_identity: ContentHash
    # Leading system messages and the first user request, and nothing else: the
    # rest of the published conversation is the answer key until the candidate
    # earns it turn by turn.
    seed_messages: tuple[ExportedMessage, ...]
    tools: tuple[FrozenDict, ...]
    turns: tuple[ScriptedTurn, ...]
    user_turns: PositiveInt
    required_tools: tuple[StrictStr, ...]
    call_order: CallOrderPolicy
    call_order_prefix: PositiveInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _freeze_tools(cls, value: Any) -> Any:
        if isinstance(value, dict) and "tools" in value:
            value = dict(value)
            validate_json_value(list(value["tools"]), label="conversation tools")
            value["tools"] = tuple(freeze_json(tool) for tool in value["tools"])
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ConversationScript:
        roles = [message.role for message in self.seed_messages]
        if not roles or roles[-1] != "user" or any(role != "system" for role in roles[:-1]):
            raise ValueError("seed_messages must be the leading system messages and the first user turn")
        if not self.turns:
            raise ValueError("an episode with no assistant turn asks the candidate nothing")
        if [turn.turn_index for turn in self.turns] != list(range(len(self.turns))):
            raise ValueError("assistant turns must be contiguous and zero-based")
        user_indexes = [turn.user_turn_index for turn in self.turns]
        if user_indexes != sorted(user_indexes) or user_indexes[0] != 0:
            raise ValueError("assistant turns must answer user requests in order, starting at the first")
        if max(user_indexes) >= self.user_turns:
            raise ValueError("an assistant turn answers a user request the conversation never makes")
        terminal = [turn.turn_index for turn in self.turns if turn.is_terminal]
        if terminal != [self.turns[-1].turn_index]:
            raise ValueError("exactly the last assistant turn ends the conversation")
        released = sum(1 for turn in self.turns if turn.releases_user_message is not None)
        if released != self.user_turns - 1:
            raise ValueError("every user request after the first must be released by exactly one turn")
        indexes = [call.call_index for turn in self.turns for call in turn.calls]
        if indexes != list(range(len(indexes))):
            raise ValueError("scripted calls must cover the row's expected calls in trace order")
        if (self.call_order == "prefix") != (self.call_order_prefix is not None):
            raise ValueError("a call_order_prefix belongs to, and only to, the prefix policy")
        if self.call_order_prefix is not None and self.call_order_prefix > len(self.required_tools):
            raise ValueError("call_order_prefix cannot exceed the required tool sequence")
        return self

    @property
    def expected_call_count(self) -> int:
        return sum(len(turn.calls) for turn in self.turns)

    def turn(self, turn_index: int) -> ScriptedTurn:
        return self.turns[turn_index]

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "source_verification_identity": self.source_verification_identity,
            "seed_messages": [message.model_dump(mode="json") for message in self.seed_messages],
            "tools": [thaw_json(tool) for tool in self.tools],
            "turns": [turn.semantic_payload() for turn in self.turns],
            "user_turns": self.user_turns,
            "required_tools": list(self.required_tools),
            "call_order": self.call_order,
            "call_order_prefix": self.call_order_prefix,
        }

    @property
    def script_hash(self) -> str:
        """One hash for "this task, replayed this way"."""
        return _sha256_json(self.semantic_payload())


class ObservedTurn(_Frozen):
    """What the candidate returned for one scripted turn, and whether it advanced."""

    turn_index: NonNegativeInt
    request_hash: ContentHash
    call_status: CallStatus
    response_hash: ContentHash | None = None
    finish_reason: StrictStr | None = None
    assistant_content: Any = None
    tool_calls: tuple[CandidateToolCall, ...] = ()
    advanced: StrictBool
    detail: StrictStr
    # Which scripted call each predicted call was paired with, in predicted order.
    # Populated only when the turn advanced, because an unpaired call has no
    # recorded result the driver may release for it.
    paired_call_indexes: tuple[NonNegativeInt, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _freeze_content(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            value["assistant_content"] = freeze_json(value.get("assistant_content"))
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ObservedTurn:
        if self.advanced and self.call_status != "completed":
            raise ValueError("a turn that did not complete its call cannot have advanced the episode")
        if self.paired_call_indexes and not self.advanced:
            raise ValueError("calls are paired with recorded results only on a turn that advanced")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "request_hash": self.request_hash,
            "call_status": self.call_status,
            "response_hash": self.response_hash,
            "finish_reason": self.finish_reason,
            "assistant_content": thaw_json(self.assistant_content),
            "tool_calls": [call.as_document() for call in self.tool_calls],
            "advanced": self.advanced,
            "detail": self.detail,
            "paired_call_indexes": list(self.paired_call_indexes),
        }


class EpisodeEvent(_Frozen):
    """One thing the driver did, in the order it did it."""

    index: NonNegativeInt
    kind: EventKind
    turn_index: NonNegativeInt | None = None
    messages_released: NonNegativeInt = 0
    detail: StrictStr | None = None

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "turn_index": self.turn_index,
            "messages_released": self.messages_released,
            "detail": self.detail,
        }


class CandidateEpisode(_Frozen):
    """One candidate's whole run of one task, as evidence rather than as a score."""

    schema_version: Literal["1.0"] = CONVERSATION_CONTRACT_VERSION
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    task_id: StrictStr
    plan_identity: ContentHash
    source_verification_identity: ContentHash
    script_hash: ContentHash
    status: EpisodeStatus
    detail: StrictStr
    assistant_turns: NonNegativeInt = Field(ge=0)
    released_user_turns: NonNegativeInt = 0
    released_tool_results: NonNegativeInt = 0
    observed: tuple[ObservedTurn, ...]
    events: tuple[EpisodeEvent, ...] = ()
    replayed: StrictBool = False

    @model_validator(mode="after")
    def _coherent(self) -> CandidateEpisode:
        if not self.observed and self.status != "episode_timeout":
            raise ValueError("only an episode budget spent before the first request has no observed turn")
        if [turn.turn_index for turn in self.observed] != list(range(len(self.observed))):
            raise ValueError("observed turns must be contiguous and zero-based")
        if self.assistant_turns != len(self.observed):
            raise ValueError("assistant_turns counts exactly the turns the candidate was asked")
        advanced = [turn.advanced for turn in self.observed]
        if self.status == "completed":
            if not advanced or not all(advanced):
                raise ValueError("a completed episode advanced through every turn it took")
        elif advanced and advanced[-1] and self.status not in {
            "max_turns_exceeded",
            "episode_timeout",
        }:
            raise ValueError("an episode that stopped for another reason did not advance its last turn")
        if [event.index for event in self.events] != list(range(len(self.events))):
            raise ValueError("episode events must be contiguous and zero-based")
        return self

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def semantic_payload(self) -> dict[str, Any]:
        """What happened, with nothing that depends on when or where it ran."""
        return {
            "schema_version": self.schema_version,
            "candidate_alias": self.candidate_alias,
            "canonical_model_identity": self.canonical_model_identity,
            "task_id": self.task_id,
            "plan_identity": self.plan_identity,
            "source_verification_identity": self.source_verification_identity,
            "script_hash": self.script_hash,
            "status": self.status,
            "detail": self.detail,
            "assistant_turns": self.assistant_turns,
            "released_user_turns": self.released_user_turns,
            "released_tool_results": self.released_tool_results,
            "observed": [turn.semantic_payload() for turn in self.observed],
            "events": [event.semantic_payload() for event in self.events],
        }

    @property
    def episode_hash(self) -> str:
        """One hash for "this candidate did this to this task".

        ``replayed`` is deliberately outside it: an episode reconstructed from the
        candidate I/O cache is the same observation as the one that paid for it,
        and a hash that disagreed would make replay untestable.
        """
        return _sha256_json(self.semantic_payload())

    def as_document(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "episode_hash": self.episode_hash,
            "replayed": self.replayed,
        }


__all__ = [
    "CONVERSATION_CONTRACT_VERSION",
    "CandidateEpisode",
    "ConversationScript",
    "EpisodeEvent",
    "EpisodeStatus",
    "EventKind",
    "ObservedTurn",
    "ScriptedCall",
    "ScriptedTurn",
]
