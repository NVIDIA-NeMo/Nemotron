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

"""Turn a published row into a script, and hold the line on what a prompt may see.

The projection is pure and total: it reads only the published row and its
conversation plan, so every candidate replays the identical episode and no new
artifact has to be trusted. The row already proves its own consistency — expected
calls in trace order, one wire call per expected call — so this module's job is to
recover *where the conversation pauses*: which assistant turn answers which user
request, and which recorded tool result answers which call.

:class:`ModelFacingConversation` is the firewall. It has no general ``append``:
the only things that can enter a candidate prompt are the answer-free seed, a turn
the candidate itself produced, a recorded result the driver decided to release,
and a scripted user request. Leaking the answer key is therefore not a mistake the
driver can make by writing the wrong line, only by adding a method here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import CandidateResponse
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedCall,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_errors import (
    ConversationAuthorizationError,
    ConversationLeakageError,
    ConversationScriptError,
    ConversationTransitionError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import VerifiedEvalSource
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
    ExportedMessage,
    thaw_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalConversationPlan,
    CanonicalExportProjection,
)


def _refuse(task_id: str, problem: str, *, expected: str, actual: Any = None) -> ConversationScriptError:
    return ConversationScriptError(
        f"task {task_id}",
        problem,
        actual=actual,
        expected=expected,
        recovery=(
            "re-publish the benchmark from a run whose benchmark replay produced "
            "the conversation, rather than editing the published parquet"
        ),
    )


def _split_seed(row: CanonicalExportRow) -> int:
    """Return the index just past the answer-free opening of the conversation."""
    cursor = 0
    while cursor < len(row.messages) and row.messages[cursor].role == "system":
        cursor += 1
    if cursor >= len(row.messages) or row.messages[cursor].role != "user":
        raise _refuse(
            row.task_id,
            "the conversation does not open with system prompts followed by a user request",
            expected="zero or more system messages, then the first user message",
            actual=[message.role for message in row.messages[: cursor + 1]],
        )
    return cursor + 1


def _scripted_calls(
    row: CanonicalExportRow,
    message: ExportedMessage,
    followers: Sequence[ExportedMessage],
    *,
    call_cursor: int,
) -> tuple[ScriptedCall, ...]:
    calls: list[ScriptedCall] = []
    if call_cursor + len(message.tool_calls) > len(row.expected_tool_calls):
        raise _refuse(
            row.task_id,
            "the messages carry more tool calls than expected_tool_calls declares",
            expected=f"{len(row.expected_tool_calls)} calls in total",
            actual=call_cursor + len(message.tool_calls),
        )
    for position, wire in enumerate(message.tool_calls):
        if position >= len(followers) or followers[position].role != "tool":
            raise _refuse(
                row.task_id,
                "an assistant tool-call message is not followed by one tool result per call",
                expected=f"{len(message.tool_calls)} consecutive tool messages",
                actual=[follower.role for follower in followers[: len(message.tool_calls)]],
            )
        result = followers[position]
        if result.tool_call_id != wire.id:
            raise _refuse(
                row.task_id,
                "a recorded tool result answers a different call than the one it follows",
                expected=f"tool_call_id {wire.id}",
                actual=result.tool_call_id,
            )
        expected = row.expected_tool_calls[call_cursor + position]
        calls.append(
            ScriptedCall(
                call_index=call_cursor + position,
                position_in_group=position,
                function_name=expected.function_name,
                arguments=expected.arguments,
                recorded_result=result.content,
            )
        )
    return tuple(calls)


def build_conversation_script(
    projection: CanonicalExportProjection,
    task_id: str,
    *,
    source: VerifiedEvalSource,
) -> ConversationScript:
    """Project one published row into the episode a driver can replay.

    The row and plan are selected from one :class:`CanonicalExportProjection`,
    whose source hash and complete task sequence must match ``source``. Taking the
    verified handle rather than a caller-supplied identity prevents a stale or
    foreign row from being stamped as if it came from the authorized publication.
    """
    artifact = source.evaluation_benchmark
    if (
        projection.source.content_hash != artifact.content_hash
        or projection.source.rows != artifact.rows
        or projection.task_ids != source.task_ids
    ):
        raise ConversationAuthorizationError(
            "eval.conversation_projection",
            "does not describe the complete benchmark the verified source authorizes",
            actual={
                "content_hash": projection.source.content_hash,
                "rows": projection.source.rows,
                "task_ids": list(projection.task_ids),
            },
            expected=(
                f"content_hash {artifact.content_hash}, {artifact.rows} rows, "
                "and the verified task sequence"
            ),
            recovery=(
                "project source.evaluation_benchmark with its expected content hash "
                "and source.task_ids, then build scripts from that projection"
            ),
        )
    try:
        row = projection.row(task_id)
        plan = projection.plan(task_id)
    except KeyError as exc:
        raise ConversationScriptError(
            f"task {task_id}",
            "is not present in the verified benchmark projection",
            actual=task_id,
            expected=f"one of the projection's {len(projection.rows)} task ids",
            recovery="iterate source.task_ids and select each row from the bound projection",
        ) from exc
    return _project_conversation_script(
        row,
        plan,
        source_verification_identity=source.verification_identity,
    )


def _project_conversation_script(
    row: CanonicalExportRow,
    plan: CanonicalConversationPlan,
    *,
    source_verification_identity: str,
) -> ConversationScript:
    """Recover the pauses of one row after its projection has been source-bound."""
    if plan.task_id != row.task_id:
        raise _refuse(
            row.task_id,
            "the conversation plan describes a different task",
            expected=f"a plan for {row.task_id}",
            actual=plan.task_id,
        )
    cursor = _split_seed(row)
    seed = row.messages[:cursor]
    turns: list[ScriptedTurn] = []
    user_turn_index = 0
    call_cursor = 0
    while cursor < len(row.messages):
        message = row.messages[cursor]
        if message.role == "user":
            if not turns:
                raise _refuse(
                    row.task_id,
                    "two user requests are separated by no assistant turn",
                    expected="every user request but the first follows an assistant turn",
                    actual=cursor,
                )
            turns[-1] = turns[-1].model_copy(update={"releases_user_message": message})
            user_turn_index += 1
            cursor += 1
            continue
        if message.role != "assistant":
            raise _refuse(
                row.task_id,
                "a tool result appears where an assistant turn or user request was expected",
                expected="assistant and user messages, with tool results attached to their calls",
                actual=cursor,
            )
        calls = _scripted_calls(row, message, row.messages[cursor + 1 :], call_cursor=call_cursor)
        turns.append(
            ScriptedTurn(
                turn_index=len(turns),
                user_turn_index=user_turn_index,
                call_group=row.expected_tool_calls[call_cursor].call_group if calls else None,
                calls=calls,
                expected_assistant_content=message.content if not calls else None,
            )
        )
        call_cursor += len(calls)
        cursor += 1 + len(calls)
    if not turns:
        raise _refuse(
            row.task_id,
            "the conversation has no assistant turn, so it asks the candidate nothing",
            expected="at least one assistant turn",
            actual=0,
        )
    if call_cursor != len(row.expected_tool_calls):
        raise _refuse(
            row.task_id,
            "the messages do not account for every expected call",
            expected=f"{len(row.expected_tool_calls)} calls carried by assistant messages",
            actual=call_cursor,
        )
    turns[-1] = turns[-1].model_copy(update={"is_terminal": True, "releases_user_message": None})
    _agree_with_plan(row, plan, turns)
    return ConversationScript(
        task_id=row.task_id,
        source_verification_identity=source_verification_identity,
        seed_messages=seed,
        tools=row.tools,
        turns=tuple(turns),
        user_turns=plan.user_turns,
        required_tools=row.required_tools,
        call_order=row.call_order,
        call_order_prefix=row.call_order_prefix,
    )


def _agree_with_plan(
    row: CanonicalExportRow,
    plan: CanonicalConversationPlan,
    turns: Sequence[ScriptedTurn],
) -> None:
    """Cross-check the walk against the independently derived conversation plan."""
    if len(turns) != plan.assistant_turns:
        raise _refuse(
            row.task_id,
            "the conversation holds a different number of assistant turns than its plan",
            expected=f"{plan.assistant_turns} assistant turns",
            actual=len(turns),
        )
    released = sum(1 for turn in turns if turn.releases_user_message is not None)
    if released + 1 != plan.user_turns:
        raise _refuse(
            row.task_id,
            "the conversation asks a different number of user requests than its plan",
            expected=f"{plan.user_turns} user turns",
            actual=released + 1,
        )
    planned = {group.turn_index: group for group in plan.groups}
    for turn in turns:
        group = planned.pop(turn.turn_index, None)
        if not turn.expects_tool_calls:
            if group is not None:
                raise _refuse(
                    row.task_id,
                    "the plan places a call group on an assistant turn that speaks in words",
                    expected=f"no call group at assistant turn {turn.turn_index}",
                    actual=group.call_group,
                )
            continue
        if group is None:
            raise _refuse(
                row.task_id,
                "an assistant turn issues calls the plan does not group",
                expected=f"a call group at assistant turn {turn.turn_index}",
                actual=None,
            )
        if group.user_turn_index != turn.user_turn_index:
            raise _refuse(
                row.task_id,
                "the plan and the messages disagree about which request a call group answers",
                expected=f"user turn {turn.user_turn_index}",
                actual=group.user_turn_index,
            )
        if [call.function_name for call in group.calls] != [call.function_name for call in turn.calls]:
            raise _refuse(
                row.task_id,
                "the plan's call group does not hold the calls the assistant turn issues",
                expected=[call.function_name for call in turn.calls],
                actual=[call.function_name for call in group.calls],
            )
    if planned:
        raise _refuse(
            row.task_id,
            "the plan groups calls onto assistant turns the conversation does not have",
            expected=f"{len(turns)} assistant turns",
            actual=sorted(planned),
        )


class ModelFacingConversation:
    """The growing prompt, restricted to what the candidate has earned.

    Nothing here is a benchmark row. The seed is the published opening, which is
    answer-free by construction; every later assistant message is the candidate's
    own output; every tool result is one the driver decided to release; and every
    user request is scripted text the trace already committed to.
    """

    __slots__ = ("_messages", "_provenance")

    def __init__(self, script: ConversationScript) -> None:
        self._messages: list[dict[str, Any]] = [_wire_message(message) for message in script.seed_messages]
        self._provenance: list[str] = ["seed"] * len(self._messages)

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(message) for message in self._messages)

    @property
    def provenance(self) -> tuple[str, ...]:
        return tuple(self._provenance)

    def audit(self) -> None:
        """Re-check, before every send, that nothing unearned is in the prompt.

        The append methods already make a leak unreachable, so this can only fire
        after a change to this class. That is exactly when it is worth having: a
        leak that reaches a provider is unrecoverable, because the run's numbers
        are then measurements of a model that was shown the answer.
        """
        required = {"assistant": "candidate", "tool": "recorded_result"}
        for index, (message, provenance) in enumerate(zip(self._messages, self._provenance, strict=True)):
            role = str(message.get("role"))
            expected = required.get(role)
            if expected is not None and provenance != expected:
                raise ConversationLeakageError(
                    f"conversation.messages[{index}]",
                    f"a {role} message in the prompt did not come from the candidate or from a released result",
                    actual=provenance,
                    expected=expected,
                    recovery=(
                        "add material to a prompt only through append_candidate_turn, "
                        "append_tool_results, or append_user_turn"
                    ),
                )
            if provenance == "seed" and role not in {"system", "user"}:
                raise ConversationLeakageError(
                    f"conversation.messages[{index}]",
                    "the answer-free opening carries a message that is not a prompt or a request",
                    actual=role,
                    expected="system or user",
                    recovery="build the script with build_conversation_script, which splits the seed itself",
                )

    def __len__(self) -> int:
        return len(self._messages)

    def append_candidate_turn(self, response: CandidateResponse) -> None:
        message: dict[str, Any] = {"role": "assistant"}
        if response.tool_calls:
            calls: list[dict[str, Any]] = []
            for call in response.tool_calls:
                if (
                    not call.id
                    or call.type != "function"
                    or not isinstance(call.raw_arguments, str)
                    or not call.function_name
                ):
                    raise ConversationTransitionError(
                        "conversation.assistant_turn",
                        "a tool call cannot be echoed back to the provider",
                        actual={
                            "id": call.id,
                            "type": call.type,
                            "function_name": call.function_name,
                        },
                        expected=(
                            "a unique id, type 'function', a function name, "
                            "and the argument string the provider returned"
                        ),
                        recovery="end the episode instead of repairing the model's own output",
                    )
                calls.append(
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {"name": call.function_name, "arguments": call.raw_arguments},
                    }
                )
            message["tool_calls"] = calls
            if response.assistant_content is not None:
                message["content"] = thaw_json(response.assistant_content)
        elif response.assistant_content is not None:
            message["content"] = thaw_json(response.assistant_content)
        else:
            raise ConversationTransitionError(
                "conversation.assistant_turn",
                "an empty assistant turn cannot be added to a prompt",
                actual=None,
                expected="content, tool calls, or both",
                recovery="end the episode instead of sending a turn the model did not make",
            )
        self._messages.append(message)
        self._provenance.append("candidate")

    def append_tool_results(self, released: Sequence[tuple[str, str]]) -> None:
        for tool_call_id, content in released:
            if not tool_call_id:
                raise ConversationTransitionError(
                    "conversation.tool_result",
                    "a recorded result has no call of the candidate's to answer",
                    actual=tool_call_id,
                    expected="the id of a tool call the candidate just made",
                    recovery="end the episode; a result addressed to nothing is not an observation",
                )
            self._messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
            self._provenance.append("recorded_result")

    def append_user_turn(self, message: ExportedMessage) -> None:
        if message.role != "user":
            raise ConversationTransitionError(
                "conversation.user_turn",
                "only a user request may be injected to continue a conversation",
                actual=message.role,
                expected="role 'user'",
                recovery="release the scripted user message the turn carries, not another message",
            )
        self._messages.append(_wire_message(message))
        self._provenance.append("scripted_user")


def _wire_message(message: ExportedMessage) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return wire


def released_results(
    turn: ScriptedTurn,
    response: CandidateResponse,
    paired: Sequence[int],
) -> tuple[tuple[str, str], ...]:
    """Address each recorded result to the call of the candidate's it answers."""
    by_index: Mapping[int, ScriptedCall] = {call.call_index: call for call in turn.calls}
    if len(paired) != len(response.tool_calls):
        raise ConversationTransitionError(
            "conversation.tool_result",
            "the pairing does not cover every call the candidate made",
            actual=len(paired),
            expected=f"{len(response.tool_calls)} paired calls",
            recovery="end the episode; an unpaired call has no recorded result to answer it",
        )
    return tuple(
        (response.tool_calls[position].id or "", by_index[call_index].recorded_result)
        for position, call_index in enumerate(paired)
    )


__all__ = [
    "ModelFacingConversation",
    "build_conversation_script",
    "released_results",
]
