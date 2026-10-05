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

"""Immutable evidence produced by one candidate's live oracle episode.

Executable evaluation interleaves candidate turns with real tool execution.  Its
evidence therefore cannot reuse :class:`CandidateEpisode`: that trace-only
contract records releases of results already published in the benchmark, while
this contract records calls made against a verified live oracle.

The models here score nothing and perform no I/O.  They preserve the candidate
turns, one normalized outcome for every proposed tool call, the final state
identity, and assertion observations.  Two boundaries are kept explicit because
collapsing either one would let evidence assert more than it observed: obtaining
a tool result is separate from admitting it to the candidate prompt, and an
oracle return that does not conform to the tool contract is recorded as such
rather than stored as if it had conformed.

Stable reason codes carry semantics; human-readable diagnostics remain in
documents but are excluded from ``episode_hash`` so rewording an error does not
create new evidence.  Only each model's own ``detail`` is excluded, so a key an
oracle happens to name ``detail`` stays part of the evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    ArgumentStatus,
    CallStatus,
    CandidateToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    FrozenDict,
    NonNegativeInt,
    freeze_json,
    json_equal,
    json_type_tag,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

EXECUTABLE_CONTRACT_VERSION: Final = "1.2"

ToolExecutionStatus = Literal[
    "not_executed",
    "completed",
    "business_rejection",
    "tool_error",
    "malformed_result",
    "timeout",
    "infrastructure_error",
    "unknown_commit_state",
]
MalformedResultType = Literal["null", "bool", "int", "float", "str", "array"]
StateCommitStatus = Literal[
    "not_started",
    "not_applicable",
    "committed",
    "not_committed",
    "unknown",
]
AssertionCategory = Literal["state", "path", "result", "final_answer", "unclassified"]
AssertionStatus = Literal["passed", "failed", "not_applicable", "infrastructure_error"]
DependencyResolutionStatus = Literal[
    "resolved",
    "producer_missing",
    "producer_ambiguous",
    "result_unavailable",
    "result_path_missing",
    "result_type_mismatch",
    "consumer_schema_invalid",
]
ExecutableEpisodeStatus = Literal[
    "completed",
    "candidate_mismatch",
    "malformed_response",
    "candidate_call_failed",
    "unusable_tool_call_ids",
    "confirmation_not_earned",
    "max_turns_exceeded",
    "episode_timeout",
    "oracle_reset_failed",
    "oracle_call_failed",
    "oracle_timeout",
    "oracle_result_malformed",
    "oracle_state_failed",
    "oracle_session_failed",
    "unknown_commit_state",
    "dependency_resolution_failed",
    "assertion_infrastructure_failed",
]
ExecutableEventKind = Literal[
    "seed",
    "oracle_reset",
    "candidate_turn",
    "tool_execution",
    "tool_results",
    "user_turn",
    "state_snapshot",
    "assertions",
    "terminal",
]

_RESULT_STATUSES: Final = frozenset({"completed", "business_rejection"})
_ORACLE_CALL_FAILURE_STATUSES: Final = frozenset({"tool_error", "infrastructure_error"})
_UNKNOWN_COMMIT_STATUSES: Final = frozenset(
    {
        "completed",
        "business_rejection",
        "tool_error",
        "malformed_result",
        "timeout",
        "infrastructure_error",
        "unknown_commit_state",
    }
)
_ATTEMPTED_STATUSES: Final = frozenset(
    {
        "completed",
        "business_rejection",
        "tool_error",
        "malformed_result",
        "timeout",
        "infrastructure_error",
        "unknown_commit_state",
    }
)
_TURN_SCOPED_EVENT_KINDS: Final = frozenset(
    {"candidate_turn", "tool_execution", "tool_results", "user_turn"}
)
# A provider envelope the client could parse is the only source of assistant
# content, a finish reason, or tool calls. Every other call status means the
# turn produced no usable envelope to read those from.
_ENVELOPE_STATUSES: Final = frozenset({"completed", "malformed_response"})


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class ExecutedToolCall(_Frozen):
    """One candidate tool call and the live oracle outcome assigned to it.

    A ``not_executed`` record is still required for a malformed, undeclared, or
    schema-invalid proposed call.  Consequently every provider tool call has one
    outcome and no invalid call silently disappears from executable evidence.

    Producing a result and admitting it to the candidate prompt are separate
    facts.  ``released_to_model`` records the second one, so a result the driver
    obtained but never released — the batch aborted after it, or the episode ran
    out of budget — stays in the evidence without claiming the candidate saw it.
    """

    execution_index: NonNegativeInt
    turn_index: NonNegativeInt
    position_in_turn: NonNegativeInt
    provider_call_index: NonNegativeInt
    call_id: StrictStr | None = None
    type: StrictStr | None = None
    function_name: StrictStr | None = None
    arguments_status: ArgumentStatus
    parsed_arguments: FrozenDict | None = None
    schema_valid: StrictBool | None = None
    schema_failures: tuple[FrozenDict, ...] = ()
    status: ToolExecutionStatus
    state_commit: StateCommitStatus
    state_before_hash: ContentHash | None = None
    state_after_hash: ContentHash | None = None
    result: FrozenDict | None = None
    result_hash: ContentHash | None = None
    released_to_model: StrictBool = False
    # A JSON value the oracle returned outside the object shape the tool contract
    # requires. It is kept separately so it cannot pose as a conforming result.
    # ``None`` is itself valid malformed evidence, so the before-validator also
    # requires the input key to be present for a malformed-result outcome.
    malformed_result: Any = None
    malformed_result_type: MalformedResultType | None = None
    malformed_result_hash: ContentHash | None = None
    reason_code: StrictStr
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_json_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        arguments = value.get("parsed_arguments")
        if arguments is not None:
            validate_json_value(arguments, label="executable call arguments")
            value["parsed_arguments"] = freeze_json(arguments)
        failures = list(value.get("schema_failures") or ())
        validate_json_value(failures, label="executable schema failures")
        value["schema_failures"] = tuple(freeze_json(failure) for failure in failures)
        result = value.get("result")
        if result is not None:
            if not isinstance(result, Mapping):
                raise ValueError(
                    "an oracle tool result must be a JSON object; record a non-object "
                    "return as a malformed_result outcome instead"
                )
            validate_json_value(result, label="oracle tool result")
            value["result"] = freeze_json(result)
        if value.get("status") == "malformed_result":
            if "malformed_result" not in value:
                raise ValueError("a malformed-result outcome preserves the JSON value the oracle returned")
            malformed = value.get("malformed_result")
            validate_json_value(malformed, label="malformed oracle result")
            if isinstance(malformed, Mapping):
                raise ValueError("an object-shaped oracle return belongs in result, not malformed_result")
            value["malformed_result"] = freeze_json(malformed)
        elif "malformed_result" in value and value.get("malformed_result") is not None:
            raise ValueError("malformed_result evidence belongs only to a malformed-result outcome")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ExecutedToolCall:
        if not self.reason_code.strip():
            raise ValueError("a tool outcome carries a stable reason code")
        if (self.arguments_status == "valid_object") != (self.parsed_arguments is not None):
            raise ValueError("exactly valid object arguments carry a parsed argument object")
        if self.schema_valid is False and not self.schema_failures:
            raise ValueError("a schema-invalid call records the constraints it violated")
        if self.schema_valid is not False and self.schema_failures:
            raise ValueError("schema failures belong only to a schema-invalid call")

        has_result = self.status in _RESULT_STATUSES
        if has_result != (self.result is not None):
            raise ValueError("exactly a completed execution or business rejection carries a result")
        if has_result:
            expected_hash = _sha256_json(thaw_json(self.result))
            if self.result_hash != expected_hash:
                raise ValueError("result_hash does not identify the canonical oracle result")
        elif self.result_hash is not None:
            raise ValueError("an outcome with no result has no result_hash")
        if self.released_to_model and not has_result:
            raise ValueError("an outcome with no result has nothing to release to the candidate")

        malformed = self.status == "malformed_result"
        if malformed:
            actual_type = json_type_tag(self.malformed_result)
            if self.malformed_result_type != actual_type:
                raise ValueError("malformed_result_type names the JSON shape the oracle returned")
            if self.malformed_result_hash != _sha256_json(thaw_json(self.malformed_result)):
                raise ValueError("malformed_result_hash identifies the canonical malformed result")
        elif self.malformed_result_type is not None or self.malformed_result_hash is not None:
            raise ValueError("malformed-result metadata belongs only to a malformed-result outcome")

        attempted = self.status in _ATTEMPTED_STATUSES
        if attempted:
            if (
                self.arguments_status != "valid_object"
                or self.schema_valid is not True
                or self.type != "function"
                or self.function_name is None
                or not self.function_name.strip()
                or self.call_id is None
                or not self.call_id.strip()
            ):
                raise ValueError("an attempted oracle call is a schema-valid named function call with an id")
            if self.state_commit == "not_started":
                raise ValueError("an attempted oracle call cannot say execution never started")
        elif self.state_commit != "not_started":
            raise ValueError("a call that was not executed cannot carry a commit verdict")

        if self.status == "business_rejection":
            result = thaw_json(self.result)
            error = result.get("error") if isinstance(result, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            if not isinstance(code, str) or not code.strip():
                raise ValueError("a business rejection is a structured error with a non-empty error.code")
        elif has_result:
            result = thaw_json(self.result)
            if isinstance(result, dict) and isinstance(result.get("error"), dict):
                raise ValueError("a structured oracle error is classified as a business rejection")
        if self.status == "unknown_commit_state" and self.state_commit != "unknown":
            raise ValueError("an unknown-commit outcome records an unknown state commit")
        if self.state_commit == "unknown" and self.status not in _UNKNOWN_COMMIT_STATUSES:
            raise ValueError("this execution outcome cannot carry an unknown state commit")
        if self.state_commit == "committed":
            if self.state_before_hash is None or self.state_after_hash is None:
                raise ValueError("a committed mutation identifies both state snapshots")
            if self.state_before_hash == self.state_after_hash:
                raise ValueError("a committed mutation changed its state hash")
        elif self.state_commit == "not_committed":
            if (self.state_before_hash is None) != (self.state_after_hash is None):
                raise ValueError("a verified non-commit identifies both state snapshots")
            if (
                self.state_before_hash is not None
                and self.state_before_hash != self.state_after_hash
            ):
                raise ValueError("a verified non-commit preserved its state hash")
        elif self.state_commit == "unknown":
            if (
                self.state_before_hash is not None
                and self.state_after_hash is not None
            ):
                raise ValueError("two state snapshots determine rather than obscure commit state")
        elif self.state_before_hash is not None or self.state_after_hash is not None:
            raise ValueError("a non-mutating outcome carries no mutation state snapshots")
        return self

    @property
    def attempted(self) -> bool:
        return self.status in _ATTEMPTED_STATUSES

    @property
    def has_result(self) -> bool:
        return self.status in _RESULT_STATUSES

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "execution_index": self.execution_index,
            "turn_index": self.turn_index,
            "position_in_turn": self.position_in_turn,
            "provider_call_index": self.provider_call_index,
            "call_id": self.call_id,
            "type": self.type,
            "function_name": self.function_name,
            "arguments_status": self.arguments_status,
            "parsed_arguments": (
                thaw_json(self.parsed_arguments) if self.parsed_arguments is not None else None
            ),
            "schema_valid": self.schema_valid,
            "schema_failures": [thaw_json(failure) for failure in self.schema_failures],
            "status": self.status,
            "state_commit": self.state_commit,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "result": thaw_json(self.result) if self.result is not None else None,
            "result_hash": self.result_hash,
            "released_to_model": self.released_to_model,
            "malformed_result": thaw_json(self.malformed_result) if self.status == "malformed_result" else None,
            "malformed_result_type": self.malformed_result_type,
            "malformed_result_hash": self.malformed_result_hash,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.semantic_payload().items() if key != "detail"}


class ExecutableTurn(_Frozen):
    """One candidate turn and the ordered live outcomes for all calls it made.

    How many live results this turn released is not recorded here.  It is a fact
    about the executions, which carry it individually, so the turn cannot restate
    it and disagree.
    """

    turn_index: NonNegativeInt
    request_hash: ContentHash
    call_status: CallStatus
    response_hash: ContentHash | None = None
    finish_reason: StrictStr | None = None
    assistant_content: Any = None
    tool_calls: tuple[CandidateToolCall, ...] = ()
    tool_call_outcome_indexes: tuple[NonNegativeInt, ...] = ()
    paired_call_indexes: tuple[NonNegativeInt, ...] = ()
    released_user_message_hash: ContentHash | None = None
    advanced: StrictBool
    reason_code: StrictStr
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_content(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            value["assistant_content"] = freeze_json(value.get("assistant_content"))
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableTurn:
        if not self.reason_code.strip():
            raise ValueError("an executable turn carries a stable reason code")
        if self.advanced and self.call_status != "completed":
            raise ValueError("a turn whose candidate call did not complete cannot advance")
        if len(self.tool_call_outcome_indexes) != len(self.tool_calls):
            raise ValueError("every proposed tool call has exactly one executable outcome")
        if [call.index for call in self.tool_calls] != list(range(len(self.tool_calls))):
            raise ValueError("provider tool calls must be contiguous and zero-based")
        if self.call_status != "completed" and (
            self.tool_calls or self.finish_reason is not None or self.assistant_content is not None
        ):
            raise ValueError("a candidate call that did not complete returned no envelope to read")
        if self.response_hash is not None and self.call_status not in _ENVELOPE_STATUSES:
            raise ValueError("only a call that received a response body can identify one")
        if self.released_user_message_hash is not None and not self.advanced:
            raise ValueError("only a turn that advanced can release its scripted user message")
        if self.paired_call_indexes and len(self.paired_call_indexes) != len(
            self.tool_calls
        ):
            raise ValueError(
                "expected-call pairings cover every candidate call in the turn"
            )
        if self.advanced and self.tool_calls and not self.paired_call_indexes:
            raise ValueError("an advanced tool-call turn identifies its expected pairings")
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
            "tool_call_outcome_indexes": list(self.tool_call_outcome_indexes),
            "paired_call_indexes": list(self.paired_call_indexes),
            "released_user_message_hash": self.released_user_message_hash,
            "advanced": self.advanced,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.semantic_payload().items() if key != "detail"}


class AssertionOutcome(_Frozen):
    """One pack assertion observation, classified without guessing from its name."""

    assertion_index: NonNegativeInt
    name: StrictStr
    category: AssertionCategory
    status: AssertionStatus
    reason_code: StrictStr
    detail: StrictStr

    @model_validator(mode="after")
    def _coherent(self) -> AssertionOutcome:
        if not self.name.strip():
            raise ValueError("an assertion name is non-empty")
        if not self.reason_code.strip():
            raise ValueError("an assertion outcome carries a stable reason code")
        return self

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def applies(self) -> bool:
        return self.status != "not_applicable"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "assertion_index": self.assertion_index,
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.semantic_payload().items() if key != "detail"}


class DependencyResolution(_Frozen):
    """How one expected downstream argument was derived from live evidence."""

    dependency_index: NonNegativeInt
    consumer_call_index: NonNegativeInt
    consumer_turn_index: NonNegativeInt
    argument_path: tuple[StrictStr | NonNegativeInt, ...]
    producer_call_index: NonNegativeInt
    producer_execution_index: NonNegativeInt | None = None
    result_path: StrictStr
    status: DependencyResolutionStatus
    resolved_value: Any = None
    resolved_value_hash: ContentHash | None = None
    reason_code: StrictStr
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_resolved_value(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("status") == "resolved":
            value = dict(value)
            validate_json_value(
                value.get("resolved_value"),
                label="resolved dependency value",
            )
            value["resolved_value"] = freeze_json(value.get("resolved_value"))
        return value

    @model_validator(mode="after")
    def _coherent(self) -> DependencyResolution:
        if not self.reason_code.strip():
            raise ValueError("a dependency resolution carries a stable reason code")
        if self.status == "resolved":
            resolved = thaw_json(self.resolved_value)
            if isinstance(resolved, (dict, list)):
                raise ValueError("a resolved dependency value is a JSON scalar")
            if (
                self.producer_execution_index is None
                or self.resolved_value_hash is None
                or self.resolved_value_hash
                != _sha256_json(resolved)
            ):
                raise ValueError(
                    "a resolved dependency binds its producer and canonical value hash"
                )
        elif self.resolved_value_hash is not None or self.resolved_value is not None:
            raise ValueError("a failed dependency does not claim a resolved value")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "dependency_index": self.dependency_index,
            "consumer_call_index": self.consumer_call_index,
            "consumer_turn_index": self.consumer_turn_index,
            "argument_path": list(self.argument_path),
            "producer_call_index": self.producer_call_index,
            "producer_execution_index": self.producer_execution_index,
            "result_path": self.result_path,
            "status": self.status,
            "resolved_value": (
                thaw_json(self.resolved_value) if self.status == "resolved" else None
            ),
            "resolved_value_hash": self.resolved_value_hash,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.semantic_payload().items()
            if key != "detail"
        }


class ExecutableEvent(_Frozen):
    """One deterministic driver action, ordered within an executable episode."""

    index: NonNegativeInt
    kind: ExecutableEventKind
    turn_index: NonNegativeInt | None = None
    execution_index: NonNegativeInt | None = None
    reason_code: StrictStr
    detail: StrictStr | None = None

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableEvent:
        if not self.reason_code.strip():
            raise ValueError("an executable event carries a stable reason code")
        if self.kind == "tool_execution" and self.execution_index is None:
            raise ValueError("a tool execution event identifies its tool outcome")
        if self.execution_index is not None and self.kind != "tool_execution":
            raise ValueError("only a tool execution event identifies a tool outcome")
        if self.kind in _TURN_SCOPED_EVENT_KINDS and self.turn_index is None:
            raise ValueError(f"a {self.kind} event identifies its executable turn")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "turn_index": self.turn_index,
            "execution_index": self.execution_index,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.semantic_payload().items() if key != "detail"}


class ExecutableEpisode(_Frozen):
    """One source-bound candidate conversation interleaved with live oracle I/O."""

    schema_version: Literal["1.2"] = EXECUTABLE_CONTRACT_VERSION
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    task_id: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    source_verification_identity: ContentHash
    oracle_verification_identity: ContentHash
    script_hash: ContentHash
    task_spec_hash: ContentHash
    status: ExecutableEpisodeStatus
    reason_code: StrictStr
    detail: StrictStr
    assistant_turns: NonNegativeInt
    observed: tuple[ExecutableTurn, ...]
    executions: tuple[ExecutedToolCall, ...] = ()
    dependencies: tuple[DependencyResolution, ...] = ()
    final_state_hash: ContentHash | None = None
    assertions: tuple[AssertionOutcome, ...] = ()
    events: tuple[ExecutableEvent, ...] = ()
    replayed: StrictBool = False

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableEpisode:
        if not self.reason_code.strip():
            raise ValueError("an executable episode carries a stable terminal reason code")
        if not self.observed and self.status not in {
            "episode_timeout",
            "oracle_reset_failed",
            "oracle_session_failed",
        }:
            raise ValueError("only a pre-conversation timeout or oracle startup failure has no candidate turn")
        if [turn.turn_index for turn in self.observed] != list(range(len(self.observed))):
            raise ValueError("executable turns must be contiguous and zero-based")
        if self.assistant_turns != len(self.observed):
            raise ValueError("assistant_turns counts exactly the turns the candidate was asked")
        if [execution.execution_index for execution in self.executions] != list(
            range(len(self.executions))
        ):
            raise ValueError("tool outcomes must be contiguous in execution order")

        cited_outcomes = [
            index for turn in self.observed for index in turn.tool_call_outcome_indexes
        ]
        if cited_outcomes != list(range(len(self.executions))):
            raise ValueError("turns cite every tool outcome once, in conversation order")
        for turn in self.observed:
            for position, (call, outcome_index) in enumerate(
                zip(turn.tool_calls, turn.tool_call_outcome_indexes, strict=True)
            ):
                outcome = self.executions[outcome_index]
                if (
                    outcome.turn_index != turn.turn_index
                    or outcome.position_in_turn != position
                    or outcome.provider_call_index != call.index
                    or outcome.call_id != call.id
                    or outcome.type != call.type
                    or outcome.function_name != call.function_name
                    or outcome.arguments_status != call.arguments_status
                    # Typed comparison, because ``==`` holds for 1 and True: an
                    # outcome must not be able to record a coerced argument and
                    # still claim it belongs to the call the provider sent.
                    or not json_equal(
                        thaw_json(outcome.parsed_arguments), thaw_json(call.parsed_arguments)
                    )
                ):
                    raise ValueError("a tool outcome must identify the provider call it records")
            terminal_without_followup = (
                self.status == "completed"
                and turn.turn_index == len(self.observed) - 1
            )
            if turn.advanced and not terminal_without_followup and not all(
                self.executions[index].released_to_model
                for index in turn.tool_call_outcome_indexes
                if self.executions[index].has_result
            ):
                raise ValueError(
                    "a nonterminal turn that advanced released every live result it obtained"
                )
            # Re-checked here because the runner marks a turn advanced with
            # ``model_copy``, which does not re-run the turn's own validators.
            if turn.paired_call_indexes and len(turn.paired_call_indexes) != len(
                turn.tool_calls
            ):
                raise ValueError(
                    "expected-call pairings cover every candidate call in the turn"
                )
            if turn.advanced and turn.tool_calls and not turn.paired_call_indexes:
                raise ValueError(
                    "an advanced tool-call turn identifies its expected pairings"
                )
        # Safety first, then the most specific observable protocol failure. In
        # practice the runner stops at the first terminal execution, but this
        # precedence also makes a partially executed multi-call turn unambiguous.
        if any(execution.state_commit == "unknown" for execution in self.executions):
            execution_terminal_status = "unknown_commit_state"
        elif any(execution.status == "malformed_result" for execution in self.executions):
            execution_terminal_status = "oracle_result_malformed"
        elif any(execution.status == "timeout" for execution in self.executions):
            execution_terminal_status = "oracle_timeout"
        elif any(
            execution.status in _ORACLE_CALL_FAILURE_STATUSES
            for execution in self.executions
        ):
            execution_terminal_status = "oracle_call_failed"
        else:
            execution_terminal_status = None
        execution_statuses = {
            "unknown_commit_state",
            "oracle_result_malformed",
            "oracle_timeout",
            "oracle_call_failed",
        }
        if (
            execution_terminal_status is not None
            and self.status != execution_terminal_status
        ) or (
            self.status in execution_statuses
            and self.status != execution_terminal_status
        ):
            raise ValueError(
                "the episode terminal status identifies its highest-priority tool outcome"
            )

        if self.status == "completed":
            if not self.observed or not all(turn.advanced for turn in self.observed):
                raise ValueError("a completed executable episode advanced through every observed turn")
            if any(execution.status not in _RESULT_STATUSES for execution in self.executions):
                raise ValueError("a completed executable episode contains only completed tool outcomes")
            if self.final_state_hash is None:
                raise ValueError("a completed executable episode identifies its final oracle state")
            if any(assertion.status == "infrastructure_error" for assertion in self.assertions):
                raise ValueError("a completed episode cannot contain an assertion infrastructure error")
        elif self.observed and self.observed[-1].advanced and self.status not in {
            "max_turns_exceeded",
            "episode_timeout",
            "assertion_infrastructure_failed",
            "oracle_state_failed",
            "oracle_session_failed",
            "dependency_resolution_failed",
        }:
            raise ValueError("an episode that stopped for another reason did not advance its last turn")

        if any(assertion.status == "infrastructure_error" for assertion in self.assertions) and (
            self.status != "assertion_infrastructure_failed"
        ):
            raise ValueError("an assertion infrastructure error is terminal episode evidence")
        if [assertion.assertion_index for assertion in self.assertions] != list(
            range(len(self.assertions))
        ):
            raise ValueError("assertion outcomes must be contiguous and zero-based")
        names = [assertion.name for assertion in self.assertions]
        if len(set(names)) != len(names):
            raise ValueError("an executable episode records each assertion once")
        if [item.dependency_index for item in self.dependencies] != list(
            range(len(self.dependencies))
        ):
            raise ValueError("dependency outcomes are an ordered declaration prefix")
        failed_dependencies = [
            item for item in self.dependencies if item.status != "resolved"
        ]
        if bool(failed_dependencies) != (self.status == "dependency_resolution_failed"):
            raise ValueError(
                "dependency_resolution_failed identifies a recorded dependency failure"
            )
        for item in self.dependencies:
            if item.producer_execution_index is not None:
                if item.producer_execution_index >= len(self.executions):
                    raise ValueError(
                        "a dependency cites an execution the episode did not record"
                    )
                producer = self.executions[item.producer_execution_index]
                if producer.turn_index >= item.consumer_turn_index:
                    raise ValueError(
                        "a dependency producer executes before its consumer turn"
                    )
        if [event.index for event in self.events] != list(range(len(self.events))):
            raise ValueError("executable events must be contiguous and zero-based")
        for event in self.events:
            if event.turn_index is not None and event.turn_index >= len(self.observed):
                raise ValueError("an executable event cites a turn the episode observed")
            if event.execution_index is not None and event.execution_index >= len(self.executions):
                raise ValueError("an executable event cites a tool outcome the episode recorded")
            if event.execution_index is not None:
                execution = self.executions[event.execution_index]
                if event.turn_index != execution.turn_index:
                    raise ValueError("a tool execution event cites its tool outcome's turn")
        return self

    @property
    def succeeded(self) -> bool:
        """Whether execution reached its planned terminal boundary, before scoring."""
        return self.status == "completed"

    @property
    def released_tool_results(self) -> int:
        """How many live results actually entered the candidate prompt."""
        return sum(1 for execution in self.executions if execution.released_to_model)

    @property
    def released_user_turns(self) -> int:
        """How many scripted user messages actually entered the candidate prompt."""
        return sum(1 for turn in self.observed if turn.released_user_message_hash is not None)

    def results_released_in(self, turn_index: int) -> int:
        """How many live results one turn admitted to the candidate prompt."""
        turn = self.observed[turn_index]
        return sum(
            1
            for index in turn.tool_call_outcome_indexes
            if self.executions[index].released_to_model
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_alias": self.candidate_alias,
            "canonical_model_identity": self.canonical_model_identity,
            "task_id": self.task_id,
            "plan_identity": self.plan_identity,
            "eval_config_hash": self.eval_config_hash,
            "source_verification_identity": self.source_verification_identity,
            "oracle_verification_identity": self.oracle_verification_identity,
            "script_hash": self.script_hash,
            "task_spec_hash": self.task_spec_hash,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "assistant_turns": self.assistant_turns,
            "released_user_turns": self.released_user_turns,
            "released_tool_results": self.released_tool_results,
            "observed": [turn.semantic_payload() for turn in self.observed],
            "executions": [execution.semantic_payload() for execution in self.executions],
            "dependencies": [
                dependency.semantic_payload() for dependency in self.dependencies
            ],
            "final_state_hash": self.final_state_hash,
            "assertions": [assertion.semantic_payload() for assertion in self.assertions],
            "events": [event.semantic_payload() for event in self.events],
        }

    def identity_payload(self) -> dict[str, Any]:
        """Semantic evidence with model diagnostics removed, not oracle result keys."""
        payload = {
            key: value
            for key, value in self.semantic_payload().items()
            if key
            not in {
                "detail",
                "observed",
                "executions",
                "dependencies",
                "assertions",
                "events",
            }
        }
        payload.update(
            {
                "observed": [turn.identity_payload() for turn in self.observed],
                "executions": [execution.identity_payload() for execution in self.executions],
                "dependencies": [
                    dependency.identity_payload() for dependency in self.dependencies
                ],
                "assertions": [assertion.identity_payload() for assertion in self.assertions],
                "events": [event.identity_payload() for event in self.events],
            }
        )
        return payload

    @property
    def episode_hash(self) -> str:
        """Identity of the live evidence, excluding replay and diagnostic prose."""
        return _sha256_json(self.identity_payload())

    def as_document(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "episode_hash": self.episode_hash,
            "replayed": self.replayed,
        }


__all__ = [
    "EXECUTABLE_CONTRACT_VERSION",
    "AssertionCategory",
    "AssertionOutcome",
    "AssertionStatus",
    "DependencyResolution",
    "DependencyResolutionStatus",
    "ExecutableEpisode",
    "ExecutableEpisodeStatus",
    "ExecutableEvent",
    "ExecutableEventKind",
    "ExecutableTurn",
    "ExecutedToolCall",
    "MalformedResultType",
    "StateCommitStatus",
    "ToolExecutionStatus",
]
