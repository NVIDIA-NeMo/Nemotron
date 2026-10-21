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

"""What a trace score is, gate by gate, for one candidate on one task.

A score is only comparable if it says what it measured. So a task score is not a
number with a label: it names every gate the scoring contract defines, says
whether that gate applied to this row, and — when it failed — which turn and
which call failed it. ``task_success`` is then derived from those gates rather
than asserted alongside them, and a validator refuses a score whose success bit
disagrees with its own gates.

Gates that do not apply are recorded as ``not_applicable`` rather than omitted or
counted as passes. A single-call row has no ordering gate, and a report that
silently dropped it could not be told apart from one where ordering was checked
and passed.

A failed gate also says whose failure it is. An unreachable endpoint and a wrong
tool both fail the task, and neither is softened, but a run whose gates failed on
infrastructure is a run to re-drive rather than a model to reject. The
classification lives on the gate because the scorer is the last place that knows
why the gate failed; every reader downstream projects it rather than re-deriving
it from a status it would have to interpret again.

Every hash here is path-free and time-free. A score cites the script and episode
it was derived from, the complete evaluation-config identity, the pinned scoring
policy, and the content hash of the prose contract that defines the comparison.
Human diagnostics remain in the document but outside the hash; stable reason
codes carry their semantics. Re-scoring the same evidence under the same rules
therefore reproduces ``score_hash`` exactly, and rewording a diagnosis does not
fork it.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_comparison import ArgumentDiff
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    ArgumentStatus,
    CallStatus,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import EpisodeStatus
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    TRACE_NON_CANDIDATE_STOPS,
    EvalFailureRecord,
    episode_failure_record,
    gate_failure_record,
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

TRACE_SCORING_CONTRACT_VERSION: Final = "1.1"

# The gates a trace score reports, in the order a report reads them: what the
# model asked for, then how it asked, then whether the conversation held together.
# The order is fixed so two scores of the same evidence hash the same.
SCORING_GATES: Final = (
    "tool_selection",
    "arguments",
    "schema_valid",
    "call_grouping",
    "call_ordering",
    "text_turn",
    "trace_completion",
)
ScoringGate = Literal[
    "tool_selection",
    "arguments",
    "schema_valid",
    "call_grouping",
    "call_ordering",
    "text_turn",
    "trace_completion",
]
GateOutcome = Literal["passed", "failed", "not_applicable"]
# Who a failed gate is about. A trace score never carries ``evidence``: evidence
# that does not line up with its script is refused as a typed error instead of
# being scored, so no gate can report it.
TraceGateFailureClass = Literal["none", "candidate", "infrastructure"]

# How the gates this scorer computes roll up into the metric names a published
# bundle declares. Several internal gates map onto one published metric: a bundle
# says a scorer measures ``arguments``, and this contract additionally separates
# "the value was wrong" from "the schema forbids the argument at all".
# ``results`` has no trace gate — it is what oracle replay measures.
EXPORT_METRIC_BY_GATE: Final = FrozenDict(
    {
        "tool_selection": "tool_selection",
        "arguments": "arguments",
        "schema_valid": "arguments",
        "call_grouping": "call_ordering",
        "call_ordering": "call_ordering",
        "text_turn": "task_success",
        "trace_completion": "task_success",
    }
)


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class GateResult(_Frozen):
    """One scoring dimension's verdict, whose failure it is, and which turn failed it.

    ``turn_index`` is the assistant turn a reader should look at, so a failure is
    actionable without diffing the whole trace. It is absent when the failure is
    about the trace as a whole — gold calls the candidate never got as far as
    being asked for.

    ``failure_class`` never changes whether the gate counts against the task; it
    only records whether the candidate or the run is answerable for it.
    """

    gate: ScoringGate
    outcome: GateOutcome
    failure_class: TraceGateFailureClass = "none"
    reason_code: StrictStr
    detail: StrictStr
    turn_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _coherent(self) -> GateResult:
        if not self.reason_code.strip():
            raise ValueError("a trace gate carries a stable reason code")
        if not self.reason_code.startswith(f"{self.gate}."):
            raise ValueError("a trace gate reason code belongs to that gate's namespace")
        if self.outcome == "failed":
            if self.failure_class == "none":
                raise ValueError("a failed gate says whose failure it is")
        else:
            if self.failure_class != "none":
                raise ValueError("only a failed gate is attributed")
            if self.turn_index is not None:
                raise ValueError("only a failed gate points at the turn that failed it")
        return self

    @property
    def counts_against(self) -> bool:
        return self.outcome == "failed"

    @property
    def applies(self) -> bool:
        return self.outcome != "not_applicable"

    @property
    def blames_the_run(self) -> bool:
        """Whether this gate failed for a reason the candidate did not choose."""
        return self.outcome == "failed" and self.failure_class == "infrastructure"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "turn_index": self.turn_index,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.semantic_payload().items() if key != "detail"}


class ScoredCall(_Frozen):
    """One call the candidate made, held against the gold call it answered.

    ``gold_call_index`` is ``None`` when no gold call in the turn's group could
    claim this call — a call to a tool the trace never asks for at this point.
    Name and argument agreement stay separate fields because they are separate
    gates: the wrong tool and the wrong amount are different failures.
    """

    turn_index: NonNegativeInt
    position_in_turn: NonNegativeInt
    gold_call_index: NonNegativeInt | None = None
    gold_function_name: StrictStr | None = None
    predicted_function_name: StrictStr | None = None
    predicted_type: StrictStr | None = None
    arguments_status: ArgumentStatus
    name_matched: StrictBool
    arguments_matched: StrictBool
    diff: ArgumentDiff | None = None
    # Which declared constraints the predicted arguments violate. Populated only
    # under the schema step, and never used to repair a call.
    schema_failures: tuple[FrozenDict, ...] = ()
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_failures(cls, value: Any) -> Any:
        if isinstance(value, dict) and "schema_failures" in value:
            value = dict(value)
            failures = list(value["schema_failures"])
            validate_json_value(failures, label="schema failures")
            value["schema_failures"] = tuple(freeze_json(failure) for failure in failures)
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ScoredCall:
        if self.arguments_matched and not self.name_matched:
            raise ValueError("arguments cannot match a gold call this call did not name")
        return self

    @property
    def matched(self) -> bool:
        return self.name_matched and self.arguments_matched

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "position_in_turn": self.position_in_turn,
            "gold_call_index": self.gold_call_index,
            "gold_function_name": self.gold_function_name,
            "predicted_function_name": self.predicted_function_name,
            "predicted_type": self.predicted_type,
            "arguments_status": self.arguments_status,
            "name_matched": self.name_matched,
            "arguments_matched": self.arguments_matched,
            "diff": self.diff.semantic_payload() if self.diff is not None else None,
            "schema_failures": [thaw_json(failure) for failure in self.schema_failures],
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        """This call's verdict without its wording, and with its evidence intact.

        Only this model's own ``detail`` is dropped. A declared constraint or an
        argument value is not diagnostic prose just because a schema author named
        it ``detail``, so nested evidence is never searched for keys to remove.
        """
        return {key: value for key, value in self.semantic_payload().items() if key != "detail"}


class ScoredTurn(_Frozen):
    """One assistant turn the candidate was actually asked, scored.

    Only turns the episode reached appear. A turn the episode never sent is not a
    turn the candidate failed at; the gold calls it would have carried are counted
    as unmatched by the coverage gates, and the reason the episode stopped is
    carried by the score's ``episode_status``.
    """

    turn_index: NonNegativeInt
    kind: Literal["tool_calls", "text"]
    call_status: CallStatus
    advanced: StrictBool
    finish_reason: StrictStr | None = None
    # Every call the candidate made on this turn, including calls made on a turn
    # the trace answers in words: a call the trace never asked for is evidence,
    # not something to drop.
    calls: tuple[ScoredCall, ...] = ()
    # ``None`` on a turn where the question does not arise: a text turn has no
    # group to size or order, a call turn has no recorded text to reproduce, and
    # ordering is unset when the row or the config does not order calls.
    group_size_matched: StrictBool | None = None
    text_matched: StrictBool | None = None
    order_respected: StrictBool | None = None
    order_detail: StrictStr | None = None
    detail: StrictStr

    @model_validator(mode="after")
    def _coherent(self) -> ScoredTurn:
        if self.kind == "text":
            if self.group_size_matched is not None or self.order_respected is not None:
                raise ValueError("a text turn has no call group to size or order")
            if any(call.gold_call_index is not None for call in self.calls):
                raise ValueError("a turn the trace answers in words has no gold call to pair a call with")
        elif self.text_matched is not None:
            raise ValueError("a turn that issues calls has no recorded text to reproduce")
        if (self.order_detail is not None) and self.order_respected is not False:
            raise ValueError("an ordering detail belongs to, and only to, an ordering failure")
        if [call.position_in_turn for call in self.calls] != list(range(len(self.calls))):
            raise ValueError("scored calls must cover the turn's calls in the order the provider returned them")
        if any(call.turn_index != self.turn_index for call in self.calls):
            raise ValueError("a scored call belongs to the turn that made it")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "kind": self.kind,
            "call_status": self.call_status,
            "advanced": self.advanced,
            "finish_reason": self.finish_reason,
            "calls": [call.semantic_payload() for call in self.calls],
            "group_size_matched": self.group_size_matched,
            "text_matched": self.text_matched,
            "order_respected": self.order_respected,
            "order_detail": self.order_detail,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.semantic_payload().items()
            if key not in {"detail", "order_detail", "calls"}
        }
        payload["calls"] = [call.identity_payload() for call in self.calls]
        return payload


class TraceTaskScore(_Frozen):
    """One candidate's score on one task, derived only from recorded evidence."""

    schema_version: Literal["1.1"] = TRACE_SCORING_CONTRACT_VERSION
    # What this number is allowed to claim. Oracle replay and pack assertions add
    # gates this scorer does not compute, so a trace score never stands in for an
    # executable one.
    scope: Literal["trace"] = "trace"
    task_id: StrictStr
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    source_verification_identity: ContentHash
    script_hash: ContentHash
    episode_hash: ContentHash
    # The prose that defines the comparison, by content. Editing it changes the
    # identity of every score taken under it.
    scoring_contract_hash: ContentHash
    scoring_policy: FrozenDict
    episode_status: EpisodeStatus
    # True when the episode ended for a reason that is not the candidate's answer:
    # an unreachable endpoint, a spent episode budget, or a turn budget below what
    # the trace needs. It never softens ``task_success`` — a task that could not be
    # finished is failed, not skipped — but it lets a report separate a broken run
    # from a wrong model.
    non_candidate_stop: StrictBool
    expected_calls: NonNegativeInt
    observed_calls: NonNegativeInt
    matched_calls: NonNegativeInt
    turns: tuple[ScoredTurn, ...] = ()
    gates: tuple[GateResult, ...]
    task_success: StrictBool
    detail: StrictStr

    @model_validator(mode="before")
    @classmethod
    def _freeze_policy(cls, value: Any) -> Any:
        if isinstance(value, dict) and "scoring_policy" in value:
            value = dict(value)
            validate_json_value(value["scoring_policy"], label="scoring policy")
            value["scoring_policy"] = freeze_json(value["scoring_policy"])
        return value

    @model_validator(mode="after")
    def _coherent(self) -> TraceTaskScore:
        if tuple(gate.gate for gate in self.gates) != SCORING_GATES:
            raise ValueError("a score reports every gate the contract defines, once, in contract order")
        if self.matched_calls > self.expected_calls:
            raise ValueError("more gold calls were matched than the trace asks for")
        if [turn.turn_index for turn in self.turns] != list(range(len(self.turns))):
            raise ValueError("scored turns must be contiguous and zero-based")
        expected_success = all(not gate.counts_against for gate in self.gates)
        if self.task_success != expected_success:
            raise ValueError("task_success must be exactly whether every applicable gate passed")
        expected_non_candidate_stop = (
            self.episode_status in TRACE_NON_CANDIDATE_STOPS
        )
        if self.non_candidate_stop != expected_non_candidate_stop:
            raise ValueError(
                "non_candidate_stop must match the trace episode status taxonomy"
            )
        # An episode that stopped for a non-candidate reason never reached the end
        # of its trace, so completion must blame the run — otherwise a broken run
        # would read as a wrong model. Nothing else may, unless the episode stopped
        # that way: a candidate's own mistake is not the run's fault.
        if self.non_candidate_stop:
            if not self.gate("trace_completion").blames_the_run:
                raise ValueError("a non-candidate stop is a completion failure the run is answerable for")
        elif any(gate.blames_the_run for gate in self.gates):
            raise ValueError("a gate blames the run only when the episode stopped for a non-candidate reason")
        return self

    @property
    def failed_gates(self) -> tuple[ScoringGate, ...]:
        return tuple(gate.gate for gate in self.gates if gate.counts_against)

    @property
    def applicable_gates(self) -> tuple[ScoringGate, ...]:
        return tuple(gate.gate for gate in self.gates if gate.applies)

    def gate(self, name: ScoringGate) -> GateResult:
        for result in self.gates:
            if result.gate == name:
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
            "source_verification_identity": self.source_verification_identity,
            "script_hash": self.script_hash,
            "episode_hash": self.episode_hash,
            "scoring_contract_hash": self.scoring_contract_hash,
            "scoring_policy": thaw_json(self.scoring_policy),
            "episode_status": self.episode_status,
            "non_candidate_stop": self.non_candidate_stop,
            "expected_calls": self.expected_calls,
            "observed_calls": self.observed_calls,
            "matched_calls": self.matched_calls,
            "turns": [turn.semantic_payload() for turn in self.turns],
            "gates": [gate.semantic_payload() for gate in self.gates],
            "task_success": self.task_success,
            "detail": self.detail,
        }

    def identity_payload(self) -> dict[str, Any]:
        """Every machine-stable verdict, with each model's own wording removed."""
        payload = {
            key: value
            for key, value in self.semantic_payload().items()
            if key not in {"detail", "turns", "gates"}
        }
        payload["turns"] = [turn.identity_payload() for turn in self.turns]
        payload["gates"] = [gate.identity_payload() for gate in self.gates]
        return payload

    @property
    def score_hash(self) -> str:
        """One hash for "this evidence, scored under these rules"."""
        return _sha256_json(self.identity_payload())

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "score_hash": self.score_hash}

    def failure_records(self) -> tuple[EvalFailureRecord, ...]:
        return trace_failure_records(self)


def trace_failure_records(score: TraceTaskScore) -> tuple[EvalFailureRecord, ...]:
    """Project one trace score's failures onto the shared eval error taxonomy.

    The episode record carries what no gate reason code can: every incomplete
    episode fails the same completion gate, and only the terminal status separates
    a spent turn budget from an endpoint that was never reachable. Gate records
    then name each dimension that failed, attributed as the scorer attributed it,
    so a report reads one vocabulary across trace and executable evaluation.
    """
    episode = episode_failure_record(score.episode_status, executable=False)
    records = [] if episode is None else [episode]
    records.extend(
        gate_failure_record(
            gate=gate.gate,
            reason_code=gate.reason_code,
            failure_class=gate.failure_class,
        )
        for gate in score.gates
        if gate.outcome == "failed"
    )
    return tuple(records)


__all__ = [
    "EXPORT_METRIC_BY_GATE",
    "SCORING_GATES",
    "TRACE_SCORING_CONTRACT_VERSION",
    "GateOutcome",
    "GateResult",
    "ScoredCall",
    "ScoredTurn",
    "ScoringGate",
    "TraceGateFailureClass",
    "TraceTaskScore",
    "trace_failure_records",
]
