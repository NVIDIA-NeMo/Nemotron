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

"""The one comparison that decides both continuation and score.

Two consumers ask the same question of a candidate turn. The conversation driver
asks it to decide which recorded tool result a call has earned; the trace scorer
asks it to decide whether the call was right. If those two answers could differ,
a release gate stricter than the scorer would end an episode the scorer would
have credited, and a correct model would fail a task on transport grounds. So the
comparison lives here once and both callers import it.

Nothing here reads a config file, contacts a provider, or produces a number. It
takes a gold call, a predicted call, the row's tool declarations, and the pinned
:class:`EvalScoringConfig`, and reports what is the same and what is not. The
distinction between *name matched* and *arguments matched* is kept rather than
collapsed, because the driver needs one bit and a report needs both: a model that
called the right tool with a wrong amount is a different failure from one that
called the wrong tool.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import ArgumentStatus
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedCall,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalScoringConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    FrozenDict,
    NonNegativeInt,
    json_equal,
    thaw_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    apply_declared_defaults,
    declared_function,
    parameter_schema,
    validate_function_arguments,
)

# How the calls of one assistant turn are lined up against the gold group.
# ``positional`` requires every gold position; ``set`` accepts any permutation of
# the group. Ordering *across* turns is not a scope: replay walks the scripted
# turns in order, so a group deferred to a later turn has no recorded result.
CallOrderScope = Literal["positional", "set"]
INCOMPLETE_FINISH_REASONS = frozenset(
    {"content_filter", "length", "max_output_tokens", "max_tokens"}
)


@runtime_checkable
class PredictedCall(Protocol):
    """The part of a recorded tool call that a comparison reads.

    Stated as a protocol so the driver can compare the call the client just
    parsed and the scorer can compare the same call after it has been flattened
    out of an episode, without either of them converting to the other's shape.
    Both read the argument object the client parsed once; neither re-parses the
    provider's argument string.
    """

    type: str | None
    function_name: str | None
    arguments_status: ArgumentStatus
    parsed_arguments: FrozenDict | None


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArgumentDiff(_Frozen):
    """Which top-level arguments the two sides disagree about."""

    missing: tuple[StrictStr, ...] = ()
    unexpected: tuple[StrictStr, ...] = ()
    differing: tuple[StrictStr, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.missing or self.unexpected or self.differing)

    def describe(self, function_name: str) -> str:
        parts = [
            f"{label} {', '.join(names)}"
            for label, names in (
                ("missing", self.missing),
                ("unexpected", self.unexpected),
                ("differing", self.differing),
            )
            if names
        ]
        return f"arguments to {function_name} do not match the trace: {'; '.join(parts)}"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "differing": list(self.differing),
        }


class CallComparison(_Frozen):
    """One predicted call held against one gold call.

    ``comparable`` is false when the provider returned something no comparison
    can be run on — a call type this contract cannot replay, or arguments that
    never parsed into an object. That is a candidate observation, not an error:
    it is reported as a mismatch rather than repaired.
    """

    comparable: StrictBool
    name_matched: StrictBool
    arguments_matched: StrictBool
    schema_valid: StrictBool | None = None
    diff: ArgumentDiff | None = None
    detail: StrictStr

    @property
    def matched(self) -> bool:
        return self.comparable and self.name_matched and self.schema_valid is not False and self.arguments_matched

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "comparable": self.comparable,
            "name_matched": self.name_matched,
            "schema_valid": self.schema_valid,
            "arguments_matched": self.arguments_matched,
            "diff": self.diff.semantic_payload() if self.diff is not None else None,
            "detail": self.detail,
        }


class CallPairing(_Frozen):
    """Which gold call each predicted call was held against, and what came of it.

    ``paired_gold_indexes`` is positional over the *predicted* calls, so a caller
    can address each recorded result to the call of the candidate's that earned
    it. ``None`` marks a predicted call no gold call in the group could claim.
    """

    paired_gold_indexes: tuple[NonNegativeInt | None, ...] = ()
    comparisons: tuple[CallComparison, ...] = ()
    unmatched_gold_indexes: tuple[NonNegativeInt, ...] = ()
    matched: StrictBool
    detail: StrictStr

    @property
    def full_pairing(self) -> tuple[int, ...]:
        """Every predicted call's gold index, valid only when fully matched."""
        return tuple(index for index in self.paired_gold_indexes if index is not None)


class TextComparison(_Frozen):
    """Whether a text-only assistant turn is the one the trace recorded."""

    matched: StrictBool
    detail: StrictStr


def turn_order_scope(script: ConversationScript, scoring: EvalScoringConfig) -> CallOrderScope:
    """How this row's calls are lined up inside one assistant turn.

    ``prefix`` lines up as a set within the turn: the prefix it declares is over
    the first appearances of ``required_tools`` across the conversation, which is
    a separate check from the shape of any single group.
    """
    if scoring.respect_call_order and script.call_order == "strict":
        return "positional"
    return "set"


def compare_group_size(gold_count: int, predicted_count: int) -> str | None:
    """Refuse a turn whose call count cannot be answered from the trace.

    This holds even with the grouping gate relaxed: replay has exactly one
    recorded result per gold call, so a differently sized turn has no faithful
    reply to hand back, whatever a scorer would award it.
    """
    if gold_count == predicted_count:
        return None
    return f"this turn's call group holds {gold_count} call(s); the candidate issued {predicted_count}"


def compare_call(
    gold: ScriptedCall,
    predicted: PredictedCall,
    *,
    tools: Sequence[Any],
    scoring: EvalScoringConfig,
) -> CallComparison:
    """Hold one predicted call against one gold call under the pinned policy."""
    if predicted.type != "function":
        return CallComparison(
            comparable=False,
            name_matched=False,
            arguments_matched=False,
            detail=f"declares type {predicted.type!r}; only 'function' can be compared",
        )
    name = predicted.function_name
    if name != gold.function_name:
        return CallComparison(
            comparable=True,
            name_matched=False,
            arguments_matched=False,
            detail=f"calls {name or '<unnamed>'}; the trace calls {gold.function_name}",
        )
    if predicted.arguments_status != "valid_object":
        return CallComparison(
            comparable=False,
            name_matched=True,
            arguments_matched=False,
            detail=f"call to {gold.function_name} has {predicted.arguments_status} arguments",
        )
    schema_valid: bool | None = None
    if scoring.argument_matching == "schema_then_canonical":
        function = declared_function(tools, gold.function_name)
        failures = (
            [{"reason": "unknown_tool", "tool": gold.function_name}]
            if function is None
            else validate_function_arguments(function, thaw_json(predicted.parsed_arguments or {}))
        )
        schema_valid = not failures
        if failures:
            reasons = ", ".join(sorted({str(failure.get("reason")) for failure in failures}))
            return CallComparison(
                comparable=True,
                name_matched=True,
                schema_valid=False,
                arguments_matched=False,
                detail=f"call to {gold.function_name} violates its declared schema: {reasons}",
            )
    diff = compare_arguments(
        gold.arguments,
        predicted.parsed_arguments or {},
        function_name=gold.function_name,
        tools=tools,
        scoring=scoring,
    )
    if diff.empty:
        return CallComparison(
            comparable=True,
            name_matched=True,
            schema_valid=schema_valid,
            arguments_matched=True,
            diff=diff,
            detail=f"call to {gold.function_name} matches the trace",
        )
    return CallComparison(
        comparable=True,
        name_matched=True,
        schema_valid=schema_valid,
        arguments_matched=False,
        diff=diff,
        detail=diff.describe(gold.function_name),
    )


def compare_arguments(
    gold_arguments: Any,
    predicted_arguments: Any,
    *,
    function_name: str,
    tools: Sequence[Any],
    scoring: EvalScoringConfig,
) -> ArgumentDiff:
    """Compare two argument objects: schema step first, then canonical JSON."""
    predicted = thaw_json(predicted_arguments)
    gold = thaw_json(gold_arguments)
    if scoring.argument_matching == "schema_then_canonical" and scoring.insert_declared_defaults:
        schema = parameter_schema(tools, function_name)
        # Fill both sides. An explicit candidate value absent from gold matches
        # only when it equals the declared default; any other value remains a
        # difference. In particular, ``confirm=true`` cannot disappear when the
        # schema default the gold omitted is ``false``.
        predicted = apply_declared_defaults(predicted, schema)
        gold = apply_declared_defaults(gold, schema)
    if json_equal(predicted, gold):
        return ArgumentDiff()
    return ArgumentDiff(
        missing=tuple(sorted(set(gold) - set(predicted))),
        unexpected=tuple(sorted(set(predicted) - set(gold))),
        differing=tuple(
            sorted(name for name in set(gold) & set(predicted) if not json_equal(predicted[name], gold[name]))
        ),
    )


def pair_turn_calls(
    gold: Sequence[ScriptedCall],
    predicted: Sequence[PredictedCall],
    *,
    tools: Sequence[Any],
    scoring: EvalScoringConfig,
    scope: CallOrderScope,
) -> CallPairing:
    """Line one turn's predicted calls up against its gold group.

    Under ``set`` scope, a full match claims its gold call before any name-only
    match does. Deciding the exact pairs before deciding the arguments is what
    lets a report say "the right two tools, one wrong amount" instead of
    attributing the whole turn to whichever gold call happened to come first.
    """
    if scope == "positional":
        return _pair_positionally(gold, predicted, tools=tools, scoring=scoring)
    return _pair_as_a_set(gold, predicted, tools=tools, scoring=scoring)


def compare_turn_order(
    turn: ScriptedTurn,
    predicted: Sequence[PredictedCall],
    *,
    script: ConversationScript,
    scoring: EvalScoringConfig,
    pairing: CallPairing,
    names_so_far: Sequence[str | None],
) -> str | None:
    """Return one ordering problem, independently of selection and arguments.

    Both continuation and scoring pass the same set pairing here. This keeps
    their per-dimension verdicts identical: a permutation contains the right
    calls but violates strict order, while a wrong call fails selection without
    also being described as an ordering error.
    """
    if not scoring.respect_call_order or script.call_order == "any":
        return None
    if script.call_order == "prefix":
        return compare_required_prefix(
            first_appearances(names_so_far, script.required_tools),
            script.required_tools,
            script.call_order_prefix or 0,
        )
    if not pairing.matched:
        return None
    positional = pair_turn_calls(
        turn.calls,
        predicted,
        tools=script.tools,
        scoring=scoring,
        scope="positional",
    )
    if positional.matched:
        return None
    return "the turn made the trace's calls in a different order than the trace issues them"


def _finish(
    gold: Sequence[ScriptedCall],
    predicted: Sequence[PredictedCall],
    paired: Sequence[int | None],
    comparisons: Sequence[CallComparison],
    *,
    success: str,
) -> CallPairing:
    """Assemble a pairing, and name the first thing that went wrong in it."""
    claimed = {index for position, index in enumerate(paired) if index is not None and comparisons[position].matched}
    unmatched = tuple(call.call_index for call in gold if call.call_index not in claimed)
    matched = len(predicted) == len(gold) and not unmatched
    if matched:
        detail = success
    else:
        failing = next(
            (position for position, comparison in enumerate(comparisons) if not comparison.matched),
            None,
        )
        if failing is not None:
            detail = f"call {failing}: {comparisons[failing].detail}"
        else:
            detail = compare_group_size(len(gold), len(predicted)) or success
    return CallPairing(
        paired_gold_indexes=tuple(paired),
        comparisons=tuple(comparisons),
        unmatched_gold_indexes=unmatched,
        matched=matched,
        detail=detail,
    )


def _pair_positionally(
    gold: Sequence[ScriptedCall],
    predicted: Sequence[PredictedCall],
    *,
    tools: Sequence[Any],
    scoring: EvalScoringConfig,
) -> CallPairing:
    paired: list[int | None] = []
    comparisons: list[CallComparison] = []
    for position, call in enumerate(predicted):
        if position >= len(gold):
            paired.append(None)
            comparisons.append(
                CallComparison(
                    comparable=True,
                    name_matched=False,
                    arguments_matched=False,
                    detail="the trace's call group has no call at this position",
                )
            )
            continue
        paired.append(gold[position].call_index)
        comparisons.append(compare_call(gold[position], call, tools=tools, scoring=scoring))
    return _finish(gold, predicted, paired, comparisons, success="every call matched the trace in order")


def _pair_as_a_set(
    gold: Sequence[ScriptedCall],
    predicted: Sequence[PredictedCall],
    *,
    tools: Sequence[Any],
    scoring: EvalScoringConfig,
) -> CallPairing:
    unclaimed = list(gold)
    paired: list[int | None] = [None] * len(predicted)
    comparisons: list[CallComparison | None] = [None] * len(predicted)

    def accepts(comparison: CallComparison, *, on_name_alone: bool) -> bool:
        return comparison.name_matched if on_name_alone else comparison.matched

    for on_name_alone in (False, True):
        for position, call in enumerate(predicted):
            if comparisons[position] is not None:
                continue
            for candidate in list(unclaimed):
                comparison = compare_call(candidate, call, tools=tools, scoring=scoring)
                if accepts(comparison, on_name_alone=on_name_alone):
                    unclaimed.remove(candidate)
                    paired[position] = candidate.call_index
                    comparisons[position] = comparison
                    break
    for position, call in enumerate(predicted):
        if comparisons[position] is not None:
            continue
        # Nothing in the group can claim this call. Compare it against a gold call
        # anyway so the report names what the trace asked for instead of only that
        # the call was unexpected.
        comparisons[position] = (
            compare_call(unclaimed[0], call, tools=tools, scoring=scoring)
            if unclaimed
            else CallComparison(
                comparable=True,
                name_matched=False,
                arguments_matched=False,
                detail="the trace's call group is already accounted for by an earlier call",
            )
        )
    resolved = [comparison for comparison in comparisons if comparison is not None]
    return _finish(gold, predicted, paired, resolved, success="every call matched the trace, in any order")


def compare_text_turn(
    turn: ScriptedTurn,
    assistant_content: Any,
    predicted_call_names: Sequence[str | None],
    *,
    scoring: EvalScoringConfig,
) -> TextComparison:
    """Decide whether a text-only assistant turn is the recorded one.

    Every text turn is held to what it did: answered in words rather than with a
    call, and said something. What the next scripted user request depends on is
    that the candidate reached this point in the conversation without calling a
    tool it had not earned, and both of those checks establish it.

    The gold sentence is one translation of one phrasing of that behavior, so
    ``structural`` matching does not ask for it back. A pack states its
    intermediate turns as milestone classes — ask for a slot, ask to confirm —
    and its own success assertions are what hold the turn to the domain meaning.
    ``verbatim`` restores the character-for-character demand, which measures
    wording rather than tool use and exists to reproduce runs scored that way.
    """
    if predicted_call_names:
        names = ", ".join(sorted({name or "<unnamed>" for name in predicted_call_names}))
        return TextComparison(
            matched=False,
            detail=f"the trace answers this request in words; the candidate called {names}",
        )
    if assistant_content is None:
        return TextComparison(matched=False, detail="the candidate returned neither content nor a tool call")
    if not _has_meaningful_text(assistant_content):
        placement = "terminal answer" if turn.is_terminal else "intermediate turn"
        return TextComparison(
            matched=False,
            detail=f"the candidate returned no non-empty textual content for the {placement}",
        )
    if (
        not turn.is_terminal
        and scoring.intermediate_text_matching == "verbatim"
        and assistant_content != turn.expected_assistant_content
    ):
        return TextComparison(
            matched=False,
            detail="the candidate did not produce the scripted intermediate assistant text",
        )
    return TextComparison(matched=True, detail="text turn, as the trace recorded")


def _has_meaningful_text(value: Any) -> bool:
    """Recognize non-empty plain or structured text without provider coercion."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_meaningful_text(child) for child in value)
    if isinstance(value, Mapping):
        text_keys = {"content", "output_text", "parts", "text", "value"}
        return any(
            _has_meaningful_text(child)
            for key, child in value.items()
            if str(key).strip().lower() in text_keys
        )
    return False


def finish_reason_problem(finish_reason: str | None) -> str | None:
    """Name explicit provider evidence that a completion did not finish."""
    if finish_reason is None:
        return None
    normalized = finish_reason.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in INCOMPLETE_FINISH_REASONS:
        return None
    return f"the provider ended the assistant turn with incomplete finish_reason {finish_reason!r}"


def first_appearances(names: Iterable[str | None], required: Sequence[str]) -> tuple[str, ...]:
    """The order in which the required tools were first called, ignoring repeats."""
    wanted = set(required)
    seen: list[str] = []
    for name in names:
        if name in wanted and name not in seen:
            seen.append(str(name))
    return tuple(seen)


def compare_required_prefix(
    seen: Sequence[str],
    required: Sequence[str],
    prefix: int,
) -> str | None:
    """Check the declared prefix of required-tool first appearances.

    Only the positions actually reached are checked, so a conversation that has
    not yet called every prefixed tool is consistent rather than wrong. Whether
    it eventually called them is the coverage question, answered elsewhere.
    """
    expected = list(required[:prefix])
    observed = list(seen[:prefix])
    if observed == expected[: len(observed)]:
        return None
    return f"required-tool prefix is {expected}; the candidate's first appearances are {observed}"


__all__ = [
    "ArgumentDiff",
    "CallComparison",
    "CallOrderScope",
    "CallPairing",
    "INCOMPLETE_FINISH_REASONS",
    "PredictedCall",
    "TextComparison",
    "compare_arguments",
    "compare_call",
    "compare_group_size",
    "compare_required_prefix",
    "compare_text_turn",
    "compare_turn_order",
    "first_appearances",
    "finish_reason_problem",
    "pair_turn_calls",
    "turn_order_scope",
]
