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

"""Output schemas for the authoring calls, shaped so the model states intent, not facts.

The design rule here is that a model should not be *able* to express a value it has no
evidence for. Where that can be enforced by shape it is: a validation case names the
*source* of each argument rather than a literal, so "make up a plausible account id" is not
a sentence this schema can say. Where shape cannot enforce it — prose fields, tool names —
`grounding.py` checks the response against the bundle afterwards.

Every draft carries `blocked_on`. At `L0` most concrete expectations are unavailable, and a
draft that admits which unknown it is waiting on is reviewable; one that quietly omits the
expectation looks finished.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

# One definition of what an identifier is, because two parties rely on it: the validator
# refuses a draft that breaks it, and the model only learns it from the field description
# below. Stated twice they drift, and a draft is then refused for a bound nobody told the
# model about.
IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9_]{2,63}$"
IDENTIFIER_DESCRIPTION = (
    "Stable identifier, unique in the pack: 3 to 64 characters, lowercase letters, digits and underscores only"
)


def _identifier() -> Any:
    """A fresh field each time, since Pydantic annotates the one it is given."""
    return Field(description=IDENTIFIER_DESCRIPTION)


# The unknown field names a draft may declare itself blocked on. Kept as a closed set so a
# typo becomes an error instead of a blocker nobody will ever resolve.
UnknownField = Literal[
    "observed_result_shapes",
    "observed_error_codes",
    "state_deltas",
    "confirmation_behavior",
    "fixture_samples",
    "tool_dependencies",
]

# The same closed set, as a value. Two parties need it: this schema constrains what a model
# may emit, and the evidence reader has to decide which of a bundle's gap records name an
# unknown a draft could declare at all. Derived from the type so they cannot drift.
UNKNOWN_FIELDS: frozenset[str] = frozenset(get_args(UnknownField))

# Where an argument value comes from. `unresolved` is the honest answer at L0 for anything
# that would otherwise be invented.
#
# `invalid_literal` exists because an error probe has to send something the tool refuses, and
# every other source is defined never to produce one. Without it that case can only be spelled
# `unresolved`, which reads identically to an unobserved value, and a reviewer then cannot tell
# a deliberate refusal probe from a gap nobody closed.
ValueSource = Literal[
    "literal",
    "invalid_literal",
    "fixture",
    "absent_id",
    "confirmation_flag",
    "unresolved",
]


class _Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCoverage(_Draft):
    """What the benchmark should exercise for one tool, and why."""

    tool: str = Field(description="Published tool name, exactly as given in the evidence")
    purpose: str = Field(description="What this tool does, in the author's own words")
    policies: list[str] = Field(
        default_factory=list,
        description="Domain rules a correct assistant must respect when using this tool",
    )
    positive_intents: list[str] = Field(
        default_factory=list,
        description="Successful outcomes worth covering, as intents rather than values",
    )
    negative_intents: list[str] = Field(
        default_factory=list,
        description="Failure situations worth covering, as intents rather than values",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Published tool names that must be called before this one can succeed",
    )
    confirmation_relevant: bool = Field(
        default=False,
        description="True when this tool is gated on an explicit confirmation argument",
    )


class CoveragePlan(_Draft):
    """The whole reviewed coverage matrix."""

    tools: list[ToolCoverage]
    cross_tool_notes: list[str] = Field(
        default_factory=list,
        description="Ordering or state interactions that span more than one tool",
    )


class ArgumentPlan(_Draft):
    """One argument, described by where its value will come from."""

    name: str = Field(description="Parameter name from the tool's own input schema")
    source: ValueSource
    literal: str | None = Field(
        default=None,
        description=(
            "The value itself: for source=literal when the parameter's own schema pins the "
            "value set, and for source=invalid_literal when that schema refuses the value"
        ),
    )
    note: str | None = Field(
        default=None,
        description="Required for source=unresolved: the value this argument still needs",
    )

    @model_validator(mode="after")
    def _unresolved_states_what_is_missing(self) -> ArgumentPlan:
        # `unresolved` is the one source naming no value at all. Left unexplained it reads
        # like a settled argument, which is the shape a reviewer has no way to question.
        if self.source == "unresolved" and not (self.note or "").strip():
            raise ValueError(
                "source=unresolved requires a note naming the value the case still needs"
            )
        return self


class ValidationCaseDraft(_Draft):
    """One probe proving a tool behaves as the pack claims."""

    case_id: str = _identifier()
    tool: str
    kind: Literal["success", "error", "confirmation_pending"]
    intent: str = Field(description="What this probe demonstrates")
    arguments: list[ArgumentPlan] = Field(default_factory=list)
    expectation: str = Field(description="The observable outcome, in words; codes and shapes come from probes")
    blocked_on: list[UnknownField] = Field(default_factory=list)


# Every plan below states a minimum for the same reason `required_tools` does. An empty list
# satisfies each field-level rule while proving nothing, and the emptiness only surfaces much
# later — at compilation, or at an assembly whose supplement can no longer name what it meant
# to reference. Stated here it reaches the model as a constraint rather than a late refusal.
class ValidationCasePlan(_Draft):
    cases: list[ValidationCaseDraft] = Field(min_length=1)


class MilestoneDraft(_Draft):
    """One thing the assistant must accomplish, in order."""

    description: str
    tool: str | None = Field(default=None, description="Tool the milestone requires, when it requires one")
    requires_confirmation: bool = False


class TaskTemplateDraft(_Draft):
    """One multi-turn task the benchmark will render."""

    template_id: str = _identifier()
    user_goal: str = Field(description="What the user wants, in the user's voice")
    # At least one, because grounding refuses a task that requires no tool and the
    # structured schema is the cheapest place to say so: stated here it reaches the model
    # as a constraint on what it may emit, rather than as a refusal after it has spent a
    # whole plan's worth of output.
    required_tools: list[str] = Field(min_length=1, description="Published tool names, in the order needed")
    milestones: list[MilestoneDraft] = Field(default_factory=list)
    policies: list[str] = Field(default_factory=list)
    blocked_on: list[UnknownField] = Field(default_factory=list)


class TaskTemplatePlan(_Draft):
    templates: list[TaskTemplateDraft] = Field(min_length=1)


class AssertionSpecDraft(_Draft):
    """One declarative predicate a successful task must satisfy."""

    assertion_id: str = _identifier()
    subject: Literal["result", "state", "trace"]
    predicate: Literal[
        "field_present",
        "field_equals_argument",
        "collection_size_changed",
        "tool_called",
        "tool_called_after",
        "tool_not_called",
    ]
    target: str = Field(description="Result field path, state collection, or published tool name")
    argument: str | None = Field(default=None, description="Argument name, for predicates that compare to an input")
    tool: str | None = Field(default=None, description="Published tool name when relevant")
    rationale: str
    blocked_on: list[UnknownField] = Field(default_factory=list)


class AssertionSpecPlan(_Draft):
    assertions: list[AssertionSpecDraft] = Field(min_length=1)
