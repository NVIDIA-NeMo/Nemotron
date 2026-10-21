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

"""Checking a model's draft against the evidence it was given.

An authoring model is useful because it generalizes and dangerous for the same reason. Left
alone it will write `expect.error_code: ACCOUNT_NOT_FOUND` because that is what such a code
usually looks like, and the pack will then certify a wrong claim about a server nobody
probed. The bundle lists that field as unknown; this module is what turns the listing from a
note into a refusal.

The strictest rule is about literals. A drafted argument may only carry a literal value when
the tool's *own* input schema constrains it — an `enum` member or a boolean. Then the value
is grounded in the schema rather than imagined. Every other value has to name its source and
wait for the probe that will supply it, because "a plausible account id" is exactly the kind
of invention that looks like progress and produces a benchmark testing nothing.

Violations are collected rather than raised one at a time: a reviewer reading why a draft was
rejected wants the whole list, not the first item.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nemotron.steps.byob.runtime.pack_authoring.bundle import EvidenceView, ToolEvidence
from nemotron.steps.byob.runtime.pack_authoring.compile_assertions import (
    COMPILABLE_PREDICATES,
)
from nemotron.steps.byob.runtime.pack_authoring.schemas import (
    IDENTIFIER_PATTERN,
    ArgumentPlan,
    AssertionSpecPlan,
    CoveragePlan,
    TaskTemplatePlan,
    ValidationCasePlan,
)
from nemotron.steps.byob.runtime.pack_authoring.untrusted_text import blocking, scan_text

_IDENTIFIER = re.compile(IDENTIFIER_PATTERN)


class GroundingError(Exception):
    """Raised when a draft asserts something the evidence does not support."""

    def __init__(self, stage: str, violations: Sequence[str]) -> None:
        self.stage = stage
        self.violations = tuple(violations)
        super().__init__(
            f"{stage} produced a draft that is not grounded in the evidence bundle:\n"
            + "\n".join(f"  - {item}" for item in violations)
        )


@dataclass(frozen=True)
class Grounding:
    """Everything a draft is allowed to refer to."""

    evidence: EvidenceView

    @property
    def tool_names(self) -> frozenset[str]:
        return self.evidence.tool_names

    @property
    def unresolved(self) -> frozenset[str]:
        return self.evidence.unresolved_unknowns

    @property
    def confirmation_parameter(self) -> str:
        return self.evidence.vocabulary["confirmation_parameter"]

    def tool(self, name: str) -> ToolEvidence | None:
        return self.evidence.tool(name) if name in self.tool_names else None

    def requires(self, field: str) -> bool:
        """True when a draft touching ``field`` must declare itself blocked on it."""
        return field in self.unresolved


def _check_prose(where: str, texts: Iterable[str]) -> list[str]:
    """Refuse model prose that carries text a reviewer cannot see.

    The model is not the untrusted party, but it read untrusted text, and a bidi override
    copied out of a tool description into a policy string defeats review just as well.
    """
    violations: list[str] = []
    for index, text in enumerate(texts):
        for finding in blocking(scan_text(text, f"{where}[{index}]")):
            violations.append(f"{finding.location}: {finding.detail}")
    return violations


def _check_identifier(value: str, where: str) -> list[str]:
    if _IDENTIFIER.match(value) is None:
        return [f"{where}: {value!r} is not a lowercase identifier of 3 to 64 characters"]
    return []


def _check_unique(values: Sequence[str], where: str) -> list[str]:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    return [f"{where}: duplicate identifier {value!r}" for value in duplicates]


def _check_tool(grounding: Grounding, name: str, where: str) -> list[str]:
    if name not in grounding.tool_names:
        return [f"{where}: {name!r} is not a published tool in the evidence bundle ({sorted(grounding.tool_names)})"]
    return []


def _check_blocked_on(
    grounding: Grounding,
    declared: Sequence[str],
    required: Iterable[str],
    where: str,
) -> list[str]:
    violations: list[str] = []
    for field in required:
        if grounding.requires(field) and field not in declared:
            violations.append(
                f"{where}: must declare blocked_on {field!r}, which the bundle lists as "
                "unknown, instead of stating an outcome it cannot know"
            )
    for field in declared:
        if not grounding.requires(field):
            violations.append(f"{where}: blocked_on {field!r} is not an open unknown in this bundle")
    return violations


def validate_coverage_plan(grounding: Grounding, plan: CoveragePlan) -> CoveragePlan:
    """Coverage must name every published tool exactly once and nothing else."""
    violations: list[str] = []
    named = [entry.tool for entry in plan.tools]
    violations.extend(_check_unique(named, "coverage.tools"))
    for entry in plan.tools:
        where = f"coverage.tools[{entry.tool}]"
        violations.extend(_check_tool(grounding, entry.tool, where))
        for dependency in entry.depends_on:
            violations.extend(_check_tool(grounding, dependency, f"{where}.depends_on"))
            if dependency == entry.tool:
                violations.append(f"{where}.depends_on: a tool cannot depend on itself")
        tool = grounding.tool(entry.tool)
        if tool is not None and entry.confirmation_relevant != tool.requires_confirmation:
            violations.append(
                f"{where}.confirmation_relevant contradicts the reviewed profile, which "
                f"declares requires_confirmation={tool.requires_confirmation}"
            )
        violations.extend(
            _check_prose(
                f"{where}.policies",
                [
                    entry.purpose,
                    *entry.policies,
                    *entry.positive_intents,
                    *entry.negative_intents,
                ],
            )
        )
    # Scanned like every other model string: these reach a reviewer, and text that hides
    # itself defeats review wherever it sits.
    violations.extend(_check_prose("coverage.cross_tool_notes", plan.cross_tool_notes))
    missing = sorted(grounding.tool_names - set(named))
    if missing:
        violations.append(
            f"coverage.tools: no coverage for published tool(s) {missing}; every tool in "
            "the benchmark surface has to be covered or explicitly removed from it"
        )
    if violations:
        raise GroundingError("coverage plan", violations)
    return plan


def validate_validation_cases(
    grounding: Grounding,
    plan: ValidationCasePlan,
) -> ValidationCasePlan:
    """Probes may describe an outcome, never assert one the bundle has not observed."""
    violations: list[str] = []
    violations.extend(_check_unique([case.case_id for case in plan.cases], "cases"))
    for case in plan.cases:
        where = f"cases[{case.case_id}]"
        violations.extend(_check_identifier(case.case_id, f"{where}.case_id"))
        violations.extend(_check_tool(grounding, case.tool, where))
        tool = grounding.tool(case.tool)
        violations.extend(
            _check_prose(
                where,
                [
                    case.intent,
                    case.expectation,
                    *(
                        argument.note
                        for argument in case.arguments
                        if argument.note is not None
                    ),
                ],
            )
        )

        required: list[str] = []
        if case.kind == "success":
            # A successful result's content cannot be asserted before it is observed.
            required.append("observed_result_shapes")
        elif case.kind == "error":
            required.append("observed_error_codes")
        else:
            required.append("confirmation_behavior")
            if tool is not None and not tool.requires_confirmation:
                violations.append(
                    f"{where}: kind=confirmation_pending, but the reviewed profile does "
                    f"not gate {case.tool!r} on confirmation"
                )

        argument_names = [argument.name for argument in case.arguments]
        violations.extend(_check_unique(argument_names, f"{where}.arguments"))
        for argument in case.arguments:
            slot = f"{where}.arguments[{argument.name}]"
            if tool is None:
                continue
            if argument.source == "invalid_literal":
                # A value the schema forbids is only a probe when the case expects the
                # tool to refuse it; on a success case it is an invented argument.
                if case.kind != "error":
                    violations.append(
                        f"{slot}: source=invalid_literal belongs to an error probe, and "
                        f"this case is kind={case.kind!r}"
                    )
            if argument.source == "confirmation_flag":
                if argument.name != grounding.confirmation_parameter:
                    violations.append(
                        f"{slot}: source=confirmation_flag is only valid for the pack's "
                        f"confirmation parameter {grounding.confirmation_parameter!r}"
                    )
                if not tool.requires_confirmation:
                    violations.append(f"{slot}: {case.tool!r} is not confirmation gated")
                continue
            if argument.name not in tool.parameter_names:
                violations.append(f"{slot}: {case.tool!r} declares no such parameter ({sorted(tool.parameter_names)})")
                continue
            if argument.source == "literal":
                violations.extend(_check_literal(tool, argument, slot))
            elif argument.source == "invalid_literal":
                violations.extend(_check_invalid_literal(tool, argument, slot))
            elif argument.source in {"fixture", "absent_id"}:
                required.append("fixture_samples")
        if tool is not None:
            omitted = [name for name in tool.required_parameters if name not in argument_names]
            if omitted:
                violations.append(f"{where}.arguments: omits required parameter(s) {omitted}")
        violations.extend(_check_blocked_on(grounding, case.blocked_on, set(required), where))
    if violations:
        raise GroundingError("validation cases", violations)
    return plan


def _check_literal(
    tool: ToolEvidence,
    argument: ArgumentPlan,
    slot: str,
) -> list[str]:
    """A literal is only grounded when the tool's own schema pins the value set."""
    schema = tool.parameters.get("properties", {}).get(argument.name)
    if not isinstance(schema, dict):
        return [f"{slot}: cannot ground a literal against a missing parameter schema"]
    if argument.literal is None:
        return [f"{slot}: source=literal requires a literal value"]
    enum = schema.get("enum")
    if isinstance(enum, list):
        allowed = [str(value) for value in enum]
        if argument.literal not in allowed:
            return [f"{slot}: literal {argument.literal!r} is not in the schema enum {allowed}"]
        return []
    if schema.get("type") == "boolean":
        if argument.literal not in {"true", "false"}:
            return [f"{slot}: boolean literal must be 'true' or 'false'"]
        return []
    return [
        f"{slot}: {argument.name!r} has no enum or boolean type, so a literal here would "
        "be invented domain data; name a fixture source instead"
    ]


def _parses_as(value: str, declared: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return number.is_integer() if declared == "integer" else True


def _check_invalid_literal(
    tool: ToolEvidence,
    argument: ArgumentPlan,
    slot: str,
) -> list[str]:
    """An invalid literal is grounded only where the schema itself refuses the value.

    This is the mirror of `_check_literal` and rests on the same rule: the tool's own input
    schema decides, not the model. Otherwise "the tool rejects this" is a claim about a
    server nobody probed, and the probe passes for whatever reason the source invents.
    """
    schema = tool.parameters.get("properties", {}).get(argument.name)
    if not isinstance(schema, dict):
        return [
            f"{slot}: cannot ground an invalid literal against a missing parameter schema"
        ]
    if argument.literal is None:
        return [f"{slot}: source=invalid_literal requires the value the tool must reject"]
    enum = schema.get("enum")
    if isinstance(enum, list):
        if argument.literal in [str(value) for value in enum]:
            return [
                f"{slot}: {argument.literal!r} is in the schema enum {sorted(str(value) for value in enum)}, "
                "so the tool has no declared reason to reject it"
            ]
        return []
    declared = schema.get("type")
    if declared == "boolean":
        if argument.literal in {"true", "false"}:
            return [
                f"{slot}: {argument.literal!r} is a valid boolean, so the tool has no "
                "declared reason to reject it"
            ]
        return []
    if declared in {"integer", "number"}:
        if _parses_as(argument.literal, str(declared)):
            return [
                f"{slot}: {argument.literal!r} is a valid {declared}, so the tool has no "
                "declared reason to reject it"
            ]
        return []
    return [
        f"{slot}: the schema for {argument.name!r} pins no enum, boolean, or numeric type, "
        "so nothing in it makes this value invalid; name a fixture or absent_id source"
    ]


def validate_task_templates(
    grounding: Grounding,
    plan: TaskTemplatePlan,
) -> TaskTemplatePlan:
    """Templates may plan an ordering, but their slot values still come from fixtures."""
    violations: list[str] = []
    violations.extend(_check_unique([template.template_id for template in plan.templates], "templates"))
    for template in plan.templates:
        where = f"templates[{template.template_id}]"
        violations.extend(_check_identifier(template.template_id, f"{where}.template_id"))
        violations.extend(_check_prose(where, [template.user_goal, *template.policies]))
        if not template.required_tools:
            violations.append(f"{where}: a task with no required tool tests nothing")
        for name in template.required_tools:
            violations.extend(_check_tool(grounding, name, f"{where}.required_tools"))
        for index, milestone in enumerate(template.milestones):
            slot = f"{where}.milestones[{index}]"
            violations.extend(_check_prose(slot, [milestone.description]))
            if milestone.tool is None:
                continue
            if milestone.tool not in template.required_tools:
                violations.append(f"{slot}: tool {milestone.tool!r} is not in required_tools")
                continue
            tool = grounding.tool(milestone.tool)
            if tool is not None and milestone.requires_confirmation and not tool.requires_confirmation:
                violations.append(f"{slot}: the reviewed profile does not gate {milestone.tool!r} on confirmation")
        # Slots need values, and ordering across tools needs an observed dependency.
        required = {"fixture_samples"}
        if len(template.required_tools) > 1:
            required.add("tool_dependencies")
        violations.extend(_check_blocked_on(grounding, template.blocked_on, required, where))
    if violations:
        raise GroundingError("task templates", violations)
    return plan


_RESULT_PREDICATES = {"field_present", "field_equals_argument"}
_STATE_PREDICATES = {"collection_size_changed"}
_TRACE_PREDICATES = {"tool_called", "tool_called_after", "tool_not_called"}


def validate_assertion_specs(
    grounding: Grounding,
    plan: AssertionSpecPlan,
) -> AssertionSpecPlan:
    """Trace predicates are grounded in BFCL's own record; result and state ones are not."""
    violations: list[str] = []
    violations.extend(_check_unique([spec.assertion_id for spec in plan.assertions], "assertions"))
    for spec in plan.assertions:
        where = f"assertions[{spec.assertion_id}]"
        violations.extend(_check_identifier(spec.assertion_id, f"{where}.assertion_id"))
        violations.extend(_check_prose(where, [spec.rationale]))

        expected_subject = {
            **{name: "result" for name in _RESULT_PREDICATES},
            **{name: "state" for name in _STATE_PREDICATES},
            **{name: "trace" for name in _TRACE_PREDICATES},
        }[spec.predicate]
        if spec.subject != expected_subject:
            violations.append(
                f"{where}: predicate {spec.predicate!r} applies to the {expected_subject}, not the {spec.subject}"
            )

        required: set[str] = set()
        if spec.predicate not in COMPILABLE_PREDICATES:
            # Grounding and the compiler have to agree on what a specification can become.
            # The compiler reads only the call trace BFCL records, and it refuses per pack
            # rather than per specification, so one predicate over a result field or over
            # oracle state costs a whole drafting run every assertion it produced.
            #
            # Treating this as a missing observation instead was the weaker reading: on a
            # bundle still listing the unknown it asked for a blocker the compiler then
            # refused, and on one that listed nothing it asked for nothing and let the
            # specification through to fail at compilation. Neither tier can express the
            # predicate, so the refusal belongs here, beside the rest of the violations.
            violations.append(
                f"{where}: predicate {spec.predicate!r} cannot become an executable "
                f"assertion; only {sorted(COMPILABLE_PREDICATES)} read the call trace, "
                "and a pack compiles all of its specifications or none of them"
            )
        elif spec.predicate == "tool_called_after":
            required.add("tool_dependencies")

        if spec.predicate in _TRACE_PREDICATES:
            violations.extend(_check_tool(grounding, spec.target, f"{where}.target"))
        if spec.predicate == "tool_called_after":
            if spec.tool is None:
                violations.append(f"{where}: tool_called_after needs the earlier tool")
            else:
                violations.extend(_check_tool(grounding, spec.tool, f"{where}.tool"))
                if spec.tool == spec.target:
                    violations.append(f"{where}: a tool cannot be required to run after itself")
        if spec.predicate == "field_equals_argument":
            if spec.argument is None or spec.tool is None:
                violations.append(f"{where}: field_equals_argument needs both tool and argument")
            else:
                violations.extend(_check_tool(grounding, spec.tool, f"{where}.tool"))
                tool = grounding.tool(spec.tool)
                if tool is not None and spec.argument not in tool.parameter_names:
                    violations.append(f"{where}.argument: {spec.tool!r} declares no parameter {spec.argument!r}")
        violations.extend(_check_blocked_on(grounding, spec.blocked_on, required, where))
    if violations:
        raise GroundingError("assertion specifications", violations)
    return plan
