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

"""The four authoring calls, each validated against the evidence before it is believed.

Every generator has the same shape: build the fenced payload, make one cached structured
call, parse the response into its schema, then hand it to `grounding.py`. A response that
fails grounding raises rather than being repaired, and nothing retries. A retry loop here
would quietly reward whichever attempt happened to pass, which is how an unreviewed claim
gets into a pack; a refusal with the full list of violations is something a human can act on.

The order is a dependency, not a preference. Coverage decides what the benchmark is trying
to exercise, and the other three refer to it, so it is produced first and passed forward as
input rather than being re-derived three times with three chances to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from nemotron.steps.byob.runtime.authoring_workflow.quota import RunQuota
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import EvidenceView
from nemotron.steps.byob.runtime.pack_authoring.grounding import (
    Grounding,
    GroundingError,
    validate_assertion_specs,
    validate_coverage_plan,
    validate_task_templates,
    validate_validation_cases,
)
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    ModelCallRecord,
    StructuredCaller,
    call_structured,
)
from nemotron.steps.byob.runtime.pack_authoring.prompts import (
    ASSERTION_PROMPT_VERSION,
    ASSERTION_TASK,
    AUTHORING_SYSTEM_PROMPT,
    COVERAGE_PROMPT_VERSION,
    COVERAGE_TASK,
    TASK_TEMPLATE_PROMPT_VERSION,
    TASK_TEMPLATE_TASK,
    VALIDATION_CASE_PROMPT_VERSION,
    VALIDATION_CASE_TASK,
    build_evidence_payload,
)
from nemotron.steps.byob.runtime.pack_authoring.schemas import (
    AssertionSpecPlan,
    CoveragePlan,
    TaskTemplatePlan,
    ValidationCasePlan,
)


@dataclass(frozen=True)
class DraftBundle:
    """Every drafted artifact, plus the record of the calls that produced them."""

    coverage: CoveragePlan
    validation_cases: ValidationCasePlan
    task_templates: TaskTemplatePlan
    assertions: AssertionSpecPlan
    calls: tuple[ModelCallRecord, ...]

    def as_documents(self) -> dict[str, Any]:
        return {
            "coverage_plan": self.coverage.model_dump(mode="json"),
            "validation_cases": self.validation_cases.model_dump(mode="json"),
            "task_templates": self.task_templates.model_dump(mode="json"),
            "assertion_specs": self.assertions.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class DraftingContext:
    """What every generator needs, gathered so each call site stays about its own stage."""

    evidence: EvidenceView
    model: AuthoringModel
    cache: ImmutableModelIOCache
    run_dir: Path
    caller: StructuredCaller | None = None
    quota: RunQuota | None = None

    @property
    def grounding(self) -> Grounding:
        return Grounding(evidence=self.evidence)


def _parse(
    stage: str,
    response: dict[str, Any],
    output_format: type[BaseModel],
) -> Any:
    try:
        return output_format.model_validate(response)
    except ValidationError as exc:
        # The structured column already constrains the shape, so this fires when a provider
        # returns something outside its own schema. It is a grounding failure in spirit.
        raise GroundingError(stage, [f"response does not satisfy {output_format.__name__}: {exc}"]) from exc


def _call(
    context: DraftingContext,
    *,
    stage: str,
    prompt_version: str,
    task: str,
    columns: dict[str, str],
    output_format: type[BaseModel],
) -> tuple[Any, ModelCallRecord]:
    response, record = call_structured(
        context.model,
        stage_name=stage,
        prompt_version=prompt_version,
        system_prompt=AUTHORING_SYSTEM_PROMPT,
        prompt=task,
        columns=columns,
        output_format=output_format,
        cache=context.cache,
        run_dir=context.run_dir,
        caller=context.caller,
        quota=context.quota,
    )
    return _parse(stage, response, output_format), record


def draft_coverage_plan(
    context: DraftingContext,
) -> tuple[CoveragePlan, ModelCallRecord]:
    """What the benchmark should exercise, for every published tool."""
    plan, record = _call(
        context,
        stage="mcp_coverage_plan",
        prompt_version=COVERAGE_PROMPT_VERSION,
        task=COVERAGE_TASK,
        columns={"evidence": build_evidence_payload(context.evidence)},
        output_format=CoveragePlan,
    )
    return validate_coverage_plan(context.grounding, plan), record


def draft_validation_cases(
    context: DraftingContext,
    coverage: CoveragePlan,
) -> tuple[ValidationCasePlan, ModelCallRecord]:
    """Probes proving each tool behaves as the pack will claim."""
    plan, record = _call(
        context,
        stage="mcp_validation_cases",
        prompt_version=VALIDATION_CASE_PROMPT_VERSION,
        task=VALIDATION_CASE_TASK,
        columns={
            "evidence": build_evidence_payload(context.evidence),
            "coverage": coverage.model_dump_json(indent=2),
        },
        output_format=ValidationCasePlan,
    )
    return validate_validation_cases(context.grounding, plan), record


def draft_task_templates(
    context: DraftingContext,
    coverage: CoveragePlan,
) -> tuple[TaskTemplatePlan, ModelCallRecord]:
    """The multi-turn tasks, their ordering, and their milestones."""
    plan, record = _call(
        context,
        stage="mcp_task_templates",
        prompt_version=TASK_TEMPLATE_PROMPT_VERSION,
        task=TASK_TEMPLATE_TASK,
        columns={
            "evidence": build_evidence_payload(context.evidence),
            "coverage": coverage.model_dump_json(indent=2),
        },
        output_format=TaskTemplatePlan,
    )
    return validate_task_templates(context.grounding, plan), record


def draft_assertion_specs(
    context: DraftingContext,
    coverage: CoveragePlan,
) -> tuple[AssertionSpecPlan, ModelCallRecord]:
    """Declarative predicates over result, state, and trace."""
    plan, record = _call(
        context,
        stage="mcp_assertion_specs",
        prompt_version=ASSERTION_PROMPT_VERSION,
        task=ASSERTION_TASK,
        columns={
            "evidence": build_evidence_payload(context.evidence),
            "coverage": coverage.model_dump_json(indent=2),
        },
        output_format=AssertionSpecPlan,
    )
    return validate_assertion_specs(context.grounding, plan), record


def draft_all(context: DraftingContext) -> DraftBundle:
    """Run the four calls in dependency order."""
    coverage, coverage_record = draft_coverage_plan(context)
    cases, cases_record = draft_validation_cases(context, coverage)
    templates, templates_record = draft_task_templates(context, coverage)
    assertions, assertions_record = draft_assertion_specs(context, coverage)
    return DraftBundle(
        coverage=coverage,
        validation_cases=cases,
        task_templates=templates,
        assertions=assertions,
        calls=(coverage_record, cases_record, templates_record, assertions_record),
    )
