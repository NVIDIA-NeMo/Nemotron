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

"""What a candidate is allowed to be scored on (schema 1.0).

Source verification proved which benchmark is being evaluated. This module
defines the object
that says *which rows of it each candidate may answer*, and why.

The shape follows three rules.

*A collision is evidence, not a flag.* Every :class:`ContaminationCollision`
names the role, the model, and the exact task ids involved, so a refusal or an
exclusion can be read back later. "Contaminated: true" would be unauditable.

*An unproven separation is recorded, never assumed away.* A comparison that
cannot settle whether the candidate is the exposed model produces an ``unknown``
collision. Those never silently shrink a task set, and they never let a run
claim publishability.

*The comparable task set is part of the plan, not of the runner.* Under
``common_intersection`` every candidate answers the same rows, chosen here and
hashed, so two scores are comparable by construction rather than by convention.
:attr:`EligibleEvalPlan.plan_identity` covers all of it and no path or timestamp,
so the same decision on another host is the same plan.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    ExposureRole,
    ExposureScope,
    ModelIdentityClaim,
    VerifiedModelExposure,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import SourceCheck
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import ContentHash
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

CONTAMINATION_CONTRACT_VERSION: Final = "1.0"

CONTAMINATION_REPORT_FILE: Final = "contamination_report.json"
# As with source verification, a refusal is written under its own name: a reader
# who finds one file must not have to parse it to learn the verdict.
CONTAMINATION_FAILURE_FILE: Final = "contamination_failure.json"

# Only two comparison outcomes are collisions. ``different`` means the candidate
# is provably not the exposed model, which is the normal case and is not
# recorded: a report listing every model a candidate is *not* would bury the one
# that matters.
CollisionVerdict = Literal["match", "unknown"]

# What a candidate is scored on, once contamination has been applied.
ComparisonSet = Literal["common_intersection", "per_candidate"]


def _sha256_json(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


class _Verified(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContaminationCollision(_Verified):
    """One candidate against one exposure, where separation was not established."""

    schema_version: Literal["1.0"] = CONTAMINATION_CONTRACT_VERSION
    role: ExposureRole
    scope: ExposureScope
    verdict: CollisionVerdict
    exposed_model: StrictStr
    reason: StrictStr
    task_ids: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def _covers_rows(self) -> ContaminationCollision:
        if not self.task_ids:
            raise ValueError("a collision over no row is not a collision")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("collision task ids must be unique")
        return self

    @property
    def task_ids_hash(self) -> str:
        return _sha256_json(list(self.task_ids))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "scope": self.scope,
            "verdict": self.verdict,
            "exposed_model": self.exposed_model,
            "reason": self.reason,
            "task_count": len(self.task_ids),
            "task_ids_hash": self.task_ids_hash,
        }

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "task_ids": list(self.task_ids)}


class CandidateEligibility(_Verified):
    """The rows one candidate may answer, and the evidence behind the decision."""

    schema_version: Literal["1.0"] = CONTAMINATION_CONTRACT_VERSION
    alias: StrictStr
    identity: ModelIdentityClaim
    canonical_model_identity: StrictStr
    collisions: tuple[ContaminationCollision, ...] = ()
    eligible_task_ids: tuple[StrictStr, ...]
    excluded_task_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> CandidateEligibility:
        if not self.eligible_task_ids:
            raise ValueError("a candidate with no eligible task cannot be scored")
        for label, ids in (
            ("eligible_task_ids", self.eligible_task_ids),
            ("excluded_task_ids", self.excluded_task_ids),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{label} must be unique")
        if overlap := sorted(set(self.eligible_task_ids) & set(self.excluded_task_ids)):
            raise ValueError(f"task(s) {overlap[:3]} are both eligible and excluded")
        return self

    @property
    def definite_collisions(self) -> tuple[ContaminationCollision, ...]:
        return tuple(collision for collision in self.collisions if collision.verdict == "match")

    @property
    def unresolved_collisions(self) -> tuple[ContaminationCollision, ...]:
        return tuple(collision for collision in self.collisions if collision.verdict == "unknown")

    @property
    def exposed(self) -> bool:
        return bool(self.definite_collisions)

    @property
    def unresolved(self) -> bool:
        return bool(self.unresolved_collisions)

    @property
    def eligible_task_ids_hash(self) -> str:
        return _sha256_json(list(self.eligible_task_ids))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "identity": self.identity.semantic_payload(),
            "canonical_model_identity": self.canonical_model_identity,
            "collisions": [collision.semantic_payload() for collision in self.collisions],
            "eligible_task_count": len(self.eligible_task_ids),
            "eligible_task_ids_hash": self.eligible_task_ids_hash,
            "excluded_task_count": len(self.excluded_task_ids),
        }

    def as_document(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "collisions": [collision.as_document() for collision in self.collisions],
            "exposed": self.exposed,
            "unresolved": self.unresolved,
            "eligible_task_ids": list(self.eligible_task_ids),
            "excluded_task_ids": list(self.excluded_task_ids),
        }


class CommonEvaluationTaskSet(_Verified):
    """The rows every candidate can answer, in publication order.

    This is computed even under ``per_candidate``, where it is not what gets
    scored: knowing how much of the benchmark the candidates still share is how
    an operator sees what contamination cost the comparison.
    """

    schema_version: Literal["1.0"] = CONTAMINATION_CONTRACT_VERSION
    comparison_set: ComparisonSet
    task_ids: tuple[StrictStr, ...]
    candidate_aliases: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def _comparable(self) -> CommonEvaluationTaskSet:
        if not self.task_ids:
            raise ValueError("candidates that share no task cannot be compared")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("common task ids must be unique")
        if not self.candidate_aliases:
            raise ValueError("a comparison covers at least one candidate")
        if len(set(self.candidate_aliases)) != len(self.candidate_aliases):
            raise ValueError("candidate aliases must be unique")
        return self

    @property
    def task_count(self) -> int:
        return len(self.task_ids)

    @property
    def task_ids_hash(self) -> str:
        return _sha256_json(list(self.task_ids))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "comparison_set": self.comparison_set,
            "task_count": self.task_count,
            "task_ids_hash": self.task_ids_hash,
            "candidate_aliases": list(self.candidate_aliases),
        }

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "task_ids": list(self.task_ids)}


class EligibleEvalPlan(_Verified):
    """The authorization a runner needs: who answers which rows, under which policy.

    Like :class:`...source_contract.VerifiedEvalSource`, this is a handle rather
    than a description. A runner that is given one cannot ask a candidate a task
    the gate excluded, because the only task list it has is the one on this
    object.
    """

    schema_version: Literal["1.0"] = CONTAMINATION_CONTRACT_VERSION
    eval_config_hash: ContentHash
    scoring_policy_hash: ContentHash
    source_verification_identity: ContentHash
    source_run_id: StrictStr
    source_task_ids_hash: ContentHash
    enforce: StrictBool
    on_violation: Literal["fail_run", "exclude_row"]
    comparison_set: ComparisonSet
    exposures: tuple[VerifiedModelExposure, ...] = ()
    candidates: tuple[CandidateEligibility, ...]
    common: CommonEvaluationTaskSet
    publication_allowed: StrictBool
    non_publication_reasons: tuple[StrictStr, ...] = ()
    checks: tuple[SourceCheck, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> EligibleEvalPlan:
        aliases = tuple(candidate.alias for candidate in self.candidates)
        if not aliases:
            raise ValueError("a plan covers at least one candidate")
        if len(set(aliases)) != len(aliases):
            raise ValueError("candidate aliases must be unique")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("candidates are ordered by alias so the plan identity does not depend on YAML order")
        if self.common.candidate_aliases != aliases:
            raise ValueError("the common task set must cover exactly the candidates in the plan")
        if self.common.comparison_set != self.comparison_set:
            raise ValueError("the common task set must be derived under the plan's comparison policy")
        shared = set(self.common.task_ids)
        for candidate in self.candidates:
            if tuple(task_id for task_id in candidate.eligible_task_ids if task_id in shared) != (
                self.common.task_ids
            ):
                raise ValueError(
                    f"candidate {candidate.alias} does not carry the common task set in publication order"
                )
        if self.publication_allowed:
            if self.non_publication_reasons:
                raise ValueError("a publishable plan cannot also state why it is not publishable")
            if any(candidate.exposed or candidate.unresolved for candidate in self.candidates):
                raise ValueError("a plan with an unresolved or exposed candidate is not publishable")
            if self.comparison_set != "common_intersection":
                raise ValueError("a publishable comparison scores every candidate on the same task set")
        return self

    @property
    def candidate_aliases(self) -> tuple[str, ...]:
        return tuple(candidate.alias for candidate in self.candidates)

    def candidate(self, alias: str) -> CandidateEligibility:
        for candidate in self.candidates:
            if candidate.alias == alias:
                return candidate
        raise KeyError(alias)

    def evaluation_task_ids(self, alias: str) -> tuple[str, ...]:
        """The rows this candidate is to be asked, in publication order.

        Under ``common_intersection`` that is the shared set, so the numbers are
        comparable. Under ``per_candidate`` it is the candidate's own eligible
        set, which is why such a run is never publishable.
        """
        candidate = self.candidate(alias)
        if self.comparison_set == "common_intersection":
            return self.common.task_ids
        return candidate.eligible_task_ids

    @property
    def excluded_any_row(self) -> bool:
        return any(candidate.excluded_task_ids for candidate in self.candidates)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "eval_config_hash": self.eval_config_hash,
            "scoring_policy_hash": self.scoring_policy_hash,
            "source_verification_identity": self.source_verification_identity,
            "source_run_id": self.source_run_id,
            "source_task_ids_hash": self.source_task_ids_hash,
            "policy": {
                "enforce": self.enforce,
                "on_violation": self.on_violation,
                "comparison_set": self.comparison_set,
            },
            "exposures": [exposure.semantic_payload() for exposure in self.exposures],
            "candidates": [candidate.semantic_payload() for candidate in self.candidates],
            "common": self.common.semantic_payload(),
            "publication_allowed": self.publication_allowed,
            "non_publication_reasons": list(self.non_publication_reasons),
        }

    @property
    def plan_identity(self) -> str:
        """One hash for "these candidates, these rows, under this policy"."""
        return _sha256_json(self.semantic_payload())

    def as_document(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "plan_identity": self.plan_identity,
            "exposures": [exposure.as_document() for exposure in self.exposures],
            "candidates": [candidate.as_document() for candidate in self.candidates],
            "common": self.common.as_document(),
            "checks": [check.as_document() for check in self.checks],
        }


class ContaminationReport(_Verified):
    """The auditable artifact a passing contamination gate writes."""

    schema_version: Literal["1.0"] = CONTAMINATION_CONTRACT_VERSION
    status: Literal["passed"] = "passed"
    decided_at: StrictStr
    plan: EligibleEvalPlan

    @property
    def plan_identity(self) -> str:
        return self.plan.plan_identity

    def as_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "decided_at": self.decided_at,
            **self.plan.as_document(),
        }
