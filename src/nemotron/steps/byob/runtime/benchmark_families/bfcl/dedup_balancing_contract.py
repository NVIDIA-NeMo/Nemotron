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

"""Versioned Stage 11 contract for semantic deduplication and balancing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

DEDUP_BALANCING_CONTRACT_VERSION = "1.0"

# These names are the complete publication-balancing surface. Implementations may
# derive helper features, but manifest targets and reports must map back to one of
# these dimensions rather than introducing pack-specific columns.
BALANCING_DIMENSIONS = (
    "intent",
    "category",
    "required_tools",
    "tools_present",
    "difficulty",
    "turn_class",
    "tool_call_count",
    "turn_policy",
)
BalancingDimensionName = Literal[
    "intent",
    "category",
    "required_tools",
    "tools_present",
    "difficulty",
    "turn_class",
    "tool_call_count",
    "turn_policy",
]
# The closed conversation-policy vocabulary. It lives with the balancing contract
# because coverage buckets and declared policy mixes are both keyed on these names, so
# a pack, a target mix, and a coverage report cannot drift apart.
TURN_POLICIES = frozenset(
    {
        "single_turn",
        "missing_slot",
        "confirmation",
        "correction",
        "multi_tool",
        "dependent_call",
        "negative_path",
        "clarify_only",
        "irrelevant",
    }
)
Stage11DropReason = Literal[
    "semantic_duplicate",
    "balance_quota",
    "max_turns_exceeded",
    "max_tool_calls_exceeded",
]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class Stage11Coverage(BaseModel):
    """Generic coverage buckets that one Stage 10 survivor can preserve.

    Deduplication may only collapse tasks that share a bucket, so these values
    also partition clusters. Values are normalized because bucket identity is
    string equality: an untrimmed variant would otherwise become a phantom
    bucket that no selected row can preserve.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    turn_policy: str
    edge_signatures: tuple[str, ...] = ()

    @field_validator("language", "turn_policy")
    @classmethod
    def normalize_bucket_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Stage 11 coverage keys must be non-empty")
        return normalized

    @field_validator("edge_signatures")
    @classmethod
    def normalize_edge_signatures(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(signature.strip() for signature in value)
        if any(not signature for signature in normalized):
            raise ValueError("Stage 11 edge signatures must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Stage 11 edge signatures must be unique per task")
        return tuple(sorted(normalized))


class DedupBalancingDecision(BaseModel):
    """One immutable Stage 11 decision for one Stage 10 survivor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0"] = DEDUP_BALANCING_CONTRACT_VERSION
    task_id: str
    selected: StrictBool
    is_duplicate: StrictBool
    duplicate_cluster_id: str | None = None
    representative_task_id: str | None = None
    drop_reason: Stage11DropReason | None = None
    balance_dimension: BalancingDimensionName | None = None
    selection_rank: NonNegativeInt

    @field_validator("task_id", "duplicate_cluster_id", "representative_task_id")
    @classmethod
    def normalize_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Stage 11 identifiers must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_decision(self) -> DedupBalancingDecision:
        if (self.duplicate_cluster_id is None) != (self.representative_task_id is None):
            raise ValueError("duplicate_cluster_id and representative_task_id must be set together")
        if self.is_duplicate:
            if self.duplicate_cluster_id is None:
                raise ValueError("a duplicate Stage 11 row requires cluster and representative ids")
            if self.representative_task_id == self.task_id:
                raise ValueError("a duplicate row cannot represent itself")
        elif self.representative_task_id is not None and self.representative_task_id != self.task_id:
            raise ValueError("a non-duplicate cluster representative must represent itself")
        if self.selected:
            if self.drop_reason is not None or self.balance_dimension is not None:
                raise ValueError("a selected Stage 11 row cannot carry drop detail")
            return self
        if self.drop_reason is None:
            raise ValueError("a dropped Stage 11 row requires drop_reason")
        if self.drop_reason == "semantic_duplicate" and not self.is_duplicate:
            raise ValueError("semantic_duplicate is valid only for a duplicate row")
        if self.drop_reason == "balance_quota":
            if self.balance_dimension is None:
                raise ValueError("a balance_quota drop requires balance_dimension")
        elif self.balance_dimension is not None:
            raise ValueError("balance_dimension is valid only for a balance_quota drop")
        return self


def validate_complete_decision_set(
    values: Sequence[DedupBalancingDecision | Mapping[str, object]],
    *,
    input_task_ids: Sequence[str],
    coverage_by_task_id: Mapping[str, Stage11Coverage | Mapping[str, object]],
    remove_duplicates: bool,
) -> list[DedupBalancingDecision]:
    """Validate one policy-consistent decision per Stage 10 input.

    A representative may itself be dropped, because balancing is allowed to drop
    a whole cluster on quota; the invariant is that a cluster has exactly one
    representative and never spans coverage buckets.
    """
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in input_task_ids):
        raise ValueError("Stage 11 input task_id values must be non-empty strings")
    if type(remove_duplicates) is not bool:
        raise ValueError("Stage 11 remove_duplicates must be a boolean")
    expected = [task_id.strip() for task_id in input_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("Stage 11 input task_id values must be unique after normalization")
    normalized_coverage: dict[str, Stage11Coverage | Mapping[str, object]] = {}
    for raw_task_id, value in coverage_by_task_id.items():
        if not isinstance(raw_task_id, str) or not raw_task_id.strip():
            raise ValueError("Stage 11 coverage task_id keys must be non-empty strings")
        task_id = raw_task_id.strip()
        if task_id in normalized_coverage:
            raise ValueError(f"duplicate Stage 11 coverage for task {task_id!r} after normalization")
        normalized_coverage[task_id] = value
    coverage_keys = set(normalized_coverage)
    expected_keys = set(expected)
    if coverage_keys != expected_keys:
        missing = [task_id for task_id in expected if task_id not in coverage_keys]
        extra = sorted(coverage_keys - expected_keys)
        raise ValueError(f"Stage 11 coverage must match inputs exactly (missing={missing}, extra={extra})")
    coverage = {
        task_id: (value if isinstance(value, Stage11Coverage) else Stage11Coverage.model_validate(value))
        for task_id, value in normalized_coverage.items()
    }
    decisions = [
        value if isinstance(value, DedupBalancingDecision) else DedupBalancingDecision.model_validate(value)
        for value in values
    ]
    by_task: dict[str, DedupBalancingDecision] = {}
    for decision in decisions:
        if decision.task_id in by_task:
            raise ValueError(f"duplicate Stage 11 decision for task {decision.task_id!r}")
        by_task[decision.task_id] = decision
    missing = [task_id for task_id in expected if task_id not in by_task]
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Stage 11 decisions must match Stage 10 survivors exactly (missing={missing}, extra={extra})"
        )
    members_by_cluster: dict[str, list[DedupBalancingDecision]] = {}
    for decision in decisions:
        representative_id = decision.representative_task_id
        if representative_id is not None:
            representative = by_task.get(representative_id)
            if representative is None:
                raise ValueError(
                    f"task {decision.task_id!r} references representative {representative_id!r} outside Stage 11"
                )
            if representative.is_duplicate:
                raise ValueError(
                    f"task {decision.task_id!r} references duplicate representative {representative_id!r}"
                )
            if representative.duplicate_cluster_id != decision.duplicate_cluster_id:
                raise ValueError(
                    f"task {decision.task_id!r} and representative {representative_id!r} "
                    "must carry the same cluster metadata"
                )
        if decision.duplicate_cluster_id is not None:
            members_by_cluster.setdefault(decision.duplicate_cluster_id, []).append(decision)
        if remove_duplicates and decision.is_duplicate:
            if decision.selected or decision.drop_reason != "semantic_duplicate":
                raise ValueError(
                    "remove_duplicates=true requires every duplicate to be dropped with semantic_duplicate"
                )
        elif not remove_duplicates and decision.drop_reason == "semantic_duplicate":
            raise ValueError("remove_duplicates=false cannot drop a row as semantic_duplicate")

    for cluster_id, members in sorted(members_by_cluster.items()):
        representatives = [member for member in members if not member.is_duplicate]
        if len(representatives) != 1:
            raise ValueError(
                f"cluster {cluster_id!r} must carry exactly one representative, got {len(representatives)}"
            )
        representative = representatives[0]
        if any(member.selected for member in members) and not representative.selected:
            raise ValueError(
                f"cluster {cluster_id!r} has selected members but its representative "
                f"{representative.task_id!r} is dropped"
            )
        bucket = coverage[representative.task_id]
        for member in members:
            if coverage[member.task_id] != bucket:
                raise ValueError(
                    f"cluster {cluster_id!r} may only group tasks that share one coverage bucket; "
                    f"task {member.task_id!r} does not share the representative bucket"
                )

    selected_ranks = sorted(decision.selection_rank for decision in decisions if decision.selected)
    if selected_ranks != list(range(len(selected_ranks))):
        raise ValueError(
            "selected Stage 11 rows must carry selection_rank 0..k-1 exactly once so publication order is total"
        )

    selected_ids = {decision.task_id for decision in decisions if decision.selected}
    required_buckets = {(item.language, item.turn_policy, item.edge_signatures) for item in coverage.values()}
    preserved_buckets = {
        (
            coverage[task_id].language,
            coverage[task_id].turn_policy,
            coverage[task_id].edge_signatures,
        )
        for task_id in selected_ids
    }
    if lost_buckets := sorted(required_buckets - preserved_buckets):
        language, turn_policy, edge_signatures = lost_buckets[0]
        raise ValueError(
            "Stage 11 must preserve at least one survivor for coverage bucket "
            f"(language={language!r}, turn_policy={turn_policy!r}, "
            f"edge_signatures={edge_signatures!r})"
        )
    return [by_task[task_id] for task_id in expected]
