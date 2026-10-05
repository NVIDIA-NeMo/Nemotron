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

"""Versioned held-out contract shared by Stage 4 binding and Stage 12 publication.

Held-out is a leakage claim, not a generalization claim: the release must not
bind the fixture rows and templates a pack reserved. Both stages therefore read
one immutable policy and compare the same reference strings, because a binding
Stage 4 blocks and a row Stage 12 scans have to mean the same thing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

HELD_OUT_CONTRACT_VERSION = "1.0"


def fixture_ref(collection: str, primary_id: Any) -> str:
    """Render the reference Stage 4 records for one bound fixture row.

    Expansion stores canonical JSON ``[collection, primary_id]`` on every task, and the
    published row carries it unchanged, so this is the only join key the two
    stages need to agree on.
    """
    normalized_collection = str(collection)
    normalized_id = str(primary_id)
    if not normalized_collection.strip() or not normalized_id.strip():
        raise ValueError("a held-out fixture reference requires a collection and a primary id")
    # A delimiter-joined string is ambiguous when either component contains the
    # delimiter. Canonical JSON preserves the exact pair while remaining a stable
    # string for task identity and parquet lineage.
    return json.dumps(
        [normalized_collection, normalized_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@lru_cache(maxsize=128)
def _reference_index(references: tuple[str, ...]) -> frozenset[str]:
    """Index one policy's references once; Stage 4 consults it per candidate row."""
    return frozenset(references)


def _normalized_strings(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value.strip() for value in normalized):
        raise ValueError(f"held-out {label} must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"held-out {label} must be unique after normalization")
    return tuple(sorted(normalized))


class HeldOutPolicy(BaseModel):
    """The reserved fixture rows and templates one run is pinned to.

    The policy is flattened to reference strings on construction so matching is
    set membership rather than a per-collection walk: Stage 4 consults it once
    per candidate row, and Stage 12 once per published row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0"] = HELD_OUT_CONTRACT_VERSION
    version: str
    fixture_refs: tuple[str, ...] = ()
    template_ids: tuple[str, ...] = ()
    fixtures_in_backend_state: StrictBool = True
    seed: StrictInt = 0
    source: str | None = None

    @field_validator("version")
    @classmethod
    def normalize_version(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("held-out policy version must be non-empty")
        return normalized

    @field_validator("fixture_refs")
    @classmethod
    def normalize_fixture_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = _normalized_strings(value, label="fixture references")
        for reference in normalized:
            try:
                pair = json.loads(reference)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"held-out fixture reference {reference!r} must be a canonical JSON pair"
                ) from exc
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(part, str) or not part.strip() for part in pair)
                or fixture_ref(pair[0], pair[1]) != reference
            ):
                raise ValueError(
                    f"held-out fixture reference {reference!r} must be a canonical JSON pair"
                )
        return normalized

    @field_validator("template_ids")
    @classmethod
    def normalize_template_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalized_strings(value, label="template ids")

    @classmethod
    def from_normalized(cls, policy: Mapping[str, Any]) -> HeldOutPolicy:
        """Build the contract from the pack loader's normalized held-out policy."""
        if not isinstance(policy, Mapping):
            raise ValueError("a held-out policy must be a mapping")
        fixtures = policy.get("fixtures") or {}
        if not isinstance(fixtures, Mapping):
            raise ValueError("held-out policy fixtures must be a mapping")
        references: list[str] = []
        for collection, identifiers in fixtures.items():
            if isinstance(identifiers, str) or not isinstance(identifiers, Sequence):
                raise ValueError(f"held-out policy fixtures.{collection} must be a list")
            references.extend(fixture_ref(collection, identifier) for identifier in identifiers)
        templates = policy.get("templates") or []
        if isinstance(templates, str) or not isinstance(templates, Sequence):
            raise ValueError("held-out policy templates must be a list")
        settings = policy.get("policy") or {}
        if not isinstance(settings, Mapping):
            raise ValueError("held-out policy settings must be a mapping")
        source = policy.get("source")
        return cls(
            version=str(policy.get("version", "")),
            fixture_refs=tuple(references),
            template_ids=tuple(str(template) for template in templates),
            fixtures_in_backend_state=bool(settings.get("fixtures_in_backend_state", True)),
            seed=int(settings.get("seed", 0)),
            source=str(source) if source is not None else None,
        )

    @property
    def reserves_nothing(self) -> bool:
        """Report a policy that declares no fixture row and no template.

        Such a policy is still enforced and still recorded, because "scanned and
        found nothing" is a different claim from "never scanned".
        """
        return not self.fixture_refs and not self.template_ids

    def blocks_template(self, template_id: Any) -> bool:
        return str(template_id) in _reference_index(self.template_ids)

    def blocks_fixture(self, reference: str) -> bool:
        return str(reference) in _reference_index(self.fixture_refs)

    def matched_fixture_refs(self, references: Sequence[str]) -> tuple[str, ...]:
        held = _reference_index(self.fixture_refs)
        return tuple(sorted({str(reference) for reference in references} & held))

    def as_lineage(self) -> dict[str, Any]:
        policy_payload = json.dumps(
            {
                "contract_version": self.contract_version,
                "version": self.version,
                "fixture_refs": self.fixture_refs,
                "template_ids": self.template_ids,
                "fixtures_in_backend_state": self.fixtures_in_backend_state,
                "seed": self.seed,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return {
            "contract_version": self.contract_version,
            "version": self.version,
            "source": self.source,
            "policy_hash": f"sha256:{hashlib.sha256(policy_payload.encode('utf-8')).hexdigest()}",
            "fixture_ref_count": len(self.fixture_refs),
            "template_count": len(self.template_ids),
            "fixtures_in_backend_state": self.fixtures_in_backend_state,
            "seed": self.seed,
        }


class HeldOutDecision(BaseModel):
    """One immutable Stage 12 verdict for one publication candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0"] = HELD_OUT_CONTRACT_VERSION
    task_id: str
    held_out_hit: StrictBool
    matched_template_id: str | None = None
    matched_fixture_refs: tuple[str, ...] = ()

    @field_validator("task_id", "matched_template_id")
    @classmethod
    def normalize_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value)
        if not normalized.strip():
            raise ValueError("held-out identifiers must be non-empty")
        return normalized

    @field_validator("matched_fixture_refs")
    @classmethod
    def normalize_matches(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalized_strings(value, label="fixture matches")

    @model_validator(mode="after")
    def validate_decision(self) -> HeldOutDecision:
        has_evidence = bool(self.matched_template_id or self.matched_fixture_refs)
        if self.held_out_hit != has_evidence:
            raise ValueError(
                "a held-out hit requires the matched template or fixture references that prove it, "
                "and a clean row must carry none"
            )
        return self


def scan_row(
    policy: HeldOutPolicy,
    *,
    task_id: str,
    template_id: Any,
    fixture_refs: Sequence[str],
) -> HeldOutDecision:
    """Decide one row against the policy, recording the evidence for the verdict."""
    matched_template = str(template_id) if policy.blocks_template(template_id) else None
    matched_fixtures = policy.matched_fixture_refs(fixture_refs)
    return HeldOutDecision(
        task_id=task_id,
        held_out_hit=bool(matched_template or matched_fixtures),
        matched_template_id=matched_template,
        matched_fixture_refs=matched_fixtures,
    )


def validate_complete_scan_set(
    values: Sequence[HeldOutDecision | Mapping[str, object]],
    *,
    expected_task_ids: Sequence[str],
) -> list[HeldOutDecision]:
    """Validate exactly one verdict per publication candidate, in input order.

    A partial scan is the failure mode that matters here: an unscanned row would
    publish as clean without ever having been compared to the policy.
    """
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in expected_task_ids):
        raise ValueError("held-out scan task_id values must be non-empty strings")
    expected = list(expected_task_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("held-out scan task_id values must be unique after normalization")
    decisions = [
        value if isinstance(value, HeldOutDecision) else HeldOutDecision.model_validate(value)
        for value in values
    ]
    by_task: dict[str, HeldOutDecision] = {}
    for decision in decisions:
        if decision.task_id in by_task:
            raise ValueError(f"duplicate held-out decision for task {decision.task_id!r}")
        by_task[decision.task_id] = decision
    missing = [task_id for task_id in expected if task_id not in by_task]
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise ValueError(
            f"held-out decisions must match publication candidates exactly (missing={missing}, extra={extra})"
        )
    return [by_task[task_id] for task_id in expected]
