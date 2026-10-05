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

"""Versioned contract for what ``benchmark_raw.parquet`` and ``benchmark.parquet`` mean.

Stage 12 writes two tables from one list of rows, and the difference between them
is the only thing a consumer can use to tell "this row was generated" from "this
row is part of the benchmark". That difference has to be a *selection*, never a
rewrite:

``benchmark_raw.parquet``
    Every schema-valid, replay-valid row, before Stage 10's quality drops and
    before Stage 11's deduplication and balancing. It is the audit table: a
    reviewer asking why a task is absent from the benchmark needs to find it
    here, otherwise the drop reasons in the stage reports point at nothing.

``benchmark.parquet``
    Only the rows that survived every publication gate, in the order Stage 11
    fixed. It carries the same Arrow schema as raw, so a consumer can swap the
    two files without changing a reader.

The contract is enforced by re-deriving the publication set from the stage
decisions and comparing it against what was actually written to disk, rather
than by trusting the in-memory list Stage 12 filtered. A published row must be
byte-identical to its raw counterpart across the whole schema — see
:data:`PUBLICATION_RESTATED_FIELDS` for why nothing is exempt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    BENCHMARK_ROW_FIELDS,
    ContentHash,
    NonNegativeInt,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

PUBLICATION_CONTRACT_VERSION = "1.0"

RAW_BENCHMARK_TABLE = "benchmark_raw.parquet"
PUBLICATION_BENCHMARK_TABLE = "benchmark.parquet"

# Columns a published row may state differently from its raw counterpart.
#
# Empty, and deliberately so. Publication decides *which* rows ship, not what
# they say: a row that is worded one way in the audit table and another way in
# the benchmark makes the audit table useless for explaining a score, and the
# only reason to allow it would be to let a late stage patch truth it should have
# rejected instead. Stage-wide facts that do change late — ``gold_eligible`` when
# Stage 11 misses a balancing target, ``held_out_hit`` when Stage 12 scans — are
# stamped on every row, so both tables move together. Adding a name here is a
# contract change and requires a version bump.
PUBLICATION_RESTATED_FIELDS: frozenset[str] = frozenset()

SurfaceGate = Literal["deterministic_guards", "surface_quality"]
PublicationOrdering = Literal["raw_order", "selection_rank"]
SurfaceQualityDecision = Literal["kept", "dropped"]


class PublicationContractError(ValueError):
    """The written tables do not match the publication semantics Stage 12 claims."""


class PublicationPlan(BaseModel):
    """The exact rows ``benchmark.parquet`` must carry, and in which order.

    Built from the stage decisions alone, so it is an independent derivation of
    the publication set rather than a description of the list Stage 12 already
    filtered. The two agreeing is the check; one of them being convenient is not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = PUBLICATION_CONTRACT_VERSION
    raw_task_ids: tuple[StrictStr, ...]
    published_task_ids: tuple[StrictStr, ...]
    surface_gate: SurfaceGate
    dedup_balancing_applied: StrictBool
    held_out_evaluated: StrictBool
    ordering: PublicationOrdering

    @model_validator(mode="after")
    def validate_plan(self) -> PublicationPlan:
        if not self.raw_task_ids:
            raise ValueError("a publication plan requires at least one raw row")
        if len(set(self.raw_task_ids)) != len(self.raw_task_ids):
            raise ValueError("raw task ids must be unique")
        if len(set(self.published_task_ids)) != len(self.published_task_ids):
            raise ValueError("published task ids must be unique")
        if unknown := sorted(set(self.published_task_ids) - set(self.raw_task_ids)):
            raise ValueError(f"published task(s) {unknown} are absent from the raw table")
        if self.dedup_balancing_applied != (self.ordering == "selection_rank"):
            raise ValueError(
                "publication order is the Stage 11 selection rank exactly when Stage 11 ran; "
                "without it the raw order is the only order the pipeline fixed"
            )
        if self.ordering == "raw_order":
            published = set(self.published_task_ids)
            in_raw_order = tuple(task_id for task_id in self.raw_task_ids if task_id in published)
            if in_raw_order != self.published_task_ids:
                raise ValueError("without Stage 11 the published rows must keep their raw order")
        return self


class PublicationSemanticsReport(BaseModel):
    """Evidence that the two written tables mean what the manifest says they mean."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = PUBLICATION_CONTRACT_VERSION
    raw_rows: NonNegativeInt
    published_rows: NonNegativeInt
    surface_gate: SurfaceGate
    dedup_balancing_applied: StrictBool
    held_out_evaluated: StrictBool
    ordering: PublicationOrdering
    restated_fields: tuple[StrictStr, ...] = ()
    raw_content_hash: ContentHash
    publication_content_hash: ContentHash

    @model_validator(mode="after")
    def validate_report(self) -> PublicationSemanticsReport:
        if self.published_rows > self.raw_rows:
            raise ValueError("publication selects from the raw table, so it cannot carry more rows")
        if self.restated_fields != tuple(sorted(PUBLICATION_RESTATED_FIELDS)):
            raise ValueError("restated_fields must report the contract's own allowance, not this run's observations")
        if self.dedup_balancing_applied != (self.ordering == "selection_rank"):
            raise ValueError("reported publication order does not match whether Stage 11 ran")
        if self.raw_content_hash == self.publication_content_hash and self.published_rows != self.raw_rows:
            raise ValueError("two tables with different row counts cannot be the same file")
        return self


def plan_publication(
    *,
    raw_task_ids: Sequence[str],
    replay_validated_rows: int,
    guard_violations: Mapping[str, bool] | None = None,
    surface_quality_decisions: Mapping[str, str] | None = None,
    dedup_decisions: Sequence[Any] | None = None,
    held_out_hits: Mapping[str, bool] | None = None,
) -> PublicationPlan:
    """Derive the publication set from the stage decisions that produced it.

    ``replay_validated_rows`` is the count Stage 9 reported. Comparing it against
    the raw table is what catches a raw table that was filtered early: once a row
    is missing from raw, nothing downstream can tell whether it failed replay or
    was dropped by a stage that had no authority to drop it.
    """
    ordered_raw = tuple(str(task_id) for task_id in raw_task_ids)
    if len(set(ordered_raw)) != len(ordered_raw):
        raise PublicationContractError("the raw table repeats a task id")
    if len(ordered_raw) != int(replay_validated_rows):
        raise PublicationContractError(
            f"the raw table carries {len(ordered_raw)} rows but Stage 9 validated "
            f"{int(replay_validated_rows)}; benchmark_raw.parquet must precede every publication drop"
        )
    if (guard_violations is None) == (surface_quality_decisions is None):
        raise PublicationContractError(
            "exactly one surface gate decides publication: Stage 10 quality decisions, "
            "or the deterministic guards used when Stage 10 is disabled"
        )

    if surface_quality_decisions is not None:
        _require_exact_coverage(surface_quality_decisions, ordered_raw, label="Stage 10 decisions")
        invalid = sorted({value for value in surface_quality_decisions.values() if value not in {"kept", "dropped"}})
        if invalid:
            raise PublicationContractError(f"Stage 10 decisions must be kept or dropped, got {invalid}")
        surface_gate: SurfaceGate = "surface_quality"
        survivors = tuple(task_id for task_id in ordered_raw if surface_quality_decisions[task_id] == "kept")
    elif guard_violations is not None:
        _require_exact_coverage(guard_violations, ordered_raw, label="surface guard results")
        if any(type(value) is not bool for value in guard_violations.values()):
            raise PublicationContractError("surface guard results must use booleans")
        surface_gate = "deterministic_guards"
        survivors = tuple(task_id for task_id in ordered_raw if not guard_violations[task_id])
    else:  # pragma: no cover - the exclusivity check above already rejected this
        raise PublicationContractError("no surface gate was supplied")

    if dedup_decisions is None:
        published = survivors
        ordering: PublicationOrdering = "raw_order"
    else:
        published = _stage_eleven_order(dedup_decisions, survivors)
        ordering = "selection_rank"

    if held_out_hits is not None:
        _require_exact_coverage(held_out_hits, ordered_raw, label="held-out decisions")
        if any(type(value) is not bool for value in held_out_hits.values()):
            raise PublicationContractError("held-out decisions must use booleans")
        if leaked := sorted(task_id for task_id in published if held_out_hits[task_id]):
            raise PublicationContractError(
                f"{len(leaked)} publication candidate(s) bind held-out material: {leaked[:5]}"
            )

    return PublicationPlan(
        raw_task_ids=ordered_raw,
        published_task_ids=published,
        surface_gate=surface_gate,
        dedup_balancing_applied=dedup_decisions is not None,
        held_out_evaluated=held_out_hits is not None,
        ordering=ordering,
    )


def _stage_eleven_order(decisions: Sequence[Any], survivors: tuple[str, ...]) -> tuple[str, ...]:
    """Order the Stage 11 selections by their rank, which is a total order."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
        DedupBalancingDecision,
    )

    parsed = [
        decision if isinstance(decision, DedupBalancingDecision) else DedupBalancingDecision.model_validate(decision)
        for decision in decisions
    ]
    by_task: dict[str, DedupBalancingDecision] = {}
    for decision in parsed:
        if decision.task_id in by_task:
            raise PublicationContractError(f"Stage 11 decided task {decision.task_id!r} twice")
        by_task[decision.task_id] = decision
    _require_exact_coverage(by_task, survivors, label="Stage 11 decisions")
    selected = [by_task[task_id] for task_id in survivors if by_task[task_id].selected]
    ranks = sorted(decision.selection_rank for decision in selected)
    if ranks != list(range(len(selected))):
        raise PublicationContractError(
            "Stage 11 must rank its selections 0..k-1 exactly once, otherwise publication order is not total"
        )
    return tuple(decision.task_id for decision in sorted(selected, key=lambda decision: decision.selection_rank))


def _require_exact_coverage(decided: Mapping[str, Any], expected: Sequence[str], *, label: str) -> None:
    """Refuse a decision set that does not answer for every row it gates."""
    wanted = set(expected)
    present = set(decided)
    if present == wanted:
        return
    missing = sorted(wanted - present)
    extra = sorted(present - wanted)
    raise PublicationContractError(f"{label} must cover every gated row exactly (missing={missing}, extra={extra})")


def verify_publication_tables(
    raw_rows: Sequence[Mapping[str, Any]],
    published_rows: Sequence[Mapping[str, Any]],
    plan: PublicationPlan,
) -> None:
    """Check both tables against the plan, raising on the first contract breach.

    Comparison is on the canonical JSON of the whole row, so a published row that
    re-orders a call, re-encodes an argument, or drops an assertion is caught even
    when the two rows would compare equal under Python's ``==`` (``1 == True``).
    """
    raw_by_task = _indexed(raw_rows, table=RAW_BENCHMARK_TABLE)
    published_by_task = _indexed(published_rows, table=PUBLICATION_BENCHMARK_TABLE)
    if tuple(raw_by_task) != plan.raw_task_ids:
        raise PublicationContractError(
            f"{RAW_BENCHMARK_TABLE} does not carry the rows the plan derived, in that order"
        )
    if tuple(published_by_task) != plan.published_task_ids:
        raise PublicationContractError(
            f"{PUBLICATION_BENCHMARK_TABLE} does not carry the rows the plan derived, in that order"
        )

    compared = tuple(field for field in BENCHMARK_ROW_FIELDS if field not in PUBLICATION_RESTATED_FIELDS)
    for task_id, row in published_by_task.items():
        raw_row = raw_by_task[task_id]
        if _fingerprint(row, compared) == _fingerprint(raw_row, compared):
            continue
        changed = sorted(field for field in compared if _fingerprint(row, (field,)) != _fingerprint(raw_row, (field,)))
        raise PublicationContractError(
            f"published task {task_id!r} restates {changed} that {RAW_BENCHMARK_TABLE} records differently; "
            "publication selects rows, it does not rewrite them"
        )

    for task_id, row in raw_by_task.items():
        hit = row["held_out_hit"]
        if plan.held_out_evaluated:
            if type(hit) is not bool:
                raise PublicationContractError(
                    f"task {task_id!r} was scanned against the held-out policy but records no verdict"
                )
        elif hit is not None:
            raise PublicationContractError(
                f"task {task_id!r} claims a held-out verdict, but no held-out policy was evaluated"
            )
    if plan.held_out_evaluated:
        if leaked := sorted(task_id for task_id, row in published_by_task.items() if row["held_out_hit"]):
            raise PublicationContractError(
                f"{PUBLICATION_BENCHMARK_TABLE} carries {len(leaked)} held-out row(s): {leaked[:5]}"
            )


def _indexed(rows: Sequence[Mapping[str, Any]], *, table: str) -> dict[str, Mapping[str, Any]]:
    """Index rows by task id, refusing an off-schema or duplicated row."""
    indexed: dict[str, Mapping[str, Any]] = {}
    expected = set(BENCHMARK_ROW_FIELDS)
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PublicationContractError(f"{table} row {position} is not a mapping")
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            raise PublicationContractError(f"{table} row {position} is off-schema (missing={missing}, extra={extra})")
        task_id = row["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise PublicationContractError(f"{table} row {position} carries no task id")
        if task_id in indexed:
            raise PublicationContractError(f"{table} repeats task {task_id!r}")
        indexed[task_id] = row
    return indexed


def _fingerprint(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Serialize the requested columns to one comparable, order-stable string."""
    return canonical_json({field: _comparable(row[field]) for field in fields})


def _comparable(value: Any) -> Any:
    """Normalize Arrow's decoded shapes so two readings of one row agree.

    A parquet map column reads back as a list of key/value pairs whose order
    Arrow does not promise, while the same column in memory is a dict. Both are
    the same JSON object, and the comparison must not mistake one for a change.
    """
    if isinstance(value, Mapping):
        return {str(key): _comparable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | memoryview):
        items = list(value)
        if items and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in items):
            return {str(key): _comparable(item) for key, item in items}
        return [_comparable(item) for item in items]
    return value


def verify_written_benchmarks(
    *,
    raw_path: Path,
    publication_path: Path,
    plan: PublicationPlan,
) -> PublicationSemanticsReport:
    """Read both parquets back from disk and hold them to the plan.

    Reading back is the point: an in-memory check confirms what Stage 12 intended
    to write, and the file is what a consumer scores against.
    """
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import benchmark_schema

    schema = benchmark_schema()
    tables = {}
    for name, path in ((RAW_BENCHMARK_TABLE, raw_path), (PUBLICATION_BENCHMARK_TABLE, publication_path)):
        if not path.is_file():
            raise PublicationContractError(f"{name} was not written to {path}")
        table = pq.read_table(path)
        if not table.schema.equals(schema):
            raise PublicationContractError(
                f"{name} was written with a schema that is not the published benchmark schema"
            )
        tables[name] = table

    verify_publication_tables(
        tables[RAW_BENCHMARK_TABLE].to_pylist(),
        tables[PUBLICATION_BENCHMARK_TABLE].to_pylist(),
        plan,
    )
    return PublicationSemanticsReport(
        raw_rows=tables[RAW_BENCHMARK_TABLE].num_rows,
        published_rows=tables[PUBLICATION_BENCHMARK_TABLE].num_rows,
        surface_gate=plan.surface_gate,
        dedup_balancing_applied=plan.dedup_balancing_applied,
        held_out_evaluated=plan.held_out_evaluated,
        ordering=plan.ordering,
        restated_fields=tuple(sorted(PUBLICATION_RESTATED_FIELDS)),
        raw_content_hash=_content_hash(raw_path),
        publication_content_hash=_content_hash(publication_path),
    )


def _content_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def publication_manifest_section(report: PublicationSemanticsReport) -> dict[str, Any]:
    """Describe the two tables' relationship for the run manifest."""
    return {
        "schema_version": PUBLICATION_CONTRACT_VERSION,
        "raw": {
            "file": RAW_BENCHMARK_TABLE,
            "rows": report.raw_rows,
            "content_hash": report.raw_content_hash,
            "contains": "schema_valid_and_replay_valid_rows",
        },
        "published": {
            "file": PUBLICATION_BENCHMARK_TABLE,
            "rows": report.published_rows,
            "content_hash": report.publication_content_hash,
            "surface_gate": report.surface_gate,
            "dedup_balancing_applied": report.dedup_balancing_applied,
            "held_out_evaluated": report.held_out_evaluated,
            "ordering": report.ordering,
        },
        "restated_fields": list(report.restated_fields),
        "verified": True,
    }
