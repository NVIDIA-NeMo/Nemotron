# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-stage record accounting, with terminal states that must add up.

``curate/audit`` can see that records disappeared. It cannot see **why**,
because an audit that runs afterwards observes only inputs and outputs and has
no way to tell a record removed on purpose from one lost to a swallowed
exception. Attribution requires the producer to say what it did, at the time it
did it. That is what a ledger is.

The contract, enforced for every stage::

    n_input == n_success + n_filtered + n_failed + n_quarantined

Anything else raises. A stage may not report success while records are
unaccounted for.

**Counting records is not enough.** The obvious completeness gate — "fail if
``n_failed + n_quarantined > 0``" — cannot detect the failure it is written for.
Those counts come from reading the shard, and a shard truncated by a killed job
reports zero rows, so a stage can lose a whole file and compute a loss of
exactly zero. :func:`assert_no_lost_units` therefore counts **units**: a shard
that could not be processed is a loss whether or not anyone can say how many
records were in it. Every record count in a failure report is a floor.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

#: Bumped when a change would alter how an existing ledger should be read.
SCHEMA_VERSION = 1

#: How many failed units a message lists before summarising the rest.
MAX_REPORTED_UNITS = 10


class TerminalState(StrEnum):
    """The only four ways a record may leave a stage."""

    SUCCESS = "success"
    FILTERED = "filtered"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class LedgerImbalanceError(RuntimeError):
    """Inputs do not equal the sum of the terminal states."""


class LedgerInvalidError(ValueError):
    """A ledger document cannot be read as one."""


@dataclass
class StageLedger:
    """Record accounting for one stage, optionally one source within it.

    Not thread-safe, deliberately: one ledger per worker, combined with
    :func:`merge_ledgers` at the end. A shared mutable counter across workers is
    how counts go missing in the first place.
    """

    stage: str
    source: str = ""
    n_input: int = 0
    n_success: int = 0
    filtered: Counter = field(default_factory=Counter)
    failed_units: list[dict[str, Any]] = field(default_factory=list)
    quarantined_units: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    # -- recording ------------------------------------------------------------

    def add_input(self, n: int) -> None:
        self.n_input += int(n)

    def add_success(self, n: int) -> None:
        self.n_success += int(n)

    def add_filtered(self, reason: str, n: int) -> None:
        """Records removed on purpose, attributed to the gate that removed them.

        The reason is the whole point. "5,187,587 records filtered" and
        "5,187,587 records filtered by language_id" are different facts, and
        only the second one can be checked against an intent.
        """
        if n:
            self.filtered[str(reason)] += int(n)

    def add_failed(self, unit: str, reason: str, n_records: int) -> None:
        """A unit — shard, source, batch — that could not be processed at all."""
        self.failed_units.append({"unit": str(unit), "reason": str(reason), "records": int(n_records)})

    def add_quarantined(self, unit: str, reason: str, n_records: int) -> None:
        """A unit set aside for inspection: unreadable input, bad schema, ..."""
        self.quarantined_units.append({"unit": str(unit), "reason": str(reason), "records": int(n_records)})

    # -- derived --------------------------------------------------------------

    @property
    def n_filtered(self) -> int:
        return int(sum(self.filtered.values()))

    @property
    def n_failed(self) -> int:
        return int(sum(u["records"] for u in self.failed_units))

    @property
    def n_quarantined(self) -> int:
        return int(sum(u["records"] for u in self.quarantined_units))

    @property
    def n_accounted(self) -> int:
        return self.n_success + self.n_filtered + self.n_failed + self.n_quarantined

    @property
    def balanced(self) -> bool:
        return self.n_input == self.n_accounted

    @property
    def lost_units(self) -> list[dict[str, Any]]:
        return list(self.failed_units) + list(self.quarantined_units)

    # -- combination ----------------------------------------------------------

    def merge(self, other: StageLedger) -> StageLedger:
        """Fold another ledger into this one, in place.

        Numeric notes **sum**. They are per-source counts, and a merge that kept
        only the first value would report one source's numbers as the total —
        which is exactly the kind of quiet undercount a ledger exists to catch.
        Non-numeric notes that disagree are recorded as disagreeing rather than
        silently resolved.
        """
        self.n_input += other.n_input
        self.n_success += other.n_success
        self.filtered.update(other.filtered)
        self.failed_units.extend(other.failed_units)
        self.quarantined_units.extend(other.quarantined_units)
        for key, value in other.notes.items():
            current = self.notes.get(key)
            if current is None:
                self.notes[key] = value
            elif _is_number(current) and _is_number(value):
                self.notes[key] = current + value
            elif current != value:
                self.notes[key] = f"<varies: {current!r} .. {value!r}>"
        return self

    # -- serialisation --------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "source": self.source,
            "n_input": self.n_input,
            "n_success": self.n_success,
            "n_filtered": self.n_filtered,
            "n_failed": self.n_failed,
            "n_quarantined": self.n_quarantined,
            "n_accounted": self.n_accounted,
            "balanced": self.balanced,
            "filtered_by_reason": dict(sorted(self.filtered.items())),
            "failed_units": self.failed_units,
            "quarantined_units": self.quarantined_units,
            "notes": dict(sorted(self.notes.items(), key=lambda kv: kv[0])),
        }

    def assert_balanced(self) -> None:
        if not self.balanced:
            raise LedgerImbalanceError(
                f"[{self.stage}/{self.source or 'all'}] record reconciliation failed: "
                f"inputs={self.n_input:,} but success={self.n_success:,} + "
                f"filtered={self.n_filtered:,} + failed={self.n_failed:,} + "
                f"quarantined={self.n_quarantined:,} = {self.n_accounted:,} "
                f"(difference {self.n_input - self.n_accounted:+,})"
            )

    def write(self, path: os.PathLike[str] | str, *, require_balanced: bool = True) -> Path:
        """Persist the ledger, atomically, refusing an unbalanced one by default.

        Written through a temporary file and ``os.replace`` so a job killed
        mid-write leaves either the old ledger or the new one, never a truncated
        document that reads as a smaller corpus.
        """
        if require_balanced:
            self.assert_balanced()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, dest)
        return dest

    def summary(self) -> str:
        return (
            f"[{self.stage}/{self.source or 'all'}] in={self.n_input:,} "
            f"success={self.n_success:,} filtered={self.n_filtered:,} "
            f"failed={self.n_failed:,} quarantined={self.n_quarantined:,} "
            f"balanced={self.balanced}"
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_ledger(path: os.PathLike[str] | str) -> StageLedger:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LedgerInvalidError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or "stage" not in data:
        raise LedgerInvalidError(f"{path}: not a ledger — no 'stage' field")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise LedgerInvalidError(
            f"{path}: schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )
    if not isinstance(data["stage"], str) or not data["stage"]:
        raise LedgerInvalidError(f"{path}: stage must be a non-empty string")
    if not isinstance(data.get("source", ""), str):
        raise LedgerInvalidError(f"{path}: source must be a string")

    for field_name in ("n_input", "n_success"):
        if not _is_count(data.get(field_name)):
            raise LedgerInvalidError(
                f"{path}: {field_name} must be a non-negative integer, got {data.get(field_name)!r}"
            )

    filtered = data.get("filtered_by_reason")
    if not isinstance(filtered, dict) or any(
        not isinstance(reason, str) or not reason or not _is_count(count) for reason, count in filtered.items()
    ):
        raise LedgerInvalidError(f"{path}: filtered_by_reason must map non-empty strings to non-negative integers")

    units_by_state: dict[str, list[dict[str, Any]]] = {}
    for field_name in ("failed_units", "quarantined_units"):
        units = data.get(field_name)
        if not isinstance(units, list):
            raise LedgerInvalidError(f"{path}: {field_name} must be a list")
        for index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise LedgerInvalidError(f"{path}: {field_name}[{index}] must be a mapping")
            if not isinstance(unit.get("unit"), str) or not unit["unit"]:
                raise LedgerInvalidError(f"{path}: {field_name}[{index}].unit must be a non-empty string")
            if not isinstance(unit.get("reason"), str) or not unit["reason"]:
                raise LedgerInvalidError(f"{path}: {field_name}[{index}].reason must be a non-empty string")
            if not _is_count(unit.get("records")):
                raise LedgerInvalidError(f"{path}: {field_name}[{index}].records must be a non-negative integer")
        units_by_state[field_name] = units

    notes = data.get("notes")
    if not isinstance(notes, dict):
        raise LedgerInvalidError(f"{path}: notes must be a mapping")

    ledger = StageLedger(stage=data["stage"], source=data.get("source", ""))
    ledger.n_input = data["n_input"]
    ledger.n_success = data["n_success"]
    ledger.filtered = Counter(filtered)
    ledger.failed_units = units_by_state["failed_units"]
    ledger.quarantined_units = units_by_state["quarantined_units"]
    ledger.notes = notes

    derived = {
        "n_filtered": ledger.n_filtered,
        "n_failed": ledger.n_failed,
        "n_quarantined": ledger.n_quarantined,
        "n_accounted": ledger.n_accounted,
        "balanced": ledger.balanced,
    }
    for field_name, expected in derived.items():
        if data.get(field_name) != expected:
            raise LedgerInvalidError(
                f"{path}: {field_name} is {data.get(field_name)!r}, expected {expected!r} "
                "from the terminal-state records"
            )
    return ledger


def merge_ledgers(stage: str, ledgers: list[StageLedger], source: str = "") -> StageLedger:
    merged = StageLedger(stage=stage, source=source)
    for ledger in ledgers:
        merged.merge(ledger)
    return merged


def quarantine_path(root: os.PathLike[str] | str, stage: str, unit_name: str) -> Path:
    """Where a quarantined unit's payload or marker is written."""
    return Path(root) / "_quarantine" / stage / unit_name


def lost_unit_report(ledgers: list[StageLedger], stage: str, where: str = "") -> str | None:
    """Describe every unit that failed or was quarantined, or ``None`` if clean.

    Counts units, not records, for the reason in the module docstring: a shard
    too damaged to open reports zero rows, so a record-count gate cannot see the
    loss it exists to catch. The record figure is reported as a floor.
    """
    units = [u for ledger in ledgers for u in ledger.lost_units]
    if not units:
        return None

    records = sum(ledger.n_failed + ledger.n_quarantined for ledger in ledgers)
    listed = "\n  ".join(f"{u['unit']}: {str(u['reason'])[:120]}" for u in units[:MAX_REPORTED_UNITS])
    more = f"\n  ... and {len(units) - MAX_REPORTED_UNITS} more" if len(units) > MAX_REPORTED_UNITS else ""
    return (
        f"[{stage}] {len(units)} unit(s) failed or were quarantined; "
        f"{records:,} records counted — and that count is a FLOOR, because a shard "
        f"too damaged to open reports 0 rows.\n  {listed}{more}" + (f"\nSee {where}" if where else "")
    )


def assert_no_lost_units(ledgers: list[StageLedger], stage: str, where: str = "") -> None:
    """Refuse to exit successfully when any unit failed or was quarantined."""
    report = lost_unit_report(ledgers, stage, where)
    if report:
        raise LedgerImbalanceError(report)


# -- attribution --------------------------------------------------------------


def attribute(ledgers: list[StageLedger], observed_delta: int) -> dict[str, Any]:
    """Explain an observed input-minus-output row delta against the ledgers.

    This is the capability ``curate/audit`` v1 does not have. v1 can say 5.2
    million records are missing; only a producer-emitted ledger can say whether
    they were removed by a language filter or lost with sixty shards.

    ``unexplained`` is the number that matters. Anything other than zero means
    records left the pipeline for a reason nobody recorded, and no amount of
    reading the output afterwards will recover it.
    """
    declared_filtered = sum(led.n_filtered for led in ledgers)
    declared_failed = sum(led.n_failed for led in ledgers)
    declared_quarantined = sum(led.n_quarantined for led in ledgers)
    by_reason: Counter = Counter()
    for ledger in ledgers:
        by_reason.update(ledger.filtered)

    accounted = declared_filtered + declared_failed + declared_quarantined
    units = [u for ledger in ledgers for u in ledger.lost_units]

    return {
        "attribution_note": (
            "filtered_by_reason attributes each document to the FIRST gate that rejected it: "
            "gates short-circuit, so a document failing two gates is counted once, under "
            "whichever ran first. These are not per-gate independent removal counts, and the "
            "order they ran in is not recorded here. Reading one figure as 'how many documents "
            "this gate rejects' will understate it."
        ),
        "observed_delta": observed_delta,
        "declared_filtered": declared_filtered,
        "declared_failed": declared_failed,
        "declared_quarantined": declared_quarantined,
        "filtered_by_reason": dict(sorted(by_reason.items())),
        "unexplained": observed_delta - accounted,
        "lost_units": len(units),
        "all_ledgers_balanced": all(led.balanced for led in ledgers),
        "record_counts_are_a_floor": bool(units),
    }
