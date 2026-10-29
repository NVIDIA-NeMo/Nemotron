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

"""Held-out enforcement shared by Stage 4 binding and Stage 12 publication.

Stage 4 refuses to bind what a pack reserved, and Stage 12 re-scans what Stage 4
produced. The second pass is not redundant: it is the only evidence that the
first one held, and without it the manifest would assert a leakage claim that
nothing checked.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HELD_OUT_CONTRACT_VERSION,
    HeldOutDecision,
    HeldOutPolicy,
    scan_row,
    validate_complete_scan_set,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)

HELD_OUT_BINDINGS = "held_out_bindings.json"
HELD_OUT_SCAN = "held_out_scan.json"


class HeldOutLeakError(RuntimeError):
    """Raised when a publication candidate binds something the pack reserved."""


@dataclass
class BindingLedger:
    """Tally what Stage 4 examined and what the policy withheld.

    Expansion records attempts as well as blocks so the manifest can distinguish
    a pack whose reservations never came up from one that was never filtered.
    """

    attempts: int = 0
    blocked_refs: set[str] = field(default_factory=set)

    def examined(self, count: int) -> None:
        self.attempts += int(count)

    def blocked(self, references: Sequence[str]) -> None:
        self.blocked_refs.update(str(reference) for reference in references)


def held_out_policy(pack: LoadedPack) -> HeldOutPolicy | None:
    """Return the run's held-out contract, or ``None`` when the pack declares none."""
    if pack.held_out is None:
        return None
    return HeldOutPolicy.from_normalized(pack.held_out)


def content_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_binding_report(
    config: BfclConfig,
    policy: HeldOutPolicy,
    *,
    blocked_templates: Sequence[str],
    blocked_fixture_refs: Sequence[str],
    bind_attempts: int,
    tasks_expanded: int,
) -> dict[str, Any]:
    """Record what Stage 4 refused to bind, so Stage 12 can attest to it.

    ``bind_attempts`` counts the candidate rows expansion considered, including
    the reserved ones, because "nothing was blocked" and "nothing was examined"
    are different states and only the first one supports the held-out claim.
    """
    report = {
        "contract_version": HELD_OUT_CONTRACT_VERSION,
        "policy": policy.as_lineage(),
        "blocked_templates": sorted({str(template) for template in blocked_templates}),
        "blocked_fixture_refs": sorted({str(reference) for reference in blocked_fixture_refs}),
        "counts": {
            "bind_attempts": int(bind_attempts),
            "templates_blocked": len({str(template) for template in blocked_templates}),
            "fixture_refs_blocked": len({str(reference) for reference in blocked_fixture_refs}),
            "tasks_expanded": int(tasks_expanded),
        },
    }
    _write_json_atomic(stage_cache_dir(config) / HELD_OUT_BINDINGS, report)
    return report


def load_binding_report(
    config: BfclConfig,
    policy: HeldOutPolicy,
    *,
    expected_tasks_expanded: int | None = None,
) -> dict[str, Any]:
    """Read Stage 4's binding report, refusing a run that cannot prove it ran."""
    path = stage_cache_dir(config) / HELD_OUT_BINDINGS
    if not path.is_file():
        raise HeldOutLeakError(
            "the pack declares a held-out policy but Stage 4 wrote no "
            f"{HELD_OUT_BINDINGS}; re-run stage=generate so binding enforcement is recorded"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HeldOutLeakError(f"{HELD_OUT_BINDINGS} is not valid JSON") from exc
    if not isinstance(report, dict):
        raise HeldOutLeakError(f"{HELD_OUT_BINDINGS} must be a JSON object")
    expected_keys = {
        "contract_version",
        "policy",
        "blocked_templates",
        "blocked_fixture_refs",
        "counts",
    }
    if set(report) != expected_keys:
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} must contain exactly {sorted(expected_keys)}"
        )
    if report.get("contract_version") != HELD_OUT_CONTRACT_VERSION:
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} was written under held-out contract "
            f"{report.get('contract_version')!r}, but this run enforces {HELD_OUT_CONTRACT_VERSION!r}"
        )
    if report.get("policy") != policy.as_lineage():
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} describes a different held-out policy than the pack now "
            "declares; re-run stage=generate after changing held_out.yaml"
        )
    blocked_templates = report.get("blocked_templates")
    blocked_fixture_refs = report.get("blocked_fixture_refs")
    if (
        not isinstance(blocked_templates, list)
        or any(not isinstance(value, str) for value in blocked_templates)
        or blocked_templates != sorted(set(blocked_templates))
        or not set(blocked_templates).issubset(policy.template_ids)
    ):
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} blocked_templates must be a sorted unique subset of the policy"
        )
    if (
        not isinstance(blocked_fixture_refs, list)
        or any(not isinstance(value, str) for value in blocked_fixture_refs)
        or blocked_fixture_refs != sorted(set(blocked_fixture_refs))
        or not set(blocked_fixture_refs).issubset(policy.fixture_refs)
    ):
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} blocked_fixture_refs must be a sorted unique subset of the policy"
        )
    counts = report.get("counts")
    expected_count_keys = {
        "bind_attempts",
        "templates_blocked",
        "fixture_refs_blocked",
        "tasks_expanded",
    }
    if (
        not isinstance(counts, dict)
        or set(counts) != expected_count_keys
        or any(
            not isinstance(counts[key], int)
            or isinstance(counts[key], bool)
            or counts[key] < 0
            for key in expected_count_keys
        )
    ):
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} counts must be non-negative integers for "
            f"{sorted(expected_count_keys)}"
        )
    if counts["templates_blocked"] != len(blocked_templates):
        raise HeldOutLeakError(f"{HELD_OUT_BINDINGS} templates_blocked count is inconsistent")
    if counts["fixture_refs_blocked"] != len(blocked_fixture_refs):
        raise HeldOutLeakError(f"{HELD_OUT_BINDINGS} fixture_refs_blocked count is inconsistent")
    if (
        expected_tasks_expanded is not None
        and counts["tasks_expanded"] != expected_tasks_expanded
    ):
        raise HeldOutLeakError(
            f"{HELD_OUT_BINDINGS} tasks_expanded does not match Stage 4 output "
            f"({counts['tasks_expanded']} != {expected_tasks_expanded})"
        )
    return report


def scan_rows(
    policy: HeldOutPolicy,
    rows: Sequence[Mapping[str, Any]],
    *,
    fixture_refs_by_task: Mapping[str, Sequence[str]],
) -> list[HeldOutDecision]:
    """Re-scan every candidate row against the policy, one verdict per row.

    Rows carry ``src`` rather than fixture references, so the bindings recorded
    by expansion are supplied separately; a row whose bindings are unknown is a
    row that cannot be cleared, and it stops the run.
    """
    decisions: list[HeldOutDecision] = []
    for row in rows:
        task_id = str(row["task_id"])
        references = fixture_refs_by_task.get(task_id)
        if references is None:
            raise HeldOutLeakError(
                f"task {task_id!r} reached publication without the fixture references needed "
                "to clear it against the held-out policy"
            )
        template_id = str(row.get("template_id") or "")
        if not template_id:
            raise HeldOutLeakError(f"task {task_id!r} reached publication without a template id")
        decisions.append(
            scan_row(
                policy,
                task_id=task_id,
                template_id=template_id,
                fixture_refs=references,
            )
        )
    return validate_complete_scan_set(
        decisions,
        expected_task_ids=[str(row["task_id"]) for row in rows],
    )


def write_scan_report(
    config: BfclConfig,
    policy: HeldOutPolicy,
    decisions: Sequence[HeldOutDecision],
    *,
    binding_report: Mapping[str, Any],
    rows_published: int,
) -> dict[str, Any]:
    """Write Stage 12's scan evidence, hits included, before any abort decision.

    The report is written even when the run is about to stop: an author fixing a
    leak needs the offending task ids, and the published outputs never exist to
    carry them.
    """
    hits = [decision for decision in decisions if decision.held_out_hit]
    aborting = bool(hits)
    report = {
        "contract_version": HELD_OUT_CONTRACT_VERSION,
        "policy": policy.as_lineage(),
        "stage_four": {
            "blocked_templates": list(binding_report.get("blocked_templates") or []),
            "blocked_fixture_refs": list(binding_report.get("blocked_fixture_refs") or []),
            "counts": dict(binding_report.get("counts") or {}),
        },
        "counts": {
            "rows_scanned": len(decisions),
            "rows_hit": len(hits),
            "rows_blocked": len(hits),
            "rows_dropped": 0,
            "planned_rows_published": int(rows_published),
            "rows_published": 0 if aborting else int(rows_published),
        },
        "action": "abort" if aborting else "publish",
        "hits": [
            {
                "task_id": decision.task_id,
                "matched_template_id": decision.matched_template_id,
                "matched_fixture_refs": list(decision.matched_fixture_refs),
            }
            for decision in sorted(hits, key=lambda decision: decision.task_id)
        ],
    }
    _write_json_atomic(stage_cache_dir(config) / HELD_OUT_SCAN, report)
    return report


def enforce_no_leak(config: BfclConfig, report: Mapping[str, Any]) -> None:
    """Stop the run when the scan found a reserved binding.

    Fail-closed: a leaked row cannot be dropped quietly, because Stage 11 already
    fixed the publication set and silently shrinking it would break the balance
    the manifest reports.
    """
    hits = list(report.get("hits") or [])
    if not hits:
        return
    offenders = ", ".join(str(hit.get("task_id")) for hit in hits[:5])
    remainder = f" (+{len(hits) - 5} more)" if len(hits) > 5 else ""
    raise HeldOutLeakError(
        f"{len(hits)} publication candidate(s) bind held-out material: {offenders}{remainder}; "
        f"inspect stage_cache/{HELD_OUT_SCAN} and release or re-declare the reserved rows"
    )


def manifest_section(
    policy: HeldOutPolicy | None,
    scan_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe held-out enforcement for the run manifest.

    A pack without a policy reports ``evaluated: false`` instead of a clean
    result, because no comparison was ever made.
    """
    if policy is None or scan_report is None:
        return {
            "contract_version": HELD_OUT_CONTRACT_VERSION,
            "source": None,
            "evaluated": False,
            "rows_scanned": 0,
            "rows_dropped": 0,
        }
    counts = dict(scan_report.get("counts") or {})
    return {
        "contract_version": HELD_OUT_CONTRACT_VERSION,
        "source": policy.source,
        "evaluated": True,
        "policy": policy.as_lineage(),
        "rows_scanned": int(counts.get("rows_scanned", 0)),
        "rows_hit": int(counts.get("rows_hit", 0)),
        "rows_dropped": int(counts.get("rows_dropped", 0)),
        "stage_four": dict(scan_report.get("stage_four") or {}),
    }
