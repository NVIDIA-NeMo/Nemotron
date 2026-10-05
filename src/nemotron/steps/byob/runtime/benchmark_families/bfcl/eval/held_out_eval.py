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

"""Private held-out slice construction and generalization reporting.

This module intentionally keeps private rows in memory.  Its public report contains
only aggregate counts, confidence intervals, strata, and content identities; task
ids, prompts, fixture values, and candidate responses are never serialized.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HeldOutPolicy,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
    expand_template,
)

HELD_OUT_EVAL_CONTRACT_VERSION: Final = "1.0"
HELD_OUT_EVAL_REPORT_VERSION: Final = "1.0"


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class HeldOutEvalError(ValueError):
    """A private held-out evaluation cannot satisfy its frozen contract."""


class HeldOutEvalConfig(BaseModel):
    """Exact held-out policy identity pinned by an eval config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0"] = HELD_OUT_EVAL_CONTRACT_VERSION
    policy_hash: StrictStr
    fixture_refs: tuple[StrictStr, ...] = ()
    template_ids: tuple[StrictStr, ...] = ()
    seed: StrictInt
    pack_version: StrictStr
    max_tasks_per_template: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def _coherent(self) -> HeldOutEvalConfig:
        if not self.policy_hash.startswith("sha256:") or len(self.policy_hash) != 71:
            raise ValueError("policy_hash must be sha256:<64 hex characters>")
        try:
            int(self.policy_hash.removeprefix("sha256:"), 16)
        except ValueError as exc:
            raise ValueError("policy_hash must be sha256:<64 hex characters>") from exc
        if not self.pack_version.strip():
            raise ValueError("pack_version must be non-empty")
        if len(set(self.fixture_refs)) != len(self.fixture_refs):
            raise ValueError("fixture_refs must be unique")
        if len(set(self.template_ids)) != len(self.template_ids):
            raise ValueError("template_ids must be unique")
        if not self.fixture_refs and not self.template_ids:
            raise ValueError("held_out_eval requires at least one fixture or template")
        return self

    @property
    def selection_mode(self) -> Literal["fixture_only", "template_only", "both"]:
        if self.fixture_refs and self.template_ids:
            return "both"
        return "fixture_only" if self.fixture_refs else "template_only"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "policy_hash": self.policy_hash,
            "fixture_refs": list(self.fixture_refs),
            "template_ids": list(self.template_ids),
            "seed": self.seed,
            "pack_version": self.pack_version,
            "max_tasks_per_template": self.max_tasks_per_template,
            "selection_mode": self.selection_mode,
        }


def verify_held_out_policy(
    config: HeldOutEvalConfig,
    policy: HeldOutPolicy | None,
    *,
    pack_version: str,
) -> HeldOutPolicy:
    """Match every policy field before any private task is expanded."""
    if policy is None:
        raise HeldOutEvalError("held_out_eval requires the exact pack held_out.yaml policy")
    expected = {
        "policy_hash": config.policy_hash,
        "fixture_refs": tuple(sorted(config.fixture_refs)),
        "template_ids": tuple(sorted(config.template_ids)),
        "seed": config.seed,
        "pack_version": config.pack_version,
    }
    actual = {
        "policy_hash": policy.as_lineage()["policy_hash"],
        "fixture_refs": policy.fixture_refs,
        "template_ids": policy.template_ids,
        "seed": policy.seed,
        "pack_version": str(pack_version),
    }
    if actual != expected:
        changed = sorted(key for key in expected if expected[key] != actual[key])
        raise HeldOutEvalError(
            "held_out_eval policy pin does not match the verified Oracle pack: "
            + ", ".join(changed)
        )
    return policy


def _private_task_id(
    task: Mapping[str, Any],
    policy_hash: str,
    pack_version: str,
) -> str:
    payload = {
        "contract_version": HELD_OUT_EVAL_CONTRACT_VERSION,
        "policy_hash": policy_hash,
        "pack_version": pack_version,
        "template_id": task["template_id"],
        "fixture_refs": sorted(task.get("fixture_refs") or []),
        "slots_initial": task.get("slots_initial") or {},
        "slot_updates": task.get("slot_updates") or [],
        "variant_index": int(task.get("variant_index", 0)),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"private-heldout-{digest}"


def _selected(task: Mapping[str, Any], policy: HeldOutPolicy) -> bool:
    return policy.blocks_template(task.get("template_id")) or bool(
        policy.matched_fixture_refs(task.get("fixture_refs") or [])
    )


def expand_private_held_out_tasks(
    pack: LoadedPack,
    config: HeldOutEvalConfig,
) -> tuple[list[dict[str, Any]], str]:
    """Deterministically expand the policy union without writing a stage artifact.

    Fixture-only means non-reserved templates bound through at least one reserved
    fixture. Template-only means reserved templates with reserved fixtures excluded.
    Both is the set union of those two slices. This prevents a template reservation
    from accidentally meaning "all fixtures are held out", or vice versa.
    """
    policy = verify_held_out_policy(
        config,
        HeldOutPolicy.from_normalized(pack.held_out) if pack.held_out is not None else None,
        pack_version=str(pack.manifest.get("version")),
    )
    limit = config.max_tasks_per_template
    by_identity: dict[str, dict[str, Any]] = {}

    if policy.fixture_refs:
        for template in pack.templates:
            tasks = expand_template(
                pack,
                template,
                limit,
                policy.seed,
                held_out=policy,
                fixture_selection="include",
                allow_held_out_template=True,
            )
            for task in tasks:
                if policy.matched_fixture_refs(task.get("fixture_refs") or []):
                    by_identity[_sha256_json(task)] = task

    if policy.template_ids:
        for template in pack.templates:
            if not policy.blocks_template(template.get("template_id")):
                continue
            tasks = expand_template(
                pack,
                template,
                limit,
                policy.seed,
                held_out=policy,
                fixture_selection="exclude",
                allow_held_out_template=True,
            )
            for task in tasks:
                by_identity[_sha256_json(task)] = task

    tasks = [by_identity[key] for key in sorted(by_identity)]
    if not tasks:
        raise HeldOutEvalError(
            f"held_out_eval {config.selection_mode} policy expanded no private task"
        )
    for task in tasks:
        if not _selected(task, policy):
            raise HeldOutEvalError("private expansion produced a task outside held_out.yaml")
        task["task_id"] = _private_task_id(
            task,
            config.policy_hash,
            config.pack_version,
        )
        task["seed"] = int(task["seed"])
    ids = [str(task["task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise HeldOutEvalError("private held-out task ids collided")
    slice_hash = _sha256_json(
        [
            {
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "fixture_refs": sorted(task.get("fixture_refs") or []),
                "seed": task["seed"],
                "turn_policy": task.get("turn_policy"),
                "required_tools": sorted(task.get("required_tools") or []),
            }
            for task in tasks
        ]
    )
    return tasks, slice_hash


def private_runtime_pack(pack: LoadedPack) -> LoadedPack:
    """Open the verified fixture inventory only for an ephemeral private run."""
    runtime_held_out = (
        {
            **pack.held_out,
            "policy": {
                **(pack.held_out.get("policy") or {}),
                "fixtures_in_backend_state": True,
            },
        }
        if pack.held_out is not None
        else None
    )
    return replace(pack, held_out=runtime_held_out)


def build_validated_private_slice(
    generation_config: BfclConfig,
    pack: LoadedPack,
    held_out_config: HeldOutEvalConfig,
) -> tuple[list[dict[str, Any]], str]:
    """Build private rows through the standard gold validation stages.

    Stage caches are redirected to an ephemeral directory and destroyed before
    return. The caller receives rows only in memory and must pass only aggregate
    results to :func:`held_out_generalization_report`.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        project_model_facing_tools,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.executable_replay import (
        run_executable_replay,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        run_expected_trace,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        build_row,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        run_render,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.schema_validation import (
        run_schema_validation,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    tasks, _task_hash = expand_private_held_out_tasks(pack, held_out_config)
    # ``fixtures_in_backend_state`` governs public generation and ordinary eval.
    # A private held-out fixture evaluation must intentionally open the complete
    # verified fixture inventory; otherwise it would measure ``not_found`` rather
    # than candidate generalization. This copy never reaches publication.
    runtime_pack = private_runtime_pack(pack)
    templates = {str(template["template_id"]): template for template in pack.templates}
    with tempfile.TemporaryDirectory(prefix="bfcl-private-heldout-") as temporary:
        private_config = replace(
            generation_config,
            output_dir=Path(temporary),
            expt_name="private-held-out",
            exports={},
            inline_eval=None,
            eval_config_path=None,
        )
        plans = run_state_machine(private_config, templates, tasks)
        surfaces, _prompt = run_render(
            private_config,
            pack,
            templates,
            tasks,
            plans,
        )
        guarded = {
            task_id: surface["guard_violations"]
            for task_id, surface in surfaces.items()
            if surface["guard_violations"]
        }
        if guarded:
            raise HeldOutEvalError(
                f"private held-out surface validation failed for {len(guarded)} task(s)"
            )
        original_ids = tuple(str(task["task_id"]) for task in tasks)
        traces, dropped = run_expected_trace(private_config, runtime_pack, tasks, plans)
        if dropped or tuple(str(task["task_id"]) for task in tasks) != original_ids:
            raise HeldOutEvalError(
                f"private held-out trace derivation dropped {len(dropped)} task(s)"
            )
        schema_failures = run_schema_validation(
            private_config,
            runtime_pack,
            tasks,
            traces,
        )
        invalid = {task_id: failures for task_id, failures in schema_failures.items() if failures}
        if invalid:
            raise HeldOutEvalError(
                f"private held-out schema validation failed for {len(invalid)} task(s)"
            )
        verdicts = run_executable_replay(
            private_config,
            runtime_pack,
            tasks,
            traces,
            schema_failures,
        )
        failed = {
            task_id: verdict
            for task_id, verdict in verdicts.items()
            if not verdict.get("passed")
        }
        if failed:
            reasons = sorted({str(verdict.get("reason")) for verdict in failed.values()})
            raise HeldOutEvalError(
                f"private held-out executable replay/assertions failed for {len(failed)} "
                f"task(s): {', '.join(reasons)}"
            )
        model_tools = project_model_facing_tools(pack.tools)
        rows = [
            build_row(
                private_config,
                pack,
                task,
                surfaces[str(task["task_id"])],
                traces[str(task["task_id"])],
                verdicts[str(task["task_id"])],
                "gold",
                model_tools,
            )
            for task in tasks
        ]
    slice_hash = _sha256_json(rows)
    return rows, slice_hash


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float | None]:
    """Return a two-sided 95% Wilson score interval."""
    if total < 0 or successes < 0 or successes > total:
        raise HeldOutEvalError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return {"low": None, "high": None}
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {"low": max(0.0, centre - radius), "high": min(1.0, centre + radius)}


def generalization_gap_interval(
    *,
    seen_successes: int,
    seen_total: int,
    held_out_successes: int,
    held_out_total: int,
) -> dict[str, float | None]:
    """Conservative Newcombe interval for ``seen rate - held-out rate``."""
    seen = wilson_interval(seen_successes, seen_total)
    held_out = wilson_interval(held_out_successes, held_out_total)
    if (
        seen["low"] is None
        or seen["high"] is None
        or held_out["low"] is None
        or held_out["high"] is None
    ):
        return {"low": None, "high": None}
    return {
        "low": float(seen["low"]) - float(held_out["high"]),
        "high": float(seen["high"]) - float(held_out["low"]),
    }


def _task_strata(task: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    policy = str(task.get("turn_policy") or "")
    tools = sorted({str(tool) for tool in task.get("required_tools") or []})
    if not tools:
        tools = ["__no_tool__"]
    return tuple((tool, policy) for tool in tools)


def held_out_generalization_report(
    *,
    seen_results: Sequence[Mapping[str, Any]],
    held_out_results: Sequence[Mapping[str, Any]],
    seen_tasks: Mapping[str, Mapping[str, Any]],
    held_out_tasks: Mapping[str, Mapping[str, Any]],
    policy: HeldOutPolicy,
    pack_version: str,
    slice_content_hash: str,
) -> dict[str, Any]:
    """Aggregate paired seen/private results without exposing private task data."""
    records = {"seen": seen_results, "held_out": held_out_results}
    tasks = {"seen": seen_tasks, "held_out": held_out_tasks}
    aliases = sorted(
        {str(row.get("candidate_alias")) for rows in records.values() for row in rows}
    )
    candidates: list[dict[str, Any]] = []
    for alias in aliases:
        aggregate: dict[str, dict[str, Any]] = {}
        stratum_counts: dict[str, dict[tuple[str, str], list[int]]] = {}
        for slice_name in ("seen", "held_out"):
            rows = [row for row in records[slice_name] if str(row.get("candidate_alias")) == alias]
            expected = set(tasks[slice_name])
            observed = [str(row.get("task_id")) for row in rows]
            if len(observed) != len(set(observed)) or set(observed) != expected:
                raise HeldOutEvalError(
                    f"candidate {alias!r} must be evaluated exactly once on every {slice_name} task"
                )
            successes = sum(bool(row.get("task_success")) for row in rows)
            total = len(rows)
            aggregate[slice_name] = {
                "successful_tasks": successes,
                "task_count": total,
                "success_rate": successes / total if total else None,
                "wilson_95": wilson_interval(successes, total),
            }
            failure_counts: dict[tuple[str, str, str], int] = defaultdict(int)
            for row in rows:
                for failure in row.get("failure_records") or []:
                    key = (
                        str(failure.get("layer")),
                        str(failure.get("code")),
                        str(failure.get("attribution")),
                    )
                    failure_counts[key] += 1
            aggregate[slice_name]["failure_taxonomy"] = [
                {
                    "layer": layer,
                    "code": code,
                    "attribution": attribution,
                    "count": count,
                }
                for (layer, code, attribution), count in sorted(failure_counts.items())
            ]
            counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
            by_id = {str(row["task_id"]): row for row in rows}
            for task_id, task in tasks[slice_name].items():
                for stratum in _task_strata(task):
                    counts[stratum][1] += 1
                    counts[stratum][0] += bool(by_id[task_id].get("task_success"))
            stratum_counts[slice_name] = counts

        matched = sorted(
            set(stratum_counts["seen"]) & set(stratum_counts["held_out"])
        )
        strata = []
        for tool, turn_policy in matched:
            entry: dict[str, Any] = {
                "applicable_tool": tool,
                "turn_policy": turn_policy,
            }
            for slice_name in ("seen", "held_out"):
                successes, total = stratum_counts[slice_name][(tool, turn_policy)]
                entry[slice_name] = {
                    "successful_tasks": successes,
                    "task_count": total,
                    "success_rate": successes / total,
                    "wilson_95": wilson_interval(successes, total),
                }
            entry["held_out_generalization_gap"] = (
                entry["seen"]["success_rate"] - entry["held_out"]["success_rate"]
            )
            entry["held_out_generalization_gap_95"] = generalization_gap_interval(
                seen_successes=entry["seen"]["successful_tasks"],
                seen_total=entry["seen"]["task_count"],
                held_out_successes=entry["held_out"]["successful_tasks"],
                held_out_total=entry["held_out"]["task_count"],
            )
            strata.append(entry)
        seen_rate = aggregate["seen"]["success_rate"]
        held_rate = aggregate["held_out"]["success_rate"]
        candidates.append(
            {
                "candidate_alias": alias,
                "seen": aggregate["seen"],
                "held_out": aggregate["held_out"],
                "held_out_generalization_gap": (
                    seen_rate - held_rate
                    if seen_rate is not None and held_rate is not None
                    else None
                ),
                "held_out_generalization_gap_95": generalization_gap_interval(
                    seen_successes=aggregate["seen"]["successful_tasks"],
                    seen_total=aggregate["seen"]["task_count"],
                    held_out_successes=aggregate["held_out"]["successful_tasks"],
                    held_out_total=aggregate["held_out"]["task_count"],
                ),
                "matched_applicable_tool_turn_policy_strata": strata,
            }
        )
    return {
        "schema_version": HELD_OUT_EVAL_REPORT_VERSION,
        "mode": "held_out_eval",
        "policy": policy.as_lineage(),
        "selection_mode": (
            "both"
            if policy.fixture_refs and policy.template_ids
            else "fixture_only"
            if policy.fixture_refs
            else "template_only"
        ),
        "private_slice": {
            "task_count": len(held_out_tasks),
            "content_hash": slice_content_hash,
            "seed": policy.seed,
            "pack_version": pack_version,
        },
        "candidates": candidates,
        "privacy": {
            "private_tasks_written": False,
            "private_prompts_written": False,
            "private_candidate_caches_written": False,
        },
    }


__all__ = [
    "HELD_OUT_EVAL_CONTRACT_VERSION",
    "HELD_OUT_EVAL_REPORT_VERSION",
    "HeldOutEvalConfig",
    "HeldOutEvalError",
    "build_validated_private_slice",
    "expand_private_held_out_tasks",
    "generalization_gap_interval",
    "held_out_generalization_report",
    "private_runtime_pack",
    "verify_held_out_policy",
    "wilson_interval",
]
