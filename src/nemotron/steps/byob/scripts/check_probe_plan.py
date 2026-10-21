#!/usr/bin/env python3
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

"""Check a probe plan against a local Python source without executing probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemotron.steps.byob.runtime.pack_authoring.probe_planning import (
    certain_mutation_conflicts,
    plan_findings,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    inspect_local_python_package,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import (
    AdapterProbePlan,
    ProbeError,
    validate_probe_plan,
)


def _plan_shape(plan: AdapterProbePlan) -> dict[str, Any]:
    counts = {"success": 0, "structured_error": 0, "timeout": 0}
    for case in plan.cases:
        counts[case.expectation] += 1
    return {
        "digest": plan.digest,
        "cases": len(plan.cases),
        "by_expectation": counts,
        "carries_fixtures": plan.fixtures is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the local Python source package",
    )
    parser.add_argument(
        "--probe-plan",
        type=Path,
        required=True,
        help="Path to the probe plan JSON to check",
    )
    args = parser.parse_args()

    # An unusable plan and a plan that merely cannot certify to A2 are different answers:
    # the first must be rewritten before intake, the second is a decision about how much
    # evidence the pack is meant to carry.
    try:
        plan = AdapterProbePlan.model_validate(json.loads(args.probe_plan.read_text(encoding="utf-8")))
        source = args.source.resolve()
        inspection = inspect_local_python_package(
            source,
            allowed_roots=(source if source.is_dir() else source.parent,),
        )
        tools = {tool.published_name: tool for tool in inspection.tools}
        validate_probe_plan(inspection.tools, plan)
        conflicts = certain_mutation_conflicts(tools, plan)
        if conflicts:
            named = ", ".join(item["case_id"] for item in conflicts)
            raise ProbeError(
                "mutation_declaration_mismatch",
                f"cases claim a state change on read-only tools: {named}",
            )
    except (OSError, ValueError) as exc:
        detail: Any = getattr(exc, "detail", str(exc))
        if isinstance(exc, ValidationError):
            detail = exc.errors(include_url=False)
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "refusal_code": getattr(exc, "code", None),
                    "reason": detail,
                },
                default=str,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    findings = plan_findings(tools, plan)
    blocked = [item for item in findings if item["impact"] == "blocks_a2"]
    report = {
        "schema_version": "1.0",
        "status": "pass",
        "plan": _plan_shape(plan),
        "source": {
            "path": str(source),
            "tools": sorted(tools),
            "mutating_tools": sorted(n for n, t in tools.items() if t.mutates),
            "confirmation_tools": sorted(n for n, t in tools.items() if t.requires_confirmation),
        },
        "intake_gate": "accept",
        "attainable_tier": "A1" if blocked else "A2",
        "findings": findings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
