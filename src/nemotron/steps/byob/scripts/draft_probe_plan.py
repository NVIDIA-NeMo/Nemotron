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

"""Draft a probe plan for a local Python source, for review before intake runs it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    call_structured,
)
from nemotron.steps.byob.runtime.pack_authoring.probe_planning import (
    PROBE_PLAN_PROMPT_VERSION,
    PROBE_PLAN_SYSTEM_PROMPT,
    PROBE_PLAN_TASK,
    ProbePlanDraft,
    ProbePlanDraftError,
    build_columns,
    certain_mutation_conflicts,
    materialize_plan,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local Python source package, read for tools.json and fixtures.json",
    )
    parser.add_argument("--domain-brief", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Where to write the plan")
    parser.add_argument(
        "--clock",
        required=True,
        help="ISO-8601 instant with an explicit timezone that probe episodes are pinned to",
    )
    parser.add_argument(
        "--error-vocabulary",
        type=Path,
        help=(
            "JSON object mapping each reviewed error code to what raises it. Without one "
            "the draft writes no error case, because a code the source does not raise "
            "fails the A1 error probe whereas an absent case is permitted"
        ),
    )
    parser.add_argument(
        "--error-path",
        default="error.code",
        help=(
            "Dotted path in a tool's return where an error code sits; must match the path "
            "the vocabulary was extracted from, since the plan is probed at whichever one "
            "it carries"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Model IO cache file; defaults to model-io-cache.jsonl beside the output",
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-canonical-id",
        required=True,
        help="Immutable model identity recorded alongside the draft",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Drafting defaults to greedy decoding so a rerun reproduces the plan",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Seconds the call may take; one answer covers every published tool",
    )
    return parser


def _fail(document: dict[str, Any]) -> None:
    print(
        json.dumps(document, default=str, ensure_ascii=False, indent=2, sort_keys=True),
        file=sys.stderr,
    )


def main() -> None:
    args = _parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    try:
        error_path = tuple(part for part in args.error_path.split(".") if part)
        if not error_path:
            raise ValueError("--error-path must name at least one field")
        tools = json.loads((source / "tools.json").read_text(encoding="utf-8"))
        fixtures = json.loads((source / "fixtures.json").read_text(encoding="utf-8"))
        brief = args.domain_brief.read_text(encoding="utf-8")
        vocabulary: dict[str, str] = {}
        if args.error_vocabulary is not None:
            vocabulary = json.loads(args.error_vocabulary.read_text(encoding="utf-8"))
            if not isinstance(vocabulary, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in vocabulary.items()
            ):
                raise ValueError("error vocabulary must map code strings to meaning strings")
    except (OSError, ValueError) as exc:
        _fail({"schema_version": "1.0", "status": "fail", "reason": str(exc)})
        raise SystemExit(1) from exc

    model = AuthoringModel(
        alias=args.model_alias,
        provider=args.model_provider,
        model=args.model,
        canonical_id=args.model_canonical_id,
        seed=args.seed,
        inference_parameters={"temperature": args.temperature},
        request_timeout_s=args.request_timeout,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or output.parent / "model-io-cache.jsonl"

    try:
        response, record = call_structured(
            model,
            stage_name="probe_plan",
            prompt_version=PROBE_PLAN_PROMPT_VERSION,
            system_prompt=PROBE_PLAN_SYSTEM_PROMPT,
            prompt=PROBE_PLAN_TASK,
            columns=build_columns(
                brief=brief,
                tools=tools,
                fixtures=fixtures,
                error_vocabulary=vocabulary,
            ),
            output_format=ProbePlanDraft,
            cache=ImmutableModelIOCache(cache_path),
            run_dir=output.parent / "model_runs",
        )
        draft = ProbePlanDraft.model_validate(response)
        document = materialize_plan(
            draft,
            tools=tools,
            fixtures=fixtures,
            clock=args.clock,
            seed=args.seed,
            error_vocabulary=vocabulary,
            error_path=error_path,
        )
        # Validated before it is written. A plan on disk is a plan somebody will run, and
        # the drafted parts are exactly the ones intake would refuse.
        plan = AdapterProbePlan.model_validate(document)
        inspection = inspect_local_python_package(
            source,
            allowed_roots=(source if source.is_dir() else source.parent,),
        )
        declared = {tool.published_name: tool for tool in inspection.tools}
        validate_probe_plan(inspection.tools, plan)
        conflicts = certain_mutation_conflicts(declared, plan)
        if conflicts:
            named = ", ".join(item["case_id"] for item in conflicts)
            raise ProbePlanDraftError(f"drafted cases claim a state change on read-only tools: {named}")
    except (OSError, ValueError, ProbeError) as exc:
        _fail(
            {
                "schema_version": "1.0",
                "status": "fail",
                "error_type": type(exc).__name__,
                "refusal_code": getattr(exc, "code", None),
                "reason": getattr(exc, "detail", str(exc)),
            }
        )
        raise SystemExit(1) from exc

    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    findings = plan_findings(declared, plan)
    blocked = [item for item in findings if item["impact"] == "blocks_a2"]
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "pass",
                "output": str(output),
                "plan_digest": plan.digest,
                "cases": len(plan.cases),
                "attainable_tier": "A1" if blocked else "A2",
                "findings": findings,
                "model_call": record.as_dict(),
                "model": model.as_provenance(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
