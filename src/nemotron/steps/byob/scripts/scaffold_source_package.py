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

"""Write the `backend.py` and `fixtures.json` a reviewed `tools.json` already determines.

Two lanes, and the difference between them is who chose the behaviour.

Without `--draft-with-model` nothing chooses it. The output is the interface the episode
runner calls, one raising handler per published tool, and a fixture row carrying every
parameter a published call has to be given. That is mechanical, so it is also reproducible:
the same catalogue writes the same bytes, and there is no model, no cache and no network.

With `--draft-with-model` a model answers in a declarative vocabulary — which collection a
tool reads, which field identifies a row, which state forbids the operation — and this
command compiles that into the same generated shape. The model does not write Python. That
is not a stylistic preference: the probes execute this file, so a model that authored the
oracle would be certifying its own output, and text it emitted directly would be arbitrary
code inside the least-privilege boundary.

Either way every generated file carries `BFCL-TODO` and intake refuses it while one remains.
The scaffold is the boilerplate; the oracle is still yours.
"""

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
    AuthoringModelError,
    call_structured,
)
from nemotron.steps.byob.runtime.pack_authoring.source_scaffolding import (
    DEFAULT_SKELETON_ROWS,
    REVIEW_MARKER,
    SOURCE_DRAFT_PROMPT_VERSION,
    SOURCE_DRAFT_SYSTEM_PROMPT,
    SOURCE_DRAFT_TASK,
    SourceDraft,
    SourceScaffoldError,
    compile_behaviours,
    compile_fixtures,
    draft_columns,
    draft_findings,
    fixture_findings,
    interface_findings,
    read_surface,
    render_backend,
    render_fixtures,
)

_LOCK = {"schema_version": "bfcl-python-dependency-lock-v1", "dependencies": []}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True, help="Reviewed tools.json to generate from")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Source package directory; existing backend.py or fixtures.json are never overwritten",
    )
    parser.add_argument(
        "--collection",
        default="records",
        help="Name for the single collection the mechanical fixture skeleton writes",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_SKELETON_ROWS,
        help="How many placeholder rows that collection gets; two error cases need two rows",
    )
    parser.add_argument(
        "--confirmation-parameter",
        default="confirm",
        help="Parameter a confirming call sets, which must match the probe plan's own",
    )
    parser.add_argument(
        "--status-field",
        default="status",
        help="Reply field a confirmation-gated tool reports its pending state in, which must "
        "match the probe plan's own",
    )
    parser.add_argument(
        "--pending-status",
        default="awaiting_confirmation",
        help="Value that field holds when a call was withheld for confirmation; the "
        "confirmation probe reads the reply by it, so a mismatch fails A2",
    )
    parser.add_argument(
        "--dependency-lock",
        action="store_true",
        help="Also write the empty dependency-lock.json a stdlib-only source needs",
    )
    parser.add_argument(
        "--draft-with-model",
        action="store_true",
        help="Opt in to a model choosing the fixture data and the declarative behaviours",
    )
    parser.add_argument("--domain-brief", type=Path, help="Required with --draft-with-model")
    parser.add_argument(
        "--error-vocabulary",
        type=Path,
        help=(
            "JSON object mapping reviewed error codes to their meaning; with one, a drafted "
            "behaviour may only name a listed code"
        ),
    )
    parser.add_argument("--cache", type=Path, help="Model IO cache; defaults to one beside the output")
    parser.add_argument("--model-alias")
    parser.add_argument("--model-provider")
    parser.add_argument("--model")
    parser.add_argument("--model-canonical-id", help="Immutable model identity recorded with the draft")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Drafting defaults to greedy decoding so a rerun reproduces the source",
    )
    parser.add_argument("--request-timeout", type=int, default=600)
    return parser


def _fail(document: dict[str, Any]) -> None:
    print(json.dumps(document, default=str, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)


def _load_vocabulary(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in vocabulary.items()
    ):
        raise ValueError("error vocabulary must map code strings to meaning strings")
    return vocabulary


def _draft(
    args: argparse.Namespace,
    *,
    surface: tuple,
    vocabulary: dict[str, str],
    output: Path,
) -> tuple[SourceDraft, dict[str, Any], dict[str, Any]]:
    missing = [
        flag
        for flag, value in (
            ("--domain-brief", args.domain_brief),
            ("--model-alias", args.model_alias),
            ("--model-provider", args.model_provider),
            ("--model", args.model),
            ("--model-canonical-id", args.model_canonical_id),
        )
        if not value
    ]
    if missing:
        raise ValueError("--draft-with-model also requires " + ", ".join(missing))
    model = AuthoringModel(
        alias=args.model_alias,
        provider=args.model_provider,
        model=args.model,
        canonical_id=args.model_canonical_id,
        seed=args.seed,
        inference_parameters={"temperature": args.temperature},
        request_timeout_s=args.request_timeout,
    )
    response, record = call_structured(
        model,
        stage_name="source_draft",
        prompt_version=SOURCE_DRAFT_PROMPT_VERSION,
        system_prompt=SOURCE_DRAFT_SYSTEM_PROMPT,
        prompt=SOURCE_DRAFT_TASK,
        columns=draft_columns(
            brief=args.domain_brief.read_text(encoding="utf-8"),
            tools=surface,
            error_vocabulary=vocabulary,
        ),
        output_format=SourceDraft,
        cache=ImmutableModelIOCache(args.cache or output / "model-io-cache.jsonl"),
        run_dir=output / "model_runs",
    )
    return SourceDraft.model_validate(response), record.as_dict(), model.as_provenance()


def main() -> None:
    args = _parser().parse_args()
    output = args.output.resolve()
    backend_path = output / "backend.py"
    fixtures_path = output / "fixtures.json"

    model_call: dict[str, Any] | None = None
    model_provenance: dict[str, Any] | None = None
    drafted: dict[str, Any] = {}
    try:
        # Checked before the model is called, because a refusal after one is a refusal that
        # cost a request, and the answer would be cached against an output nobody can write.
        existing = [path.name for path in (backend_path, fixtures_path) if path.exists()]
        if existing:
            raise ValueError(
                f"{output} already holds {', '.join(existing)}; generating over reviewed work "
                "would lose it, so write to a new directory and merge deliberately"
            )
        surface = read_surface(json.loads(args.tools.read_text(encoding="utf-8")))
        vocabulary = _load_vocabulary(args.error_vocabulary)
        output.mkdir(parents=True, exist_ok=True)

        if args.draft_with_model:
            draft, model_call, model_provenance = _draft(
                args,
                surface=surface,
                vocabulary=vocabulary,
                output=output,
            )
            fixtures = compile_fixtures(draft)
            behaviours = compile_behaviours(
                draft,
                tools=surface,
                fixtures=fixtures,
                error_vocabulary=vocabulary,
                confirmation_parameter=args.confirmation_parameter,
            )
            drafted = {
                "collections": sorted(fixtures),
                "behaviours": {name: item.operation for name, item in sorted(behaviours.items())},
            }
        else:
            fixtures = render_fixtures(
                surface,
                collection=args.collection,
                rows=args.rows,
                confirmation_parameter=args.confirmation_parameter,
            )
            behaviours = {}

        backend = render_backend(
            surface,
            behaviours=behaviours,
            confirmation_parameter=args.confirmation_parameter,
            status_field=args.status_field,
            pending_status=args.pending_status,
        )
        findings = [
            *interface_findings(backend, published=[tool.name for tool in surface]),
            *fixture_findings(fixtures),
            *draft_findings(surface, behaviours),
        ]
    # AuthoringModelError is listed because a provider that times out or answers off-schema is
    # an ordinary outcome of this command, not a defect in it, and a caller parsing stdout
    # deserves the same document for it as for a bad tools.json.
    except (OSError, ValueError, SourceScaffoldError, AuthoringModelError) as exc:
        _fail(
            {
                "schema_version": "1.0",
                "status": "fail",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        )
        raise SystemExit(1) from exc

    backend_path.write_text(backend, encoding="utf-8")
    fixtures_path.write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written = [str(backend_path), str(fixtures_path)]
    if args.dependency_lock:
        lock_path = output / "dependency-lock.json"
        if not lock_path.exists():
            lock_path.write_text(json.dumps(_LOCK, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(str(lock_path))
    if drafted:
        draft_path = output / "source-draft.json"
        draft_path.write_text(
            json.dumps(drafted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(str(draft_path))

    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "pass",
                "written": sorted(written),
                "tools": [tool.name for tool in surface],
                "drafted_with_model": bool(args.draft_with_model),
                # Always true on a fresh scaffold, and stated rather than implied: this is
                # the one field that says the output is not yet a source.
                "review_required": REVIEW_MARKER in backend,
                "findings": findings,
                "model_call": model_call,
                "model": model_provenance,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
