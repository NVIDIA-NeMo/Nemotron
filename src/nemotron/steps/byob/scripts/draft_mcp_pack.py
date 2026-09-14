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

"""Draft BFCL pack artifacts from an approved MCP evidence bundle.

Exit codes: 0 when every drafted artifact compiled, 2 when drafts were written but remain
blocked on unknowns that only executable probes can resolve, and 1 on failure. Exit 2 is the
normal outcome at L0 and is not an error: it means the drafts are honest about what nobody
has observed yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    resolved_config_digest,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import BundleError
from nemotron.steps.byob.runtime.pack_authoring.grounding import GroundingError
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    AuthoringModelError,
)
from nemotron.steps.byob.runtime.pack_authoring.runner import run_drafting
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="evidence_bundle.json")
    # Only a bundle migrated from legacy v1 has these; native v2 intake produces neither,
    # and requiring them would put the current path behind a compatibility artifact.
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--migration-record", type=Path)
    parser.add_argument("--certification-report", type=Path, required=True)
    parser.add_argument("--source-observations", type=Path)
    parser.add_argument("--domain-brief-source", type=Path, required=True)
    parser.add_argument("--domain-brief-report", type=Path, required=True)
    parser.add_argument("--held-out-redaction-report", type=Path, required=True)
    parser.add_argument("--held-out-policy", type=Path)
    parser.add_argument("--held-out-content", type=Path)
    parser.add_argument("--certification-public-key", type=Path, required=True)
    parser.add_argument("--certification-key-id", required=True)
    parser.add_argument("--exposure-authorization", type=Path, required=True)
    parser.add_argument("--organizational-policy-digest")
    parser.add_argument("--resolved-authoring-config", type=Path)
    parser.add_argument("--parent-evidence", type=Path)
    parser.add_argument("--open-questions", type=Path)
    parser.add_argument("--answer-set", type=Path)
    parser.add_argument(
        "--approval",
        type=Path,
        required=True,
        help="Reviewer approval naming the bundle digest and every flagged finding",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for drafts")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-canonical-id",
        required=True,
        help="Immutable model identity recorded in provenance",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Authoring defaults to greedy decoding so a rerun reproduces the draft",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help=(
            "Seconds one drafting call may take. Each of the four calls emits a whole plan "
            "covering every published tool, so the answer grows with the reviewed surface "
            "while the client's own default assumes a single short completion"
        ),
    )
    return parser


def _print(document: dict) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))

def _fail(document: dict) -> None:
    print(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        file=sys.stderr,
    )


def main() -> None:
    args = _parser().parse_args()
    model = AuthoringModel(
        alias=args.model_alias,
        provider=args.model_provider,
        model=args.model,
        canonical_id=args.model_canonical_id,
        seed=args.seed,
        inference_parameters={"temperature": args.temperature},
        request_timeout_s=args.request_timeout,
    )
    try:
        result = run_drafting(
            args.bundle,
            args.approval,
            args.output,
            model,
            certification_report_path=args.certification_report,
            trusted_certification_keys=load_trusted_certification_key(
                args.certification_public_key,
                key_id=args.certification_key_id,
            ),
            domain_brief_source_path=args.domain_brief_source,
            domain_brief_report_path=args.domain_brief_report,
            held_out_redaction_report_path=args.held_out_redaction_report,
            held_out_policy_path=args.held_out_policy,
            held_out_content_path=args.held_out_content,
            source_bundle_path=args.source_bundle,
            migration_record_path=args.migration_record,
            source_observations_path=args.source_observations,
            parent_evidence_path=args.parent_evidence,
            open_questions_path=args.open_questions,
            answer_set_path=args.answer_set,
            exposure_authorization_path=args.exposure_authorization,
            organizational_policy_digest=args.organizational_policy_digest,
            resolved_authoring_config_digest=(
                resolved_config_digest(args.resolved_authoring_config)
                if args.resolved_authoring_config is not None
                else None
            ),
        )
    except (BundleError, GroundingError, AuthoringModelError, OSError, ValueError) as exc:
        _fail(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "violations": list(getattr(exc, "violations", ())),
            }
        )
        raise SystemExit(1) from exc

    document = result.provenance.document
    _print(
        {
            "status": "drafted",
            "pack": document["pack"],
            "output_root": str(result.output_root),
            "draft_root": str(result.draft_root),
            "model": document["model"]["canonical_id"],
            "calls": [
                {
                    "stage": call["stage"],
                    "served_from_cache": call["served_from_cache"],
                }
                for call in document["calls"]
            ],
            "assertions_compiled": document["assertions_compiled"],
            "compilation_refusals": document["compilation_refusals"],
            "blocked_on": document["blocked_on"],
        }
    )
    if document["blocked_on"] or not document["assertions_compiled"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
