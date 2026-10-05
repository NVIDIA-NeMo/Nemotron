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

"""Build transport-neutral v2 intake for local Python or HTTP source packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    load_resolved_authoring_config,
    verify_resolved_authoring_inputs,
)
from nemotron.steps.byob.runtime.authoring_workflow.rollout import (
    require_adapter_rollout,
    require_no_rollout_revocation,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    load_certification_authority,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import PackIdentity
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
    load_required_held_out_policy,
)
from nemotron.steps.byob.runtime.source_adapters.intake import run_conventional_intake
from nemotron.steps.byob.runtime.source_adapters.probe_engine import AdapterProbePlan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        choices=("local_python", "http_package"),
        required=True,
    )
    parser.add_argument("--domain-brief", type=Path, required=True)
    parser.add_argument("--domain-brief-language", default="en")
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--pack-version", required=True)
    parser.add_argument("--certification-private-key", type=Path, required=True)
    parser.add_argument("--certification-key-id", required=True)
    parser.add_argument("--required-tier", choices=("A0", "A1", "A2"), default="A0")
    # A1 and A2 are observation tiers, so they are only reachable when the reviewer
    # supplies the bounded probe plan whose execution the certification report covers.
    # The plan is transport-neutral: the same document probes a local package or an
    # endpoint, because the questions on the ladder are the same either way.
    parser.add_argument("--probe-plan", type=Path)
    held_out = parser.add_mutually_exclusive_group(required=True)
    held_out.add_argument("--held-out-policy", type=Path)
    held_out.add_argument("--held-out-not-applicable-reason")
    parser.add_argument("--held-out-reviewed-by", required=True)
    parser.add_argument("--held-out-content", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-authoring-config", type=Path)
    return parser


def _print(document: dict[str, object]) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))

def _fail(document: dict[str, object]) -> None:
    print(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        file=sys.stderr,
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        source = args.source.resolve()
        resolved_config_digest = None
        if args.resolved_authoring_config is not None:
            resolved = load_resolved_authoring_config(args.resolved_authoring_config)
            verify_resolved_authoring_inputs(
                resolved,
                adapter_kind=args.adapter,
                source=source,
                domain_brief=args.domain_brief,
            )
            semantic = resolved.semantic_payload
            if (
                semantic.adapter_kind.value != args.adapter
                or semantic.pack_id.value != args.pack_id
                or semantic.pack_version.value != args.pack_version
                or semantic.required_certification_tier.value != args.required_tier
            ):
                raise ValueError(
                    "intake arguments do not match resolved authoring configuration"
                )
            resolved_config_digest = resolved.resolved_authoring_config_digest
            if (
                semantic.rollout_policy is None
                or not semantic.rollout_policy.live_authoring_enabled.value
            ):
                raise ValueError(
                    "resolved configuration does not enable this source adapter"
                )
            require_no_rollout_revocation(args.adapter)
        else:
            require_adapter_rollout(args.adapter)
        if args.held_out_policy is None and args.held_out_content is not None:
            raise ValueError("--held-out-content requires --held-out-policy")
        probe_plan = None
        if args.probe_plan is not None:
            probe_plan = AdapterProbePlan.model_validate(
                json.loads(args.probe_plan.read_text(encoding="utf-8"))
            )
        decision = (
            load_required_held_out_policy(
                args.held_out_policy,
                reviewed_by=args.held_out_reviewed_by,
            )
            if args.held_out_policy is not None
            else build_not_applicable_decision(
                args.held_out_not_applicable_reason,
                reviewed_by=args.held_out_reviewed_by,
            )
        )
        result = run_conventional_intake(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                args.adapter: {"path": str(source)},
            },
            args.output,
            source_base_dir=source.parent,
            allowed_roots=(source if source.is_dir() else source.parent,),
            pack=PackIdentity(pack_id=args.pack_id, version=args.pack_version),
            domain_brief_path=args.domain_brief,
            domain_brief_language=args.domain_brief_language,
            certification_authority=load_certification_authority(
                args.certification_private_key,
                key_id=args.certification_key_id,
            ),
            held_out_decision=decision,
            held_out_policy_path=args.held_out_policy,
            held_out_content_path=args.held_out_content,
            required_tier=AdapterTier(args.required_tier),
            probe_plan=probe_plan,
            resolved_authoring_config_digest=resolved_config_digest,
        )
    except (OSError, ValueError) as exc:
        _fail(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "code": getattr(exc, "code", "source_intake_failed"),
                "reason": str(exc),
                "recovery": getattr(
                    exc,
                    "recovery",
                    "fix the reviewed source or intake arguments and retry",
                ),
            }
        )
        raise SystemExit(1) from exc
    _print(
        {
            "status": "intake_complete",
            "adapter": args.adapter,
            "output": str(result.output_root),
            "evidence": str(result.evidence_path),
            "evidence_digest": result.finalized.evidence.bundle_digest,
            "certification_tier": result.finalized.certification.attained_tier.value,
            "next_commands": [
                "answer",
                "authorize",
                "approve --boundary evidence",
            ],
        }
    )


if __name__ == "__main__":
    main()
