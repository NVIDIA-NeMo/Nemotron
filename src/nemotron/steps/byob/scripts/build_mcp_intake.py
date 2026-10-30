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

"""Derive a reviewable BFCL pack draft and evidence bundle from an MCP server.

Exit codes: 0 when the draft needs no human attention, 2 when hygiene flagged text a
reviewer must approve before the draft is used, and 1 on failure. The draft is never
publication ready on its own: fixtures, task templates, validation cases, and assertions
still need evidence this phase cannot observe.
"""

from __future__ import annotations

import argparse
import asyncio
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
from nemotron.steps.byob.runtime.mcp.authoring.intake import load_mcp_intake
from nemotron.steps.byob.runtime.mcp.authoring.pack_artifacts import (
    PENDING_PACK_ARTIFACTS,
)
from nemotron.steps.byob.runtime.mcp.authoring.runner import IntakeResult, run_intake
from nemotron.steps.byob.runtime.mcp.config import load_trusted_executable_policies
from nemotron.steps.byob.runtime.mcp.errors import McpIntegrationError
from nemotron.steps.byob.runtime.pack_authoring.untrusted_text import (
    UNTRUSTED_TAG,
    ProseHygieneError,
    quote_untrusted,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    load_certification_authority,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
    load_required_held_out_policy,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import AdapterProbePlan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intake",
        type=Path,
        required=True,
        help="Path to reviewed mcp_intake.yaml",
    )
    held_out = parser.add_mutually_exclusive_group(required=True)
    held_out.add_argument(
        "--held-out-policy",
        type=Path,
        help="Reviewed held-out policy; its canonical digest enters v2 evidence",
    )
    held_out.add_argument(
        "--held-out-not-applicable-reason",
        help="Reviewed reason this benchmark has no held-out requirement",
    )
    parser.add_argument(
        "--held-out-reviewed-by",
        required=True,
        help="Stable reviewer name or email for the held-out decision",
    )
    parser.add_argument(
        "--held-out-content",
        type=Path,
        help="Optional runtime-only YAML mapping containing reserved content to scan",
    )
    parser.add_argument(
        "--domain-brief",
        type=Path,
        required=True,
        help="Reviewed source/domain intent; sanitized and bound into v2 evidence",
    )
    parser.add_argument("--domain-brief-language", default="en")
    parser.add_argument(
        "--certification-private-key",
        type=Path,
        required=True,
        help="BFCL Ed25519 private key used to issue the A0 certification report",
    )
    parser.add_argument(
        "--certification-key-id",
        required=True,
        help="Allowlisted identifier for the matching BFCL public key",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the pack draft, evidence bundle, and provenance",
    )
    # A1 and A2 are observation tiers, so they are only reachable when the reviewer
    # supplies the bounded probe plan whose execution the certification report covers.
    # The plan is the same transport-neutral document a local package or an endpoint is
    # probed with; only mode A can be probed, because only mode A can be reset.
    parser.add_argument("--probe-plan", type=Path)
    parser.add_argument("--resolved-authoring-config", type=Path)
    parser.add_argument(
        "--trusted-executables",
        type=Path,
        help="Host-owned policy file required for stdio transport",
    )
    parser.add_argument(
        "--allow-insecure-localhost",
        action="store_true",
        help="Allow debug-only http:// loopback MCP transport; never publication eligible",
    )
    return parser


def _flagged_text(result: IntakeResult) -> list[dict[str, str]]:
    """Show a reviewer the exact text that was flagged, fenced as data."""
    descriptions = {
        f"tools.{tool['published_name']}.description": tool["description"][UNTRUSTED_TAG]
        for tool in result.bundle.document["tools"]
    }
    quoted: list[dict[str, str]] = []
    for finding in result.bundle.document["review"]["advisory"]:
        text = descriptions.get(finding["location"])
        if text is None:
            continue
        quoted.append(
            {
                "location": finding["location"],
                "code": finding["code"],
                "text": quote_untrusted(text),
            }
        )
    return quoted


async def _run(args: argparse.Namespace) -> IntakeResult:
    if args.held_out_policy is None and args.held_out_content is not None:
        raise ValueError("--held-out-content requires --held-out-policy")
    probe_plan = (
        AdapterProbePlan.model_validate(
            json.loads(args.probe_plan.read_text(encoding="utf-8"))
        )
        if args.probe_plan is not None
        else None
    )
    policies = (
        load_trusted_executable_policies(args.trusted_executables)
        if args.trusted_executables is not None
        else None
    )
    held_out = (
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
    resolved_config_digest = None
    required_tier = AdapterTier.A0
    if args.resolved_authoring_config is not None:
        resolved = load_resolved_authoring_config(args.resolved_authoring_config)
        verify_resolved_authoring_inputs(
            resolved,
            adapter_kind="mcp_mode_a",
            source=args.intake,
            domain_brief=args.domain_brief,
        )
        intake = load_mcp_intake(
            args.intake,
            allow_insecure_localhost=args.allow_insecure_localhost,
        )
        semantic = resolved.semantic_payload
        if (
            semantic.adapter_kind.value != "mcp_mode_a"
            or semantic.pack_id.value != intake.value.pack.pack_id
            or semantic.pack_version.value != intake.value.pack.version
        ):
            raise ValueError(
                "MCP intake pack does not match resolved authoring configuration"
            )
        resolved_config_digest = resolved.resolved_authoring_config_digest
        required_tier = AdapterTier(
            resolved.semantic_payload.required_certification_tier.value
        )
        if (
            semantic.rollout_policy is None
            or not semantic.rollout_policy.live_authoring_enabled.value
        ):
            raise ValueError("resolved configuration does not enable MCP authoring")
        require_no_rollout_revocation("mcp_mode_a")
    else:
        require_adapter_rollout("mcp_mode_a")
    return await run_intake(
        args.intake,
        args.output,
        executable_policies=policies,
        allow_insecure_localhost=args.allow_insecure_localhost,
        domain_brief_path=args.domain_brief,
        domain_brief_language=args.domain_brief_language,
        certification_authority=load_certification_authority(
            args.certification_private_key,
            key_id=args.certification_key_id,
        ),
        held_out_decision=held_out,
        held_out_policy_path=args.held_out_policy,
        held_out_content_path=args.held_out_content,
        probe_plan=probe_plan,
        resolved_authoring_config_digest=resolved_config_digest,
        required_tier=required_tier,
    )


def _print(document: dict) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))

def _fail(document: dict) -> None:
    print(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        file=sys.stderr,
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        result = asyncio.run(_run(args))
    except (McpIntegrationError, ProseHygieneError, OSError, ValueError) as exc:
        _fail(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        )
        raise SystemExit(1) from exc

    advisory = result.bundle.document["review"]["advisory"]
    assert result.source_evidence is not None
    _print(
        {
            "status": result.bundle.document["status"],
            "attained_tier": result.source_evidence.certification.attained_tier,
            "pack": result.bundle.document["pack"],
            "oracle": result.bundle.document["oracle"],
            "output_root": str(result.output_root),
            "artifacts": [
                artifact.as_dict(root=result.output_root)
                for artifact in result.artifacts
            ],
            "evidence_bundle_digest": result.source_evidence.bundle_digest,
            "tool_count": len(result.bundle.document["tools"]),
            "advisory_findings": advisory,
            "flagged_text": _flagged_text(result),
            "unknowns": [item["field"] for item in result.bundle.document["unknowns"]],
            "pending_artifacts": list(PENDING_PACK_ARTIFACTS),
        }
    )
    if advisory:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
