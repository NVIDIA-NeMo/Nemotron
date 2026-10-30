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

"""Freeze an approved canonical MCP pack into a read-only BFCL release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.release.freeze import (
    FreezeError,
    FreezeInputs,
    freeze_canonical_pack,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "pack",
        "mcp-config",
        "evidence",
        "intake-provenance",
        "gateway-attestation",
        "draft-provenance",
        "validation-report",
        "review-packet",
        "review-approval",
        "output",
        "source-bundle",
        "migration-record",
        "certification-report",
        "domain-brief-source",
        "domain-brief-report",
        "held-out-redaction-report",
        "source-observations",
        "certification-public-key",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--certification-key-id", required=True)
    parser.add_argument("--held-out-policy", type=Path)
    parser.add_argument("--held-out-content", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    inputs = FreezeInputs(
        pack_root=args.pack,
        mcp_config_path=args.mcp_config,
        evidence_path=args.evidence,
        intake_provenance_path=args.intake_provenance,
        gateway_attestation_path=args.gateway_attestation,
        draft_provenance_path=args.draft_provenance,
        validation_report_path=args.validation_report,
        review_packet_path=args.review_packet,
        review_approval_path=args.review_approval,
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
    )
    try:
        release = freeze_canonical_pack(inputs, args.output)
    except (FreezeError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                "status": "frozen",
                "release": str(release.root),
                "pack": str(release.pack_root),
                "frozen_pack_fingerprint": release.pack_fingerprint,
                "freeze_manifest_digest": release.manifest["record_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
