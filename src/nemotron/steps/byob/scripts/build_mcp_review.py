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

"""Build the deterministic domain-review packet for an MCP-backed BFCL pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.release.review import (
    ReviewError,
    build_review_packet,
    write_review_packet,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--migration-record", type=Path, required=True)
    parser.add_argument("--certification-report", type=Path, required=True)
    parser.add_argument("--domain-brief-source", type=Path, required=True)
    parser.add_argument("--domain-brief-report", type=Path, required=True)
    parser.add_argument("--held-out-redaction-report", type=Path, required=True)
    parser.add_argument("--held-out-policy", type=Path)
    parser.add_argument("--held-out-content", type=Path)
    parser.add_argument("--certification-public-key", type=Path, required=True)
    parser.add_argument("--certification-key-id", required=True)
    parser.add_argument("--intake-provenance", type=Path, required=True)
    parser.add_argument("--gateway-attestation", type=Path, required=True)
    parser.add_argument("--draft-provenance", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--mcp-config", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--held-out", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        packet = build_review_packet(
            args.evidence,
            args.intake_provenance,
            args.gateway_attestation,
            args.draft_provenance,
            args.validation_report,
            args.mcp_config,
            args.pack,
            held_out_path=args.held_out,
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
        )
        path = write_review_packet(packet, args.output)
    except (ReviewError, OSError, ValueError) as exc:
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
                "status": packet.document["status"],
                "packet_digest": packet.digest,
                "output": str(path),
                "blockers": packet.document["blockers"],
                "risk_ids": [
                    item["id"] for item in packet.document["metadata_risks"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if packet.document["status"] != "ready_for_approval":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
