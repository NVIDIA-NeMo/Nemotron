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

"""Build an adapter-neutral v2 review packet from verified authoring records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_release.assembly import (
    ReviewContext,
    assemble_review,
)
from nemotron.steps.byob.runtime.authoring_release.review import write_review_packet
from nemotron.steps.byob.runtime.pack_authoring.artifacts import write_canonical_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"named path must be unique NAME=PATH, got {value!r}")
        result[name] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-kind",
        choices=("local_python", "http_package", "mcp_mode_a"),
        required=True,
    )
    parser.add_argument("--pack", type=Path, required=True)
    for name in (
        "evidence",
        "certification-report",
        "certification-public-key",
        "domain-brief-source",
        "domain-brief-report",
        "held-out-redaction-report",
        "source-observations",
        "intake-provenance",
        "draft-provenance",
        "validation-report",
        "validation-config",
        "resolved-authoring-config",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--certification-key-id", required=True)
    parser.add_argument("--exposure-authorization", type=Path)
    parser.add_argument("--evidence-approval", type=Path)
    parser.add_argument("--held-out-policy", type=Path)
    parser.add_argument("--held-out-content", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--migration-record", type=Path)
    parser.add_argument("--parent-evidence", type=Path)
    parser.add_argument("--open-questions", type=Path)
    parser.add_argument("--answer-set", type=Path)
    parser.add_argument("--organizational-policy-digest")
    parser.add_argument("--adapter-record", action="append", default=[])
    parser.add_argument("--freeze-sidecar", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-inputs-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        adapter_records = _named_paths(args.adapter_record)
        sidecars = _named_paths(args.freeze_sidecar)
        context = ReviewContext(
            evidence_path=args.evidence,
            certification_report_path=args.certification_report,
            trusted_certification_keys=load_trusted_certification_key(
                args.certification_public_key,
                key_id=args.certification_key_id,
            ),
            domain_brief_source_path=args.domain_brief_source,
            domain_brief_report_path=args.domain_brief_report,
            held_out_redaction_report_path=args.held_out_redaction_report,
            source_observations_path=args.source_observations,
            intake_provenance_path=args.intake_provenance,
            draft_provenance_path=args.draft_provenance,
            validation_report_path=args.validation_report,
            validation_config_path=args.validation_config,
            resolved_authoring_config_path=args.resolved_authoring_config,
            exposure_authorization_path=args.exposure_authorization,
            evidence_approval_path=args.evidence_approval,
            held_out_policy_path=args.held_out_policy,
            held_out_content_path=args.held_out_content,
            source_bundle_path=args.source_bundle,
            migration_record_path=args.migration_record,
            parent_evidence_path=args.parent_evidence,
            open_questions_path=args.open_questions,
            answer_set_path=args.answer_set,
            adapter_records=adapter_records,
            freeze_sidecars=sidecars,
            organizational_policy_digest=args.organizational_policy_digest,
        )
        assembled = assemble_review(
            adapter_kind=args.adapter_kind,
            pack_root=args.pack,
            context=context,
        )
        packet_path = write_review_packet(assembled.packet, args.output)
        freeze_inputs_path = write_canonical_json(
            {
                "schema_version": "bfcl-authoring-freeze-inputs-v1",
                "adapter_kind": args.adapter_kind,
                "pack_root": str(args.pack.resolve()),
                "review_packet": str(packet_path),
                "source_records": {
                    name: str(path.resolve())
                    for name, path in sorted(assembled.source_records.items())
                },
                "freeze_sidecars": {
                    name: str(path.resolve())
                    for name, path in sorted(sidecars.items())
                },
            },
            args.freeze_inputs_output,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "code": getattr(exc, "code", "review_assembly_failed"),
                    "reason": str(exc),
                    "recovery": getattr(
                        exc,
                        "recovery",
                        "repair the verified authoring records and retry",
                    ),
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
                "status": (
                    "blocked" if assembled.packet.document["blockers"] else "review_ready"
                ),
                "adapter_kind": args.adapter_kind,
                "packet_digest": assembled.packet.digest,
                "output": str(packet_path),
                "freeze_inputs": str(freeze_inputs_path),
                "blockers": assembled.packet.document["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if assembled.packet.document["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
