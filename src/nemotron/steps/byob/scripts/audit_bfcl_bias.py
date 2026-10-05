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

"""Recompute the versioned BFCL B1-B16 bias audit from frozen artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.bfcl.bias_audit import (
    AuditInputs,
    BiasAuditError,
    build_bias_audit_report,
    write_bias_audit_reports,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument(
        "--expanded",
        type=Path,
        help=(
            "Rendered-conversation Parquet or stage_cache directory containing "
            "task_instances.parquet and rendered_conversations.parquet"
        ),
    )
    parser.add_argument("--pack-manifest", type=Path)
    parser.add_argument(
        "--contamination-report",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--distractor-evidence", type=Path)
    parser.add_argument("--judge-evidence", type=Path)
    parser.add_argument("--portability-evidence", type=Path)
    parser.add_argument("--exceptions", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    # A failing audit and an audit that could not run are different outcomes, so
    # they get different exit codes: automation that retries on a crash must not
    # also retry on a verdict it should report to a human instead.
    try:
        report = build_bias_audit_report(
            AuditInputs(
                run_manifest=args.run_manifest,
                output_dir=args.output_dir,
                published=args.published,
                raw=args.raw,
                expanded=args.expanded,
                pack_manifest=args.pack_manifest,
                contamination_reports=tuple(args.contamination_report),
                distractor_evidence=args.distractor_evidence,
                judge_evidence=args.judge_evidence,
                portability_evidence=args.portability_evidence,
                exceptions=args.exceptions,
            )
        )
        json_path, markdown_path = write_bias_audit_reports(
            report,
            args.output_dir,
        )
    except (BiasAuditError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(
        json.dumps(
            {
                "status": report["summary"]["status"],
                "report_hash": report["report_hash"],
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "failed_bias_ids": report["summary"]["failed_bias_ids"],
                "unexcepted_failure_bias_ids": report["summary"]["unexcepted_failure_bias_ids"],
            },
            sort_keys=True,
        )
    )
    if report["summary"]["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
