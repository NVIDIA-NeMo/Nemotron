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

"""Discover and verify an MCP server against a strict BFCL MCP oracle profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.mcp import (
    discover_mcp_oracle,
    load_mcp_oracle_config,
    load_trusted_executable_policies,
    write_discovery_report,
)
from nemotron.steps.byob.runtime.mcp.errors import (
    McpIntegrationError,
)
from nemotron.steps.byob.runtime.mcp.rollout import require_mcp_feature


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to strict mcp_oracle.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for deterministic mcp_discovery_report.json",
    )
    parser.add_argument(
        "--trusted-executables",
        type=Path,
        help="Host-owned policy file required for stdio transport",
    )
    parser.add_argument(
        "--allow-insecure-localhost",
        action="store_true",
        help="Allow debug-only http:// loopback transport; never publication eligible",
    )
    parser.add_argument(
        "--bootstrap-catalog-digest",
        action="store_true",
        help="Write a pre-L0 report containing the observed digest instead of failing on its placeholder",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict:
    require_mcp_feature()
    loaded = load_mcp_oracle_config(
        args.config,
        allow_insecure_localhost=args.allow_insecure_localhost,
    )
    policies = (
        load_trusted_executable_policies(args.trusted_executables)
        if args.trusted_executables is not None
        else None
    )
    report = await discover_mcp_oracle(
        loaded,
        executable_policies=policies,
        verify_catalog_digest=not args.bootstrap_catalog_digest,
    )
    write_discovery_report(report, args.output)
    return report.to_dict()


def main() -> None:
    args = _parser().parse_args()
    try:
        report = asyncio.run(_run(args))
    except (McpIntegrationError, OSError, ValueError) as exc:
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
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report.get("status") != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
