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

"""Run the MCP-to-BFCL Oracle HTTP v1 gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette

from nemotron.steps.byob.runtime.mcp import (
    load_mcp_oracle_config,
    load_trusted_executable_policies,
)
from nemotron.steps.byob.runtime.mcp.errors import McpIntegrationError
from nemotron.steps.byob.runtime.mcp.gateway import (
    GatewayArtifacts,
    GatewayError,
    GatewayService,
    create_gateway_app,
)
from nemotron.steps.byob.runtime.mcp.gateway.conformance import (
    load_conformance_evidence,
)
from nemotron.steps.byob.runtime.mcp.rollout import require_mcp_feature
from nemotron.steps.byob.runtime.release_seal import (
    load_trusted_release_seal_key,
)

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trusted-executables", type=Path)
    parser.add_argument("--gateway-artifact-digest", required=True)
    parser.add_argument("--shim-artifact-digest")
    parser.add_argument("--snapshot-digest")
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--gateway-suite", type=Path)
    parser.add_argument("--evidence-issuer")
    parser.add_argument("--evidence-public-key", type=Path)
    parser.add_argument("--evidence-key-id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tls-certfile", type=Path)
    parser.add_argument("--tls-keyfile", type=Path)
    parser.add_argument(
        "--allow-insecure-loopback",
        action="store_true",
        help="Debug only: allow cleartext gateway and/or upstream on explicit loopback",
    )
    parser.add_argument(
        "--client-bearer-token-env",
        help="Optional environment variable clients must present as a bearer token",
    )
    parser.add_argument("--max-request-bytes", type=int, default=10 * 1024 * 1024)
    return parser


def _client_token(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    token = os.environ.get(env_name)
    if not token or "\r" in token or "\n" in token:
        raise ValueError(f"gateway client bearer environment variable {env_name!r} is missing or unsafe")
    return token


def _validate_bind(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.max_request_bytes <= 0:
        raise ValueError("--max-request-bytes must be positive")
    if (args.probe_report is None) != (args.gateway_suite is None):
        raise ValueError("--probe-report and --gateway-suite must be provided together")
    evidence_trust = (
        args.evidence_issuer,
        args.evidence_public_key,
        args.evidence_key_id,
    )
    if any(value is not None for value in evidence_trust) and not all(value is not None for value in evidence_trust):
        raise ValueError("evidence issuer, public key, and key ID must be supplied together")
    has_cert = args.tls_certfile is not None
    has_key = args.tls_keyfile is not None
    if has_cert != has_key:
        raise ValueError("--tls-certfile and --tls-keyfile must be provided together")
    if not has_cert and not (args.allow_insecure_loopback and args.host in _LOOPBACK):
        raise ValueError("TLS is required unless --allow-insecure-loopback binds an explicit loopback host")
    for value, label in (
        (args.tls_certfile, "--tls-certfile"),
        (args.tls_keyfile, "--tls-keyfile"),
    ):
        if value is not None and not value.resolve().is_file():
            raise ValueError(f"{label} is not a file: {value.resolve()}")


def _load_unique_json(path: Path) -> object:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"{path} repeats JSON key {key!r}")
            document[key] = value
        return document

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_pairs,
    )


def _build_app(args: argparse.Namespace) -> Starlette:
    require_mcp_feature()
    _validate_bind(args)
    loaded = load_mcp_oracle_config(
        args.config,
        allow_insecure_localhost=args.allow_insecure_loopback,
    )
    policies = (
        load_trusted_executable_policies(args.trusted_executables) if args.trusted_executables is not None else None
    )
    conformance_evidence = None
    if args.probe_report is not None and args.gateway_suite is not None:
        trusted_keys = (
            load_trusted_release_seal_key(
                args.evidence_public_key,
                key_id=args.evidence_key_id,
            )
            if args.evidence_public_key is not None
            else None
        )
        conformance_evidence = load_conformance_evidence(
            _load_unique_json(args.probe_report.resolve()),
            _load_unique_json(args.gateway_suite.resolve()),
            trusted_public_keys=trusted_keys,
            expected_issuer=args.evidence_issuer,
            gateway_artifact_digest=args.gateway_artifact_digest,
        )
    service = GatewayService(
        loaded,
        artifacts=GatewayArtifacts(
            gateway_artifact_digest=args.gateway_artifact_digest,
            shim_artifact_digest=args.shim_artifact_digest,
            snapshot_digest=args.snapshot_digest,
        ),
        executable_policies=policies,
        conformance_evidence=conformance_evidence,
    )
    return create_gateway_app(
        service,
        max_request_bytes=args.max_request_bytes,
        client_bearer_token=_client_token(args.client_bearer_token_env),
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        app = _build_app(args)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_certfile=(str(args.tls_certfile.resolve()) if args.tls_certfile is not None else None),
            ssl_keyfile=(str(args.tls_keyfile.resolve()) if args.tls_keyfile is not None else None),
            access_log=False,
        )
    except (GatewayError, McpIntegrationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
