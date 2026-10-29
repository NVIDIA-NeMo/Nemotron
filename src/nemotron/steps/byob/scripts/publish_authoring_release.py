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

"""Freshly validate and publish one adapter-neutral v2 authoring release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_release.handoff import (
    AuthoringHandoffError,
    handoff_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.publication import (
    publication_adapter_for_release,
)
from nemotron.steps.byob.runtime.authoring_release.revocation import (
    RevocationRegistryVerifier,
    load_trusted_revocation_key,
)
from nemotron.steps.byob.runtime.release_seal import (
    load_trusted_release_seal_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seal-issuer", required=True)
    parser.add_argument("--seal-public-key", type=Path, required=True)
    parser.add_argument("--seal-key-id", required=True)
    parser.add_argument("--revocation-registry", type=Path)
    parser.add_argument("--revocation-issuer")
    parser.add_argument("--revocation-public-key", type=Path)
    parser.add_argument("--revocation-key-id")
    parser.add_argument("--revocation-minimum-generation", type=int, default=1)
    args = parser.parse_args()
    try:
        revocation_values = (
            args.revocation_registry,
            args.revocation_issuer,
            args.revocation_public_key,
            args.revocation_key_id,
        )
        if any(value is not None for value in revocation_values) and not all(
            value is not None for value in revocation_values
        ):
            raise ValueError("revocation registry, issuer, public key, and key ID must be supplied together")
        revocation_check = None
        if args.revocation_registry is not None:
            revocation_check = RevocationRegistryVerifier(
                path=args.revocation_registry,
                expected_issuer=args.revocation_issuer,
                trusted_public_keys=load_trusted_revocation_key(
                    args.revocation_public_key,
                    key_id=args.revocation_key_id,
                ),
                minimum_generation=args.revocation_minimum_generation,
            )
        trusted_seal_keys = load_trusted_release_seal_key(
            args.seal_public_key,
            key_id=args.seal_key_id,
        )
        adapter = publication_adapter_for_release(
            args.release,
            trusted_seal_keys=trusted_seal_keys,
            expected_seal_issuer=args.seal_issuer,
        )
        result = handoff_frozen_release(
            args.release,
            args.config,
            adapter=adapter,
            trusted_seal_keys=trusted_seal_keys,
            expected_seal_issuer=args.seal_issuer,
            revocation_check=revocation_check,
        )
    except (AuthoringHandoffError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "code": getattr(exc, "code", "publication_failed"),
                    "reason": str(exc),
                    "recovery": getattr(
                        exc,
                        "recovery",
                        "repair the frozen release or publication config and retry",
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
                "status": "published",
                "adapter": result.release.adapter_kind,
                "validation_report": str(result.validation_report_path),
                "benchmark": str(result.benchmark_path),
                "benchmark_raw": str(result.raw_benchmark_path),
                "run_manifest": str(result.run_manifest_path),
                "frozen_pack_fingerprint": result.release.pack_fingerprint,
                "run_id": result.run_manifest.get("run_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
