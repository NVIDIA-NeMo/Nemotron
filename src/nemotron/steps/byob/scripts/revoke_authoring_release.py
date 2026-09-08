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

"""Issue a signed revocation or supersession registry snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_release.revocation import (
    ReleaseRevocationError,
    build_revocation_record,
    build_revocation_registry,
    exclusive_revocation_registry,
    load_revocation_authority,
    load_revocation_registry,
    revocation_target_from_release,
    write_revocation_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--action", choices=("revoke", "supersede"), required=True)
    parser.add_argument("--replacement-release", type=Path)
    parser.add_argument("--reason-code", required=True)
    parser.add_argument("--valid-days", type=int, default=30)
    args = parser.parse_args()
    try:
        if args.valid_days <= 0:
            raise ValueError("--valid-days must be positive")
        if (args.action == "supersede") != (args.replacement_release is not None):
            raise ValueError("--replacement-release is required only for action supersede")
        authority = load_revocation_authority(
            args.private_key,
            issuer=args.issuer,
            key_id=args.key_id,
        )
        now = datetime.now(timezone.utc)
        with exclusive_revocation_registry(args.registry):
            if args.registry.exists():
                current = load_revocation_registry(
                    args.registry,
                    expected_issuer=args.issuer,
                    trusted_public_keys={args.key_id: authority.public_key},
                    now=now,
                )
                records = current.records
                generation = current.generation + 1
            else:
                records = ()
                generation = 1
            seal_verification = {
                "expected_seal_issuer": args.issuer,
                "trusted_seal_keys": {args.key_id: authority.public_key},
            }
            target = revocation_target_from_release(
                args.release,
                **seal_verification,
            )
            prior = next(
                (
                    existing
                    for existing in reversed(records)
                    if existing.target.frozen_pack_fingerprint == target.frozen_pack_fingerprint
                ),
                None,
            )
            replacement = (
                revocation_target_from_release(
                    args.replacement_release,
                    **seal_verification,
                ).frozen_pack_fingerprint
                if args.replacement_release is not None
                else None
            )
            record = build_revocation_record(
                target,
                authority=authority,
                action=args.action,
                reason_code=args.reason_code,
                effective_at=now,
                prior=prior,
                replacement_frozen_pack_fingerprint=replacement,
            )
            registry = build_revocation_registry(
                (*records, record),
                authority=authority,
                generation=generation,
                generated_at=now,
                valid_until=now + timedelta(days=args.valid_days),
            )
            write_revocation_registry(registry, args.registry)
    except (OSError, ValueError, ReleaseRevocationError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "code": getattr(exc, "code", "revocation_failed"),
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
                "status": record.action,
                "target_frozen_pack_fingerprint": (record.target.frozen_pack_fingerprint),
                "replacement_frozen_pack_fingerprint": (record.replacement_frozen_pack_fingerprint),
                "record_digest": record.record_digest,
                "registry_digest": registry.registry_digest,
                "registry_generation": registry.generation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
