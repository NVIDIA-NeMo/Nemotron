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

"""Record final approval of one exact adapter-neutral review packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_release.review import (
    REQUIRED_CHECKLIST_V2,
    ReviewPacketV2,
    build_review_approval,
    load_review_packet,
    write_review_approval,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--acknowledge-risk", action="append", default=[])
    parser.add_argument("--note")
    parser.add_argument("--output", type=Path, required=True)
    for name in sorted(REQUIRED_CHECKLIST_V2):
        parser.add_argument(
            f"--accept-{name.replace('_', '-')}",
            action="store_true",
            required=True,
        )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        packet = load_review_packet(args.packet)
        if not isinstance(packet, ReviewPacketV2):
            raise ValueError("adapter-neutral approval requires a v2 review packet")
        approval = build_review_approval(
            packet,
            approved_by=args.approved_by,
            reviewed_at=args.reviewed_at,
            checklist={
                name: bool(getattr(args, f"accept_{name}"))
                for name in REQUIRED_CHECKLIST_V2
            },
            acknowledged_risks=args.acknowledge_risk,
            note=args.note,
        )
        path = write_review_approval(approval, args.output)
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "code": getattr(exc, "code", "release_approval_failed"),
                    "reason": str(exc),
                    "recovery": getattr(
                        exc,
                        "recovery",
                        "review the exact packet and every explicit checklist item",
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
                "status": "release_approved",
                "approval_digest": approval.digest,
                "review_packet_digest": packet.digest,
                "output": str(path),
                "note": "Final release approval is not model-exposure authorization.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
