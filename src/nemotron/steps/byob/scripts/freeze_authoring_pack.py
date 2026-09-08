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

"""Freeze one approved adapter-neutral v2 authoring release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_release.assembly import (
    release_adapter_for_packet,
)
from nemotron.steps.byob.runtime.authoring_release.freeze import (
    FreezeInputsV2,
    freeze_canonical_pack,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    ReviewPacketV2,
    load_json_mapping,
    load_review_packet,
)
from nemotron.steps.byob.runtime.release_seal import (
    load_release_seal_authority,
)

_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "adapter_kind",
        "pack_root",
        "review_packet",
        "source_records",
        "freeze_sidecars",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-inputs", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--signing-key-id", required=True)
    parser.add_argument("--seal-issuer", required=True)
    return parser


def _paths(value: object, label: str) -> dict[str, Path]:
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(path, str) for name, path in value.items()
    ):
        raise ValueError(f"{label} must map names to paths")
    return {name: Path(path) for name, path in value.items()}


def main() -> None:
    args = _parser().parse_args()
    try:
        document = load_json_mapping(args.freeze_inputs, "freeze inputs")
        if set(document) != _INPUT_KEYS or document.get("schema_version") != "bfcl-authoring-freeze-inputs-v1":
            raise ValueError("freeze inputs do not match the v1 handoff contract")
        packet_path = Path(str(document["review_packet"]))
        packet = load_review_packet(packet_path)
        if not isinstance(packet, ReviewPacketV2):
            raise ValueError("adapter-neutral freeze requires a v2 review packet")
        if packet.document["adapter_kind"] != document["adapter_kind"]:
            raise ValueError("freeze inputs adapter differs from its review packet")
        source_records = _paths(document["source_records"], "source_records")
        sidecars = _paths(document["freeze_sidecars"], "freeze_sidecars")
        adapter = release_adapter_for_packet(
            packet,
            freeze_sidecars=sidecars,
        )
        release = freeze_canonical_pack(
            FreezeInputsV2(
                pack_root=Path(str(document["pack_root"])),
                review_packet_path=packet_path,
                review_approval_path=args.approval,
                source_records=source_records,
                seal_authority=load_release_seal_authority(
                    args.signing_key,
                    issuer=args.seal_issuer,
                    key_id=args.signing_key_id,
                ),
            ),
            args.output,
            adapter=adapter,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "code": getattr(exc, "code", "authoring_freeze_failed"),
                    "reason": str(exc),
                    "recovery": getattr(
                        exc,
                        "recovery",
                        "restore the approved review inputs and retry",
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
                "status": "frozen",
                "adapter_kind": release.adapter_kind,
                "pack_fingerprint": release.pack_fingerprint,
                "manifest_digest": release.manifest["manifest_digest"],
                "output": str(release.root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
