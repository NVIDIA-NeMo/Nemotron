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

"""Fresh validation and publication handoff for v2 authoring releases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.authoring_release.contracts import PublicationAdapter
from nemotron.steps.byob.runtime.authoring_release.freeze import (
    FrozenReleaseV2,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.review import load_json_mapping


class AuthoringHandoffError(RuntimeError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


@dataclass(frozen=True)
class PublicationHandoffV2:
    release: FrozenReleaseV2
    validation_report_path: Path
    benchmark_path: Path
    raw_benchmark_path: Path
    run_manifest_path: Path
    run_manifest: Mapping[str, Any]


def handoff_frozen_release(
    release_root: Path,
    config_path: Path,
    *,
    adapter: PublicationAdapter,
    trusted_seal_keys: Mapping[str, Ed25519PublicKey] | None = None,
    expected_seal_issuer: str | None = None,
    revocation_check: Callable[[str], object] | None = None,
) -> PublicationHandoffV2:
    verification = {
        "trusted_seal_keys": trusted_seal_keys,
        "expected_seal_issuer": expected_seal_issuer,
    }

    def load_verified_release() -> object:
        if trusted_seal_keys is None and expected_seal_issuer is None:
            return load_frozen_release(release_root)
        return load_frozen_release(release_root, **verification)

    loaded = load_verified_release()
    if not isinstance(loaded, FrozenReleaseV2):
        raise AuthoringHandoffError(
            "release_version_mismatch",
            "v2 handoff cannot publish a v1 MCP release",
            recovery="use the MCP compatibility handoff for v1",
        )
    release = loaded
    if revocation_check is not None:
        revocation_check(release.pack_fingerprint)
    if adapter.kind != release.adapter_kind:
        raise AuthoringHandoffError(
            "release_adapter_mismatch",
            "publication adapter differs from the frozen release",
            recovery="use the adapter recorded in the freeze manifest",
        )
    observed = adapter.validate_pack(release.pack_root)
    if observed != release.pack_fingerprint:
        raise AuthoringHandoffError(
            "frozen_release_tampered",
            "pack fingerprint changed after freeze",
            recovery="restore the immutable frozen release",
        )
    adapter.bind_config(
        config_path,
        release.pack_root,
        release.pack_fingerprint,
    )
    report_path = adapter.prepare(config_path)
    report = adapter.load_validation_report(report_path)
    adapter.require_fresh_gold(report, release.pack_fingerprint)
    load_verified_release()
    if revocation_check is not None:
        revocation_check(release.pack_fingerprint)

    benchmark_path = adapter.generate(config_path)
    load_verified_release()
    if revocation_check is not None:
        revocation_check(release.pack_fingerprint)
    publication_root = benchmark_path.parent
    raw_path = publication_root / "benchmark_raw.parquet"
    manifest_path = publication_root / "run_manifest.json"
    manifest = load_json_mapping(manifest_path, "BFCL run manifest")
    pack = manifest.get("pack")
    if not isinstance(pack, Mapping) or pack.get("content_hash") != release.pack_fingerprint:
        raise AuthoringHandoffError(
            "publication_pack_mismatch",
            "published manifest pins a different frozen pack",
            recovery="publish from the exact sealed release",
        )
    ineligibility = manifest.get("gold_ineligibility_reasons")
    if (
        manifest.get("tier") != "gold"
        or manifest.get("gold_eligible") is not True
        or not isinstance(ineligibility, list)
        or ineligibility
    ):
        raise AuthoringHandoffError(
            "publication_not_gold_eligible",
            "generated manifest is not eligible for Gold publication",
            recovery="use strict publication lineage and resolve every Gold blocker",
        )
    adapter.verify_publication_manifest(manifest, release.pack_fingerprint)
    for path in (benchmark_path, raw_path, manifest_path):
        if not path.is_file():
            raise AuthoringHandoffError(
                "publication_artifact_missing",
                f"generation did not publish required artifact {path}",
                recovery="repair stage=all publication and retry",
            )
    return PublicationHandoffV2(
        release=release,
        validation_report_path=report_path,
        benchmark_path=benchmark_path,
        raw_benchmark_path=raw_path,
        run_manifest_path=manifest_path,
        run_manifest=manifest,
    )
