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

"""Fresh Gold revalidation and handoff to BFCL's existing generation path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_fingerprint,
    resolve_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
    generate_bfcl,
    prepare_bfcl,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
    derive_pack_tier,
)
from nemotron.steps.byob.runtime.mcp.release.freeze import (
    FrozenRelease,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.mcp.release.review import load_json_mapping


class HandoffError(RuntimeError):
    """Raised when a frozen release is not the source the BFCL config will run."""


@dataclass(frozen=True)
class PublicationHandoff:
    release: FrozenRelease
    validation_report_path: Path
    benchmark_path: Path
    raw_benchmark_path: Path
    run_manifest_path: Path
    run_manifest: dict[str, Any]


def _bind_config_to_release(config: BfclConfig, release: FrozenRelease) -> str:
    paths = resolve_pack_paths(config)
    if paths.pack_root != release.pack_root:
        raise HandoffError(
            f"BFCL config resolves pack {paths.pack_root}, not frozen pack {release.pack_root}"
        )
    observed = f"sha256:{pack_fingerprint(paths)}"
    if observed != release.pack_fingerprint:
        raise HandoffError(
            "BFCL config resolves a pack whose fingerprint differs from the freeze manifest"
        )
    return observed


def _require_fresh_gold(report: dict[str, Any], release: FrozenRelease) -> None:
    gold, tier = derive_pack_tier(report)
    if not gold or tier != "gold":
        raise HandoffError(f"fresh prepare did not reach Gold (tier={tier!r})")
    if f"sha256:{report.get('pack_fingerprint')}" != release.pack_fingerprint:
        raise HandoffError("fresh validation report covers a different pack fingerprint")
    conformance = next(
        (
            check
            for check in report.get("extra_checks") or []
            if isinstance(check, dict) and check.get("id") == "A1"
        ),
        None,
    )
    verdict = conformance.get("conformance") if isinstance(conformance, dict) else None
    if (
        not isinstance(verdict, dict)
        or verdict.get("effective_level") != "L2"
        or verdict.get("publishable") is not True
    ):
        raise HandoffError("fresh prepare did not independently verify endpoint L2")


def handoff_frozen_release(
    release_root: Path,
    config_path: Path,
) -> PublicationHandoff:
    """Force a fresh prepare, then invoke existing BFCL generation."""
    release = load_frozen_release(release_root)
    config = BfclConfig.from_yaml(config_path)
    _bind_config_to_release(config, release)

    # force_validation bypasses the same-process memoized verdict. The report consumed below
    # was computed from this frozen tree now, never copied from drafting or pre-freeze review.
    report_path = prepare_bfcl(config_path, force_validation=True)
    report = load_json_mapping(report_path, "fresh oracle validation report")
    _require_fresh_gold(report, release)
    load_frozen_release(release_root)

    benchmark_path = generate_bfcl(config_path)
    load_frozen_release(release_root)
    # Follow the artifact generation actually returned rather than recomputing the layout, so
    # a future change to publication paths cannot make this check silently inspect nothing.
    publication_root = benchmark_path.parent
    manifest_path = publication_root / "run_manifest.json"
    raw_path = publication_root / "benchmark_raw.parquet"
    manifest = load_json_mapping(manifest_path, "BFCL run manifest")
    if manifest.get("pack", {}).get("content_hash") != release.pack_fingerprint:
        raise HandoffError("published run manifest pins a different frozen pack")
    mcp = manifest.get("oracle", {}).get("mcp")
    if not isinstance(mcp, dict):
        raise HandoffError("published run manifest omitted MCP origin provenance")
    if mcp.get("frozen_pack_fingerprint") != release.pack_fingerprint:
        raise HandoffError("published MCP provenance pins a different frozen pack")
    for path in (benchmark_path, raw_path, manifest_path):
        if not path.is_file():
            raise HandoffError(f"existing stage=all did not publish required artifact {path}")
    return PublicationHandoff(
        release=release,
        validation_report_path=report_path,
        benchmark_path=benchmark_path,
        raw_benchmark_path=raw_path,
        run_manifest_path=manifest_path,
        run_manifest=manifest,
    )
