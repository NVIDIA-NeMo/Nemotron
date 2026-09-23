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

"""MCP hooks for the adapter-neutral v2 release kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.authoring_release.contracts import (
    AdapterReviewContribution,
    FreezeHookContext,
)
from nemotron.steps.byob.runtime.authoring_release.handoff import (
    AuthoringHandoffError,
)
from nemotron.steps.byob.runtime.authoring_release.review import load_json_mapping
from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
    BfclConfig,
    OraclePackRef,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_fingerprint,
    resolve_declared_pack_paths,
    resolve_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
    generate_bfcl,
    prepare_bfcl,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
    derive_pack_tier,
)


@dataclass(frozen=True)
class McpReleaseAdapter:
    """MCP implementation of typed release hooks.

    Review facts remain explicit constructor inputs: the MCP review assembler still
    sits outside the generic review contract, so the caller supplies them.
    """

    identity_digest: str
    certification_tier: str = "A2"
    review_data: Mapping[str, Any] = field(default_factory=dict)
    blockers: Sequence[Mapping[str, Any]] = ()
    risks: Sequence[Mapping[str, Any]] = ()
    sidecars: Mapping[str, bytes] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "mcp_mode_a"

    def validate_pack(self, pack_root: Path) -> str:
        root = pack_root.resolve()
        paths = resolve_declared_pack_paths(
            OraclePackRef(manifest_path=root / "manifest.yaml"),
            (root,),
        )
        if paths.endpoint_config_path is None or paths.backend_path is not None:
            raise ValueError(
                "an MCP release must use endpoint_config.yaml, not backend.py"
            )
        return f"sha256:{pack_fingerprint(paths)}"

    def review(
        self,
        pack_root: Path,
        source_digests: Mapping[str, str],
    ) -> AdapterReviewContribution:
        del pack_root, source_digests
        return AdapterReviewContribution(
            identity_digest=self.identity_digest,
            certification_tier=self.certification_tier,
            review_data=self.review_data,
            blockers=self.blockers,
            risks=self.risks,
        )

    def freeze_sidecars(
        self,
        context: FreezeHookContext,
    ) -> Mapping[str, bytes]:
        del context
        return self.sidecars

    def bind_config(
        self,
        config_path: Path,
        pack_root: Path,
        frozen_pack_fingerprint: str,
    ) -> None:
        config = BfclConfig.from_yaml(config_path)
        paths = resolve_pack_paths(config)
        if paths.pack_root != pack_root.resolve():
            raise AuthoringHandoffError(
                "publication_config_mismatch",
                f"BFCL config resolves {paths.pack_root}, not {pack_root.resolve()}",
                recovery="point the BFCL config at the frozen pack",
            )
        if f"sha256:{pack_fingerprint(paths)}" != frozen_pack_fingerprint:
            raise AuthoringHandoffError(
                "publication_config_mismatch",
                "BFCL config resolves a different pack fingerprint",
                recovery="point the BFCL config at the exact frozen pack",
            )

    def prepare(self, config_path: Path) -> Path:
        return prepare_bfcl(config_path, force_validation=True)

    def load_validation_report(self, report_path: Path) -> Mapping[str, Any]:
        return load_json_mapping(report_path, "fresh oracle validation report")

    def require_fresh_gold(
        self,
        report: Mapping[str, Any],
        frozen_pack_fingerprint: str,
    ) -> None:
        gold, tier = derive_pack_tier(dict(report))
        if not gold or tier != "gold":
            raise AuthoringHandoffError(
                "fresh_gold_required",
                f"fresh prepare did not reach Gold (tier={tier!r})",
                recovery="repair validation failures and create a new release",
            )
        if f"sha256:{report.get('pack_fingerprint')}" != frozen_pack_fingerprint:
            raise AuthoringHandoffError(
                "fresh_validation_stale",
                "fresh validation covers a different frozen pack",
                recovery="validate the exact frozen release",
            )
        conformance = next(
            (
                check
                for check in report.get("extra_checks") or []
                if isinstance(check, Mapping) and check.get("id") == "A1"
            ),
            None,
        )
        verdict = (
            conformance.get("conformance")
            if isinstance(conformance, Mapping)
            else None
        )
        if (
            not isinstance(verdict, Mapping)
            or verdict.get("effective_level") != "L2"
            or verdict.get("publishable") is not True
        ):
            raise AuthoringHandoffError(
                "fresh_conformance_required",
                "fresh prepare did not independently verify endpoint L2",
                recovery="repair MCP conformance and create a new release",
            )

    def generate(self, config_path: Path) -> Path:
        return generate_bfcl(config_path)

    def verify_publication_manifest(
        self,
        manifest: Mapping[str, Any],
        frozen_pack_fingerprint: str,
    ) -> None:
        oracle = manifest.get("oracle")
        mcp = oracle.get("mcp") if isinstance(oracle, Mapping) else None
        origin = oracle.get("origin") if isinstance(oracle, Mapping) else None
        legacy_matches = (
            isinstance(mcp, Mapping)
            and mcp.get("frozen_pack_fingerprint") == frozen_pack_fingerprint
        )
        generic_matches = (
            isinstance(origin, Mapping)
            and origin.get("provider_kind") == "bfcl_authoring"
            and origin.get("adapter_kind") == "mcp_mode_a"
            and origin.get("frozen_pack_fingerprint") == frozen_pack_fingerprint
        )
        if not legacy_matches and not generic_matches:
            raise AuthoringHandoffError(
                "publication_origin_missing",
                "published manifest omitted matching MCP authoring provenance",
                recovery="publish through the authoring-aware BFCL final-output stage",
            )
