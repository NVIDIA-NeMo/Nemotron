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

"""BFCL publication adapters and frozen-release dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nemotron.steps.byob.runtime.authoring_release.contracts import PublicationAdapter
from nemotron.steps.byob.runtime.authoring_release.freeze import (
    FrozenReleaseV2,
    load_frozen_release,
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

PublicationKind = Literal["local_python", "http_package"]


@dataclass(frozen=True)
class BfclPublicationAdapter:
    """Publication hooks shared by conventional BFCL oracle packs."""

    adapter_kind: PublicationKind

    @property
    def kind(self) -> str:
        return self.adapter_kind

    def validate_pack(self, pack_root: Path) -> str:
        root = pack_root.resolve()
        paths = resolve_declared_pack_paths(
            OraclePackRef(manifest_path=root / "manifest.yaml"),
            (root,),
        )
        if self.adapter_kind == "local_python":
            valid_shape = paths.backend_path is not None and paths.endpoint_config_path is None
        else:
            valid_shape = paths.endpoint_config_path is not None and paths.backend_path is None
        if not valid_shape:
            raise AuthoringHandoffError(
                "publication_pack_adapter_mismatch",
                f"pack shape does not match {self.adapter_kind}",
                recovery="publish with the adapter recorded in the frozen release",
            )
        return f"sha256:{pack_fingerprint(paths)}"

    def bind_config(
        self,
        config_path: Path,
        pack_root: Path,
        frozen_pack_fingerprint: str,
    ) -> None:
        config = BfclConfig.from_yaml(config_path)
        paths = resolve_pack_paths(config)
        if paths.pack_root != pack_root.resolve() or f"sha256:{pack_fingerprint(paths)}" != frozen_pack_fingerprint:
            raise AuthoringHandoffError(
                "publication_config_mismatch",
                "BFCL config does not resolve the exact frozen pack",
                recovery="point the BFCL config at the sealed release pack",
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

    def generate(self, config_path: Path) -> Path:
        return generate_bfcl(config_path)

    def verify_publication_manifest(
        self,
        manifest: Mapping[str, Any],
        frozen_pack_fingerprint: str,
    ) -> None:
        oracle = manifest.get("oracle")
        origin = oracle.get("origin") if isinstance(oracle, Mapping) else None
        if (
            not isinstance(origin, Mapping)
            or origin.get("provider_kind") != "bfcl_authoring"
            or origin.get("adapter_kind") != self.adapter_kind
            or origin.get("frozen_pack_fingerprint") != frozen_pack_fingerprint
        ):
            raise AuthoringHandoffError(
                "publication_origin_missing",
                f"manifest omitted matching {self.adapter_kind} authoring provenance",
                recovery="publish through the authoring-aware BFCL final-output stage",
            )


def publication_adapter_for_release(
    release_root: Path,
    *,
    trusted_seal_keys: Mapping[str, Ed25519PublicKey],
    expected_seal_issuer: str,
) -> PublicationAdapter:
    """Resolve a built-in publication adapter without contacting the source."""
    release = load_frozen_release(
        release_root,
        trusted_seal_keys=trusted_seal_keys,
        expected_seal_issuer=expected_seal_issuer,
    )
    if not isinstance(release, FrozenReleaseV2):
        raise AuthoringHandoffError(
            "release_version_mismatch",
            "generic publication requires a v2 authoring release",
            recovery="use the MCP v1 compatibility publisher for legacy releases",
        )
    if release.adapter_kind == "mcp_mode_a":
        from nemotron.steps.byob.runtime.mcp.release.adapter import McpReleaseAdapter

        return McpReleaseAdapter(identity_digest="")
    if release.adapter_kind == "local_python":
        return BfclPublicationAdapter(cast(PublicationKind, release.adapter_kind))
    if release.adapter_kind == "http_package":
        raise AuthoringHandoffError(
            "publication_adapter_unsupported",
            "HTTP-package publication origin verification is not available",
            recovery="retain the frozen release until HTTP origin hooks are installed",
        )
    raise AuthoringHandoffError(
        "release_adapter_unsupported",
        f"unsupported publication adapter {release.adapter_kind!r}",
        recovery="install a publication adapter for the frozen release kind",
    )
