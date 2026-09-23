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

"""The pack files intake can derive without a model: tools, manifest, and endpoint.

These three are mechanical. `tools.json` is the normalized catalog that discovery already
validated and digested, so re-deriving it here would risk publishing something the pinned
digest does not cover; it is copied through instead. `endpoint_config.yaml` pins the same
effective identity the gateway will answer with at `GET /v1/metadata`, which is why it is
built from the gateway's own identity function rather than from a second calculation that
could drift. `manifest.yaml` carries the pack identity plus the confirmation vocabulary the
MCP profile chose, so a reviewer sees that vocabulary in the pack instead of having to
remember it lives in another file.

What is deliberately absent: fixtures, task templates, validation cases, and assertions.
Those need evidence this path cannot produce at `L0`, and emitting empty stubs would turn a
missing input into a file that looks authored.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    PROTOCOL_VERSION,
)
from nemotron.steps.byob.runtime.mcp.authoring.intake import LoadedMcpIntake
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.errors import McpConfigError
from nemotron.steps.byob.runtime.mcp.gateway.identity import GatewayIdentity
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_text,
    write_canonical_json,
    write_text_atomic,
)

# The pack files this path still needs from a later phase, named so the draft cannot be
# mistaken for a loadable pack.
PENDING_PACK_ARTIFACTS: tuple[str, ...] = (
    "fixtures.json",
    "task_templates.yaml",
    "validation_cases.yaml",
    "assertions.py",
)

_CA_BUNDLE_NAME = "oracle_ca.pem"


@dataclass(frozen=True)
class EmittedArtifact:
    path: Path
    digest: str

    def as_dict(self, *, root: Path) -> dict[str, str]:
        return {
            "path": self.path.relative_to(root).as_posix(),
            "digest": self.digest,
        }


def _dump_yaml(document: dict[str, Any]) -> str:
    # Sorted keys are the determinism guarantee: the same intake must produce the same
    # bytes, so a reviewer's diff shows server change rather than dictionary order.
    return str(
        yaml.safe_dump(
            document,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        )
    )


def _endpoint_document(
    intake: LoadedMcpIntake,
    identity: GatewayIdentity,
    attestation_document: dict[str, Any],
    *,
    ca_bundle_written: bool,
) -> dict[str, Any]:
    gateway = intake.value.gateway
    # Pin what the live gateway actually served, never a discovery-only prediction. The
    # latter creates a cycle: once P4-P11 turn the route into an L2 document its digest no
    # longer matches the L0/L1 prediction emitted here.
    document: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "base_url": gateway.base_url,
        "expected": {
            "oracle_id": identity.oracle_id,
            "oracle_version": identity.oracle_version,
            "content_digest": identity.content_digest,
            **(
                {
                    "principal_digest": gateway.principal_digest,
                    "permission_digest": gateway.permission_digest,
                    "authorization_context_digest": (
                        gateway.authorization_context_digest
                    ),
                }
                if gateway.authorization_context_digest is not None
                else {}
            ),
        },
        "attestation": {
            "kind": ATTESTATION_KIND,
            "expected_digest": attestation_digest(attestation_document),
        },
        "max_request_bytes": gateway.max_request_bytes,
        "max_response_bytes": gateway.max_response_bytes,
    }
    auth: dict[str, Any] = {}
    if gateway.auth.bearer_token_env is not None:
        auth["bearer_token_env"] = gateway.auth.bearer_token_env
    elif gateway.auth.bearer_token_ref is not None:
        auth["bearer_token_ref"] = gateway.auth.bearer_token_ref.model_dump(mode="json")
    if gateway.auth.headers:
        auth["headers"] = dict(gateway.auth.headers)
    if gateway.auth.header_refs:
        auth.setdefault("headers", {}).update(
            {
                header: reference.model_dump(mode="json")
                for header, reference in gateway.auth.header_refs.items()
            }
        )
    if auth:
        document["auth"] = auth
    if ca_bundle_written:
        # Relative to the pack, because the endpoint loader requires the bundle to live
        # inside the pack tree where the fingerprint can see it.
        document["tls"] = {"ca_bundle_path": f"./{_CA_BUNDLE_NAME}"}
    if identity.protocol_version != PROTOCOL_VERSION:
        raise McpConfigError(
            "gateway identity declares protocol "
            f"{identity.protocol_version!r}, which this pack cannot consume"
        )
    return document


def _manifest_document(intake: LoadedMcpIntake) -> dict[str, Any]:
    results = intake.oracle.value.results
    return {
        "pack_id": intake.value.pack.pack_id,
        "version": intake.value.pack.version,
        "confirmation": {
            "parameter": results.confirmation_parameter,
            "status_field": results.status_field,
            "pending_status": results.pending_status,
        },
    }


def emit_pack_artifacts(
    intake: LoadedMcpIntake,
    report: DiscoveryReport,
    identity: GatewayIdentity,
    attestation_document: dict[str, Any],
    pack_root: Path,
) -> list[EmittedArtifact]:
    """Write tools.json, manifest.yaml, endpoint_config.yaml, and any pinned CA bundle."""
    report.verify_digest()
    root = pack_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    tools = report.document["catalog"]["tools"]
    if not tools:
        raise McpConfigError(
            "the normalized catalog selected no tools; a pack with no tool surface "
            "cannot produce a benchmark"
        )

    emitted: list[EmittedArtifact] = []

    ca_source = intake.value.gateway.ca_bundle_path
    ca_written = False
    if ca_source is not None:
        resolved = (
            ca_source
            if ca_source.is_absolute()
            else intake.path.parent / ca_source
        ).resolve()
        destination = root / _CA_BUNDLE_NAME
        shutil.copyfile(resolved, destination)
        ca_written = True
        emitted.append(
            EmittedArtifact(
                path=destination,
                digest=sha256_text(destination.read_text(encoding="utf-8")),
            )
        )

    tools_path = write_canonical_json(tools, root / "tools.json")
    emitted.append(
        EmittedArtifact(
            path=tools_path,
            digest=sha256_text(tools_path.read_text(encoding="utf-8")),
        )
    )

    for name, document in (
        ("manifest.yaml", _manifest_document(intake)),
        (
            "endpoint_config.yaml",
            _endpoint_document(
                intake,
                identity,
                attestation_document,
                ca_bundle_written=ca_written,
            ),
        ),
    ):
        text = _dump_yaml(document)
        path = write_text_atomic(text, root / name)
        emitted.append(EmittedArtifact(path=path, digest=sha256_text(text)))

    return sorted(emitted, key=lambda item: item.path.name)
