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

"""Run reviewed source-adapter probes through a Mode A gateway."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron.steps.byob.runtime.authoring_workflow.credentials import CredentialResolver
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.source_adapters.certification import (
    CertificationProbe,
    ProbeExecutionRecord,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import (
    AdapterProbePlan,
    ProbeError,
    probe_record,
    run_probe_suite,
    validate_probe_plan,
)


@dataclass(frozen=True)
class GatewayProbeTool:
    """The reviewed facts about one published tool the probes may rely on."""

    published_name: str
    mutates: bool
    requires_confirmation: bool


@dataclass(frozen=True)
class McpProbeRun:
    plan_digest: str
    records: tuple[ProbeExecutionRecord, ...]


def reviewed_probe_tools(
    catalog_tools: Sequence[Mapping[str, Any]],
) -> tuple[GatewayProbeTool, ...]:
    """Read the probe-relevant claims out of the discovered BFCL tool definitions."""
    tools: list[GatewayProbeTool] = []
    for definition in catalog_tools:
        function = definition["function"]
        tools.append(
            GatewayProbeTool(
                published_name=str(function["name"]),
                mutates=bool(definition.get("x-mutates", False)),
                requires_confirmation=bool(definition.get("x-requires-confirmation", False)),
            )
        )
    return tuple(tools)


def discovery_identity_record(
    *,
    identity_document: Mapping[str, Any],
    attestation_digest: str,
    started: float,
) -> ProbeExecutionRecord:
    """Carry the A0 pin that discovery and the gateway attestation already established.

    Re-deriving identity here would observe less than intake already did: discovery
    negotiated a protocol and read the server identity, and the attestation proved the
    running gateway serves that same catalog. The probes inherit that observation and are
    only responsible for showing it did not move while they ran. ``started`` is the
    monotonic clock reading from before discovery opened its connection, so the profile's
    deadline is applied to the work that actually pinned the identity.
    """
    return probe_record(
        CertificationProbe.IDENTITY_INTEGRITY,
        started=started,
        calls=1,
        status="pass",
        evidence={
            "identity": dict(identity_document),
            "gateway_attestation_digest": attestation_digest,
        },
        # Discovery closed the MCP connection it opened and the attestation fetch closed
        # its own request, so nothing is left behind to clean up.
        cleanup_status="passed",
    )


def run_mcp_gateway_probes(
    *,
    endpoint_config: EndpointConfig,
    tools: Sequence[GatewayProbeTool],
    identity_record: ProbeExecutionRecord,
    catalog_pinned: bool,
    plan: AdapterProbePlan,
    environ: Mapping[str, str] | None = None,
    credential_resolver: CredentialResolver | None = None,
    held_out_sensitive_terms: Sequence[str] = (),
    timeout_s: float = 30.0,
    worker_startup_s: float = 30.0,
    timeout_probe_s: float = 0.25,
) -> McpProbeRun:
    """Probe a reviewed Mode A gateway and return observations, never a tier."""
    validate_probe_plan(tools, plan)
    if plan.fixtures is None:
        # Mode A hands fixtures to the reset control tool at every session open, so a
        # plan with none cannot promise the reset it is about to claim was deterministic.
        raise ProbeError(
            "probe_evidence_invalid",
            "an MCP probe plan must carry the fixtures each session is opened with",
        )
    headers = resolve_endpoint_headers(
        endpoint_config,
        environ,
        credential_resolver=credential_resolver,
    )
    worker = ProcessWorker(default_timeout_s=timeout_s, worker="process")

    def episode(
        task_id: str,
        steps: list[dict[str, Any]],
        *,
        tool_timeout: float | None = None,
    ) -> list[Any]:
        deadline = timeout_s if tool_timeout is None else tool_timeout
        return worker.run_episode(
            endpoint_config=endpoint_config,
            endpoint_headers_override=headers,
            fixtures=copy.deepcopy(plan.fixtures),
            clock_iso=plan.clock,
            seed=plan.seed,
            task_id=task_id,
            steps=steps,
            # Starting the isolated worker is the host's cost, not the oracle's, so it is
            # budgeted apart from the deadlines the gateway is being held to. Charging a
            # slow interpreter start to the source would report a timeout it never caused.
            import_timeout_s=worker_startup_s,
            reset_timeout_s=timeout_s,
            tool_timeout_s=deadline,
            assertion_timeout_s=timeout_s,
            episode_timeout_s=(worker_startup_s + timeout_s + deadline * max(1, len(steps))),
        )

    def catalog_probe() -> tuple[bool, dict[str, Any], int]:
        # Discovery decides whether the catalog matches the digest the operator reviewed;
        # the gateway decides what it will actually serve. A tier that ignored either one
        # would be certifying a catalog nobody agreed to.
        (listed,) = episode("probe-catalog", [{"op": "list_tools"}])
        listed_names = (
            sorted(listed)
            if isinstance(listed, list)
            and all(isinstance(name, str) for name in listed)
            and len(listed) == len(set(listed))
            else []
        )
        reviewed_names = sorted(tool.published_name for tool in tools)
        return (
            catalog_pinned and listed_names == reviewed_names,
            {
                "listed_names": listed_names,
                "reviewed_names": reviewed_names,
                "discovery_catalog_pinned": catalog_pinned,
            },
            1,
        )

    def identity_drifted() -> bool:
        try:
            episode("probe-identity-recheck", [{"op": "metadata"}])
        except Exception:  # noqa: BLE001 - any refusal means it is no longer the same
            return True
        return False

    records = run_probe_suite(
        plan=plan,
        tools=tools,
        episode=episode,
        identity_record=identity_record,
        catalog_probe=catalog_probe,
        identity_drifted=identity_drifted,
        held_out_sensitive_terms=held_out_sensitive_terms,
        timeout_probe_s=timeout_probe_s,
    )
    return McpProbeRun(plan_digest=plan.digest, records=records)
