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

"""The HTTP half of the probe ladder: reviewed sessions against a live oracle.

A reviewed HTTP package could always be checked for identity and catalog, which is A0, but
nothing above that, because intake never called a tool. That left two of the three
transports permanently below the ladder they were said to share.

Nothing about the questions on the ladder needed a local backend, though, and neither does
the worker: it already reaches an HTTP oracle by opening a session, sending calls, and
deleting the session afterwards. So this module supplies only what HTTP means — one episode
is one endpoint session, the catalog is what `/v1/tools` serves, and identity is the
metadata the endpoint still agrees to — and `probe_engine.py` asks the questions.

An endpoint that stops matching its reviewed identity part-way through is the reason
identity is re-checked at the end: the calls in between were made against something else.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron.steps.byob.runtime.authoring_workflow.credentials import CredentialResolver
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.source_adapters.certification import (
    CertificationProbe,
    ProbeExecutionRecord,
)
from nemotron.steps.byob.runtime.source_adapters.contract import AdapterDescriptor
from nemotron.steps.byob.runtime.source_adapters.http_package import (
    HttpPackageInspection,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import (
    AdapterProbePlan,
    ProbeError,
    run_probe_suite,
    validate_probe_plan,
)


@dataclass(frozen=True)
class HttpProbeRun:
    descriptor: AdapterDescriptor
    plan_digest: str
    records: tuple[ProbeExecutionRecord, ...]


def _identity_record(inspection: HttpPackageInspection) -> ProbeExecutionRecord:
    for record in inspection.execution_records:
        if record.observation.probe is CertificationProbe.IDENTITY_INTEGRITY:
            return record
    raise ProbeError(
        "probe_evidence_invalid",
        "HTTP inspection produced no identity observation to probe against",
    )


def run_http_package_probes(
    inspection: HttpPackageInspection,
    plan: AdapterProbePlan,
    *,
    environ: Mapping[str, str] | None = None,
    credential_resolver: CredentialResolver | None = None,
    held_out_sensitive_terms: Sequence[str] = (),
    timeout_s: float = 15.0,
    timeout_probe_s: float = 0.25,
) -> HttpProbeRun:
    """Probe a reviewed HTTP oracle and return observations, never a tier or report."""
    validate_probe_plan(inspection.tools, plan)
    config = inspection.endpoint_config
    if plan.fixtures is None:
        # A local package owns fixtures.json; an endpoint is handed its fixtures at
        # session open, so a plan with none cannot promise a reproducible reset.
        raise ProbeError(
            "probe_evidence_invalid",
            "an HTTP probe plan must carry the fixtures each session is opened with",
        )
    headers = resolve_endpoint_headers(
        config,
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
            endpoint_config=config,
            endpoint_headers_override=headers,
            fixtures=copy.deepcopy(plan.fixtures),
            clock_iso=plan.clock,
            seed=plan.seed,
            task_id=task_id,
            steps=steps,
            import_timeout_s=timeout_s,
            reset_timeout_s=timeout_s,
            tool_timeout_s=deadline,
            assertion_timeout_s=timeout_s,
            episode_timeout_s=max(
                timeout_s + 2.0,
                timeout_s + deadline * max(1, len(steps)),
            ),
        )

    def catalog_probe() -> tuple[bool, dict[str, Any], int]:
        # `list_tools` re-checks metadata first, so a catalog served by an endpoint that
        # no longer matches its reviewed identity cannot pass this probe.
        (listed,) = episode("probe-catalog", [{"op": "list_tools"}])
        listed_names = (
            sorted(listed)
            if isinstance(listed, list)
            and all(isinstance(name, str) for name in listed)
            and len(listed) == len(set(listed))
            else []
        )
        reviewed_names = sorted(tool.published_name for tool in inspection.tools)
        return (
            listed_names == reviewed_names,
            {
                "listed_names": listed_names,
                "reviewed_names": reviewed_names,
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
        tools=inspection.tools,
        episode=episode,
        # Identity was pinned by the A0 inspection, which reached the endpoint and closed
        # the connection it opened. Re-deriving that here would only restate it less
        # accurately, so the probe run carries the observation that was actually made.
        identity_record=_identity_record(inspection),
        catalog_probe=catalog_probe,
        identity_drifted=identity_drifted,
        held_out_sensitive_terms=held_out_sensitive_terms,
        timeout_probe_s=timeout_probe_s,
    )
    return HttpProbeRun(
        descriptor=inspection.descriptor,
        plan_digest=plan.digest,
        records=records,
    )
