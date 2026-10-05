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

"""The local-Python half of the probe ladder: execution policy and a process episode.

The choreography these probes follow lives in `probe_engine.py`, because it is the same
for every transport that can be reset. What is genuinely local is here: the least
privilege check that decides whether this package may be imported at all, and the episode
runner that reaches the backend through a child process.
"""

from __future__ import annotations

import ast
import copy
import io
import time
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    ProcessWorker,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    CertificationProbe,
    ProbeExecutionRecord,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapability,
    AdapterDescriptor,
    CleanupKind,
    CleanupSemantics,
    FixtureAccessKind,
    FixtureAccessPolicy,
    ProbeSafetyKind,
    ProbeSafetyPolicy,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    LocalPythonInspection,
    inspect_local_python_package,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import (
    AdapterProbeCase,
    AdapterProbePlan,
    ProbeError,
    probe_record,
    run_probe_suite,
    validate_probe_plan,
)

# The plan and its refusals were named for this adapter before they were shared. The
# aliases keep reviewed plans and existing callers reading the same way.
LocalProbeCase = AdapterProbeCase
LocalProbePlan = AdapterProbePlan
LocalProbeError = ProbeError


_SAFE_STDLIB = frozenset(
    {
        "__future__",
        "bisect",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "time",
        "typing",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)


@dataclass(frozen=True)
class LocalProbeRun:
    descriptor: AdapterDescriptor
    plan_digest: str
    records: tuple[ProbeExecutionRecord, ...]


def local_runtime_descriptor(timeout_s: float = 10.0) -> AdapterDescriptor:
    """Return the expanded A2 descriptor, whose digest is distinct from the A0 one."""
    return AdapterDescriptor(
        contract_version=ADAPTER_CONTRACT_VERSION,
        kind="local_python",
        implementation_name="bfcl.local_python",
        implementation_version="1.0.0+a2",
        capabilities=tuple(sorted(AdapterCapability, key=lambda item: item.value)),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.READ_ONLY,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.RESET_ISOLATED,
            max_calls=128,
            timeout_s=timeout_s,
        ),
        cleanup=CleanupSemantics(
            kind=CleanupKind.PROCESS,
            timeout_s=timeout_s,
        ),
    )


def _validate_execution_surface(inspection: LocalPythonInspection) -> dict[str, Any]:
    root = inspection.package_root
    local_top_levels = {Path(relative).parts[0].removesuffix(".py") for relative in inspection.import_closure}
    checked: list[str] = []
    for relative in inspection.import_closure:
        path = root / relative
        raw = path.read_bytes()
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        tree = ast.parse(raw.decode(encoding), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "__builtins__":
                raise LocalProbeError(
                    "probe_unsafe",
                    f"execution policy rejects __builtins__ access in {relative}",
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise LocalProbeError(
                    "probe_unsafe",
                    f"execution policy rejects dunder access in {relative}",
                )
            if isinstance(node, ast.Import):
                names = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [] if node.level else [(node.module or "").partition(".")[0]]
            else:
                names = []
            for name in names:
                if name and name not in local_top_levels and name not in _SAFE_STDLIB:
                    raise LocalProbeError(
                        "probe_unsafe",
                        f"execution policy rejects import {name!r} in {relative}",
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                    raise LocalProbeError(
                        "probe_unsafe",
                        f"execution policy rejects {node.func.id}() in {relative}",
                    )
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "fork",
                    "open",
                    "popen",
                    "run",
                    "spawn",
                    "system",
                }:
                    raise LocalProbeError(
                        "probe_unsafe",
                        f"execution policy rejects .{node.func.attr}() in {relative}",
                    )
        checked.append(relative)
    return {
        "policy": "bfcl-local-least-privilege-v1",
        "checked_files": checked,
        "allowed_stdlib": sorted(_SAFE_STDLIB),
    }


def run_local_python_probes(
    inspection: LocalPythonInspection,
    plan: LocalProbePlan,
    *,
    allowed_roots: tuple[Path, ...],
    held_out_sensitive_terms: Sequence[str] = (),
    timeout_s: float = 10.0,
    timeout_probe_s: float = 0.25,
) -> LocalProbeRun:
    """Run bounded local probes and return observations, never a tier or report."""
    validate_probe_plan(inspection.tools, plan)
    current = inspect_local_python_package(
        inspection.package_root,
        allowed_roots=allowed_roots,
        timeout_s=timeout_s,
    )
    if current.identity != inspection.identity:
        raise LocalProbeError(
            "identity_drift",
            "local source identity changed before probes",
        )
    policy_evidence = _validate_execution_surface(current)
    worker = ProcessWorker(default_timeout_s=timeout_s, worker="process")
    fixture_source = current.package_root / "fixtures.json"
    fixture_source_path = fixture_source if fixture_source.is_file() else None

    def episode(
        task_id: str,
        steps: list[dict[str, Any]],
        *,
        tool_timeout: float | None = None,
    ) -> list[Any]:
        deadline = timeout_s if tool_timeout is None else tool_timeout
        return worker.run_episode(
            backend_path=current.backend_path,
            fixtures=copy.deepcopy(plan.fixtures),
            clock_iso=plan.clock,
            seed=plan.seed,
            task_id=task_id,
            steps=steps,
            import_root=current.package_root,
            import_timeout_s=timeout_s,
            reset_timeout_s=timeout_s,
            tool_timeout_s=deadline,
            assertion_timeout_s=timeout_s,
            episode_timeout_s=max(
                timeout_s + 2.0,
                timeout_s + deadline * max(1, len(steps)),
            ),
            fixture_source_path=fixture_source_path,
        )

    def catalog_probe() -> tuple[bool, dict[str, Any], int]:
        inspection_output, listed = episode(
            "probe-catalog",
            [{"op": "inspect_backend"}, {"op": "list_tools"}],
        )
        required_symbols = {"call_tool", "get_state", "list_tools", "reset"}
        symbols_ok = isinstance(inspection_output, Mapping) and all(
            inspection_output.get(name) is True for name in required_symbols
        )
        listed_names = (
            sorted(listed)
            if isinstance(listed, list)
            and all(isinstance(name, str) for name in listed)
            and len(listed) == len(set(listed))
            else []
        )
        reviewed_names = sorted(tool.published_name for tool in current.tools)
        return (
            symbols_ok and listed_names == reviewed_names,
            {
                "backend_symbols_complete": symbols_ok,
                "listed_names": listed_names,
                "reviewed_names": reviewed_names,
            },
            1,
        )

    def identity_drifted() -> bool:
        final = inspect_local_python_package(
            current.package_root,
            allowed_roots=allowed_roots,
            timeout_s=timeout_s,
        )
        return final.identity != current.identity

    records = run_probe_suite(
        plan=plan,
        tools=current.tools,
        episode=episode,
        identity_record=probe_record(
            CertificationProbe.IDENTITY_INTEGRITY,
            started=time.monotonic(),
            calls=0,
            status="pass",
            evidence={
                "source_identity_digest": inspection.source_identity_digest,
                "execution_policy": policy_evidence,
            },
            # Reading a file digest starts nothing that has to be torn down.
            cleanup_status="not_required",
        ),
        catalog_probe=catalog_probe,
        identity_drifted=identity_drifted,
        held_out_sensitive_terms=held_out_sensitive_terms,
        timeout_probe_s=timeout_probe_s,
    )
    return LocalProbeRun(
        descriptor=local_runtime_descriptor(timeout_s),
        plan_digest=plan.digest,
        records=records,
    )
