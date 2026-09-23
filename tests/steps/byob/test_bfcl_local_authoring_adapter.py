from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    ProcessWorker,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    certification_input_digest,
    derive_attained_tier,
    local_python_reference_profile,
    project_probe_executions,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    CleanupKind,
    ProbeSafetyKind,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    LocalPythonError,
    inspect_local_python_package,
)
from nemotron.steps.byob.runtime.source_adapters.local_python_probes import (
    LocalProbeCase,
    LocalProbePlan,
    run_local_python_probes,
)

# The probed backend sleeps for whole seconds, so an episode costs real time rather
# than work, and the production default leaves the episode ceiling at twelve seconds.
# A loaded machine has been observed spending 12.6 seconds there, which times a probe
# out and drops the derived tier. These tests assert tier derivation, not latency, so
# they buy headroom while leaving the forced-deadline probe at its own budget.
_PROBE_TIMEOUT_S = 60.0


def _package(tmp_path: Path, *, backend: str = "import helper\n") -> Path:
    (tmp_path / "backend.py").write_text(backend, encoding="utf-8")
    (tmp_path / "helper.py").write_text(
        "import json\n\nVALUE = json.dumps({'ok': True})\n",
        encoding="utf-8",
    )
    (tmp_path / "tools.json").write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up one item.",
                        "parameters": {
                            "type": "object",
                            "properties": {"item_id": {"type": "string"}},
                            "required": ["item_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dependency-lock.json").write_text(
        json.dumps(
            {
                "schema_version": "bfcl-python-dependency-lock-v1",
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _runtime_package(tmp_path: Path, *, undeclared_mutation: bool = False) -> Path:
    mutates = "False" if undeclared_mutation else "True"
    backend = f"""
import copy
import time

_state = {{}}

def list_tools():
    return ["lookup", "slow", "update"]

def reset(*, ctx, fixtures):
    global _state
    _state = copy.deepcopy(fixtures or {{}})

def get_state():
    return copy.deepcopy(_state)

def call_tool(name, arguments, *, ctx):
    if name == "lookup":
        item_id = arguments.get("item_id")
        item = _state.get("items", {{}}).get(item_id)
        if item is None:
            return {{"error": {{"code": "not_found"}}}}
        return {{"item": copy.deepcopy(item)}}
    if name == "slow":
        if arguments.get("delay"):
            time.sleep(5)
        return {{"ok": True}}
    if name == "update":
        if not arguments.get("confirm", False):
            return {{"status": "awaiting_confirmation"}}
        _state.setdefault("items", {{}})["A"] = {{"value": arguments["value"]}}
        return {{"updated": True, "declared": {mutates}}}
    raise KeyError(name)
"""
    (tmp_path / "backend.py").write_text(backend, encoding="utf-8")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "string"}},
                    "required": ["item_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "slow",
                "parameters": {
                    "type": "object",
                    "properties": {"delay": {"type": "boolean"}},
                    "required": ["delay"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "x-mutates": not undeclared_mutation,
            "x-requires-confirmation": True,
            "function": {
                "name": "update",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                        "confirm": {"type": "boolean", "default": False},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    (tmp_path / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    (tmp_path / "dependency-lock.json").write_text(
        json.dumps(
            {
                "schema_version": "bfcl-python-dependency-lock-v1",
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _probe_plan(*, include_timeout: bool = True) -> LocalProbePlan:
    cases = [
        LocalProbeCase(
            case_id="a_lookup",
            tool="lookup",
            arguments={"item_id": "A"},
            expectation="success",
            expected_state_change=False,
        ),
        LocalProbeCase(
            case_id="b_slow",
            tool="slow",
            arguments={"delay": False},
            expectation="success",
            expected_state_change=False,
        ),
        LocalProbeCase(
            case_id="c_update",
            tool="update",
            arguments={"value": 2, "confirm": True},
            expectation="success",
            expected_state_change=True,
        ),
        LocalProbeCase(
            case_id="d_error",
            tool="lookup",
            arguments={"item_id": "missing"},
            expectation="structured_error",
            expected_error_code="not_found",
        ),
    ]
    if include_timeout:
        cases.append(
            LocalProbeCase(
                case_id="e_timeout",
                tool="slow",
                arguments={"delay": True},
                expectation="timeout",
            )
        )
    return LocalProbePlan(
        schema_version="bfcl-local-probe-plan-v1",
        clock="2026-03-02T09:00:00+07:00",
        seed=7,
        fixtures={"items": {"A": {"value": 1}}},
        cases=tuple(cases),
    )


def test_local_python_static_closure_reaches_a0_without_importing_code(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    inspection = inspect_local_python_package(
        package,
        allowed_roots=(tmp_path,),
    )
    profile = local_python_reference_profile()
    input_digest = certification_input_digest(
        inspection.descriptor,
        source_identity_digest=inspection.source_identity_digest,
        profile=profile,
    )
    outcomes = project_probe_executions(
        profile,
        inspection.execution_records,
        input_digest=input_digest,
    )

    assert inspection.import_closure == ("backend.py", "helper.py")
    assert [tool.published_name for tool in inspection.tools] == ["lookup"]
    assert inspection.descriptor.capabilities == (
        AdapterCapability.DESCRIBE_TOOLS,
        AdapterCapability.PIN_IDENTITY,
    )
    assert inspection.descriptor.probe_safety.kind is ProbeSafetyKind.IDENTITY_ONLY
    assert inspection.descriptor.cleanup.kind is CleanupKind.NONE
    assert derive_attained_tier(profile, outcomes) is AdapterTier.A0


def test_local_identity_tracks_closure_fixtures_and_lock_but_not_unrelated_files(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    first = inspect_local_python_package(package, allowed_roots=(tmp_path,))

    (package / "notes.txt").write_text("not executable", encoding="utf-8")
    unrelated = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert unrelated.identity == first.identity

    (package / "helper.py").write_text("VALUE = 'changed'\n", encoding="utf-8")
    changed_source = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert changed_source.identity.effective_content_digest != (
        first.identity.effective_content_digest
    )

    (package / "fixtures.json").write_text('{"items":[{"id":"A"}]}', encoding="utf-8")
    changed_fixtures = inspect_local_python_package(
        package,
        allowed_roots=(tmp_path,),
    )
    assert changed_fixtures.identity.effective_content_digest != (
        changed_source.identity.effective_content_digest
    )


@pytest.mark.parametrize(
    ("backend", "code"),
    [
        ("module = __import__('helper')\n", "dynamic_import"),
        ("import third_party\n", "undeclared_import"),
        ("import importlib\nmodule = importlib.import_module('helper')\n", "dynamic_import"),
        ("import importlib as il\nmodule = il.import_module('helper')\n", "dynamic_import"),
        ("from importlib import import_module\nmodule = import_module('helper')\n", "dynamic_import"),
        ("from builtins import __import__ as load\nmodule = load('helper')\n", "dynamic_import"),
    ],
)
def test_local_python_rejects_dynamic_or_undeclared_imports(
    tmp_path: Path,
    backend: str,
    code: str,
) -> None:
    package = _package(tmp_path, backend=backend)

    with pytest.raises(LocalPythonError) as error:
        inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert error.value.code == code


def test_local_python_requires_canonical_dependency_lock(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "dependency-lock.json").unlink()
    with pytest.raises(LocalPythonError) as missing:
        inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert missing.value.code == "dependency_lock_missing"

    (package / "dependency-lock.json").write_text(
        json.dumps(
            {
                "schema_version": "bfcl-python-dependency-lock-v1",
                "dependencies": [
                    {
                        "import_name": "zeta",
                        "distribution": "zeta",
                        "version": "1.0",
                        "artifact_digest": "sha256:" + "a" * 64,
                    },
                    {
                        "import_name": "alpha",
                        "distribution": "alpha",
                        "version": "1.0",
                        "artifact_digest": "sha256:" + "b" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LocalPythonError) as invalid:
        inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert invalid.value.code == "dependency_lock_invalid"


def test_local_python_rejects_namespace_packages_and_symlink_escape(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path, backend="import helpers.value\n")
    (package / "helpers").mkdir()
    (package / "helpers" / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(LocalPythonError) as namespace:
        inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert namespace.value.code == "namespace_package_ambiguous"

    outside = tmp_path.parent / "outside-helper.py"
    outside.write_text("VALUE = 2\n", encoding="utf-8")
    (package / "helper.py").unlink()
    (package / "helper.py").symlink_to(outside)
    (package / "backend.py").write_text("import helper\n", encoding="utf-8")
    with pytest.raises(LocalPythonError) as escaped:
        inspect_local_python_package(package, allowed_roots=(tmp_path,))
    assert escaped.value.code == "source_path_escape"


def test_local_process_probes_derive_a2_without_adapter_assigned_tier(
    tmp_path: Path,
) -> None:
    package = _runtime_package(tmp_path)
    inspection = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    run = run_local_python_probes(
        inspection,
        _probe_plan(),
        allowed_roots=(tmp_path,),
        timeout_s=_PROBE_TIMEOUT_S,
    )
    profile = local_python_reference_profile()
    input_digest = certification_input_digest(
        run.descriptor,
        source_identity_digest=inspection.source_identity_digest,
        profile=profile,
        execution_inputs_digest=run.plan_digest,
    )
    outcomes = project_probe_executions(
        profile,
        run.records,
        input_digest=input_digest,
    )

    assert run.descriptor != inspection.descriptor
    assert run.descriptor.probe_safety.kind is ProbeSafetyKind.RESET_ISOLATED
    assert run.descriptor.cleanup.kind is CleanupKind.PROCESS
    assert derive_attained_tier(profile, outcomes) is AdapterTier.A2


def test_local_a2_requires_timeout_cleanup_and_truthful_mutation(
    tmp_path: Path,
) -> None:
    package = _runtime_package(tmp_path)
    inspection = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    profile = local_python_reference_profile()

    without_timeout = run_local_python_probes(
        inspection,
        _probe_plan(include_timeout=False),
        allowed_roots=(tmp_path,),
        timeout_s=_PROBE_TIMEOUT_S,
    )
    outcomes = project_probe_executions(
        profile,
        without_timeout.records,
        input_digest=certification_input_digest(
            without_timeout.descriptor,
            source_identity_digest=inspection.source_identity_digest,
            profile=profile,
            execution_inputs_digest=without_timeout.plan_digest,
        ),
    )
    assert derive_attained_tier(profile, outcomes) is AdapterTier.A1
    assert next(
        item
        for item in outcomes
        if item.probe.value == "timeout_cleanup"
    ).reason == "probe_missing"

    package = _runtime_package(tmp_path, undeclared_mutation=True)
    changed = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    mutation_run = run_local_python_probes(
        changed,
        _probe_plan(),
        allowed_roots=(tmp_path,),
        timeout_s=_PROBE_TIMEOUT_S,
    )
    mutation_outcomes = project_probe_executions(
        profile,
        mutation_run.records,
        input_digest=certification_input_digest(
            mutation_run.descriptor,
            source_identity_digest=changed.source_identity_digest,
            profile=profile,
            execution_inputs_digest=mutation_run.plan_digest,
        ),
    )
    mutation = next(
        item
        for item in mutation_outcomes
        if item.probe.value == "mutation_declaration"
    )
    assert mutation.reason == "mutation_declaration_mismatch"
    assert derive_attained_tier(profile, mutation_outcomes) is AdapterTier.A1


def test_timeout_probe_separates_a_broken_transport_from_an_ignored_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that never reached a deadline records why, not just that it failed.

    A source that runs past its deadline and a transport that broke before the
    deadline could be observed both record `timeout_observed: false`, so without a
    reason an operator would go looking for a cleanup defect in a source whose real
    problem is that it never answered.

    The projected refusal code stays `cleanup_failed`, because a probe that never
    completed cannot claim its session was cleaned up and the projection refuses
    fail-closed on `cleanup_status`. The diagnosis lives in the observation the
    outcome carries.
    """
    package = _runtime_package(tmp_path)
    inspection = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    profile = local_python_reference_profile()

    original = ProcessWorker.run_episode

    def failing_run_episode(self, **kwargs):  # type: ignore[no-untyped-def]
        if str(kwargs.get("task_id", "")).startswith("probe-timeout-"):
            raise ConnectionResetError("worker connection lost")
        return original(self, **kwargs)

    monkeypatch.setattr(ProcessWorker, "run_episode", failing_run_episode)

    run = run_local_python_probes(
        inspection,
        _probe_plan(),
        allowed_roots=(tmp_path,),
        timeout_s=_PROBE_TIMEOUT_S,
    )
    outcomes = project_probe_executions(
        profile,
        run.records,
        input_digest=certification_input_digest(
            run.descriptor,
            source_identity_digest=inspection.source_identity_digest,
            profile=profile,
            execution_inputs_digest=run.plan_digest,
        ),
    )

    raw = next(
        record
        for record in run.records
        if record.observation.probe.value == "timeout_cleanup"
    )
    assert raw.observation.reason == "probe_failed"
    assert raw.observation.evidence["timeout_observed"] is False
    assert raw.observation.evidence["probe_error"] == "ConnectionResetError"

    timeout = next(
        item for item in outcomes if item.probe.value == "timeout_cleanup"
    )
    assert timeout.status == "fail"
    observed = timeout.evidence["observation"]
    assert observed["reason"] == "probe_failed"
    assert observed["evidence"]["probe_error"] == "ConnectionResetError"
    # The finding costs the top tier, exactly as an ignored deadline would.
    assert derive_attained_tier(profile, outcomes) is AdapterTier.A1


def test_local_probe_inputs_cannot_overlap_held_out_material(
    tmp_path: Path,
) -> None:
    package = _runtime_package(tmp_path)
    inspection = inspect_local_python_package(package, allowed_roots=(tmp_path,))
    plan = _probe_plan().model_copy(
        update={"fixtures": {"items": {"RESERVED-42": {"value": 1}}}}
    )

    with pytest.raises(ValueError, match="held-out"):
        run_local_python_probes(
            inspection,
            plan,
            allowed_roots=(tmp_path,),
            held_out_sensitive_terms=("RESERVED-42",),
            timeout_s=_PROBE_TIMEOUT_S,
        )


@pytest.mark.parametrize(
    "unsafe_source",
    [
        "\nimport os\nHOST = os.environ\n",
        "\nHOST = getattr(__builtins__, 'open')('/etc/hosts').read()\n",
    ],
)
def test_local_execution_policy_rejects_host_and_process_access(
    tmp_path: Path,
    unsafe_source: str,
) -> None:
    package = _runtime_package(tmp_path)
    (package / "backend.py").write_text(
        (package / "backend.py").read_text(encoding="utf-8")
        + unsafe_source,
        encoding="utf-8",
    )
    inspection = inspect_local_python_package(package, allowed_roots=(tmp_path,))

    with pytest.raises(ValueError, match="probe_unsafe"):
        run_local_python_probes(
            inspection,
            _probe_plan(),
            allowed_roots=(tmp_path,),
            timeout_s=_PROBE_TIMEOUT_S,
        )
