from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    PackTrustError,
    ProcessWorker,
    assert_pack_allowed,
    run_with_timeout,
)

# Budgets that are scaffolding rather than the thing under test. Spawning a worker
# and importing a backend competes with every other test process for CPU, so a value
# chosen to look brisk fails on a loaded machine and says nothing when it does. These
# are large enough that only a genuine hang trips them; where a timeout *is* the
# assertion, the test states its own tight value instead.
GENEROUS_TIMEOUT_S = 30.0
GENEROUS_EPISODE_TIMEOUT_S = 60.0


def _double() -> int:
    return 21 * 2


def _increment(value: int) -> int:
    return value + 1


def _large_payload(size: int) -> str:
    return "x" * size


def _exit_without_result() -> None:
    import os

    os._exit(0)


def test_assert_pack_allowed_accepts_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "data"
    pack = root / "tiny_oracle_pack" / "manifest.yaml"
    pack.parent.mkdir(parents=True)
    pack.write_text("pack_id: tiny_library\n", encoding="utf-8")

    resolved = assert_pack_allowed(pack, [root])
    assert resolved == pack.resolve()


def test_assert_pack_allowed_rejects_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    outsider = tmp_path / "uploaded_pack" / "manifest.yaml"
    outsider.parent.mkdir()
    outsider.write_text("pack_id: evil\n", encoding="utf-8")

    with pytest.raises(PackTrustError, match="outside allowlisted"):
        assert_pack_allowed(outsider, [root])


def test_run_with_timeout_returns_value() -> None:
    assert run_with_timeout(_double, 5.0, worker="process") == 42


def test_run_with_timeout_kills_hung_call() -> None:
    with pytest.raises(TimeoutError, match="timeout_s"):
        run_with_timeout(time.sleep, 0.5, 30, worker="process")


def test_process_worker_call() -> None:
    worker = ProcessWorker(default_timeout_s=5.0)
    assert worker.call(_increment, 41) == 42


def test_a_result_larger_than_the_pipe_buffer_still_returns() -> None:
    """Joining before draining the queue would deadlock until the deadline instead."""
    started = time.monotonic()
    payload = run_with_timeout(_large_payload, 20.0, 5_000_000, worker="process")

    assert len(payload) == 5_000_000
    assert time.monotonic() - started < 15.0


def test_a_worker_that_exits_silently_is_not_reported_as_a_timeout() -> None:
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="without returning a result"):
        run_with_timeout(_exit_without_result, 20.0, worker="process")

    assert time.monotonic() - started < 15.0


def test_episode_deadline_terminates_a_tool_that_never_returns() -> None:
    """The timeout gate must cover the path pack tools actually run through."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        SLOW_BACKEND_PATH,
        SLOW_BACKEND_TOOL,
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="call_tool"):
        ProcessWorker().run_episode(
            backend_path=SLOW_BACKEND_PATH,
            fixtures=None,
            clock_iso="2026-03-02T09:00:00+07:00",
            seed=0,
            task_id="slow-tool",
            steps=[{"op": "call_tool", "name": SLOW_BACKEND_TOOL, "arguments": {}}],
            import_timeout_s=10.0,
            tool_timeout_s=0.5,
            episode_timeout_s=12.0,
        )

    assert time.monotonic() - started < 12.0


def test_episode_deadline_covers_receiving_the_complete_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable message header must not make an unbounded recv look complete."""
    from multiprocessing.connection import Connection

    backend = tmp_path / "backend.py"
    backend.write_text(
        """
def list_tools():
    return ["probe"]

def reset(*, ctx, fixtures):
    return None

def call_tool(name, arguments, *, ctx):
    return {"ok": True}

def get_state():
    return {}
""",
        encoding="utf-8",
    )
    real_recv = Connection.recv
    calls = 0

    stalled_recv_s = 10.0

    def delayed_second_recv(connection):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(stalled_recv_s)
        return real_recv(connection)

    monkeypatch.setattr(Connection, "recv", delayed_second_recv)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="call_tool"):
        ProcessWorker().run_episode(
            backend_path=backend,
            fixtures={},
            clock_iso="2026-03-02T09:00:00+07:00",
            seed=0,
            task_id="slow-receive",
            steps=[{"op": "call_tool", "name": "probe", "arguments": {}}],
            import_timeout_s=30.0,
            tool_timeout_s=0.1,
            episode_timeout_s=60.0,
        )
    # The deadline has to abort the stalled recv rather than notice afterwards that it
    # took too long. The bound is a fraction of the stall so worker startup, which this
    # test does not measure, cannot decide the outcome on a loaded machine.
    assert time.monotonic() - started < stalled_recv_s / 2


def test_debug_thread_mode_leaves_the_caller_environment_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thread mode shares this process, so it must not strip the host environment.

    Stripping it here would outlive the episode: a thread that hangs cannot be
    stopped, so nothing would put the variables back.
    """
    backend = tmp_path / "backend.py"
    backend.write_text(
        """
def list_tools():
    return ["probe"]

def reset(*, ctx, fixtures=None):
    return None

def call_tool(name, arguments, *, ctx):
    return {"ok": True}

def get_state():
    return {}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("BFCL_TEST_HOST_VAR", "keep-me")

    ProcessWorker(default_timeout_s=5.0, worker="thread").run_episode(
        backend_path=backend,
        fixtures={},
        clock_iso="2026-03-02T09:00:00+07:00",
        seed=7,
        task_id="thread-env",
        steps=[{"op": "reset"}, {"op": "call_tool", "name": "probe", "arguments": {}}],
        episode_timeout_s=5.0,
    )

    import os

    assert os.environ.get("BFCL_TEST_HOST_VAR") == "keep-me"
    assert "PATH" in os.environ


def test_debug_thread_mode_restores_import_state(tmp_path: Path) -> None:
    helper_name = "bfcl_thread_only_helper"
    (tmp_path / f"{helper_name}.py").write_text("VALUE = 7\n", encoding="utf-8")
    backend = tmp_path / "backend.py"
    backend.write_text(
        f"""
import {helper_name}

def list_tools():
    return ["probe"]

def reset(*, ctx, fixtures=None):
    return None

def call_tool(name, arguments, *, ctx):
    return {{"value": {helper_name}.VALUE}}

def get_state():
    return {{}}
""",
        encoding="utf-8",
    )
    original_path = list(sys.path)
    original_bytecode_flag = sys.dont_write_bytecode

    outputs = ProcessWorker(default_timeout_s=5.0, worker="thread").run_episode(
        backend_path=backend,
        fixtures={},
        clock_iso="2026-03-02T09:00:00+07:00",
        seed=7,
        task_id="thread-import-state",
        steps=[{"op": "reset"}, {"op": "call_tool", "name": "probe", "arguments": {}}],
        episode_timeout_s=5.0,
    )

    assert outputs[1] == {"value": 7}
    assert sys.path == original_path
    assert sys.dont_write_bytecode is original_bytecode_flag
    assert helper_name not in sys.modules


def test_pack_worker_sanitizes_environment_and_enforces_reset_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = tmp_path / "backend.py"
    backend.write_text(
        """
import os
import time

if "BFCL_TEST_SECRET" in os.environ:
    raise RuntimeError("host secret leaked into pack import")

def list_tools():
    return ["probe"]

def reset(*, ctx, fixtures=None):
    time.sleep(5)

def call_tool(name, arguments, *, ctx):
    return {"ok": True}

def get_state():
    return {}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("BFCL_TEST_SECRET", "must-not-leak")
    worker = ProcessWorker(default_timeout_s=5.0)

    assert worker.run_episode(
        backend_path=backend,
        fixtures={},
        clock_iso="2026-03-02T09:00:00+07:00",
        seed=7,
        task_id="sanitized-import",
        steps=[{"op": "list_tools"}],
        import_timeout_s=GENEROUS_TIMEOUT_S,
        reset_timeout_s=0.1,
        tool_timeout_s=1.0,
        episode_timeout_s=GENEROUS_EPISODE_TIMEOUT_S,
    ) == [["probe"]]

    with pytest.raises(TimeoutError, match="reset"):
        worker.run_episode(
            backend_path=backend,
            fixtures={},
            clock_iso="2026-03-02T09:00:00+07:00",
            seed=7,
            task_id="reset-timeout",
            steps=[{"op": "reset"}],
            import_timeout_s=GENEROUS_TIMEOUT_S,
            reset_timeout_s=0.1,
            tool_timeout_s=1.0,
            episode_timeout_s=GENEROUS_EPISODE_TIMEOUT_S,
        )


def test_assertion_signature_is_checked_in_worker(tmp_path: Path) -> None:
    assertions = tmp_path / "assertions.py"
    assertions.write_text(
        """
def assert_valid(*, state, trace, task, ctx):
    return None

def assert_invalid(*, state):
    return None

ASSERTIONS = {
    "assert_valid": assert_valid,
    "assert_invalid": assert_invalid,
}

ASSERTION_CAPABILITIES = {
    "assert_valid": {"trace": True, "executable": True, "category": "result"},
}
""",
        encoding="utf-8",
    )

    report = ProcessWorker().inspect_assertions(assertions, timeout_s=GENEROUS_TIMEOUT_S)

    assert report["assert_valid"]["valid"] is True
    assert report["assert_valid"]["capabilities"] == {
        "trace": True,
        "executable": True,
        "category": "result",
    }
    assert report["assert_invalid"]["valid"] is False
    assert report["assert_invalid"]["capabilities"] == {
        "trace": False,
        "executable": True,
        "category": "unclassified",
    }
    assert report["assert_valid"]["capability_reason"] is None


def test_a_malformed_capability_is_not_reported_as_a_signature_defect(
    tmp_path: Path,
) -> None:
    assertions = tmp_path / "assertions.py"
    assertions.write_text(
        """
def assert_valid(*, state, trace, task, ctx):
    return None

ASSERTIONS = {"assert_valid": assert_valid}

ASSERTION_CAPABILITIES = {
    "assert_valid": {"trace": True, "executable": True, "category": "spelling"},
}
""",
        encoding="utf-8",
    )

    report = ProcessWorker().inspect_assertions(assertions, timeout_s=GENEROUS_TIMEOUT_S)

    assert report["assert_valid"]["valid"] is True
    assert report["assert_valid"]["reason"] is None
    assert report["assert_valid"]["capabilities"] is None
    assert "category 'spelling'" in report["assert_valid"]["capability_reason"]


def test_a_computed_capability_mapping_is_refused_at_validation(
    tmp_path: Path,
) -> None:
    assertions = tmp_path / "assertions.py"
    assertions.write_text(
        """
def assert_valid(*, state, trace, task, ctx):
    return None

ASSERTIONS = {"assert_valid": assert_valid}

ASSERTION_CAPABILITIES = dict.fromkeys(ASSERTIONS, {"category": "state"})
""",
        encoding="utf-8",
    )

    report = ProcessWorker().inspect_assertions(assertions, timeout_s=GENEROUS_TIMEOUT_S)

    assert report["assert_valid"]["valid"] is True
    assert report["assert_valid"]["capabilities"] is None
    assert "computed" in report["assert_valid"]["capability_reason"]


def test_episode_propagates_each_call_turn_index(tmp_path: Path) -> None:
    backend = tmp_path / "backend.py"
    backend.write_text(
        """
def list_tools():
    return ["probe"]

def reset(*, ctx, fixtures=None):
    return None

def call_tool(name, arguments, *, ctx):
    return {"turn_index": ctx.turn_index}

def get_state():
    return {}
""",
        encoding="utf-8",
    )
    outputs = ProcessWorker().run_episode(
        backend_path=backend,
        fixtures={},
        clock_iso="2026-03-02T09:00:00+07:00",
        seed=7,
        task_id="turn-index",
        steps=[
            {"op": "reset"},
            {"op": "call_tool", "name": "probe", "arguments": {}, "turn_index": 2},
            {"op": "call_tool", "name": "probe", "arguments": {}, "turn_index": 4},
        ],
        episode_timeout_s=GENEROUS_EPISODE_TIMEOUT_S,
    )
    assert outputs[1:] == [{"turn_index": 2}, {"turn_index": 4}]


def test_episode_resolves_pack_helper_modules(tmp_path: Path) -> None:
    """A multi-file pack must load the same way from any calling process."""
    pack = tmp_path / "pack"
    (pack / "backend").mkdir(parents=True)
    (pack / "vocabulary.py").write_text("STATUS = 'on_loan'\n", encoding="utf-8")
    (pack / "backend" / "records.py").write_text("ITEMS = ['B-1']\n", encoding="utf-8")
    backend = pack / "backend" / "main.py"
    backend.write_text(
        """
import records
import vocabulary


def list_tools():
    return ["probe"]

def reset(*, ctx, fixtures=None):
    return None

def call_tool(name, arguments, *, ctx):
    return {"items": records.ITEMS, "status": vocabulary.STATUS}

def get_state():
    return {}
""",
        encoding="utf-8",
    )

    outputs = ProcessWorker().run_episode(
        backend_path=backend,
        fixtures={},
        clock_iso="2026-03-02T09:00:00+07:00",
        seed=0,
        task_id="helpers",
        steps=[{"op": "reset"}, {"op": "call_tool", "name": "probe", "arguments": {}}],
        import_root=pack,
        episode_timeout_s=GENEROUS_EPISODE_TIMEOUT_S,
    )

    assert outputs[1] == {"items": ["B-1"], "status": "on_loan"}
