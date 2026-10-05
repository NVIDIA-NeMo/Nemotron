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

"""Trust + timeout helpers for oracle-pack execution.

Pipeline owns reset/timeouts; packs only expose entrypoints.
Gold claims require process workers. Each episode (reset + tools + state)
must run inside a single worker process so fixture state is coherent.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from nemotron.steps.byob.runtime.benchmark_families.bfcl.assertion_capabilities import (
    ASSERTION_CAPABILITIES_SYMBOL,
    AssertionCapabilityError,
    normalized_assertion_capability,
    read_literal_assertion_capabilities,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
_FIRST_STEP = object()


def _advance_episode(
    iterator: Any,
    output: Any = _FIRST_STEP,
) -> dict[str, Any]:
    """Advance a static step iterator or feed output back to an interactive generator."""
    if output is _FIRST_STEP:
        return next(iterator)
    sender = getattr(iterator, "send", None)
    return sender(output) if sender is not None else next(iterator)


@contextmanager
def _restored_import_state() -> Iterator[None]:
    """Keep debug-thread pack imports from leaking into later host imports."""
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    original_bytecode_flag = sys.dont_write_bytecode
    try:
        yield
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode_flag
        for name in set(sys.modules) - original_modules:
            sys.modules.pop(name, None)


class PackTrustError(PermissionError):
    """Raised when an oracle pack path is outside the configured allowlist."""


def assert_pack_allowed(path: Path | str, allowed_roots: Sequence[Path | str]) -> Path:
    """Return the resolved pack path if it sits under an allowlisted root."""
    resolved = Path(path).resolve()
    roots = [Path(root).resolve() for root in allowed_roots]
    if not roots:
        raise PackTrustError("oracle_runtime.allowed_roots is empty; refusing pack load")

    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    roots_display = ", ".join(str(root) for root in roots)
    raise PackTrustError(f"Oracle pack path {resolved} is outside allowlisted roots: {roots_display}")


def _load_module(path: Path, module_name: str, import_root: Path | str | None = None) -> ModuleType:
    """Import a pack module from its file, resolving the pack's own helper modules.

    A pack may split its backend or assertions across several files, and the fingerprint
    already covers the whole pack tree. Importing by file location alone would leave
    ``import helper`` resolving against whatever ``sys.path`` the calling process happened
    to have, so the same pack would load in one invocation and fail in another. The
    module's directory and the pack root are what resolve those imports instead.
    """
    # Importing pack code must not write into the pack: a __pycache__ directory
    # appearing next to backend.py changes the tree the fingerprint covers.
    sys.dont_write_bytecode = True
    # Inserted last-to-first so the module's own directory outranks the pack root.
    roots = [] if import_root is None else [Path(import_root)]
    roots.append(path.parent)
    for root in roots:
        entry = str(root.resolve())
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sanitize_pack_environment() -> None:
    """Remove inherited host secrets before executing allowlisted pack code."""
    keep = {
        key: value
        for key, value in os.environ.items()
        if key in {"LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP"}
    }
    os.environ.clear()
    os.environ.update(keep)


def _timeout_target(
    queue: mp.Queue,
    fn: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        queue.put(("ok", fn(*args, **kwargs)))
    except BaseException:  # noqa: BLE001 — re-raise in parent with remote traceback
        queue.put(("err", traceback.format_exc()))


def run_with_timeout(
    fn: Callable[..., T],
    timeout_s: float,
    /,
    *args: Any,
    worker: str = "process",
    **kwargs: Any,
) -> T:
    """Run ``fn`` under a deadline.

    ``worker="process"`` is required for gold claims. ``worker="thread"`` is
    debug-only: Python cannot terminate a running thread, so a timed-out callable
    may continue in the background and thread mode cannot claim a hard timeout.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    if worker == "thread":
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(
                f"debug thread exceeded timeout_s={timeout_s}; the callable may still be running"
            ) from exc
        finally:
            pool.shutdown(wait=future.done(), cancel_futures=True)

    if worker != "process":
        raise ValueError(f"unsupported worker {worker!r}; expected 'process' or 'thread'")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_timeout_target, args=(queue, fn, args, kwargs))
    proc.start()

    # Read before joining. A result larger than the pipe buffer leaves the child
    # blocked in its feeder thread until the parent drains the queue, so joining
    # first would deadlock until the deadline and report a timeout that never was.
    payload: Any = None
    received = False
    exited = False
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                payload = queue.get(timeout=min(0.05, timeout_s))
                received = True
                break
            except queue_module.Empty:
                if not proc.is_alive():
                    # The child is gone; give its feeder thread a last chance to
                    # flush before calling this a crash rather than a timeout.
                    try:
                        payload = queue.get(timeout=0.5)
                        received = True
                    except queue_module.Empty:
                        exited = True
                    break
                if time.monotonic() >= deadline:
                    break
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
        proc.join(1.0)
        queue.close()

    if exited:
        raise RuntimeError("worker exited without returning a result")
    if not received:
        raise TimeoutError(f"call exceeded timeout_s={timeout_s}")
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError("worker exited without returning a result")
    if payload[0] == "ok":
        return payload[1]
    raise RuntimeError(f"worker failed:\n{payload[1]}")


def _run_backend_episode_sync_inner(
    backend_path: str | None,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    task_id: str,
    tool_timeout_s: float,
    steps: Iterable[dict[str, Any]],
    *,
    sanitize_environment: bool,
    assertions_path: str | None = None,
    import_root: str | None = None,
    endpoint_config: Any = None,
    endpoint_headers: dict[str, str] | None = None,
    oracle_holder: list[Any] | None = None,
) -> list[Any]:
    """Run an episode synchronously (debug-thread path only).

    ``sanitize_environment`` is only safe when this runs in a worker process of its
    own. Thread mode shares the parent's environment: clearing it here would strip
    the host session, and a timed-out thread cannot be stopped to restore it.
    """
    from datetime import datetime

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.run_context import RunContext

    if sanitize_environment:
        _sanitize_pack_environment()
    if endpoint_config is not None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
            EndpointOracleClient,
        )

        module = EndpointOracleClient(
            endpoint_config,
            headers=endpoint_headers or {},
            timeout_s=tool_timeout_s,
        )
    else:
        if backend_path is None:
            raise ValueError("backend_path is required for a local oracle")
        module = _load_module(
            Path(backend_path),
            f"_bfcl_episode_{Path(backend_path).stem}_{task_id}",
            import_root,
        )
    if oracle_holder is not None:
        oracle_holder.append(module)
    assertions_module = None
    if assertions_path is not None:
        assertions_module = _load_module(
            Path(assertions_path),
            f"_bfcl_episode_assertions_{Path(assertions_path).stem}_{task_id}",
            import_root,
        )
    ctx = RunContext(
        clock=datetime.fromisoformat(clock_iso),
        seed=seed,
        timeout_s=tool_timeout_s,
        task_id=task_id,
        turn_index=0,
    )
    outputs: list[Any] = []
    trace: list[dict[str, Any]] = []
    iterator = iter(steps)
    try:
        step = _advance_episode(iterator)
    except StopIteration:
        return outputs
    while True:
        op = step["op"]
        if op == "reset":
            module.reset(ctx=ctx, fixtures=fixtures)
            trace.clear()
            outputs.append(None)
        elif op == "get_state":
            outputs.append(module.get_state())
        elif op == "call_tool":
            arguments = step.get("arguments") or {}
            ctx = replace(ctx, turn_index=int(step.get("turn_index", 0)))
            result = module.call_tool(step["name"], arguments, ctx=ctx)
            trace.append(
                {
                    "tool": step["name"],
                    "arguments": arguments,
                    "result": result,
                    "turn_index": ctx.turn_index,
                }
            )
            outputs.append(result)
        elif op == "list_tools":
            outputs.append(module.list_tools())
        elif op == "metadata":
            metadata = getattr(module, "metadata", None)
            outputs.append(metadata() if callable(metadata) else None)
        elif op == "inspect_backend":
            outputs.append(
                {
                    name: callable(getattr(module, name, None))
                    for name in ("list_tools", "reset", "call_tool", "get_state")
                }
            )
        elif op == "run_assertion":
            if assertions_module is None:
                raise RuntimeError("run_assertion requires assertions_path on the worker")
            outputs.append(
                _assertion_verdict(
                    _resolve_assertion(assertions_module, step["name"]),
                    name=step["name"],
                    state=module.get_state(),
                    trace=step.get("trace") if step.get("trace") is not None else trace,
                    task=step.get("task") or {},
                    ctx=ctx,
                )
            )
        else:
            raise ValueError(f"unknown episode op {op!r}")
        try:
            step = _advance_episode(iterator, outputs[-1])
        except StopIteration:
            break
    return outputs


def _run_backend_episode_sync(
    backend_path: str | None,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    task_id: str,
    tool_timeout_s: float,
    steps: Iterable[dict[str, Any]],
    *,
    sanitize_environment: bool,
    assertions_path: str | None = None,
    import_root: str | None = None,
    endpoint_config: Any = None,
    endpoint_headers: dict[str, str] | None = None,
) -> list[Any]:
    """Run the debug-thread episode and restore process-global import state."""
    oracle_holder: list[Any] = []
    with _restored_import_state():
        try:
            return _run_backend_episode_sync_inner(
                backend_path,
                fixtures,
                clock_iso,
                seed,
                task_id,
                tool_timeout_s,
                steps,
                sanitize_environment=sanitize_environment,
                assertions_path=assertions_path,
                import_root=import_root,
                endpoint_config=endpoint_config,
                endpoint_headers=endpoint_headers,
                oracle_holder=oracle_holder,
            )
        finally:
            close = getattr(oracle_holder[0], "close", None) if oracle_holder else None
            if callable(close):
                close()


def _resolve_assertion(module: ModuleType, name: str) -> Callable[..., Any]:
    exported = getattr(module, "ASSERTIONS", None)
    if isinstance(exported, dict) and name in exported:
        candidate = exported[name]
    else:
        candidate = getattr(module, name, None)
    if not callable(candidate):
        raise LookupError(f"assertion {name!r} is not defined")
    return candidate


def _tracked_assertion_value(
    value: Any,
    *,
    path: tuple[str, ...],
    accessed: set[tuple[str, ...]],
) -> Any:
    if isinstance(value, dict):
        return _AssertionTaskDict(value, path=path, accessed=accessed)
    if isinstance(value, list):
        return _AssertionTaskList(value, path=path, accessed=accessed)
    return value


class _AssertionTaskList(list[Any]):
    """A list whose nested containers retain their assertion-task paths."""

    def __init__(
        self,
        value: list[Any],
        *,
        path: tuple[str, ...],
        accessed: set[tuple[str, ...]],
    ) -> None:
        super().__init__(
            _tracked_assertion_value(
                child,
                path=(*path, str(index)),
                accessed=accessed,
            )
            for index, child in enumerate(value)
        )


class _AssertionTaskDict(dict[str, Any]):
    """A plain-dict-compatible assertion payload that records field reads."""

    def __init__(
        self,
        value: dict[str, Any],
        *,
        path: tuple[str, ...] = (),
        accessed: set[tuple[str, ...]],
    ) -> None:
        self._path = path
        self._accessed = accessed
        super().__init__(
            {
                str(key): _tracked_assertion_value(
                    child,
                    path=(*path, str(key)),
                    accessed=accessed,
                )
                for key, child in value.items()
            }
        )

    def _read(self, key: Any) -> None:
        name = str(key)
        self._accessed.add((*self._path, name))
        child = dict.get(self, name)
        # ``task.get("slots") or {}`` is the ordinary pack idiom, and an empty
        # child is falsy, so every later read would land on an untracked literal.
        # Record the whole child here, while the caller can still be observed.
        # Emptiness goes through ``dict`` rather than ``not child`` so that asking
        # the question does not itself count as reading every field of the child.
        if isinstance(child, _AssertionTaskDict) and dict.__len__(child) == 0:
            self._accessed.add((*self._path, name, "*"))

    def __getitem__(self, key: Any) -> Any:
        self._read(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._read(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self._read(key)
        return super().__contains__(key)

    def _read_all(self) -> None:
        self._accessed.add((*self._path, "*"))

    def __iter__(self) -> Iterator[str]:
        self._read_all()
        return super().__iter__()

    def items(self) -> Any:
        self._read_all()
        return super().items()

    def keys(self) -> Any:
        self._read_all()
        return super().keys()

    def values(self) -> Any:
        self._read_all()
        return super().values()

    def copy(self) -> dict[str, Any]:
        self._read_all()
        return super().copy()

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self._read(key)
        return super().setdefault(key, default)

    def pop(self, key: Any, *default: Any) -> Any:
        self._read(key)
        return super().pop(key, *default)

    def popitem(self) -> tuple[str, Any]:
        self._read_all()
        return super().popitem()

    def __len__(self) -> int:
        # A count drops with the slot that could not be bound, so counting depends
        # on every field even though it names none of them.
        self._read_all()
        return super().__len__()

    def __bool__(self) -> bool:
        # Truthiness is not a count. ``task.get("slots") or {}`` is the ordinary
        # pack idiom, and on a mapping that still holds anything its answer is the
        # same whether or not a slot was dropped, so asking cannot be charged as
        # reading every slot. Emptiness is the case that does depend on the missing
        # fields, and ``_read`` already records the whole child when it sees one.
        # Without this, ``__len__`` would answer for truthiness too and every pack
        # assertion would look like it had read every unresolved slot.
        return dict.__len__(self) > 0

    def __eq__(self, other: object) -> bool:
        self._read_all()
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        self._read_all()
        return super().__ne__(other)


def _assertion_verdict(
    assertion: Callable[..., Any],
    *,
    name: str,
    state: dict[str, Any],
    trace: list[dict[str, Any]],
    task: dict[str, Any],
    ctx: Any,
) -> dict[str, Any]:
    """Run one assertion and keep missing reconstructed slots infrastructural."""
    accessed: set[tuple[str, ...]] = set()
    tracked_task = _AssertionTaskDict(task, accessed=accessed)
    unresolved_by_field = {
        "slots": {
            str(slot)
            for slot in task.get("unresolved_slots") or []
            if isinstance(slot, str)
        },
        "slots_initial": {
            str(slot)
            for slot in task.get("unresolved_slots_initial") or []
            if isinstance(slot, str)
        },
    }
    unresolved_updates = [
        (entry.get("update_index"), str(slot))
        for entry in task.get("unresolved_slot_updates") or []
        if isinstance(entry, dict)
        and isinstance(entry.get("update_index"), int)
        and not isinstance(entry.get("update_index"), bool)
        for slot in entry.get("slots") or []
        if isinstance(slot, str)
    ]
    assertion_error: AssertionError | None = None
    returned: Any = None
    try:
        returned = assertion(state=state, trace=trace, task=tracked_task, ctx=ctx)
    except AssertionError as exc:
        assertion_error = exc
    except Exception as exc:  # noqa: BLE001 — an assertion exception is verdict data
        return {
            "name": name,
            "status": "infrastructure_error",
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }

    missing_reads = sorted(
        f"{field}.{slot}"
        for field, unresolved in unresolved_by_field.items()
        for slot in unresolved
        if (field, slot) in accessed or (field, "*") in accessed
    )
    missing_reads.extend(
        sorted(
            f"slot_updates[{update_index}].values.{slot}"
            for update_index, slot in unresolved_updates
            if (
                ("slot_updates", str(update_index), "values", slot) in accessed
                or ("slot_updates", str(update_index), "values", "*") in accessed
                or ("slot_updates", str(update_index), "*") in accessed
            )
        )
    )
    if missing_reads:
        return {
            "name": name,
            "status": "infrastructure_error",
            "passed": False,
            "detail": (
                "UnresolvedAssertionBinding: assertion accessed unresolved slot(s): "
                + ", ".join(missing_reads)
            ),
        }
    if assertion_error is not None:
        return {
            "name": name,
            "status": "failed",
            "passed": False,
            "detail": str(assertion_error),
        }
    if isinstance(returned, Mapping):
        status = returned.get("status")
        detail = returned.get("detail")
        if status == "not_applicable" and isinstance(detail, str) and detail.strip():
            return {
                "name": name,
                "status": "not_applicable",
                "passed": False,
                "detail": detail,
            }
        return {
            "name": name,
            "status": "infrastructure_error",
            "passed": False,
            "detail": (
                "AssertionContractError: a verdict mapping must declare "
                "status='not_applicable' and a non-empty detail"
            ),
        }
    if returned is not None:
        return {
            "name": name,
            "status": "infrastructure_error",
            "passed": False,
            "detail": (
                "AssertionContractError: assertions return None, or an explicit "
                "not_applicable verdict mapping"
            ),
        }
    return {
        "name": name,
        "status": "passed",
        "passed": True,
        "detail": None,
    }


def _backend_worker(
    connection: Any,
    backend_path: str | None,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    task_id: str,
    tool_timeout_s: float,
    assertions_path: str | None = None,
    import_root: str | None = None,
    endpoint_config: Any = None,
    endpoint_headers: dict[str, str] | None = None,
) -> None:
    """Persistent backend worker; parent enforces import and per-operation deadlines."""
    from datetime import datetime

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.run_context import RunContext

    try:
        _sanitize_pack_environment()
        if endpoint_config is not None:
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
                EndpointOracleClient,
            )

            module = EndpointOracleClient(
                endpoint_config,
                headers=endpoint_headers or {},
                timeout_s=tool_timeout_s,
            )
        else:
            if backend_path is None:
                raise ValueError("backend_path is required for a local oracle")
            module = _load_module(
                Path(backend_path),
                f"_bfcl_backend_{Path(backend_path).stem}_{abs(hash((backend_path, task_id)))}",
                import_root,
            )
        assertions_module = None
        if assertions_path is not None:
            assertions_module = _load_module(
                Path(assertions_path),
                f"_bfcl_worker_assertions_{abs(hash((assertions_path, task_id)))}",
                import_root,
            )
        ctx = RunContext(
            clock=datetime.fromisoformat(clock_iso),
            seed=seed,
            timeout_s=tool_timeout_s,
            task_id=task_id,
            turn_index=0,
        )
        connection.send(("ready", None, None))
        trace: list[dict[str, Any]] = []
        while True:
            step = connection.recv()
            op = step["op"]
            if op == "close":
                close = getattr(module, "close", None)
                if callable(close):
                    close()
                connection.send(("ok", None, None))
                return
            if op == "reset":
                value = module.reset(ctx=ctx, fixtures=fixtures)
                trace.clear()
            elif op == "get_state":
                value = module.get_state()
            elif op == "call_tool":
                arguments = step.get("arguments") or {}
                ctx = replace(ctx, turn_index=int(step.get("turn_index", 0)))
                value = module.call_tool(step["name"], arguments, ctx=ctx)
                trace.append(
                    {
                        "tool": step["name"],
                        "arguments": arguments,
                        "result": value,
                        "turn_index": ctx.turn_index,
                    }
                )
            elif op == "list_tools":
                value = module.list_tools()
            elif op == "metadata":
                metadata = getattr(module, "metadata", None)
                value = metadata() if callable(metadata) else None
            elif op == "inspect_backend":
                value = {
                    name: callable(getattr(module, name, None))
                    for name in ("list_tools", "reset", "call_tool", "get_state")
                }
            elif op == "run_assertion":
                if assertions_module is None:
                    raise RuntimeError("run_assertion requires assertions_path on the worker")
                value = _assertion_verdict(
                    _resolve_assertion(assertions_module, step["name"]),
                    name=step["name"],
                    state=module.get_state(),
                    trace=step.get("trace") if step.get("trace") is not None else trace,
                    task=step.get("task") or {},
                    ctx=ctx,
                )
            else:
                raise ValueError(f"unknown episode op {op!r}")
            connection.send(("ok", value, getattr(module, "session_id", None)))
    except EOFError:
        return
    except BaseException:  # noqa: BLE001
        try:
            connection.send(("err", traceback.format_exc(), getattr(locals().get("module"), "session_id", None)))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        close = getattr(locals().get("module"), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        connection.close()


def _inspect_assertions_module_inner(
    path: str,
    *,
    sanitize_environment: bool,
    import_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Import assertions inside a worker and validate the callable contract."""
    if sanitize_environment:
        _sanitize_pack_environment()
    module = _load_module(Path(path), f"_bfcl_assertions_{abs(hash(path))}", import_root)
    exported = getattr(module, "ASSERTIONS", None)
    if isinstance(exported, dict):
        assertions = {str(name): fn for name, fn in exported.items() if callable(fn)}
    else:
        assertions = {
            name: getattr(module, name)
            for name in dir(module)
            if name.startswith("assert_") and callable(getattr(module, name))
        }
    # Read from the file rather than the imported module so validation admits
    # exactly the declarations executable projection can read back later.
    declared: dict[str, Any] | None = None
    module_reason: str | None = None
    try:
        declared = read_literal_assertion_capabilities(Path(path))
        if declared is None and hasattr(module, ASSERTION_CAPABILITIES_SYMBOL):
            raise AssertionCapabilityError(
                f"{ASSERTION_CAPABILITIES_SYMBOL} is computed rather than "
                "declared literally, so projection cannot read it back"
            )
        if unknown := sorted(set(declared or {}) - set(assertions)):
            raise AssertionCapabilityError(
                f"{ASSERTION_CAPABILITIES_SYMBOL} names unknown assertion(s) {unknown}"
            )
    except AssertionCapabilityError as exc:
        declared, module_reason = None, str(exc)

    required = {"state", "trace", "task", "ctx"}
    report: dict[str, dict[str, Any]] = {}
    for name, fn in assertions.items():
        signature = inspect.signature(fn)
        params = signature.parameters
        has_var_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
        missing = sorted(
            arg
            for arg in required
            if arg not in params or params[arg].kind is inspect.Parameter.POSITIONAL_ONLY
        )
        required_extras = sorted(
            param.name
            for param in params.values()
            if param.name not in required
            and param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            and param.default is inspect.Parameter.empty
        )
        # A malformed capability and a malformed signature are separate defects
        # with separate fixes, so neither is reported as the other.
        capability: dict[str, Any] | None = None
        capability_reason = module_reason
        if module_reason is None:
            try:
                capability = normalized_assertion_capability(
                    name, (declared or {}).get(name, {})
                )
            except AssertionCapabilityError as exc:
                capability_reason = str(exc)
        valid = (has_var_kwargs or not missing) and not required_extras
        report[name] = {
            "valid": valid,
            "reason": None
            if valid
            else f"missing keyword args={missing}; unsupported required args={required_extras}",
            "capabilities": capability,
            "capability_reason": capability_reason,
        }
    return report


def _inspect_assertions_module(
    path: str,
    *,
    sanitize_environment: bool,
    import_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Inspect assertions without retaining pack modules in thread-debug mode."""
    if sanitize_environment:
        return _inspect_assertions_module_inner(
            path,
            sanitize_environment=True,
            import_root=import_root,
        )
    with _restored_import_state():
        return _inspect_assertions_module_inner(
            path,
            sanitize_environment=False,
            import_root=import_root,
        )


def _stop_process(proc: mp.Process) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join(1.0)


@contextmanager
def _projected_pack_tree(
    *,
    backend_path: Path | str | None,
    assertions_path: Path | str | None,
    import_root: Path | str | None,
    fixture_source_path: Path | str,
    fixtures: dict[str, Any] | None,
) -> Iterator[tuple[Path | None, Path | None, Path]]:
    """Mirror a local pack with only the fixture rows this episode may observe.

    This closes the common ``Path(__file__).with_name("fixtures.json")`` bypass.
    It is not an OS sandbox: allowlisted pack code remains trusted and could access
    unrelated host paths if it deliberately names them.
    """
    if import_root is None:
        raise ValueError("fixture-file isolation requires import_root")
    if fixtures is None:
        raise ValueError("fixture-file isolation requires projected fixtures")
    source_root = Path(import_root).resolve()

    def relative(path: Path | str | None, *, label: str) -> Path | None:
        if path is None:
            return None
        try:
            return Path(path).resolve().relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"fixture-file isolation requires {label} to be inside import_root"
            ) from exc

    backend_relative = relative(backend_path, label="backend_path")
    assertions_relative = relative(assertions_path, label="assertions_path")
    fixtures_relative = relative(fixture_source_path, label="fixture_source_path")
    assert fixtures_relative is not None
    with tempfile.TemporaryDirectory(prefix="bfcl-oracle-pack-") as temporary:
        projected_root = Path(temporary) / "pack"
        shutil.copytree(source_root, projected_root, symlinks=True)
        projected_fixtures = projected_root / fixtures_relative
        projected_fixtures.write_text(
            json.dumps(fixtures, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yield (
            None if backend_relative is None else projected_root / backend_relative,
            None if assertions_relative is None else projected_root / assertions_relative,
            projected_root,
        )


@dataclass
class ProcessWorker:
    """Process-worker facade used by oracle validation / replay."""

    default_timeout_s: float = 5.0
    worker: str = "process"

    def __post_init__(self) -> None:
        if self.worker == "thread":
            logger.warning(
                "BFCL oracle_runtime.worker=thread imports pack code into this process with the "
                "host environment visible to it. Only episode_timeout_s bounds the caller's wait; "
                "per-operation timeouts are hard deadlines only in process mode, and a timed-out "
                "thread may keep running. Use thread mode to debug and process mode to certify."
            )

    def call(self, fn: Callable[..., T], /, *args: Any, timeout_s: float | None = None, **kwargs: Any) -> T:
        return run_with_timeout(
            fn,
            timeout_s if timeout_s is not None else self.default_timeout_s,
            *args,
            worker=self.worker,
            **kwargs,
        )

    def run_episode(
        self,
        *,
        backend_path: Path | str | None = None,
        endpoint_config: Any = None,
        endpoint_headers_override: Mapping[str, str] | None = None,
        fixtures: dict[str, Any] | None,
        clock_iso: str,
        seed: int,
        task_id: str,
        steps: Iterable[dict[str, Any]],
        assertions_path: Path | str | None = None,
        import_root: Path | str | None = None,
        import_timeout_s: float | None = None,
        reset_timeout_s: float | None = None,
        tool_timeout_s: float | None = None,
        assertion_timeout_s: float | None = None,
        episode_timeout_s: float | None = None,
        fixture_source_path: Path | str | None = None,
    ) -> list[Any]:
        """Run one backend episode.

        ``steps`` may be a regular iterable or an interactive generator. Interactive
        generators receive each operation result through ``send`` before yielding the
        next step, which lets a later tool call bind arguments from an earlier result
        without restarting the backend. Process workers enforce each operation timeout;
        debug-thread workers can only bound the caller's wait for the whole episode.
        """
        if fixture_source_path is not None:
            if endpoint_config is not None:
                raise ValueError("fixture-file isolation applies only to local oracle packs")
            with _projected_pack_tree(
                backend_path=backend_path,
                assertions_path=assertions_path,
                import_root=import_root,
                fixture_source_path=fixture_source_path,
                fixtures=fixtures,
            ) as (projected_backend, projected_assertions, projected_root):
                return self.run_episode(
                    backend_path=projected_backend,
                    fixtures=fixtures,
                    clock_iso=clock_iso,
                    seed=seed,
                    task_id=task_id,
                    steps=steps,
                    assertions_path=projected_assertions,
                    import_root=projected_root,
                    import_timeout_s=import_timeout_s,
                    reset_timeout_s=reset_timeout_s,
                    tool_timeout_s=tool_timeout_s,
                    assertion_timeout_s=assertion_timeout_s,
                    episode_timeout_s=episode_timeout_s,
                )
        import_timeout = self.default_timeout_s if import_timeout_s is None else import_timeout_s
        reset_timeout = self.default_timeout_s if reset_timeout_s is None else reset_timeout_s
        tool_timeout = self.default_timeout_s if tool_timeout_s is None else tool_timeout_s
        assertion_timeout = self.default_timeout_s if assertion_timeout_s is None else assertion_timeout_s
        episode_timeout = self.default_timeout_s if episode_timeout_s is None else episode_timeout_s
        if (backend_path is None) == (endpoint_config is None):
            raise ValueError("run_episode requires exactly one of backend_path or endpoint_config")
        endpoint_headers: dict[str, str] | None = None
        if endpoint_config is not None:
            if endpoint_headers_override is not None:
                # Intake resolves credentials against a reviewed environment mapping, not
                # the ambient one this process happens to have.
                endpoint_headers = dict(endpoint_headers_override)
            else:
                from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
                    resolve_endpoint_headers,
                )

                endpoint_headers = resolve_endpoint_headers(endpoint_config)

        if self.worker == "thread":
            return run_with_timeout(
                _run_backend_episode_sync,
                episode_timeout,
                None if backend_path is None else str(backend_path),
                fixtures,
                clock_iso,
                seed,
                task_id,
                tool_timeout,
                steps,
                sanitize_environment=False,
                assertions_path=None if assertions_path is None else str(assertions_path),
                import_root=None if import_root is None else str(import_root),
                endpoint_config=endpoint_config,
                endpoint_headers=endpoint_headers,
                worker="thread",
            )
        if self.worker != "process":
            raise ValueError(f"unsupported worker {self.worker!r}")

        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        proc = context.Process(
            target=_backend_worker,
            args=(
                child,
                None if backend_path is None else str(backend_path),
                fixtures,
                clock_iso,
                seed,
                task_id,
                tool_timeout,
                None if assertions_path is None else str(assertions_path),
                None if import_root is None else str(import_root),
                endpoint_config,
                endpoint_headers,
            ),
        )
        started = time.monotonic()
        proc.start()
        child.close()
        remote_session_id: str | None = None

        def cleanup_endpoint_session() -> None:
            nonlocal remote_session_id
            if endpoint_config is None or remote_session_id is None:
                return
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
                EndpointOracleClient,
            )

            client = EndpointOracleClient(
                endpoint_config,
                headers=endpoint_headers or {},
                timeout_s=tool_timeout,
            )
            client.session_id = remote_session_id
            client.close()
            remote_session_id = None

        def exchange(
            timeout_s: float,
            label: str,
            step: dict[str, Any] | None = None,
            *,
            respect_episode_deadline: bool = True,
        ) -> Any:
            """Send one step and receive its reply without blocking past the deadline.

            ``Connection.poll`` only proves that the first bytes of a message are
            readable. ``Connection.recv`` can then block while a large payload is
            still being written, so the complete exchange has to run behind the same
            deadline as the backend operation.
            """
            nonlocal remote_session_id
            remaining = episode_timeout - (time.monotonic() - started)
            timeout = min(timeout_s, remaining) if respect_episode_deadline else timeout_s
            if timeout <= 0:
                _stop_process(proc)
                error = TimeoutError(f"{label} exceeded timeout_s={timeout_s}")
                try:
                    cleanup_endpoint_session()
                except Exception as cleanup_exc:  # noqa: BLE001
                    error.add_note(
                        "endpoint session cleanup also failed: "
                        f"{type(cleanup_exc).__name__}"
                    )
                raise error

            replies: queue_module.Queue[tuple[str, Any]] = queue_module.Queue(maxsize=1)

            def communicate() -> None:
                try:
                    if step is not None:
                        parent.send(step)
                    replies.put(("reply", parent.recv()))
                except BaseException as exc:  # noqa: BLE001 — transferred to caller
                    replies.put(("exception", exc))

            receiver = threading.Thread(target=communicate, daemon=True)
            receiver.start()
            receiver.join(timeout)
            if receiver.is_alive():
                _stop_process(proc)
                parent.close()
                # The receiver is daemonized and may be stalled inside a partial
                # frame read. Waiting for it here would extend a hard tool
                # deadline by an unrelated cleanup second; closing the pipe is
                # sufficient to make the thread unwind when that read resumes.
                error = TimeoutError(f"{label} exceeded timeout_s={timeout_s}")
                try:
                    cleanup_endpoint_session()
                except Exception as cleanup_exc:  # noqa: BLE001
                    error.add_note(
                        "endpoint session cleanup also failed: "
                        f"{type(cleanup_exc).__name__}"
                    )
                raise error
            kind, value = replies.get_nowait()
            if kind == "exception":
                raise RuntimeError(f"worker connection failed during {label}: {value}") from value
            status, payload, session_id = value
            remote_session_id = session_id
            if status == "err":
                _stop_process(proc)
                error = RuntimeError(f"worker failed during {label}:\n{payload}")
                try:
                    cleanup_endpoint_session()
                except Exception as cleanup_exc:  # noqa: BLE001
                    error.add_note(
                        "endpoint session cleanup also failed: "
                        f"{type(cleanup_exc).__name__}"
                    )
                raise error
            return payload

        try:
            exchange(import_timeout, "backend import")
            outputs: list[Any] = []
            operation_timeouts = {"reset": reset_timeout, "run_assertion": assertion_timeout}
            iterator = iter(steps)
            try:
                step = _advance_episode(iterator)
                while True:
                    op = step["op"]
                    output = exchange(operation_timeouts.get(op, tool_timeout), op, step)
                    outputs.append(output)
                    step = _advance_episode(iterator, output)
            except StopIteration:
                pass
            exchange(
                tool_timeout,
                "session close",
                {"op": "close"},
                respect_episode_deadline=False,
            )
            proc.join(min(tool_timeout, max(0.0, episode_timeout - (time.monotonic() - started))))
            _stop_process(proc)
            return outputs
        finally:
            _stop_process(proc)
            parent.close()
            # Covers parent-side generator failures and broken IPC after a reset:
            # neither path passes through exchange's timeout/error cleanup.
            cleanup_endpoint_session()

    def inspect_assertions(
        self,
        assertions_path: Path | str,
        *,
        import_root: Path | str | None = None,
        timeout_s: float | None = None,
        fixture_source_path: Path | str | None = None,
        fixtures: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Discover assertions and validate signatures without importing in the parent."""
        if fixture_source_path is not None:
            with _projected_pack_tree(
                backend_path=None,
                assertions_path=assertions_path,
                import_root=import_root,
                fixture_source_path=fixture_source_path,
                fixtures=fixtures,
            ) as (_, projected_assertions, projected_root):
                assert projected_assertions is not None
                return self.inspect_assertions(
                    projected_assertions,
                    import_root=projected_root,
                    timeout_s=timeout_s,
                )
        return self.call(
            _inspect_assertions_module,
            str(assertions_path),
            sanitize_environment=self.worker == "process",
            import_root=None if import_root is None else str(import_root),
            timeout_s=timeout_s,
        )
