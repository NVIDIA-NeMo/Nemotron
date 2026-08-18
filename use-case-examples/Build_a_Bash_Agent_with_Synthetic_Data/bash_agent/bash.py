"""Validated LangGraph CLI execution with no intervening shell."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from .commands import (
    LangGraphInvocation,
    invocation_from_argv,
    invocation_from_payload,
)
from .config import Config


_SECRET_ENV_NAMES = frozenset(
    {
        "API_KEY",
        "CREDENTIALS",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGGRAPH_CLOUD_LICENSE_KEY",
        "PASSWORD",
        "SECRET",
        "TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    }
)
_SECRET_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIALS",
)
_READ_CHUNK_BYTES = 64 * 1024
_TRUNCATION_MARKER = b"\n...[output truncated at configured byte limit]...\n"


class _BoundedBuffer:
    """Drain an entire stream while retaining at most ``limit`` bytes."""

    def __init__(self, limit: int):
        self.limit = limit
        self._data = bytearray()
        self._truncated = False
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            if self._truncated:
                return
            remaining = self.limit - len(self._data)
            self._data.extend(data[:remaining])
            if len(data) > remaining:
                self._truncated = True
                marker = _TRUNCATION_MARKER[: self.limit]
                self._data[-len(marker) :] = marker

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace").strip()


class _BoundedLog:
    """Write a live log whose on-disk size never exceeds ``limit`` bytes."""

    def __init__(self, log_file: BinaryIO, limit: int):
        self.log_file = log_file
        self.limit = limit
        self._written = 0
        self._truncated = False

    def append(self, data: bytes) -> None:
        if self._truncated:
            return
        remaining = self.limit - self._written
        retained = data[:remaining]
        if retained:
            self.log_file.write(retained)
            self._written += len(retained)
        if len(data) > remaining:
            self._truncated = True
            marker = _TRUNCATION_MARKER[: self.limit]
            self.log_file.seek(self.limit - len(marker))
            self.log_file.write(marker)
            self.log_file.truncate(self.limit)
            self.log_file.seek(0, os.SEEK_END)
            self._written = self.limit
        self.log_file.flush()


class _OutputDrainer:
    """Drain a binary subprocess pipe without retaining unbounded output."""

    def __init__(self, stream: BinaryIO, sink: _BoundedBuffer | _BoundedLog):
        self.stream = stream
        self.sink = sink
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="langgraph-output-drainer",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            os.set_blocking(self.stream.fileno(), False)
            while not self._stop.is_set():
                try:
                    data = os.read(self.stream.fileno(), _READ_CHUNK_BYTES)
                except BlockingIOError:
                    self._stop.wait(0.05)
                    continue
                if not data:
                    break
                self.sink.append(data)
        except (OSError, ValueError):
            # Closing a pipe during bounded shutdown can race with this thread.
            pass

    def finish(self, timeout: float) -> None:
        """Drain through EOF, then stop within a small bounded interval."""
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=0.1)
        try:
            self.stream.close()
        except OSError:
            pass


@dataclass(slots=True)
class _BackgroundProcess:
    process: subprocess.Popen[bytes]
    log_file: BinaryIO
    log_path: str
    drainer: _OutputDrainer


def _is_secret_environment_name(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in _SECRET_ENV_NAMES or upper_name.endswith(_SECRET_ENV_SUFFIXES)


class Bash:
    """Prepare and execute the tutorial's narrowly scoped CLI invocations."""

    def __init__(self, config: Config):
        if (
            isinstance(config.output_limit_bytes, bool)
            or not isinstance(config.output_limit_bytes, int)
            or config.output_limit_bytes < len(_TRUNCATION_MARKER)
        ):
            raise ValueError(
                f"output_limit_bytes must be an integer of at least {len(_TRUNCATION_MARKER)}"
            )
        self.config = config
        self.cwd = str(Path(config.root_dir).resolve())
        self._background_processes: dict[int, _BackgroundProcess] = {}
        self._temporary_logs: set[Path] = set()

    def prepare_payload(self, payload: Mapping[str, Any]) -> LangGraphInvocation:
        """Turn the trained model's structured output into validated argv."""
        return invocation_from_payload(payload, self.cwd)

    def prepare_argv(self, argv: Sequence[str]) -> LangGraphInvocation:
        """Validate argv received from an API tool call."""
        return invocation_from_argv(argv, self.cwd)

    def child_environment(self) -> dict[str, str]:
        """Copy ordinary environment values while excluding credentials by default."""
        explicitly_allowed = set(self.config.pass_environment)
        return {
            name: value
            for name, value in os.environ.items()
            if name in explicitly_allowed or not _is_secret_environment_name(name)
        }

    @staticmethod
    def _is_background(invocation: LangGraphInvocation) -> bool:
        command = invocation.argv[1]
        return command == "dev" or (command == "up" and "--wait" not in invocation.argv)

    def _spawn(self, invocation: LangGraphInvocation, **kwargs: Any) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            list(invocation.argv),
            shell=False,
            cwd=self.cwd,
            env=self.child_environment(),
            start_new_session=True,
            **kwargs,
        )

    @staticmethod
    def _send_process_group_signal(process: subprocess.Popen[bytes], sig: int) -> bool:
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            # This fallback is only for platforms where group signaling is
            # unavailable. Supported Unix systems use the branch above.
            if process.poll() is not None:
                return False
            try:
                process.send_signal(sig)
                return True
            except ProcessLookupError:
                return False

    @staticmethod
    def _process_group_exists(process: subprocess.Popen[bytes]) -> bool:
        # Reap the group leader if it has exited, then probe the PGID. The PGID
        # can remain alive after its leader exits when descendants still run.
        process.poll()
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _wait_for_process_group_exit(
        self, process: subprocess.Popen[bytes], timeout: float
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while self._process_group_exists(process):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return True

    @staticmethod
    def _reap_process(process: subprocess.Popen[bytes]) -> None:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> None:
        """Stop all session descendants, even when their group leader has exited."""
        term_sent = self._send_process_group_signal(process, signal.SIGTERM)
        if term_sent and not self._wait_for_process_group_exit(
            process, self.config.background_shutdown_grace_seconds
        ):
            kill_sent = self._send_process_group_signal(process, signal.SIGKILL)
            if kill_sent:
                self._wait_for_process_group_exit(
                    process, self.config.background_shutdown_grace_seconds
                )
        self._reap_process(process)

    def _run_finite(self, invocation: LangGraphInvocation) -> dict[str, Any]:
        try:
            process = self._spawn(
                invocation,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            return self._spawn_error("The 'langgraph' executable was not found.")
        except OSError as exc:
            return self._spawn_error(str(exc))

        stdout_capture = _BoundedBuffer(self.config.output_limit_bytes)
        stderr_capture = _BoundedBuffer(self.config.output_limit_bytes)
        drainers: list[_OutputDrainer] = []
        timed_out = False
        try:
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Failed to create output pipes for the LangGraph command.")
            drainers.append(_OutputDrainer(process.stdout, stdout_capture))
            drainers.append(_OutputDrainer(process.stderr, stderr_capture))
            process.wait(timeout=self.config.command_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process_group(process)
        except BaseException:
            # Ctrl-C and other interruptions must not leave the command or its
            # descendants running after control returns to the launcher.
            self._terminate_process_group(process)
            raise
        finally:
            for drainer in drainers:
                drainer.finish(self.config.background_shutdown_grace_seconds)
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

        stdout = stdout_capture.text()
        stderr = stderr_capture.text()
        if timed_out:
            timeout_message = (
                "Command timed out after "
                f"{self.config.command_timeout_seconds:g} seconds; "
                "its process group was terminated."
            )
            stderr = f"{stderr}\n{timeout_message}".strip()
        elif not stdout and not stderr and process.returncode == 0:
            stdout = "Command executed successfully, without any output."
        return {
            "stdout": stdout,
            "stderr": stderr,
            "cwd": self.cwd,
            "returncode": process.returncode,
            "output_limit_bytes_per_stream": self.config.output_limit_bytes,
        }

    def _remove_log(self, log_path: str | Path) -> None:
        path = Path(log_path)
        path.unlink(missing_ok=True)
        self._temporary_logs.discard(path)

    def _start_background(self, invocation: LangGraphInvocation) -> dict[str, Any]:
        log_file = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="langgraph-agent-",
            suffix=".log",
            delete=False,
        )
        log_path = Path(log_file.name)
        self._temporary_logs.add(log_path)
        try:
            process = self._spawn(
                invocation,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            log_file.close()
            self._remove_log(log_path)
            return self._spawn_error("The 'langgraph' executable was not found.")
        except OSError as exc:
            log_file.close()
            self._remove_log(log_path)
            return self._spawn_error(str(exc))

        drainer: _OutputDrainer | None = None
        try:
            if process.stdout is None:
                raise RuntimeError("Failed to create a background output pipe.")
            log_sink = _BoundedLog(log_file, self.config.output_limit_bytes)
            drainer = _OutputDrainer(process.stdout, log_sink)
            try:
                returncode = process.wait(timeout=self.config.background_startup_grace_seconds)
            except subprocess.TimeoutExpired:
                managed = _BackgroundProcess(
                    process=process,
                    log_file=log_file,
                    log_path=str(log_path),
                    drainer=drainer,
                )
                self._background_processes[process.pid] = managed
                return {
                    "stdout": (
                        f"Started background process {process.pid}. Combined output is in "
                        f"{log_path} (capped at {self.config.output_limit_bytes} bytes and "
                        "deleted when this executor closes)."
                    ),
                    "stderr": "",
                    "cwd": self.cwd,
                    "returncode": None,
                    "pid": process.pid,
                    "log_path": str(log_path),
                    "background": True,
                    "log_limit_bytes": self.config.output_limit_bytes,
                    "log_retention": "deleted_on_close",
                }
        except BaseException:
            try:
                self._terminate_process_group(process)
            finally:
                if drainer is not None:
                    drainer.finish(self.config.background_shutdown_grace_seconds)
                elif process.stdout is not None:
                    process.stdout.close()
                log_file.close()
                self._remove_log(log_path)
            raise

        assert drainer is not None
        drainer.finish(self.config.background_shutdown_grace_seconds)
        log_file.flush()
        log_file.seek(0)
        output = log_file.read().decode("utf-8", errors="replace").strip()
        log_file.close()
        return {
            "stdout": output,
            "stderr": (
                ""
                if returncode == 0
                else f"Process exited during startup with status {returncode}."
            ),
            "cwd": self.cwd,
            "returncode": returncode,
            "pid": process.pid,
            "log_path": str(log_path),
            "background": False,
            "log_limit_bytes": self.config.output_limit_bytes,
            "log_retention": "deleted_on_close",
        }

    def _spawn_error(self, message: str) -> dict[str, Any]:
        return {
            "stdout": "",
            "stderr": message,
            "cwd": self.cwd,
            "returncode": None,
        }

    def exec_langgraph(self, invocation: LangGraphInvocation) -> dict[str, Any]:
        """Execute a previously validated invocation without a shell."""
        if not isinstance(invocation, LangGraphInvocation):
            raise TypeError("exec_langgraph() requires a prepared LangGraphInvocation.")
        # Revalidate at the execution boundary so manually constructed instances
        # and paths changed after confirmation cannot bypass the command grammar.
        invocation = self.prepare_argv(invocation.argv)
        if self._is_background(invocation):
            return self._start_background(invocation)
        return self._run_finite(invocation)

    def close(self) -> None:
        """Stop background groups and delete their size-limited temporary logs."""
        for managed in list(self._background_processes.values()):
            try:
                self._terminate_process_group(managed.process)
            finally:
                managed.drainer.finish(self.config.background_shutdown_grace_seconds)
                managed.log_file.close()
        self._background_processes.clear()
        for log_path in list(self._temporary_logs):
            self._remove_log(log_path)

    def to_json_schema(self) -> dict[str, Any]:
        """Return the OpenAI-compatible schema for validated argv calls."""
        return {
            "type": "function",
            "function": {
                "name": "run_langgraph",
                "description": (
                    "Run one LangGraph CLI command. Supply a process argument array, "
                    "not a shell command string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 8,
                            "description": (
                                "A LangGraph argv array, for example "
                                "['langgraph', 'dev', '--port', '2024', "
                                "'--no-browser']."
                            ),
                        }
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
            },
        }
