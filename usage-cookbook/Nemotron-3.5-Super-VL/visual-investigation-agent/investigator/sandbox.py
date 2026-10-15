# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-written Python with no network, no writes outside a scratch folder, and reads limited to the data.

Two backends: macOS sandbox-exec, and on Linux bubblewrap (bwrap), which runs the code in its own namespaces with only
the system libraries, the Python interpreter, and the world's data mounted read-only. Where neither is available,
run_python refuses to run code rather than running it unsandboxed.
"""

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import world

TIMEOUT_SECONDS = 60
SYSTEM_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc/ld.so.cache", "/etc/ld.so.conf",
                "/etc/ld.so.conf.d", "/etc/alternatives")


def available():
    """Name of the sandbox backend on this machine, or None."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    if sys.platform.startswith("linux") and _bwrap_works():
        return "bwrap"
    return None


@functools.lru_cache(maxsize=None)
def _bwrap_works():
    """bubblewrap is installed and this kernel lets an unprivileged user create the namespaces it needs."""
    if not shutil.which("bwrap"):
        return False
    try:
        probe = subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "true"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _interpreter_dirs():
    return sorted({str(Path(p).resolve()) for p in (sys.prefix, sys.base_prefix, Path(sys.executable).resolve().parent.parent)})


def _profile(data_dir, scratch):
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{scratch}") (literal "/dev/null"))'
            f'(deny file-read-data (subpath "{Path.home()}"))'
            f'(allow file-read-data (subpath "{data_dir}") (subpath "{scratch}")'
            + "".join(f' (subpath "{p}")' for p in _interpreter_dirs()) + ")")


def _bwrap_args(data_dir, scratch):
    """A new, read-only root with nothing but the system libraries, the interpreter, and the data; only the scratch
    folder is writable, and it is also where temporary files go."""
    args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "TMPDIR", scratch, "--proc", "/proc", "--dev", "/dev"]
    for path in SYSTEM_PATHS:
        if os.path.islink(path):
            args += ["--symlink", os.readlink(path), path]
        elif os.path.exists(path):
            args += ["--ro-bind", path, path]
    for path in _interpreter_dirs() + [str(data_dir)]:
        args += ["--ro-bind", path, path]
    return args + ["--bind", scratch, scratch, "--remount-ro", "/", "--chdir", scratch]


def run_python(scenario, code):
    backend = available()
    if backend is None:
        return {"error": "Python is unavailable: this machine has no sandbox for model-written code, so it was not run."}
    data_dir = (world.WORLD / scenario).resolve()
    scratch_root = world.WORLD / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
        scratch = str(Path(scratch).resolve())
        prelude = f"import sqlite3\nconn = sqlite3.connect('file:{world.db_path(scenario).resolve()}?mode=ro', uri=True)\n"
        wrapper = ["sandbox-exec", "-p", _profile(data_dir, scratch)] if backend == "sandbox-exec" else _bwrap_args(data_dir, scratch)
        try:
            done = subprocess.run(wrapper + [sys.executable, "-I", "-B", "-c", prelude + code],
                                  capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=scratch, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return {"error": f"Python timed out after {TIMEOUT_SECONDS} seconds"}
    return {"stdout": done.stdout[-8000:], "stderr": done.stderr[-3000:], "exit_code": done.returncode}
