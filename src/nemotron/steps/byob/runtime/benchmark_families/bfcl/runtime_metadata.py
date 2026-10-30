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

"""Portable execution-environment metadata shared by BFCL manifests.

Every field is best effort by design. A manifest that cannot be written because
provenance collection hit an unreadable file would discard a completed run, so a
detail that cannot be established is reported as null instead.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_STEP_PACKAGE = "byob"


def _step_package_root() -> Path:
    """Locate the step package that owns this pipeline, not just this family.

    Hashing only the family directory would report an unchanged pipeline after a
    change to the shared runtime, process worker, or oracle isolation code that
    this family executes through.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == _STEP_PACKAGE and (parent / "__init__.py").is_file():
            return parent
    return here.parent


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _in_pipeline_worktree() -> bool:
    """Refuse to report a SHA from a repository that does not hold this code.

    An installed wheel can sit inside an unrelated checkout, where ``rev-parse``
    happily answers with a commit that never contained this pipeline.
    """
    if _git("rev-parse", "--is-inside-work-tree") != "true":
        return False
    toplevel = _git("rev-parse", "--show-toplevel")
    if not toplevel:
        return False
    try:
        package_root = _step_package_root()
        relative_root = package_root.relative_to(Path(toplevel).resolve())
    except ValueError:
        return False
    marker = (relative_root / "__init__.py").as_posix()
    # A wheel installed in <unrelated-repo>/.venv is physically below that repo,
    # but its bytes were not sourced from the repo's HEAD. Only tracked source may
    # inherit the surrounding checkout's revision.
    return _git("ls-files", "--error-unmatch", "--", marker) == marker


def _dependency_lock_hash() -> str | None:
    for parent in Path(__file__).resolve().parents:
        lock = parent / "uv.lock"
        if lock.is_file():
            try:
                return f"sha256:{hashlib.sha256(lock.read_bytes()).hexdigest()}"
            except OSError:
                return None
    return None


def _pipeline_source_hash() -> str | None:
    root = _step_package_root()
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*.py")):
            logical_name = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(logical_name).to_bytes(8, "big"))
            digest.update(logical_name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    except OSError:
        return None
    return f"sha256:{digest.hexdigest()}"


def runtime_metadata() -> dict[str, Any]:
    """Describe the interpreter, platform, source, lock, and worker image."""
    in_worktree = _in_pipeline_worktree()
    environment_sha = next(
        (
            value.strip()
            for name in ("GIT_COMMIT", "CI_COMMIT_SHA")
            if (value := os.environ.get(name))
            if value.strip()
        ),
        None,
    )
    git_sha = environment_sha or (_git("rev-parse", "HEAD") if in_worktree else None)
    dirty = _git("status", "--porcelain") if in_worktree else None
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "pipeline_git_sha": git_sha,
        "pipeline_git_dirty": None if dirty is None else bool(dirty),
        "pipeline_source_hash": _pipeline_source_hash(),
        "dependency_lock_hash": _dependency_lock_hash(),
        "worker_image_digest": os.environ.get("BFCL_WORKER_IMAGE_DIGEST"),
    }


__all__ = ["runtime_metadata"]
