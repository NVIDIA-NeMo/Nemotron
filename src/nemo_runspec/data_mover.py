# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ship local ``src/`` to remote pods.

Per-executor strategy:

* **lepton** / **dgxcloud** — chunk tarball across env vars. The env field
  bypasses argv / Args limits, and each pod reassembles the source locally.
* **anything else** — nemo-run's native packager extraction.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import typer

try:
    from nemo_run.core.packaging.base import Packager as _BasePackager
except ImportError:
    _BasePackager = object  # type: ignore[assignment,misc]


_EXCLUDE_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".venv",
        "node_modules",
    }
)
_EXCLUDE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pyd",
    # Common data/model artifacts should not ride along in the source tarball.
    ".parquet",
    ".arrow",
    ".bin",
    ".idx",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".h5",
    ".hdf5",
)
_SCOPED_COLLECTIONS = frozenset({"recipes", "steps"})
_DEFAULT_TARBALL_WARN_BYTES = 1_000_000


def _tar_filter(info):
    base = os.path.basename(info.name)
    if base in _EXCLUDE_NAMES or base.endswith(_EXCLUDE_SUFFIXES):
        return None
    return info


def _tarball_warn_bytes() -> int:
    raw = os.environ.get("NEMOTRON_SRC_TARBALL_WARN_BYTES")
    if raw is None:
        return _DEFAULT_TARBALL_WARN_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_TARBALL_WARN_BYTES


def _format_bytes(size: int) -> str:
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _warn_if_large_tarball(path: str) -> None:
    limit = _tarball_warn_bytes()
    if not limit:
        return
    size = os.path.getsize(path)
    if size <= limit:
        return
    typer.secho(
        "[stage] warning: source tarball is "
        f"{_format_bytes(size)}; this may exceed cloud job/env limits. "
        "Move large artifacts outside src/ or extend the data_mover exclude "
        "suffix list. Set NEMOTRON_SRC_TARBALL_WARN_BYTES=0 to disable.",
        fg=typer.colors.YELLOW,
        err=True,
    )


@dataclass(frozen=True)
class _ScriptLocation:
    package: str
    collection: str | None
    branch: str | None


def _repo_relative_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(repo_root)
        except ValueError:
            if "src" in candidate.parts:
                return Path(*candidate.parts[candidate.parts.index("src") :])
    return candidate


def _script_location(repo_root: Path, script_path: str | None) -> _ScriptLocation | None:
    if not script_path:
        return None
    rel = _repo_relative_path(repo_root, script_path)
    parts = rel.parts
    if len(parts) < 3 or parts[0] != "src":
        return None
    collection = parts[2] if len(parts) >= 4 else None
    branch = parts[3] if len(parts) >= 5 else None
    return _ScriptLocation(
        package=parts[1],
        collection=collection,
        branch=branch,
    )


def _include_collection(
    includes: list[str],
    *,
    pkg_name: str,
    collection_name: str,
    collection_dir: Path,
    branch: str | None = None,
) -> None:
    prefix = f"src/{pkg_name}/{collection_name}"
    for child in sorted(collection_dir.iterdir()):
        if child.name in _EXCLUDE_NAMES:
            continue
        if branch is None or child.is_file() or child.name == branch or child.name.startswith("_"):
            includes.append(f"{prefix}/{child.name}")


def _auto_includes(repo_root: Path, script_path: str | None) -> list[str]:
    """Discover repo-relative paths to ship.

    Walks ``<repo>/src/*`` and ships every top-level package. For packages with
    large source collections such as ``recipes/`` or ``steps/``, only the active
    branch from ``script_path`` is included when possible. This keeps the
    tarball small because unrelated runnable collections can weigh many MiB.
    """
    src = repo_root / "src"
    if not src.is_dir():
        raise ValueError(f"No src/ under {repo_root}. Set repo_root in env.toml.")

    includes: list[str] = []
    script = _script_location(repo_root, script_path)

    for pkg in sorted(p for p in src.iterdir() if p.is_dir() and p.name not in _EXCLUDE_NAMES):
        active_collection = (
            script.collection
            if script and script.package == pkg.name and script.collection and script.branch
            else None
        )
        collection_names = {name for name in _SCOPED_COLLECTIONS if (pkg / name).is_dir()}
        if active_collection and (pkg / active_collection).is_dir():
            collection_names.add(active_collection)

        if collection_names:
            for child in sorted(pkg.iterdir()):
                if child.name in _EXCLUDE_NAMES or child.name in collection_names:
                    continue
                includes.append(f"src/{pkg.name}/{child.name}")

            for collection_name in sorted(collection_names):
                collection_dir = pkg / collection_name
                is_active = active_collection == collection_name and script and script.branch
                if active_collection and not is_active:
                    continue
                _include_collection(
                    includes,
                    pkg_name=pkg.name,
                    collection_name=collection_name,
                    collection_dir=collection_dir,
                    branch=script.branch if is_active else None,
                )
        else:
            includes.append(f"src/{pkg.name}")
    return list(dict.fromkeys(includes))


@dataclass(kw_only=True)
class SourcePackager(_BasePackager):
    """Tarballs local src/ into ``job_dir``.

    Returns the ``.tar.gz`` path so nemo-run's native flow extracts it into
    ``job_dir/code`` (Slurm). For Lepton / DGXCloud we instead read the
    tarball bytes and chunk them across env vars — see :func:`plan_for`.
    """

    repo_root: str
    script_path: str | None = None
    # Accepted for backwards-compatibility with cached fiddle Configs that
    # were serialized while the DGXCloud file-staging path existed. The
    # field is no longer read — DGXCloud now uses env-var chunks like Lepton.
    fixed_output_name: str | None = None

    def package(self, path, job_dir, name):  # type: ignore[override]
        out = os.path.join(job_dir, f"{name}.tar.gz")
        if not os.path.exists(out):
            root = Path(self.repo_root)
            with tarfile.open(out, "w:gz") as tf:
                for rel in _auto_includes(root, self.script_path):
                    tf.add(root / rel, arcname=rel, filter=_tar_filter)
        _warn_if_large_tarball(out)
        return out


@dataclass
class Plan:
    packager: _BasePackager
    pod_src_root: str
    pre_script_cmds: list[str] = field(default_factory=list)
    needs_pwd_symlinks: bool = False
    source_ready_marker: str | None = None


def plan_for(
    *,
    executor_type: str,
    env_vars: dict[str, str],
    script_path: str | None,
    pod_nemotron_home: str,
    repo_root: str | Path | None = None,
) -> Plan:
    """Build a :class:`Plan` for ``executor_type``. Mutates ``env_vars``."""
    from nemo_runspec.run import (
        patch_cloud_data_mover_skip_configs,
        patch_dgxcloud_accept_legacy_kwargs,
        patch_dgxcloud_strip_source_chunks_from_exports,
    )

    patch_cloud_data_mover_skip_configs()
    # DGXCloud needs chunks delivered via ``environmentVariables`` only, not
    # re-baked into ``torchrun_job.sh`` — otherwise the file balloons and
    # ``move_data`` spends ~12 min chunking it into ~46 workloads.
    patch_dgxcloud_strip_source_chunks_from_exports()
    # Silence the noisy "unexpected keyword argument 'app_id'" traceback that
    # fiddle's error-message formatter emits on every status poll when it
    # re-hydrates cached DGXCloudExecutor Configs with legacy kwargs.
    patch_dgxcloud_accept_legacy_kwargs()

    root = Path(repo_root or Path(__file__).resolve().parents[2])
    common = {"repo_root": str(root), "script_path": script_path}

    # Both Lepton and DGXCloud deliver env vars via the Job spec's structured
    # field, which bypasses the per-workload Args cap entirely. Use the same
    # chunked-env-var transport for both.
    if executor_type in ("lepton", "dgxcloud"):
        # Chunk the tarball across env vars; pod reassembles via python3.
        # Per-value caps differ by platform:
        #   * Lepton: 128 KiB MAX_ARG_STRLEN → 96 KiB leaves 32 KiB headroom.
        #   * DGX Cloud: **10 000 chars per env-var value** (hard 400 otherwise)
        #     → 9000 bytes base64, with 1 KiB headroom.
        chunk_bytes = 9_000 if executor_type == "dgxcloud" else 96 * 1024
        with tempfile.TemporaryDirectory() as td:
            raw = Path(SourcePackager(**common).package(None, td, "nemotron-src")).read_bytes()
        source_digest = hashlib.sha256(raw).hexdigest()[:16]
        source_marker_id = f"{source_digest}-{uuid.uuid4().hex[:8]}"
        pod_src = f"{pod_nemotron_home}/src-{source_marker_id}"
        b64 = base64.b64encode(raw).decode("ascii")
        chunks = [b64[i : i + chunk_bytes] for i in range(0, len(b64), chunk_bytes)]
        env_vars["_NEMOTRON_SRC_CHUNKS"] = str(len(chunks))
        env_vars["_NEMOTRON_SRC_SHA256"] = source_digest
        for i, c in enumerate(chunks):
            env_vars[f"_NEMOTRON_SRC_CHUNK_{i}"] = c
        typer.echo(
            f"[stage] {executor_type}: {len(raw) // 1024} KiB raw → "
            f"{len(chunks)} env-var chunks (sha256={source_digest})"
        )
        import nemo_run as run

        # Multi-pod NFS race: when N pods share a filesystem, use a unique
        # destination per submission and gate extraction on NODE_RANK. This
        # avoids clobbering another active run's import tree.
        pod_src_q = shlex.quote(pod_src)
        ready_marker = f"{pod_src}/.nemotron-src-ready"
        ready_marker_q = shlex.quote(ready_marker)
        failed_marker = f"{pod_src}/.nemotron-src-failed"
        failed_marker_q = shlex.quote(failed_marker)
        extract_cmd = (
            "python3 -c 'import os,sys,base64;"
            'n=int(os.environ["_NEMOTRON_SRC_CHUNKS"]);'
            'sys.stdout.buffer.write(base64.b64decode("".join('
            'os.environ[f"_NEMOTRON_SRC_CHUNK_{i}"] for i in range(n))))\''
            f" | tar -xz -C {pod_src_q} --strip-components=1"
        )
        return Plan(
            packager=run.Packager(),
            pod_src_root=pod_src,
            needs_pwd_symlinks=True,
            source_ready_marker=ready_marker,
            pre_script_cmds=[
                'if [ "${NODE_RANK:-0}" = "0" ]; then'
                f" rm -rf {pod_src_q} && mkdir -p {pod_src_q} && {extract_cmd}"
                f" && touch {ready_marker_q}"
                f" || {{ status=$?; mkdir -p {pod_src_q}; touch {failed_marker_q}; exit $status; }};"
                " else i=0;"
                " while [ \"$i\" -lt 600 ]; do"
                f" [ -f {ready_marker_q} ] && break;"
                f" [ -f {failed_marker_q} ] && echo 'source extraction failed on rank 0' >&2 && exit 1;"
                " i=$((i + 1)); sleep 2;"
                " done;"
                f" [ -f {ready_marker_q} ] || "
                f"{{ echo 'timed out waiting for {ready_marker_q}' >&2; exit 1; }};"
                " fi"
            ],
        )

    # Fallback: nemo-run extracts into /nemo_run/code/src (Slurm / others).
    typer.echo("[stage] native packager: /nemo_run/code/src")
    return Plan(
        packager=SourcePackager(**common),
        pod_src_root="/nemo_run/code/src",
        needs_pwd_symlinks=True,
    )


__all__ = ["SourcePackager", "Plan", "plan_for"]
