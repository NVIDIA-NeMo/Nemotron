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

Per-executor strategy (one function, three branches):

* **lepton** — chunk tarball across env vars (envp bypasses 128 KiB argv limit).
* **dgxcloud** — drop tarball in ``job_dir`` as one file; run:AI's ``move_data``
  chunks only that file (its API caps each workload at 10 000 chars).
* **anything else** — nemo-run's native packager extraction.
"""

from __future__ import annotations

import base64
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import typer

try:
    from nemo_run.core.packaging.base import Packager as _BasePackager
except ImportError:
    _BasePackager = object  # type: ignore[assignment,misc]


_EXCLUDE_NAMES = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".git", ".venv", "node_modules",
})
_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pyd")


def _tar_filter(info):
    base = os.path.basename(info.name)
    if base in _EXCLUDE_NAMES or base.endswith(_EXCLUDE_SUFFIXES):
        return None
    return info


def _auto_includes(repo_root: Path, script_path: str | None) -> list[str]:
    """Discover repo-relative paths to ship.

    Walks ``<repo>/src/*`` and ships every top-level package. For a package
    that has a ``recipes/`` subdir (conventional in this repo), only the
    *active* recipe family from ``script_path`` is included (e.g. ``nano3``)
    to keep the tarball small — unrelated families can weigh many MiB.
    """
    src = repo_root / "src"
    if not src.is_dir():
        raise ValueError(f"No src/ under {repo_root}. Set repo_root in env.toml.")

    includes: list[str] = []
    family = None
    if script_path:
        parts = Path(script_path).parts
        if "recipes" in parts:
            idx = parts.index("recipes")
            if idx + 1 < len(parts):
                family = parts[idx + 1]

    for pkg in sorted(p for p in src.iterdir() if p.is_dir() and p.name not in _EXCLUDE_NAMES):
        recipes = pkg / "recipes"
        if recipes.is_dir():
            # Ship every non-recipes child + top-level recipe files + one family.
            for child in sorted(pkg.iterdir()):
                if child.name in _EXCLUDE_NAMES or child == recipes:
                    continue
                includes.append(f"src/{pkg.name}/{child.name}")
            for child in sorted(recipes.iterdir()):
                if child.is_file():
                    includes.append(f"src/{pkg.name}/recipes/{child.name}")
            chosen_families: list[str] = [family] if family and (recipes / family).is_dir() else [
                c.name for c in recipes.iterdir() if c.is_dir() and c.name not in _EXCLUDE_NAMES
            ]
            for fam in sorted(chosen_families):
                includes.append(f"src/{pkg.name}/recipes/{fam}")
        else:
            includes.append(f"src/{pkg.name}")
    return includes


@dataclass(kw_only=True)
class SourcePackager(_BasePackager):
    """Tarballs local src/ into ``job_dir``.

    With ``fixed_output_name`` set, writes under that name and returns ``None``
    so nemo-run skips its auto-extract step (DGXCloud file-in-job_dir flow).
    Otherwise returns the ``.tar.gz`` path and nemo-run extracts into
    ``job_dir/code`` (Slurm flow).
    """

    repo_root: str
    script_path: str | None = None
    fixed_output_name: str | None = None

    def package(self, path, job_dir, name):  # type: ignore[override]
        out = os.path.join(job_dir, self.fixed_output_name or f"{name}.tar.gz")
        if not os.path.exists(out):
            root = Path(self.repo_root)
            with tarfile.open(out, "w:gz") as tf:
                for rel in _auto_includes(root, self.script_path):
                    tf.add(root / rel, arcname=rel, filter=_tar_filter)
        return None if self.fixed_output_name else out


@dataclass
class Plan:
    packager: _BasePackager
    pod_src_root: str
    pre_script_cmds: list[str] = field(default_factory=list)
    needs_pwd_symlinks: bool = False


def plan_for(
    *,
    executor_type: str,
    env_vars: dict[str, str],
    script_path: str | None,
    pod_nemotron_home: str,
    repo_root: str | Path | None = None,
) -> Plan:
    """Build a :class:`Plan` for ``executor_type``. Mutates ``env_vars``."""
    from nemo_runspec.run import patch_cloud_data_mover_skip_configs
    patch_cloud_data_mover_skip_configs()

    root = Path(repo_root or Path(__file__).resolve().parents[2])
    pod_src = f"{pod_nemotron_home}/src"
    common = {"repo_root": str(root), "script_path": script_path}

    if executor_type == "lepton":
        # Chunk the tarball across env vars; pod reassembles via python3.
        chunk_bytes = 96 * 1024
        with tempfile.TemporaryDirectory() as td:
            raw = Path(SourcePackager(**common).package(None, td, "nemotron-src")).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        chunks = [b64[i : i + chunk_bytes] for i in range(0, len(b64), chunk_bytes)]
        env_vars["_NEMOTRON_SRC_CHUNKS"] = str(len(chunks))
        for i, c in enumerate(chunks):
            env_vars[f"_NEMOTRON_SRC_CHUNK_{i}"] = c
        typer.echo(f"[stage] lepton: {len(raw) // 1024} KiB raw → {len(chunks)} env-var chunks")
        import nemo_run as run
        # Multi-pod NFS race: when N pods share the same dest on NFS, each
        # would otherwise ``rm -rf && tar -xz`` concurrently and clobber each
        # other. Gate on NODE_RANK: rank-0 extracts and drops a marker; others
        # wait for it. The marker uses the chunk count so stale markers from
        # prior runs with different source can't mislead waiters.
        ready_marker = f"{pod_src}/.nemotron-src-ready-${{_NEMOTRON_SRC_CHUNKS}}"
        extract_cmd = (
            'python3 -c \'import os,sys,base64;'
            'n=int(os.environ["_NEMOTRON_SRC_CHUNKS"]);'
            'sys.stdout.buffer.write(base64.b64decode("".join('
            'os.environ[f"_NEMOTRON_SRC_CHUNK_{i}"] for i in range(n))))\''
            f" | tar -xz -C {pod_src} --strip-components=1"
        )
        return Plan(
            packager=run.Packager(),
            pod_src_root=pod_src,
            pre_script_cmds=[
                'if [ "${NODE_RANK:-0}" = "0" ]; then'
                f" rm -rf {pod_src} && mkdir -p {pod_src} && {extract_cmd}"
                f" && touch {ready_marker};"
                f' else while [ ! -f {ready_marker} ]; do sleep 2; done; fi'
            ],
        )

    if executor_type == "dgxcloud":
        # Packager writes one tarball to job_dir; move_data chunks only it.
        src_file = "nemotron-src.tgz"
        typer.echo(f"[stage] dgxcloud: packaged as /nemo_run/{src_file}")
        return Plan(
            packager=SourcePackager(fixed_output_name=src_file, **common),
            pod_src_root=pod_src,
            pre_script_cmds=[
                f"rm -rf {pod_src} && mkdir -p {pod_src}",
                f"tar -xzf /nemo_run/{src_file} -C {pod_src} --strip-components=1",
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
