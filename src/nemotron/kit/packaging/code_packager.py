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

# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""nemo-run packager that bundles the nemotron repo for remote execution."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError

from nemo_run.core.packaging import Packager


@dataclass(kw_only=True)
class CodePackager(Packager):
    """Package the repo codebase for remote execution.

    This is intended for recipes that need local imports (e.g. data prep).
    It bundles the full repo tree while excluding a small set of directories,
    and writes a thin `main.py` launcher that ensures `src/` is on `sys.path`.
    """

    script_path: str
    train_path: str
    exclude_dirs: tuple[str, ...] = ("usage-cookbook", "use-case-examples")
    dependencies: list[str] | None = None
    """Dependencies to install in the container.

    - Local paths (e.g., "libraries/Automodel") are installed with pip install -e
    - Package names (e.g., "nemo-automodel") are installed from PyPI
    """

    def package(self, path: Path, job_dir: str, name: str) -> str:
        repo_root = Path(path)
        output_file = Path(job_dir) / f"{name}.tar.gz"
        if output_file.exists():
            return str(output_file)

        job_dir_path = Path(job_dir).resolve()
        launcher = self._build_launcher(repo_root)

        with tarfile.open(output_file, "w:gz") as tf:
            tf.addfile(self._tarinfo_for_text("main.py", launcher), self._bytes_io(launcher))
            tf.add(Path(self.train_path), arcname="config.yaml")

            for rel in self._iter_repo_paths(repo_root):
                if self._is_excluded(rel):
                    continue
                file_path = repo_root / rel

                # Avoid accidentally packaging local run outputs and the tarball itself.
                try:
                    file_resolved = file_path.resolve()
                    if job_dir_path == file_resolved or job_dir_path in file_resolved.parents:
                        continue
                except Exception:
                    pass

                if not file_path.exists():
                    continue
                tf.add(file_path, arcname=str(rel), recursive=False)

        return str(output_file)

    def _is_excluded(self, rel: Path) -> bool:
        parts = rel.parts
        for d in self.exclude_dirs:
            if parts and parts[0] == d:
                return True
        return False

    def _build_launcher(self, repo_root: Path) -> str:
        script_file = Path(self.script_path)
        if not script_file.is_absolute():
            script_file = repo_root / self.script_path

        rel_script = script_file.relative_to(repo_root)

        # Build dependency installation code
        deps_code = self._build_deps_code()

        # The launcher runs in the extracted package root.
        # Change working directory to ROOT so ${oc.env:PWD} resolves correctly.
        # Always run script in subprocess to ensure clean Python environment
        # and visibility of any newly installed packages.
        return (
            "from __future__ import annotations\n\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n\n"
            "ROOT = os.path.dirname(__file__)\n"
            "os.chdir(ROOT)\n"
            "print('[launcher] Starting...', file=sys.stderr)\n\n"
            f"{deps_code}"
            "# Run script in subprocess for clean environment\n"
            "script_path = os.path.join(ROOT, " + repr(rel_script.as_posix()) + ")\n"
            "config_path = os.path.join(ROOT, 'config.yaml')\n"
            "env = os.environ.copy()\n"
            "env['PYTHONPATH'] = os.pathsep.join([ROOT, os.path.join(ROOT, 'src')]) + os.pathsep + env.get('PYTHONPATH', '')\n"
            "result = subprocess.run(\n"
            "    [sys.executable, script_path, '--config', config_path] + sys.argv[1:],\n"
            "    env=env,\n"
            ")\n"
            "sys.exit(result.returncode)\n"
        )

    def _build_deps_code(self) -> str:
        """Build dependency installation code for the launcher.

        Returns Python code that installs dependencies if not already available.
        """
        if not self.dependencies:
            return ""

        deps_str = ", ".join(self.dependencies)
        lines = [
            "# Install dependencies if not already available\n",
            "def _ensure_deps():\n",
            f"    print('[launcher] Installing dependencies: {deps_str}', file=sys.stderr)\n",
        ]

        for dep in self.dependencies:
            # Determine if this is a local path or a package name
            # Local paths start with ./ or contain / and exist in the repo
            if "/" in dep and not dep.startswith("http"):
                # Local path - install with pip install -e
                # Convert path to use forward slashes for consistency
                dep_path = dep.replace("\\", "/")
                # Derive a simple name for logging from the path
                dep_name = dep_path.rstrip("/").split("/")[-1]
                lines.append(
                    f"    dep_path = os.path.join(ROOT, {dep_path!r})\n"
                    f"    if os.path.exists(dep_path):\n"
                    f"        print(f'[launcher] Installing {dep_name} from {{dep_path}}...', file=sys.stderr)\n"
                    f"        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', dep_path], check=True)\n"
                    f"    else:\n"
                    f"        print(f'[launcher] WARNING: Path not found: {{dep_path}}', file=sys.stderr)\n"
                )
            else:
                # Package name - install from PyPI
                lines.append(
                    f"    print('[launcher] Installing {dep} from PyPI...', file=sys.stderr)\n"
                    f"    subprocess.run([sys.executable, '-m', 'pip', 'install', {dep!r}], check=True)\n"
                )

        lines.append("\n_ensure_deps()\n\n")
        return "".join(lines)

    def _iter_repo_paths(self, repo_root: Path):
        """Yield repo-relative paths to include.

        Prefer git-aware file listing when available to avoid packaging ignored
        run outputs (e.g. output/, artifacts/, wandb/).
        Also includes submodule contents.
        """
        git_dir = repo_root / ".git"
        if git_dir.exists():
            try:
                tracked = self._git_ls_files(repo_root)
                others = self._git_ls_files(repo_root, others=True)
                # Also get submodule contents
                submodule_files = self._git_ls_files_with_submodules(repo_root)
                yield from sorted({*tracked, *others, *submodule_files})
                return
            except (CalledProcessError, FileNotFoundError, OSError):
                pass

        for p in repo_root.rglob("*"):
            yield p.relative_to(repo_root)

    @staticmethod
    def _git_ls_files(repo_root: Path, *, others: bool = False) -> list[Path]:
        import subprocess

        cmd = ["git", "-C", str(repo_root), "ls-files", "-z"]
        if others:
            cmd.extend(["--others", "--exclude-standard"])

        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
        result: list[Path] = []
        for chunk in out.split(b"\x00"):
            if not chunk:
                continue
            result.append(Path(chunk.decode("utf-8")))
        return result

    @staticmethod
    def _git_ls_files_with_submodules(repo_root: Path) -> list[Path]:
        """Get files from git submodules.

        Uses git ls-files --recurse-submodules to get all tracked files
        including those in submodules.
        """
        import subprocess

        try:
            cmd = ["git", "-C", str(repo_root), "ls-files", "-z", "--recurse-submodules"]
            out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            result: list[Path] = []
            for chunk in out.split(b"\x00"):
                if not chunk:
                    continue
                result.append(Path(chunk.decode("utf-8")))
            return result
        except (CalledProcessError, FileNotFoundError, OSError):
            # Fallback: manually iterate submodules
            return CodePackager._iter_submodule_files(repo_root)

    @staticmethod
    def _iter_submodule_files(repo_root: Path) -> list[Path]:
        """Fallback: iterate submodule directories directly."""
        import subprocess

        result: list[Path] = []
        try:
            # Get list of submodule paths
            cmd = ["git", "-C", str(repo_root), "submodule", "foreach", "--quiet", "echo $sm_path"]
            out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            submodule_paths = [p.strip() for p in out.decode("utf-8").split("\n") if p.strip()]

            for sm_path in submodule_paths:
                sm_dir = repo_root / sm_path
                if sm_dir.exists():
                    # Get tracked files in this submodule
                    sm_files = CodePackager._git_ls_files(sm_dir)
                    for f in sm_files:
                        result.append(Path(sm_path) / f)
        except (CalledProcessError, FileNotFoundError, OSError):
            pass

        return result

    @staticmethod
    def _tarinfo_for_text(name: str, text: str) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name=name)
        data = text.encode("utf-8")
        info.size = len(data)
        info.mode = 0o644
        return info

    @staticmethod
    def _bytes_io(text: str):
        import io

        return io.BytesIO(text.encode("utf-8"))


__all__ = ["CodePackager"]
