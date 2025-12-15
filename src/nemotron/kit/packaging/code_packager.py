# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

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

        # The launcher runs in the extracted package root.
        # Add `src/` so `import nemotron` works without installation.
        # Change working directory to ROOT so ${oc.env:PWD} resolves correctly.
        return (
            "from __future__ import annotations\n\n"
            "import os\n"
            "import runpy\n"
            "import sys\n\n"
            "ROOT = os.path.dirname(__file__)\n"
            "os.chdir(ROOT)\n"
            "sys.path.insert(0, ROOT)\n"
            "sys.path.insert(0, os.path.join(ROOT, 'src'))\n\n"
            f"runpy.run_path(os.path.join(ROOT, {rel_script.as_posix()!r}), run_name='__main__')\n"
        )

    def _iter_repo_paths(self, repo_root: Path):
        """Yield repo-relative paths to include.

        Prefer git-aware file listing when available to avoid packaging ignored
        run outputs (e.g. output/, artifacts/, wandb/).
        """
        git_dir = repo_root / ".git"
        if git_dir.exists():
            try:
                tracked = self._git_ls_files(repo_root)
                others = self._git_ls_files(repo_root, others=True)
                yield from sorted({*tracked, *others})
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
