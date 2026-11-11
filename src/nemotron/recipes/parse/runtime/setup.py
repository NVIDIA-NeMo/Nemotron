"""Install the isolated Parse runtime with the validated upstream framework patches.

Run from the Nemotron source checkout on Linux:
    PYTHONPATH=src python3 -m nemotron.recipes.parse.runtime.setup --runtime-dir .nemotron/parse-runtime

This command does not install the root Nemotron project or change its environment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .launch import default_runtime_dir, runtime_environment

RUNTIME = Path(__file__).resolve().parent
UPSTREAM_REVISION = "2c2be873c05d904de38c80e55ef5ac754aea180d"
SOURCES = (
    ("nemo-rl", "https://github.com/NVIDIA-NeMo/RL.git", "b0b7dd5597c4bc5c9a728d6d31e2058c698b7423"),
    ("automodel", "https://github.com/NVIDIA-NeMo/Automodel.git", "b916107a5ce8e77a3f18682f8b0eaa46a4b54389"),
)


def run(*command: str | Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def output(*command: str | Path, **kwargs) -> str:
    return subprocess.check_output(list(map(str, command)), text=True, **kwargs).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def installation_lock(root: Path) -> Iterator[None]:
    """A second setup process must not mutate this environment concurrently."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another Parse runtime installation holds {root / '.setup.lock'}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _matches_tree(directory: Path, tree: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(directory), "diff", "--quiet", tree, "--"], check=False
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot inspect framework source at {directory}")
    return result.returncode == 0


def apply_patch(name: str, directory: Path) -> str:
    """Accept precisely upstream HEAD or HEAD plus our patch; never discard edits."""
    patch = RUNTIME / "patches" / f"{name}.patch"
    # A temporary index constructs the exact expected tree without changing the
    # user's index or working files. Reverse-apply alone misses unrelated edits.
    with tempfile.TemporaryDirectory(prefix="parse-index-") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        run("git", "-C", directory, "read-tree", "HEAD", env=env)
        run("git", "-C", directory, "apply", "--cached", patch, env=env)
        expected = output("git", "-C", directory, "write-tree", env=env)
    if _matches_tree(directory, expected):
        return expected
    if not _matches_tree(directory, "HEAD"):
        raise RuntimeError(f"Unexpected framework edits in {directory}; use a new runtime directory")
    run("git", "-C", directory, "apply", patch)
    if not _matches_tree(directory, expected):
        raise RuntimeError(f"Framework patch verification failed in {directory}")
    return expected


def ensure_source(root: Path, name: str, repository: str, revision: str) -> dict[str, str]:
    directory = root / "sources" / name
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", repository, directory)
        run("git", "-C", directory, "checkout", "--detach", revision)
    found = output("git", "-C", directory, "rev-parse", "HEAD")
    if found != revision:
        raise RuntimeError(f"{directory} must use pinned revision {revision}, found {found}")
    expected = apply_patch(name, directory)
    return {"repository": repository, "revision": revision, "patched_tree": expected}


def install_request(torch_backend: str, python_version: str) -> dict:
    files = [
        "setup.py", "launch.py", "requirements.txt", "launcher-requirements.txt",
        *[f"patches/{s[0]}.patch" for s in SOURCES],
    ]
    return {
        "schema": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "torch_backend": torch_backend,
        "python_version": python_version,
        "input_sha256": {name: sha256(RUNTIME / name) for name in files},
        "framework_revisions": {name: revision for name, _, revision in SOURCES},
    }


def install(root: Path, *, torch_backend: str = "auto", python_version: str = "3.13.16") -> Path:
    """Install once, or verify an identical completed installation without upgrading it."""
    if sys.platform != "linux":
        raise RuntimeError("The Parse training runtime requires Linux and a compatible NVIDIA driver")
    root = root.expanduser().resolve()
    # Validate the source boundary before cloning frameworks or writing a runtime.
    env = runtime_environment(root)
    requested = install_request(torch_backend, python_version)
    with installation_lock(root):
        request_path = root / "requested.json"
        if request_path.exists() and json.loads(request_path.read_text()) != requested:
            raise RuntimeError(f"Runtime settings changed at {root}; choose a new runtime directory")
        request_path.write_text(json.dumps(requested, indent=2) + "\n")
        sources = {name: ensure_source(root, name, repo, rev) for name, repo, rev in SOURCES}
        python = root / "venv" / "bin" / "python"
        env["UV_NO_CONFIG"] = "1"
        env.setdefault("UV_CACHE_DIR", str(root / "cache" / "uv"))
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / "python"))
        uv_version = output("uv", "--version", env=env)
        if uv_version.split()[1] != "0.13.1":
            raise RuntimeError(f"Use uv==0.13.1 for this profile; found {uv_version}")
        receipt_path = root / "runtime.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["request"] != requested or receipt["sources"] != sources:
                raise RuntimeError("Installed runtime receipt does not match the requested inputs")
            if sha256(root / "environment.txt") != receipt["environment_sha256"]:
                raise RuntimeError("Recorded package freeze differs from its installation receipt")
            frozen = output("uv", "--no-config", "pip", "freeze", "--python", python, env=env)
            if frozen != (root / "environment.txt").read_text().strip():
                raise RuntimeError("Installed packages differ from the completed runtime receipt")
            run("uv", "--no-config", "pip", "check", "--python", python, env=env)
            print(f"Verified existing Parse runtime: {python}")
            return python
        if not python.exists():
            run("uv", "--no-config", "venv", "--python", python_version, root / "venv", env=env)
        run(
            "uv", "--no-config", "pip", "install", "--python", python,
            "--link-mode", "copy", f"--torch-backend={torch_backend}",
            "-r", RUNTIME / "requirements.txt", "-r", RUNTIME / "launcher-requirements.txt",
            "-e", root / "sources" / "nemo-rl", "-e", root / "sources" / "automodel",
            env=env,
        )
        run("uv", "--no-config", "pip", "check", "--python", python, env=env)
        frozen = output("uv", "--no-config", "pip", "freeze", "--python", python, env=env)
        (root / "environment.txt").write_text(frozen + "\n")
        metadata = json.loads(output(
            python, "-c",
            "import importlib.metadata as m,json,platform; "
            "print(json.dumps({'python':platform.python_version(),'packages':"
            "{n:m.version(n) for n in ['torch','torchvision','torchao','torchaudio',"
            "'transformers','vllm','ray','Pillow']}}))",
            env=env,
        ))
        receipt = {
            "request": requested,
            "sources": sources,
            "uv": uv_version,
            "resolved": metadata,
            "environment_sha256": sha256(root / "environment.txt"),
            "worker_environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
            "completed_at": datetime.now(UTC).isoformat(),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Parse runtime ready: {python}")
        return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=default_runtime_dir())
    parser.add_argument(
        "--torch-backend", default="auto", help="Use uv's GPU-aware package build selection (default: auto)"
    )
    parser.add_argument("--python-version", default="3.13.16")
    args = parser.parse_args()
    install(args.runtime_dir, torch_backend=args.torch_backend, python_version=args.python_version)


if __name__ == "__main__":
    main()
