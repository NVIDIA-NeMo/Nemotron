#!/usr/bin/env python3
"""Shared uv runner for container and Slurm stage execution.

Existing wrappers call main(stage_dir=...). Alternatively invoke this module
with --stage-dir. The default inherited mode uses image packages and configured
exclusions; isolated mode installs the stage's complete locked environment.
Stage metadata lives in [tool.nemotron] in pyproject.toml.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from nemo_runspec._pyproject import _write_temp_pyproject

_BASE_EXCLUDE = [
    "torch",
    "torchvision",
    "flash-attn",
    "triton",
    "pyarrow",
    "scipy",
    "opencv-python-headless",
]


def _inherit_container_packages(child_site: Path, parent_sites: list[Path]) -> None:
    # addsitedir processes parent .pth files too. Merely adding its directory
    # loses editable framework installs supplied by the container image.
    (child_site / "nemotron-container.pth").write_text(
        "".join(f"import site; site.addsitedir({str(path.resolve())!r})\n" for path in parent_sites)
    )


def main(stage_dir: Path | None = None) -> None:
    """Run the UV dependency wrapper for a given stage directory."""
    # Existing stage wrappers can still pass stage_dir; new callers use one
    # shared entry point, without copying a wrapper into every stage.
    forwarded = sys.argv[1:]
    if stage_dir is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--stage-dir", required=True, type=Path)
        options, forwarded = parser.parse_known_args()
        stage_dir = options.stage_dir
    stage_dir = stage_dir.resolve()
    print("[run_uv.py] Starting wrapper script")
    print(f"[run_uv.py] Working directory: {os.getcwd()}")

    # 1. Read stage pyproject.toml
    pyproject_path = stage_dir / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"[run_uv.py] ERROR: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    # 2. Extract nemotron config
    nemotron_cfg = pyproject_data.get("tool", {}).get("nemotron", {})
    entry_point = nemotron_cfg.get("entry-point")
    exclude_deps = nemotron_cfg.get("container-exclude-dependencies", _BASE_EXCLUDE)
    extras = nemotron_cfg.get("container-extras", [])
    mode = os.environ.get("NEMOTRON_UV_MODE", nemotron_cfg.get("container-mode", "inherit"))
    if mode not in ("isolated", "inherit"):
        raise ValueError("NEMOTRON_UV_MODE must be isolated or inherit")
    container_project = stage_dir if mode == "isolated" else None
    if container_project:
        container_project = (stage_dir / container_project).resolve()
        if not (container_project / "uv.lock").is_file():
            raise FileNotFoundError(f"Missing container dependency lock: {container_project}")

    if not entry_point:
        print("[run_uv.py] ERROR: [tool.nemotron] entry-point not set in pyproject.toml")
        sys.exit(1)

    target_script = stage_dir / entry_point
    if not target_script.exists():
        print(f"[run_uv.py] ERROR: entry-point {entry_point!r} not found at {target_script}")
        sys.exit(1)

    print(f"[run_uv.py] Target script: {target_script}")

    # 3. Configure environment and remember the package roots supplied by the
    # container interpreter. Some images launch from their own /opt/venv, which
    # must remain intact while the stage-specific child environment is built.
    env = os.environ.copy()
    env["PYTHONPATH"] = "/nemo_run/code/src:" + env.get("PYTHONPATH", "")
    container_site_paths = [
        Path(path)
        for path in sys.path
        if path and Path(path).is_dir() and Path(path).name in {"site-packages", "dist-packages"}
    ]

    constraints = nemotron_cfg.get("container-requires", {})
    if constraints and mode == "inherit":
        from importlib.metadata import version

        from packaging.specifiers import SpecifierSet

        for package, constraint in constraints.items():
            installed = version(package)
            if installed not in SpecifierSet(constraint):
                raise RuntimeError(
                    f"Container {package}=={installed} does not satisfy {constraint}; use the stage's validated image"
                )
    # 4. Configure the shared stage-local environment. Torchrun starts this
    # wrapper once per GPU, so one local worker prepares /opt/nemotron-venv
    # while its peers wait and then reuse the completed environment.
    # Opt-in stage identity avoids reusing another stage's incompatible env.
    # Keep the historical path for existing wrapper callers.
    identity = nemotron_cfg.get("environment-name") or (stage_dir.name if mode == "isolated" else None)
    if identity:
        inputs = pyproject_path.read_bytes() + sys.version.encode() + mode.encode()
        if container_project:
            inputs += (container_project / "pyproject.toml").read_bytes()
            inputs += (container_project / "uv.lock").read_bytes()
        fingerprint = hashlib.sha256(inputs).hexdigest()[:12]
        venv_path = Path(os.environ.get("NEMOTRON_VENV_ROOT", "/opt/nemotron-venvs")) / f"{identity}-{fingerprint}"
    else:
        venv_path = Path("/opt/nemotron-venv")
    venv_python = venv_path / "bin" / "python3"
    env["VIRTUAL_ENV"] = str(venv_path)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    env["PATH"] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"
    if mode == "isolated":
        # The lock owns this environment; image-specific constraints must not replace it.
        for name in ("UV_OVERRIDE", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT"):
            env.pop(name, None)
        # Some images reserve /opt/uv/python for root-owned image interpreters.
        env["UV_PYTHON_INSTALL_DIR"] = str(venv_path.parent / ".python")

    ready_path = venv_path / ".nemotron-ready"
    child_site = venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if identity:
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = venv_path.parent / f".{venv_path.name}.lock"
    else:
        lock_path = Path("/tmp/nemotron-run-uv.lock")
    with open(lock_path, "w") as lock_file:
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(f"[run_uv.py] Local rank {local_rank} waiting for environment setup lock")
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        # Find UV — bootstrap via pip if not already installed.
        uv_cmd = shutil.which("uv")
        if not uv_cmd:
            print("[run_uv.py] UV not found, installing via pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "uv"],
                capture_output=True,
            )
            if result.returncode != 0:
                print("[run_uv.py] ERROR: Failed to install uv via pip")
                print(result.stderr.decode() if result.stderr else "")
                sys.exit(1)
            uv_cmd = shutil.which("uv")
            if not uv_cmd:
                # pip may install to a location not yet on PATH; check common spots
                for candidate in [
                    Path(sys.prefix) / "bin" / "uv",
                    Path.home() / ".local" / "bin" / "uv",
                ]:
                    if candidate.exists():
                        uv_cmd = str(candidate)
                        break
            if not uv_cmd:
                print("[run_uv.py] ERROR: UV not found after installation")
                sys.exit(1)
            print(f"[run_uv.py] UV installed at {uv_cmd}")

        # Earlier wrappers could mark an inherited environment ready after uv
        # replaced its interpreter. Repair that incomplete cache on reuse.
        if mode == "inherit" and ready_path.exists() and not child_site.is_dir():
            ready_path.unlink()
        if ready_path.exists():
            print(f"[run_uv.py] Reusing environment prepared at {venv_path}")
        else:
            if venv_path.exists():
                print(f"[run_uv.py] Removing existing venv at {venv_path}")
                shutil.rmtree(venv_path)

            print(f"[run_uv.py] Creating {mode} environment at {venv_path}")
            print(f"[run_uv.py] Base interpreter: {sys.executable}")
            result = subprocess.run(
                [
                    uv_cmd,
                    "venv",
                    "--python",
                    sys.executable,
                    *(["--system-site-packages"] if mode == "inherit" else []),
                    "--seed",
                    str(venv_path),
                ]
            )
            if result.returncode != 0:
                print("[run_uv.py] ERROR: Failed to create venv")
                sys.exit(1)

            # Always sync the stage-local project. Container base images may
            # already include major packages such as nemo-automodel, but stages
            # can add smaller dependencies that must still be installed.
            print("[run_uv.py] Syncing packages using pyproject.toml (injecting exclude-dependencies)...")
            temp_dir = container_project or _write_temp_pyproject(pyproject_data, stage_dir, exclude_deps)
            print(f"[run_uv.py] Dependency project: {temp_dir / 'pyproject.toml'}")

            sync_cmd = [
                uv_cmd,
                "sync",
                "--active",
                "--project",
                str(temp_dir),
            ]
            if container_project:
                sync_cmd.append("--locked")
            if mode == "inherit":
                # Image packages must use the image's Python ABI, even when a
                # shared profile sets UV_PYTHON for an isolated training stage.
                sync_cmd.extend(["--python", sys.executable])
            for extra in extras:
                sync_cmd.extend(["--extra", extra])
            print(f"[run_uv.py] Running: {' '.join(sync_cmd)}")
            result = subprocess.run(sync_cmd, env=env, cwd=str(temp_dir))
            if result.returncode != 0:
                print("[run_uv.py] ERROR: Package sync failed")
                sys.exit(1)

        if mode == "inherit":
            parent_sites = [path for path in container_site_paths if path != child_site]
            if parent_sites:
                _inherit_container_packages(child_site, parent_sites)
                print(f"[run_uv.py] Inherited {len(parent_sites)} container package root(s)")

        ready_path.touch()
        print(f"[run_uv.py] Environment setup complete at {venv_path}")

    # 9. Execute target script
    print("[run_uv.py] Dependencies installed successfully")
    cmd = [str(venv_python), str(target_script)] + forwarded
    print(f"[run_uv.py] Executing: {' '.join(cmd)}")
    print(f"[run_uv.py] Args: {sys.argv[1:]}")

    result = subprocess.run(cmd, env=env, capture_output=False)
    print(f"[run_uv.py] Exit code: {result.returncode}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
