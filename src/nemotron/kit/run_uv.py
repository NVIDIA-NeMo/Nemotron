#!/usr/bin/env python3
"""Shared UV dependency wrapper for container/Slurm execution.

Creates a venv with --system-site-packages to access container packages
(torch, transformers, flash-attn, etc.) while installing stage-specific
dependencies using UV's exclude-dependencies.

Each stage's pyproject.toml declares its configuration in [tool.nemotron]:

    [tool.nemotron]
    entry-point = "train.py"
    container-exclude-dependencies = [
        "torch", "torchvision", "flash-attn", ...
    ]
"""
from __future__ import annotations

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


def main(stage_dir: Path) -> None:
    """Run the UV dependency wrapper for a given stage directory."""
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

    if not entry_point:
        print("[run_uv.py] ERROR: [tool.nemotron] entry-point not set in pyproject.toml")
        sys.exit(1)

    target_script = stage_dir / entry_point
    if not target_script.exists():
        print(f"[run_uv.py] ERROR: entry-point {entry_point!r} not found at {target_script}")
        sys.exit(1)

    print(f"[run_uv.py] Target script: {target_script}")

    # 3. Configure environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "/nemo_run/code/src:" + env.get("PYTHONPATH", "")

    # 4. Find UV — bootstrap via pip if not already installed
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

    # 5. Create venv with system-site-packages (always recreate for correctness)
    venv_path = Path("/opt/venv")
    if venv_path.exists():
        print(f"[run_uv.py] Removing existing venv at {venv_path}")
        shutil.rmtree(venv_path)

    print(f"[run_uv.py] Creating venv with system-site-packages at {venv_path}")
    result = subprocess.run([
        uv_cmd, "venv",
        "--system-site-packages",
        "--seed",
        str(venv_path),
    ])
    if result.returncode != 0:
        print("[run_uv.py] ERROR: Failed to create venv")
        sys.exit(1)

    # 6. Configure env for venv
    venv_python = venv_path / "bin" / "python3"
    env["VIRTUAL_ENV"] = str(venv_path)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    env["PATH"] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"

    # 7. Check if packages are already available (fast path)
    print("[run_uv.py] Checking if packages are already available...")
    check_result = subprocess.run(
        [str(venv_python), "-c", "import nemo_automodel, omegaconf"],
        capture_output=True,
    )

    if check_result.returncode == 0:
        print("[run_uv.py] Required packages already available, skipping installation")
    else:
        # 8. Write temp pyproject.toml with container exclude-dependencies
        print("[run_uv.py] Syncing packages using pyproject.toml (injecting exclude-dependencies)...")
        temp_dir = _write_temp_pyproject(pyproject_data, stage_dir, exclude_deps)
        print(f"[run_uv.py] Created temporary pyproject.toml at {temp_dir / 'pyproject.toml'}")

        # 9. Run uv sync
        sync_cmd = [
            uv_cmd, "sync",
            "--active",
            "--project", str(temp_dir),
        ]
        print(f"[run_uv.py] Running: {' '.join(sync_cmd)}")
        result = subprocess.run(sync_cmd, env=env, cwd=str(temp_dir))
        if result.returncode != 0:
            print("[run_uv.py] ERROR: Package sync failed")
            sys.exit(1)

    # 10. Execute target script
    print("[run_uv.py] Dependencies installed successfully")
    cmd = [str(venv_python), str(target_script)] + sys.argv[1:]
    print(f"[run_uv.py] Executing: {' '.join(cmd)}")
    print(f"[run_uv.py] Args: {sys.argv[1:]}")

    result = subprocess.run(cmd, env=env, capture_output=False)
    print(f"[run_uv.py] Exit code: {result.returncode}")
    sys.exit(result.returncode)
