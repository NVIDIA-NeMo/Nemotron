"""Shared helper for local UV execution with torch detection.

When torch is already importable (e.g., inside an NVIDIA container), we create
a venv with --system-site-packages and exclude torch from UV resolution. This
avoids the CUDA version mismatch where UV's torch-backend=auto detects the
kernel driver's CUDA version (via nvidia-smi) but the container's libcuda.so
is an older version.

When torch is NOT importable (bare machine), we fall back to the standard
``uv run --with torch`` approach with ``UV_TORCH_BACKEND=auto``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def _torch_is_importable() -> bool:
    """Check if torch is importable in the current Python."""
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
    )
    return result.returncode == 0


def execute_uv_local(
    *,
    script_path: str,
    stage_dir: Path,
    repo_root: Path,
    train_path: Path,
    passthrough: list[str],
    extra_with: list[str] | None = None,
    pre_script_args: list[str] | None = None,
) -> int:
    """Execute a stage script locally, handling torch installation correctly.

    When torch is already available in the current Python (e.g., inside a
    container), creates a venv with --system-site-packages and excludes torch
    from resolution. Otherwise, uses ``uv run --with torch``.

    Args:
        script_path: Relative path to the stage script.
        stage_dir: Absolute path to the stage directory (contains pyproject.toml).
        repo_root: Absolute path to the repo root (installed via --with).
        train_path: Path to the resolved training config YAML.
        passthrough: Extra CLI arguments to forward to the script.
        extra_with: Additional ``--with`` packages for uv run (e.g., ["tensorrt"]).
        pre_script_args: Arguments inserted before the script path
            (e.g., ["-m", "torch.distributed.run", "--nproc_per_node=gpu"]).

    Returns:
        Process exit code.
    """
    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        print("Error: 'uv' command not found. Please install uv.", file=sys.stderr)
        return 1

    script_abs = (stage_dir / Path(script_path).name) if not Path(script_path).is_absolute() else Path(script_path)

    if _torch_is_importable():
        return _execute_with_system_torch(
            uv_cmd=uv_cmd,
            stage_dir=stage_dir,
            repo_root=repo_root,
            script_abs=script_abs,
            train_path=train_path,
            passthrough=passthrough,
            pre_script_args=pre_script_args or [],
        )
    else:
        return _execute_with_uv_torch(
            uv_cmd=uv_cmd,
            stage_dir=stage_dir,
            repo_root=repo_root,
            script_abs=script_abs,
            train_path=train_path,
            passthrough=passthrough,
            extra_with=extra_with or [],
            pre_script_args=pre_script_args or [],
        )


def _execute_with_system_torch(
    *,
    uv_cmd: str,
    stage_dir: Path,
    repo_root: Path,
    script_abs: Path,
    train_path: Path,
    passthrough: list[str],
    pre_script_args: list[str],
) -> int:
    """Execute using system torch via --system-site-packages venv."""
    print("Detected system torch — using system-site-packages to avoid CUDA mismatch")

    # Read stage pyproject.toml for exclude-dependencies
    pyproject_path = stage_dir / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    nemotron_cfg = pyproject_data.get("tool", {}).get("nemotron", {})
    exclude_deps = nemotron_cfg.get("container-exclude-dependencies", [
        "torch", "torchvision", "flash-attn", "triton",
        "pyarrow", "scipy", "opencv-python-headless",
    ])

    # Import the temp pyproject writer from run_uv
    from nemotron.kit.run_uv import _write_temp_pyproject

    temp_dir = _write_temp_pyproject(pyproject_data, stage_dir, exclude_deps)

    # Create venv with system-site-packages
    venv_path = Path(tempfile.mkdtemp()) / "venv"
    result = subprocess.run([
        uv_cmd, "venv",
        "--system-site-packages",
        "--seed",
        str(venv_path),
    ])
    if result.returncode != 0:
        print("Error: Failed to create venv", file=sys.stderr)
        return 1

    venv_python = str(venv_path / "bin" / "python3")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["VIRTUAL_ENV"] = str(venv_path)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    env["PATH"] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"

    # Sync stage dependencies (excluding torch)
    sync_cmd = [
        uv_cmd, "sync",
        "--active",
        "--project", str(temp_dir),
    ]
    print(f"Syncing dependencies (torch excluded): {' '.join(sync_cmd)}")
    result = subprocess.run(sync_cmd, env=env, cwd=str(temp_dir))
    if result.returncode != 0:
        print("Error: Package sync failed", file=sys.stderr)
        return 1

    # Install the repo package
    result = subprocess.run(
        [uv_cmd, "pip", "install", "--no-deps", str(repo_root)],
        env=env,
    )
    if result.returncode != 0:
        print("Error: Failed to install repo package", file=sys.stderr)
        return 1

    # Run the script
    cmd = [venv_python, *pre_script_args, str(script_abs), "--config", str(train_path), *passthrough]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _execute_with_uv_torch(
    *,
    uv_cmd: str,
    stage_dir: Path,
    repo_root: Path,
    script_abs: Path,
    train_path: Path,
    passthrough: list[str],
    extra_with: list[str],
    pre_script_args: list[str],
) -> int:
    """Execute using uv run with torch installed via UV_TORCH_BACKEND."""
    cmd = [
        uv_cmd, "run",
        "--with", str(repo_root),
        "--with", "torch",
    ]
    for pkg in extra_with:
        cmd += ["--with", pkg]

    if pre_script_args:
        cmd += ["--project", str(stage_dir), *pre_script_args]
    else:
        cmd += ["--project", str(stage_dir), "python"]

    cmd += [str(script_abs), "--config", str(train_path), *passthrough]

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.setdefault("UV_TORCH_BACKEND", "auto")

    print(f"Executing with uv isolated environment: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode
