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

import io
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

_BASE_EXCLUDE = [
    "torch",
    "torchvision",
    "flash-attn",
    "triton",
    "pyarrow",
    "scipy",
    "opencv-python-headless",
]


def _write_temp_pyproject(
    pyproject_data: dict, stage_dir: Path, exclude_deps: list[str]
) -> Path:
    """Write a temporary pyproject.toml with container exclude-dependencies."""
    temp_dir = Path(tempfile.mkdtemp())
    buf = io.StringIO()

    # [project]
    proj = pyproject_data["project"]
    buf.write("[project]\n")
    buf.write(f'name = "{proj["name"]}"\n')
    buf.write(f'version = "{proj["version"]}"\n')
    buf.write(f'requires-python = "{proj["requires-python"]}"\n')
    buf.write("dependencies = [\n")
    for dep in proj.get("dependencies", []):
        buf.write(f'  "{dep}",\n')
    buf.write("]\n\n")

    uv = pyproject_data.get("tool", {}).get("uv", {})

    # [tool.uv.sources] — convert relative paths to absolute
    if "sources" in uv:
        buf.write("[tool.uv.sources]\n")
        for key, value in uv["sources"].items():
            if "path" in value:
                source_path = Path(value["path"])
                if not source_path.is_absolute():
                    source_path = (stage_dir / source_path).resolve()
                buf.write(f'{key} = {{ path = "{source_path}" }}\n')
        buf.write("\n")

    # [tool.uv.extra-build-dependencies]
    if "extra-build-dependencies" in uv:
        buf.write("[tool.uv.extra-build-dependencies]\n")
        for key, deps in uv["extra-build-dependencies"].items():
            deps_str = "[" + ", ".join(f'"{d}"' for d in deps) + "]"
            buf.write(f"{key} = {deps_str}\n")
        buf.write("\n")

    # [tool.uv]
    buf.write("[tool.uv]\n")
    if "override-dependencies" in uv:
        buf.write("override-dependencies = [\n")
        for dep in uv["override-dependencies"]:
            buf.write(f'  "{dep}",\n')
        buf.write("]\n")
    buf.write("exclude-dependencies = [\n")
    for dep in exclude_deps:
        buf.write(f'  "{dep}",\n')
    buf.write("]\n")

    (temp_dir / "pyproject.toml").write_text(buf.getvalue())
    return temp_dir


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

    # 4. Find UV
    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        print("[run_uv.py] ERROR: UV not found")
        sys.exit(1)

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
