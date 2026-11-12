"""Start each recipe stage in its dedicated dependency environment."""

from __future__ import annotations

import os
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path


def default_runtime_dir() -> Path:
    """Use a project-local runtime unless the caller provides an explicit path."""
    return Path(os.environ.get("NEMOTRON_PARSE_RUNTIME", ".nemotron/parse-runtime")).expanduser().resolve()


def runtime_python(runtime_dir: str | Path | None = None) -> Path:
    root = Path(runtime_dir).expanduser().resolve() if runtime_dir is not None else default_runtime_dir()
    python = root / "venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"Parse runtime is not installed at {root}. Run "
            f"python -m nemotron.recipes.parse.runtime.setup --runtime-dir {root} first."
        )
    return python


def checkout_source_root(source_root: str | Path | None = None) -> Path:
    """Reject wheel installs instead of exposing their host site-packages to workers."""
    src = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[4]
    project = src.parent / "pyproject.toml"
    required = (
        src / "nemotron" / "__init__.py",
        src / "nemo_runspec" / "__init__.py",
        src / "nemotron" / "recipes" / "parse" / "runtime" / "launch.py",
    )
    if src.name == "src" and project.is_file() and all(path.is_file() for path in required):
        with project.open("rb") as stream:
            if tomllib.load(stream).get("project", {}).get("name") == "nemotron":
                return src
    raise RuntimeError(
        "Parse v1 requires a Nemotron source checkout (or editable install) with "
        "src/nemotron, src/nemo_runspec and the root pyproject.toml. "
        f"Refusing to expose a wheel/host package directory to the training runtime: {src}"
    )


def runtime_environment(
    runtime_dir: str | Path | None = None,
    *,
    source_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Expose this checkout and the native sampler to the stage and Ray workers.

    Host PYTHONPATH and user-site packages must not reintroduce the launcher's
    incompatible Transformers/Torch installations. Other task-specific settings,
    including existing model/cache locations and TMPDIR, are retained.
    """
    root = Path(runtime_dir).expanduser().resolve() if runtime_dir is not None else default_runtime_dir()
    src = checkout_source_root(source_root)
    env = dict(os.environ if environ is None else environ)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(src)
    env["PYTHONNOUSERSITE"] = "1"
    env["VIRTUAL_ENV"] = str(root / "venv")
    env["UV_PROJECT_ENVIRONMENT"] = str(root / "venv")
    env["NEMOTRON_PARSE_RUNTIME"] = str(root)
    env["PATH"] = os.pathsep.join([str(root / "venv" / "bin"), env.get("PATH", "")])
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def run_module(
    module: str,
    args: Sequence[str | Path] = (),
    *,
    runtime_dir: str | Path | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a stage as a separate process and propagate its exit status."""
    return subprocess.run(
        [str(runtime_python(runtime_dir)), "-m", module, *map(str, args)],
        env=runtime_environment(runtime_dir),
        cwd=cwd,
        check=True,
    )
