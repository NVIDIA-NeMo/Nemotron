#!/usr/bin/env python3
"""Create a pyproject-driven Curator/BYOB runtime, then exec the step.

The dependency source of truth is a compact runtime manifest and requirements
file shipped with the bootstrap source tree. Local fallback can still read
``pyproject.toml`` directly. The bootstrap only decides which runtime profile
to install and where to create the container-local virtual environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback for older containers.
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CURATOR_PATH = Path(os.environ.get("NEMOTRON_CURATOR_PATH", "/opt/Curator"))
DEFAULT_VENV_ROOT = Path(os.environ.get("NEMOTRON_CURATOR_VENV_ROOT", "/tmp/nemotron-curator-runtime"))
DEFAULT_METADATA_ROOT = Path(
    os.environ.get("NEMOTRON_CURATOR_METADATA_ROOT", str(DEFAULT_VENV_ROOT / "metadata"))
)


@dataclass(frozen=True)
class ProjectMetadata:
    root: Path
    pyproject: Path | None
    lockfile: Path | None
    data: Mapping[str, Any]
    manifest: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    venv_name: str
    extras: tuple[str, ...]
    extra_index_urls: tuple[str, ...]
    torch_backend: str | None
    omit_packages: tuple[str, ...]
    required_modules: tuple[str, ...]
    spec_only_modules: tuple[str, ...]
    pyproject_digest: str
    requirements_file: Path | None = None
    constraints_file: Path | None = None
    overrides_file: Path | None = None

    @property
    def stamp(self) -> str:
        payload = repr(
            (
                self.name,
                self.venv_name,
                self.extras,
                self.extra_index_urls,
                self.torch_backend,
                self.omit_packages,
                self.required_modules,
                self.spec_only_modules,
                self.pyproject_digest,
                self.constraints_file.name if self.constraints_file else None,
                self.overrides_file.name if self.overrides_file else None,
                sys.version_info[:2],
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def _quote(argv: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _run(argv: Sequence[str | Path], *, env: dict[str, str] | None = None) -> None:
    print(f"[curator-runtime] $ {_quote(argv)}", flush=True)
    subprocess.run([str(part) for part in argv], check=True, env=env)


def _run_capture(argv: Sequence[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    print(f"[curator-runtime] $ {_quote(argv)}", flush=True)
    result = subprocess.run(
        [str(part) for part in argv],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed with exit code {result.returncode}: {_quote(argv)}\n{message}")
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.stdout


def _runtime_env(venv_dir: Path, curator_path: Path = DEFAULT_CURATOR_PATH) -> dict[str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
    env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"

    pythonpath = [part for part in env.get("PYTHONPATH", "").split(":") if part]
    if curator_path.exists():
        pythonpath.insert(0, str(curator_path))

    seen: set[str] = set()
    env["PYTHONPATH"] = ":".join(part for part in pythonpath if not (part in seen or seen.add(part)))
    return env


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _venv_uv(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "uv"


def _ensure_venv(venv_dir: Path, *, recreate: bool) -> Path:
    python = _venv_python(venv_dir)
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not python.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
    return python


def _ensure_uv(venv_python: Path, venv_dir: Path, env: dict[str, str]) -> Path:
    uv = _venv_uv(venv_dir)
    if uv.exists():
        return uv
    _run([venv_python, "-m", "pip", "install", "--quiet", "uv"], env=env)
    if uv.exists():
        return uv
    found = shutil.which("uv", path=env.get("PATH"))
    if found:
        return Path(found)
    raise RuntimeError("uv was installed but no uv executable was found")


def _candidate_metadata_dirs(project_metadata: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if project_metadata is not None:
        candidates.append(project_metadata)
    env_metadata = os.environ.get("NEMOTRON_PROJECT_METADATA_DIR")
    if env_metadata:
        candidates.append(Path(env_metadata))

    for start in (Path.cwd(), Path(__file__).resolve()):
        candidates.extend(start.parents)
    return candidates


def _find_env_project_metadata() -> ProjectMetadata | None:
    from nemotron.steps._bootstrap.runtime_payloads import (
        RUNTIME_PAYLOAD_SHA256_ENV,
        write_runtime_payloads_from_env,
    )

    digest = os.environ.get(RUNTIME_PAYLOAD_SHA256_ENV)
    if not digest:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise RuntimeError(f"Invalid Curator runtime payload digest: {digest!r}")

    root = DEFAULT_METADATA_ROOT / digest[:16]
    if not write_runtime_payloads_from_env(root, os.environ):
        return None
    manifest = root / "runtime.json"
    if not manifest.exists():
        raise FileNotFoundError("Curator runtime payload is missing runtime.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return ProjectMetadata(root=root, pyproject=None, lockfile=None, data={}, manifest=data)


def _find_project_metadata(project_metadata: Path | None = None) -> ProjectMetadata:
    env_metadata = _find_env_project_metadata()
    if env_metadata is not None:
        return env_metadata

    for candidate in _candidate_metadata_dirs(project_metadata):
        for root in (candidate, candidate / ".nemotron_runtime", candidate / "runtime"):
            manifest = root / "runtime.json"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                return ProjectMetadata(
                    root=root,
                    pyproject=None,
                    lockfile=None,
                    data={},
                    manifest=data,
                )
            pyproject = root / "pyproject.toml"
            if pyproject.exists():
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                lockfile = root / "uv.lock"
                return ProjectMetadata(
                    root=root,
                    pyproject=pyproject,
                    lockfile=lockfile if lockfile.exists() else None,
                    data=data,
                )
    raise FileNotFoundError(
        "Could not find .nemotron_runtime/runtime.json or pyproject.toml. "
        "Ensure the source packager ships runtime metadata."
    )


def _runtime_config(data: Mapping[str, Any], runtime_name: str) -> Mapping[str, Any]:
    runtime_table = data.get("tool", {}).get("nemotron", {}).get("runtime", {})
    if not isinstance(runtime_table, Mapping):
        return {}
    config = runtime_table.get(runtime_name, {})
    if not isinstance(config, Mapping):
        return {}
    parent_name = config.get("extends")
    if parent_name and isinstance(parent_name, str):
        parent = dict(_runtime_config(data, parent_name))
        parent.update(config)
        return parent
    return config


def _as_tuple(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise TypeError(f"Expected string or list of strings, got {type(value).__name__}")


def _manifest_path(root: Path, config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Runtime profile field {key!r} must be a file name")
    return root / value


def load_runtime_spec(
    runtime_name: str,
    metadata: ProjectMetadata,
    *,
    cli_extras: Sequence[str] = (),
) -> RuntimeSpec:
    resolved_name = runtime_name
    if metadata.manifest is not None:
        profiles = metadata.manifest.get("profiles", {})
        if not isinstance(profiles, Mapping) or resolved_name not in profiles:
            raise ValueError(f"Runtime profile {resolved_name!r} is not defined in runtime manifest")
        config = profiles[resolved_name]
        if not isinstance(config, Mapping):
            raise ValueError(f"Runtime profile {resolved_name!r} must be a mapping")
        requirements = config.get("requirements")
        if not isinstance(requirements, str) or not requirements:
            raise ValueError(f"Runtime profile {resolved_name!r} is missing a requirements file")
        return RuntimeSpec(
            name=str(config.get("name") or resolved_name),
            venv_name=str(config.get("venv_name") or resolved_name),
            extras=tuple(cli_extras) or _as_tuple(config.get("extras"), default=(resolved_name,)),
            extra_index_urls=_as_tuple(config.get("extra_index_urls")),
            torch_backend=str(config["torch_backend"]) if config.get("torch_backend") else None,
            omit_packages=(),
            required_modules=_as_tuple(config.get("required_modules")),
            spec_only_modules=_as_tuple(config.get("spec_only_modules")),
            pyproject_digest=str(config.get("digest") or "runtime-manifest"),
            requirements_file=metadata.root / requirements,
            constraints_file=_manifest_path(metadata.root, config, "constraints"),
            overrides_file=_manifest_path(metadata.root, config, "overrides"),
        )

    if metadata.pyproject is None:
        raise ValueError("Runtime metadata is missing both manifest and pyproject data")
    config = _runtime_config(metadata.data, resolved_name)
    if not config:
        raise ValueError(f"Runtime profile {resolved_name!r} is not defined in pyproject.toml")
    extras = tuple(cli_extras) or _as_tuple(config.get("extras"), default=(resolved_name,))
    digest = hashlib.sha256(metadata.pyproject.read_bytes()).hexdigest()[:16]
    if metadata.lockfile:
        digest = hashlib.sha256((digest + metadata.lockfile.read_text(encoding="utf-8")).encode()).hexdigest()[:16]
    return RuntimeSpec(
        name=resolved_name,
        venv_name=str(config.get("venv-name") or resolved_name),
        extras=extras,
        extra_index_urls=_as_tuple(config.get("extra-index-urls")),
        torch_backend=str(config["torch-backend"]) if config.get("torch-backend") else None,
        omit_packages=tuple(_normalize_package_name(name) for name in _as_tuple(config.get("omit-packages"))),
        required_modules=_as_tuple(config.get("required-imports")),
        spec_only_modules=_as_tuple(config.get("spec-only-imports")),
        pyproject_digest=digest,
    )


def _normalize_package_name(name: str) -> str:
    return name.replace("_", "-").replace(".", "-").lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        return ""
    return _normalize_package_name(match.group(1))


def _write_list(path: Path, values: Sequence[str]) -> Path | None:
    if not values:
        return None
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path


def _filter_requirement_text(requirements: str, omit_packages: Sequence[str]) -> str:
    omit = set(omit_packages)
    lines: list[str] = []
    skip_continuation = False
    for line in requirements.splitlines():
        stripped = line.strip()
        if skip_continuation:
            skip_continuation = line.rstrip().endswith("\\")
            continue
        if stripped and not stripped.startswith(("#", "-")) and _requirement_name(stripped) in omit:
            skip_continuation = line.rstrip().endswith("\\")
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _build_locked_requirement_files(
    uv: Path,
    metadata: ProjectMetadata,
    spec: RuntimeSpec,
    work_dir: Path,
    env: dict[str, str],
) -> dict[str, Path | None] | None:
    if metadata.lockfile is None:
        return None

    command: list[str | Path] = [
        uv,
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-annotate",
        "--no-hashes",
        "--no-header",
        "--format",
        "requirements.txt",
    ]
    for extra in spec.extras:
        command.extend(["--extra", extra])

    requirements = _filter_requirement_text(_run_capture(command, cwd=metadata.root, env=env), spec.omit_packages)
    if not requirements.strip():
        raise RuntimeError(f"Runtime {spec.name!r} exported no requirements from extras {spec.extras!r}")

    path = work_dir / "requirements.locked.txt"
    path.write_text(requirements, encoding="utf-8")
    return {"requirements": path, "constraints": None, "overrides": None}


def _build_direct_requirement_files(
    metadata: ProjectMetadata,
    spec: RuntimeSpec,
    work_dir: Path,
) -> dict[str, Path | None]:
    optional_deps = metadata.data.get("project", {}).get("optional-dependencies", {})
    if not isinstance(optional_deps, Mapping):
        raise ValueError("pyproject.toml is missing [project.optional-dependencies]")

    requirements: list[str] = []
    for extra in spec.extras:
        deps = optional_deps.get(extra)
        if deps is None:
            raise ValueError(f"Runtime extra {extra!r} is not defined in pyproject.toml")
        for dep in deps:
            dep_text = str(dep)
            if _requirement_name(dep_text) not in spec.omit_packages:
                requirements.append(dep_text)

    if not requirements:
        raise ValueError(f"Runtime {spec.name!r} produced no requirements from extras {spec.extras!r}")

    tool_uv = metadata.data.get("tool", {}).get("uv", {})
    if not isinstance(tool_uv, Mapping):
        tool_uv = {}

    return {
        "requirements": _write_list(work_dir / "requirements.in", requirements),
        "constraints": _write_list(
            work_dir / "constraints.txt",
            tuple(str(item) for item in tool_uv.get("constraint-dependencies", []) or []),
        ),
        "overrides": _write_list(
            work_dir / "overrides.txt",
            tuple(str(item) for item in tool_uv.get("override-dependencies", []) or []),
        ),
    }


def _build_requirement_files(
    metadata: ProjectMetadata,
    spec: RuntimeSpec,
    work_dir: Path,
    *,
    uv: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Path | None]:
    if spec.requirements_file is not None:
        files = {
            "requirements": spec.requirements_file,
            "constraints": spec.constraints_file,
            "overrides": spec.overrides_file,
        }
        for file_type, path in files.items():
            if path is not None and not path.exists():
                raise FileNotFoundError(f"Runtime {file_type} file not found: {path}")
        return files
    if uv is not None and env is not None:
        locked = _build_locked_requirement_files(uv, metadata, spec, work_dir, env)
        if locked is not None:
            return locked
    return _build_direct_requirement_files(metadata, spec, work_dir)


def _stamp_path(venv_dir: Path) -> Path:
    return venv_dir / ".nemotron-curator-runtime.stamp"


def _stamp_matches(venv_dir: Path, spec: RuntimeSpec) -> bool:
    stamp = _stamp_path(venv_dir)
    return stamp.exists() and stamp.read_text(encoding="utf-8").strip() == spec.stamp


def _write_stamp(venv_dir: Path, spec: RuntimeSpec) -> None:
    _stamp_path(venv_dir).write_text(f"{spec.stamp}\n", encoding="utf-8")


def _verify_profile(venv_python: Path, spec: RuntimeSpec, env: dict[str, str]) -> bool:
    if not spec.required_modules and not spec.spec_only_modules:
        return True
    code = """
import importlib.util
import sys

required = sys.argv[1].split(",") if sys.argv[1] else []
spec_only = sys.argv[2].split(",") if sys.argv[2] else []
missing = [name for name in required + spec_only if importlib.util.find_spec(name) is None]
if missing:
    print("missing modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
"""
    result = subprocess.run(
        [str(venv_python), "-c", code, ",".join(spec.required_modules), ",".join(spec.spec_only_modules)],
        env=env,
    )
    return result.returncode == 0


def ensure_runtime(
    spec: RuntimeSpec,
    *,
    metadata: ProjectMetadata,
    venv_root: Path = DEFAULT_VENV_ROOT,
    curator_path: Path = DEFAULT_CURATOR_PATH,
    recreate: bool = False,
    skip_install: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Ensure a runtime exists and return ``(python, env)`` for exec."""
    venv_dir = venv_root / spec.venv_name
    venv_python = _ensure_venv(venv_dir, recreate=recreate)
    env = _runtime_env(venv_dir, curator_path)

    if _stamp_matches(venv_dir, spec) and _verify_profile(venv_python, spec, env):
        print(f"[curator-runtime] reusing {spec.name} runtime at {venv_dir}", flush=True)
        return venv_python, env

    if skip_install:
        if _verify_profile(venv_python, spec, env):
            return venv_python, env
        raise RuntimeError(f"{spec.name} runtime is missing packages and --skip-install was set")

    uv = _ensure_uv(venv_python, venv_dir, env)
    with tempfile.TemporaryDirectory(prefix="nemotron-runtime-") as td:
        requirement_files = _build_requirement_files(metadata, spec, Path(td), uv=uv, env=env)
        command: list[str | Path] = [
            uv,
            "pip",
            "install",
            "--python",
            venv_python,
            "--quiet",
            "--no-cache",
            "--requirements",
            requirement_files["requirements"],
        ]
        if requirement_files["constraints"]:
            command.extend(["--constraints", requirement_files["constraints"]])
        if requirement_files["overrides"]:
            command.extend(["--overrides", requirement_files["overrides"]])
        for index_url in spec.extra_index_urls:
            command.extend(["--extra-index-url", index_url])
        if spec.torch_backend:
            command.extend(["--torch-backend", spec.torch_backend])
        _run(command, env=env)

    if not _verify_profile(venv_python, spec, env):
        raise RuntimeError(f"{spec.name} runtime installation completed but import checks failed")
    _write_stamp(venv_dir, spec)
    return venv_python, env


def _normalize_command(command: Sequence[str], venv_python: Path) -> list[str]:
    if not command:
        raise ValueError("missing command after '--'")
    argv = list(command)
    if argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ValueError("missing command after '--'")
    if Path(argv[0]).name in {"python", "python3"}:
        argv[0] = str(venv_python)
    return argv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="byob", help="Runtime profile under [tool.nemotron.runtime].")
    parser.add_argument("--extra", action="append", default=[], help="Override runtime extras from pyproject.toml.")
    parser.add_argument("--project-metadata", type=Path, default=None, help="Directory containing runtime metadata.")
    parser.add_argument("--venv-root", type=Path, default=DEFAULT_VENV_ROOT)
    parser.add_argument("--curator-path", type=Path, default=DEFAULT_CURATOR_PATH)
    parser.add_argument("--recreate", action="store_true", help="Recreate the runtime venv before installing.")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Use the venv/container as-is and fail if required modules are missing.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to exec after '--'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = _find_project_metadata(args.project_metadata)
    spec = load_runtime_spec(args.profile, metadata, cli_extras=args.extra)
    venv_python, env = ensure_runtime(
        spec,
        metadata=metadata,
        venv_root=args.venv_root,
        curator_path=args.curator_path,
        recreate=args.recreate,
        skip_install=args.skip_install,
    )
    command = _normalize_command(args.command, venv_python)
    print(f"[curator-runtime] exec {_quote(command)}", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
