"""Build compact runtime metadata for remote bootstrap execution."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_PAYLOAD_CHUNKS_ENV = "NEMOTRON_CURATOR_RUNTIME_CHUNKS"
RUNTIME_PAYLOAD_SHA256_ENV = "NEMOTRON_CURATOR_RUNTIME_SHA256"
RUNTIME_PAYLOAD_CHUNK_PREFIX = "NEMOTRON_CURATOR_RUNTIME_CHUNK_"
_ENV_CHUNK_SIZE = 9_000


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _normalize_package_name(name: str) -> str:
    return name.replace("_", "-").replace(".", "-").lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    return _normalize_package_name(match.group(1)) if match else ""


def _filter_requirement_text(requirements: str, omit_packages: list[str]) -> str:
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


def _direct_extra_requirements(data: dict[str, Any], extras: list[str], omit_packages: list[str]) -> str:
    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    requirements: list[str] = []
    for extra in extras:
        deps = optional_deps.get(extra)
        if deps is None:
            raise ValueError(f"Runtime extra {extra!r} is not defined in pyproject.toml")
        if not isinstance(deps, list):
            raise ValueError(f"Runtime extra {extra!r} must be a list in pyproject.toml")
        for dep in deps:
            dep_text = str(dep)
            if _requirement_name(dep_text) not in omit_packages:
                requirements.append(dep_text)
    return "\n".join(requirements).strip() + "\n"


def _runtime_dependency_list(data: dict[str, Any], config: dict[str, Any], key: str) -> list[str]:
    explicit = _as_str_list(config.get(key))
    if explicit:
        return explicit
    tool_uv = data.get("tool", {}).get("uv", {})
    if not isinstance(tool_uv, dict):
        return []
    return _as_str_list(tool_uv.get(key))


def _add_payload(
    payloads: list[tuple[str, bytes]],
    payload_names_by_content: dict[bytes, str],
    name: str,
    content: bytes,
) -> str:
    existing = payload_names_by_content.get(content)
    if existing is not None:
        return existing
    payloads.append((name, content))
    payload_names_by_content[content] = name
    return name


def _add_text_payload(
    payloads: list[tuple[str, bytes]],
    payload_names_by_content: dict[bytes, str],
    name: str,
    values: list[str],
) -> str | None:
    if not values:
        return None
    return _add_payload(payloads, payload_names_by_content, name, ("\n".join(values) + "\n").encode("utf-8"))


def _validate_payload_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or name in {"", ".", ".."}:
        raise ValueError(f"Runtime payload name must be a single file name: {name!r}")


def encode_runtime_payload_env(
    payloads: list[tuple[str, bytes]],
    *,
    chunk_size: int = _ENV_CHUNK_SIZE,
) -> dict[str, str]:
    """Encode generated runtime payload files into chunked env vars."""
    files = []
    for name, content in payloads:
        _validate_payload_name(name)
        files.append({"name": name, "content": base64.b64encode(content).decode("ascii")})

    raw = json.dumps({"version": 1, "files": files}, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode("ascii")
    chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
    env = {
        RUNTIME_PAYLOAD_CHUNKS_ENV: str(len(chunks)),
        RUNTIME_PAYLOAD_SHA256_ENV: digest,
    }
    for idx, chunk in enumerate(chunks):
        env[f"{RUNTIME_PAYLOAD_CHUNK_PREFIX}{idx}"] = chunk
    return env


def _payload_digest_context(env: Mapping[str, str]) -> str:
    digest = env.get(RUNTIME_PAYLOAD_SHA256_ENV)
    return f" sha256={digest}" if digest else ""


def decode_runtime_payload_env(env: Mapping[str, str]) -> list[tuple[str, bytes]]:
    """Decode runtime payload files from env vars, returning an empty list if absent."""
    chunks_value = env.get(RUNTIME_PAYLOAD_CHUNKS_ENV)
    if not chunks_value:
        if env.get(RUNTIME_PAYLOAD_SHA256_ENV):
            raise RuntimeError(
                "Curator runtime payload is incomplete: "
                f"{RUNTIME_PAYLOAD_CHUNKS_ENV} is missing. "
                "Check whether the executor filters NEMOTRON_CURATOR_RUNTIME_* environment variables."
            )
        return []

    digest_context = _payload_digest_context(env)
    if not env.get(RUNTIME_PAYLOAD_SHA256_ENV):
        raise RuntimeError(
            "Curator runtime payload is incomplete: "
            f"{RUNTIME_PAYLOAD_SHA256_ENV} is missing. "
            "Check whether the executor filters NEMOTRON_CURATOR_RUNTIME_* environment variables."
        )
    try:
        count = int(chunks_value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid Curator runtime payload chunk count {chunks_value!r}.{digest_context}") from exc
    if count < 1:
        raise RuntimeError(f"Invalid Curator runtime payload chunk count {count}.{digest_context}")

    missing = [idx for idx in range(count) if f"{RUNTIME_PAYLOAD_CHUNK_PREFIX}{idx}" not in env]
    if missing:
        missing_text = ", ".join(str(idx) for idx in missing)
        raise RuntimeError(
            "Curator runtime payload is incomplete: "
            f"missing chunk index(es) {missing_text} of {count}.{digest_context} "
            "Check whether the executor filters NEMOTRON_CURATOR_RUNTIME_* environment variables."
        )

    encoded = "".join(env[f"{RUNTIME_PAYLOAD_CHUNK_PREFIX}{idx}"] for idx in range(count))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise RuntimeError(f"Curator runtime payload is not valid base64.{digest_context}") from exc
    expected = env.get(RUNTIME_PAYLOAD_SHA256_ENV)
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("Curator runtime payload digest mismatch")

    payload = json.loads(raw.decode("utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Curator runtime payload version")

    decoded: list[tuple[str, bytes]] = []
    for item in payload.get("files", []):
        name = str(item["name"])
        _validate_payload_name(name)
        decoded.append((name, base64.b64decode(item["content"])))
    return decoded


def write_runtime_payloads_from_env(output_dir: Path, env: Mapping[str, str]) -> bool:
    """Write env-encoded runtime payloads into ``output_dir`` if present."""
    payloads = decode_runtime_payload_env(env)
    if not payloads:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads:
        (output_dir / name).write_bytes(content)
    return True


def read_runtime_payloads(runtime_dir: Path = DEFAULT_OUTPUT_DIR) -> list[tuple[str, bytes]]:
    """Read a packaged runtime payload directory if one is present."""
    manifest = runtime_dir / "runtime.json"
    if not manifest.is_file():
        return []

    data = json.loads(manifest.read_text(encoding="utf-8"))
    names = {"runtime.json"}
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"Runtime manifest has invalid profiles table: {manifest}")
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        for key in ("requirements", "constraints", "overrides"):
            name = profile.get(key)
            if name:
                _validate_payload_name(str(name))
                names.add(str(name))

    payloads: list[tuple[str, bytes]] = []
    for name in sorted(names):
        _validate_payload_name(name)
        path = runtime_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Runtime manifest references missing payload file: {path}")
        payloads.append((name, path.read_bytes()))
    return payloads


def _export_extra_requirements(root: Path, data: dict[str, Any], extras: list[str], omit_packages: list[str]) -> str:
    uv = shutil.which("uv")
    if not uv or not (root / "uv.lock").exists():
        return _direct_extra_requirements(data, extras, omit_packages)

    command = [
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
    for extra in extras:
        command.extend(["--extra", extra])
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "uv export failed while building Curator runtime metadata. "
            f"Command: {' '.join(command)}\n{message}"
        )
    return _filter_requirement_text(result.stdout, omit_packages)


def build_runtime_payloads(root: Path) -> list[tuple[str, bytes]]:
    """Return runtime payload files for profiles declared in ``pyproject.toml``."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return []
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    runtime_table = data.get("tool", {}).get("nemotron", {}).get("runtime", {})
    if not isinstance(runtime_table, dict):
        return []

    manifest: dict[str, Any] = {"version": 1, "profiles": {}}
    payloads: list[tuple[str, bytes]] = []
    payload_names_by_content: dict[bytes, str] = {}
    for name, config in sorted(runtime_table.items()):
        if not isinstance(config, dict):
            continue
        extras = _as_str_list(config.get("extras")) or [name]
        omit_packages = [_normalize_package_name(item) for item in _as_str_list(config.get("omit-packages"))]
        requirements = _export_extra_requirements(root, data, extras, omit_packages)
        if not requirements.strip():
            continue
        requirements_name = f"{name}.requirements.txt"
        constraints_name = _add_text_payload(
            payloads,
            payload_names_by_content,
            f"{name}.constraints.txt",
            _runtime_dependency_list(data, config, "constraint-dependencies"),
        )
        overrides_name = _add_text_payload(
            payloads,
            payload_names_by_content,
            f"{name}.overrides.txt",
            _runtime_dependency_list(data, config, "override-dependencies"),
        )
        digest = hashlib.sha256()
        digest.update(pyproject.read_bytes())
        lockfile = root / "uv.lock"
        if lockfile.exists():
            digest.update(lockfile.read_bytes())
        digest.update(requirements.encode("utf-8"))
        requirements_name = _add_payload(
            payloads,
            payload_names_by_content,
            requirements_name,
            requirements.encode("utf-8"),
        )
        manifest["profiles"][name] = {
            "name": name,
            "venv_name": str(config.get("venv-name") or name),
            "extras": extras,
            "requirements": requirements_name,
            "extra_index_urls": _as_str_list(config.get("extra-index-urls")),
            "torch_backend": config.get("torch-backend"),
            "required_modules": _as_str_list(config.get("required-imports")),
            "spec_only_modules": _as_str_list(config.get("spec-only-imports")),
            "digest": digest.hexdigest()[:16],
        }
        if constraints_name:
            manifest["profiles"][name]["constraints"] = constraints_name
        if overrides_name:
            manifest["profiles"][name]["overrides"] = overrides_name

    if not manifest["profiles"]:
        return []
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    return [("runtime.json", manifest_bytes), *payloads]


def write_runtime_payloads(root: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in build_runtime_payloads(root):
        (output_dir / name).write_bytes(content)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    write_runtime_payloads(args.repo_root, args.output_dir)


if __name__ == "__main__":
    main()
