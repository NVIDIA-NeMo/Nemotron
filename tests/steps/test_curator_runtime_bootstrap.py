from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.steps._bootstrap import curator_runtime, runtime_payloads


def _write_pyproject(root: Path) -> Path:
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "demo"
version = "0"

[project.optional-dependencies]
byob = [
    "data-designer==0.5.5",
    "nemo-curator[translation_all] @ git+https://example.invalid/Curator.git",
    "cupy-cuda12x==14.0.1",
]

[tool.uv]
constraint-dependencies = ["transformers>=4.56.0,<5.0"]
override-dependencies = ["torch==2.10.0"]

[tool.nemotron.runtime.byob]
extras = ["byob"]
venv-name = "byob"
extra-index-urls = ["https://pypi.nvidia.com"]
torch-backend = "cu128"
omit-packages = ["nemo-curator"]
required-imports = ["data_designer"]
spec-only-imports = ["cupy"]

[tool.nemotron.runtime.translate]
extras = ["byob"]
venv-name = "translate"
extra-index-urls = ["https://pypi.nvidia.com"]
torch-backend = "cu128"
omit-packages = ["nemo-curator"]
required-imports = ["nemo_curator", "yaml"]

[tool.nemotron.runtime.curate]
extras = ["byob"]
venv-name = "curate"
extra-index-urls = ["https://pypi.nvidia.com"]
torch-backend = "cu128"
omit-packages = ["nemo-curator"]
required-imports = ["huggingface_hub", "nemo_curator", "yaml"]
""".lstrip(),
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("# fake lock\n", encoding="utf-8")
    return pyproject


def _write_runtime_manifest(root: Path) -> Path:
    runtime_dir = root / ".nemotron_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "byob.requirements.txt").write_text(
        "data-designer==0.5.5\ntransitive-curator-dependency==1.0.0\n",
        encoding="utf-8",
    )
    (runtime_dir / "byob.constraints.txt").write_text("transformers>=4.56.0,<5.0\n", encoding="utf-8")
    (runtime_dir / "byob.overrides.txt").write_text("huggingface-hub>=0.34,<1.0\n", encoding="utf-8")
    manifest = {
        "version": 1,
        "profiles": {
            name: {
                "name": name,
                "venv_name": name,
                "extras": ["byob"],
                "requirements": "byob.requirements.txt",
                "constraints": "byob.constraints.txt",
                "overrides": "byob.overrides.txt",
                "extra_index_urls": ["https://pypi.nvidia.com"],
                "torch_backend": "cu128",
                "required_modules": ["data_designer"] if name == "byob" else ["nemo_curator"],
                "spec_only_modules": ["cupy"] if name == "byob" else [],
                "digest": "abc123",
            }
            for name in ("byob", "translate", "curate")
        },
    }
    (runtime_dir / "runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
    return runtime_dir


def test_runtime_manifest_drives_profile_without_pyproject(tmp_path: Path) -> None:
    runtime_dir = _write_runtime_manifest(tmp_path)
    metadata = curator_runtime._find_project_metadata(tmp_path)  # noqa: SLF001

    spec = curator_runtime.load_runtime_spec("byob", metadata)
    paths = curator_runtime._build_requirement_files(metadata, spec, tmp_path)  # noqa: SLF001

    assert metadata.root == runtime_dir
    assert spec.name == "byob"
    assert spec.requirements_file == runtime_dir / "byob.requirements.txt"
    assert paths["requirements"] == runtime_dir / "byob.requirements.txt"
    assert paths["constraints"] == runtime_dir / "byob.constraints.txt"
    assert paths["overrides"] == runtime_dir / "byob.overrides.txt"


def test_unknown_runtime_profile_fails_from_pyproject(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    metadata = curator_runtime._find_project_metadata(tmp_path)  # noqa: SLF001

    with pytest.raises(ValueError, match="Runtime profile 'translation' is not defined"):
        curator_runtime.load_runtime_spec("translation", metadata)


def test_named_curator_runtime_profiles_from_pyproject(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    metadata = curator_runtime._find_project_metadata(tmp_path)  # noqa: SLF001

    profiles = {name: curator_runtime.load_runtime_spec(name, metadata) for name in ("byob", "translate", "curate")}

    assert set(profiles) == {"byob", "translate", "curate"}
    assert profiles["translate"].venv_name == "translate"
    assert profiles["curate"].venv_name == "curate"
    assert profiles["translate"].extras == ("byob",)
    assert profiles["curate"].extras == ("byob",)


def test_build_requirement_files_from_pyproject_extra(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    metadata = curator_runtime._find_project_metadata(tmp_path)  # noqa: SLF001
    spec = curator_runtime.load_runtime_spec("byob", metadata)

    work_dir = tmp_path / "out"
    work_dir.mkdir()
    paths = curator_runtime._build_requirement_files(metadata, spec, work_dir)  # noqa: SLF001

    requirements = paths["requirements"].read_text(encoding="utf-8")
    constraints = paths["constraints"].read_text(encoding="utf-8")
    overrides = paths["overrides"].read_text(encoding="utf-8")

    assert "data-designer==0.5.5" in requirements
    assert "cupy-cuda12x==14.0.1" in requirements
    assert "nemo-curator" not in requirements
    assert "transformers>=4.56.0,<5.0" in constraints
    assert "torch==2.10.0" in overrides


def test_runtime_payloads_ship_uv_constraints_and_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_pyproject(tmp_path)
    monkeypatch.setattr(runtime_payloads.shutil, "which", lambda _: None)
    output_dir = tmp_path / "runtime"

    runtime_payloads.write_runtime_payloads(tmp_path, output_dir=output_dir)

    manifest = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8"))
    byob = manifest["profiles"]["byob"]

    assert set(manifest["profiles"]) == {"byob", "curate", "translate"}
    assert byob["requirements"] == "byob.requirements.txt"
    assert byob["constraints"] == "byob.constraints.txt"
    assert byob["overrides"] == "byob.overrides.txt"
    assert manifest["profiles"]["translate"]["requirements"] == "byob.requirements.txt"
    assert manifest["profiles"]["curate"]["requirements"] == "byob.requirements.txt"
    assert (output_dir / "byob.constraints.txt").read_text(encoding="utf-8") == "transformers>=4.56.0,<5.0\n"
    assert (output_dir / "byob.overrides.txt").read_text(encoding="utf-8") == "torch==2.10.0\n"


def test_runtime_payload_build_fails_when_uv_export_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_pyproject(tmp_path)
    monkeypatch.setattr(runtime_payloads.shutil, "which", lambda _: "/bin/uv")

    class FailedExport:
        returncode = 1
        stdout = ""
        stderr = "lockfile is stale"

    monkeypatch.setattr(runtime_payloads.subprocess, "run", lambda *args, **kwargs: FailedExport())

    with pytest.raises(RuntimeError, match="uv export failed"):
        runtime_payloads.build_runtime_payloads(tmp_path)


def test_runtime_payload_env_drives_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime_dir = _write_runtime_manifest(tmp_path)
    payloads = [(path.name, path.read_bytes()) for path in sorted(runtime_dir.iterdir())]
    env_vars = runtime_payloads.encode_runtime_payload_env(payloads, chunk_size=32)

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("NEMOTRON_CURATOR_METADATA_ROOT", str(tmp_path / "metadata"))
    monkeypatch.setattr(curator_runtime, "DEFAULT_METADATA_ROOT", tmp_path / "metadata")

    metadata = curator_runtime._find_project_metadata()  # noqa: SLF001
    spec = curator_runtime.load_runtime_spec("byob", metadata)
    paths = curator_runtime._build_requirement_files(metadata, spec, tmp_path)  # noqa: SLF001

    assert metadata.root == tmp_path / "metadata" / env_vars[runtime_payloads.RUNTIME_PAYLOAD_SHA256_ENV][:16]
    assert spec.name == "byob"
    assert paths["requirements"].name == "byob.requirements.txt"
    assert paths["constraints"].name == "byob.constraints.txt"
    assert paths["overrides"].name == "byob.overrides.txt"


def test_runtime_payload_env_reports_missing_chunks(tmp_path: Path) -> None:
    runtime_dir = _write_runtime_manifest(tmp_path)
    payloads = [(path.name, path.read_bytes()) for path in sorted(runtime_dir.iterdir())]
    env_vars = runtime_payloads.encode_runtime_payload_env(payloads, chunk_size=32)
    missing_key = f"{runtime_payloads.RUNTIME_PAYLOAD_CHUNK_PREFIX}1"
    env_vars.pop(missing_key)

    with pytest.raises(RuntimeError, match="missing chunk index\\(es\\) 1"):
        runtime_payloads.decode_runtime_payload_env(env_vars)


def test_runtime_payload_env_reports_missing_chunk_count(tmp_path: Path) -> None:
    runtime_dir = _write_runtime_manifest(tmp_path)
    payloads = [(path.name, path.read_bytes()) for path in sorted(runtime_dir.iterdir())]
    env_vars = runtime_payloads.encode_runtime_payload_env(payloads, chunk_size=32)
    env_vars.pop(runtime_payloads.RUNTIME_PAYLOAD_CHUNKS_ENV)

    with pytest.raises(RuntimeError, match=runtime_payloads.RUNTIME_PAYLOAD_CHUNKS_ENV):
        runtime_payloads.decode_runtime_payload_env(env_vars)


def test_locked_requirement_files_use_uv_export_and_filter_omits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_pyproject(tmp_path)
    metadata = curator_runtime._find_project_metadata(tmp_path)  # noqa: SLF001
    spec = curator_runtime.load_runtime_spec("byob", metadata)
    calls = []

    def fake_run_capture(argv, *, cwd, env):  # noqa: ANN001
        calls.append((argv, cwd, env))
        return "\n".join(
            [
                "data-designer==0.5.5",
                "nemo-curator @ git+https://example.invalid/Curator.git",
                "transitive-curator-dependency==1.0.0",
            ]
        )

    monkeypatch.setattr(curator_runtime, "_run_capture", fake_run_capture)

    work_dir = tmp_path / "locked"
    work_dir.mkdir()
    paths = curator_runtime._build_requirement_files(  # noqa: SLF001
        metadata,
        spec,
        work_dir,
        uv=Path("/venv/bin/uv"),
        env={"PATH": "/venv/bin"},
    )

    requirements = paths["requirements"].read_text(encoding="utf-8")
    assert "data-designer==0.5.5" in requirements
    assert "transitive-curator-dependency==1.0.0" in requirements
    assert "nemo-curator" not in requirements
    assert paths["constraints"] is None
    assert paths["overrides"] is None
    assert calls[0][1] == metadata.root
    assert "--extra" in calls[0][0]
    assert "byob" in calls[0][0]


def test_normalize_command_replaces_python_with_runtime_python(tmp_path: Path) -> None:
    runtime_python = tmp_path / "venv" / "bin" / "python"

    assert curator_runtime._normalize_command(  # noqa: SLF001
        ["--", "python", "-m", "nemotron.steps.byob.step"],
        runtime_python,
    ) == [str(runtime_python), "-m", "nemotron.steps.byob.step"]


def test_normalize_command_requires_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing command"):
        curator_runtime._normalize_command(["--"], tmp_path / "python")  # noqa: SLF001
