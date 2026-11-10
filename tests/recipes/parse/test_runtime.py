"""The Parse environment must stay isolated and reject unrecorded changes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nemotron.recipes.parse.runtime import launch, setup


def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "nemotron"\n')
    src = root / "src"
    for name in ("nemotron/__init__.py", "nemo_runspec/__init__.py", "nemotron/recipes/parse/runtime/launch.py"):
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n")
    return src


def git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


@pytest.fixture
def framework(tmp_path, monkeypatch):
    repo = tmp_path / "framework"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "module.py").write_text("VALUE = 1\n")
    (repo / "other.py").write_text("UNCHANGED = True\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "core.hooksPath=/dev/null",
        "commit", "-qm", "initial")
    (repo / "module.py").write_text("VALUE = 2\n")
    patch = git(repo, "diff") + "\n"
    (repo / "module.py").write_text("VALUE = 1\n")
    runtime = tmp_path / "runtime-inputs"
    (runtime / "patches").mkdir(parents=True)
    (runtime / "patches" / "fixture.patch").write_text(patch)
    monkeypatch.setattr(setup, "RUNTIME", runtime)
    return repo


def test_framework_patch_is_exact_and_idempotent(framework):
    original_index = git(framework, "write-tree")
    tree = setup.apply_patch("fixture", framework)
    assert (framework / "module.py").read_text() == "VALUE = 2\n"
    assert setup.apply_patch("fixture", framework) == tree
    assert git(framework, "write-tree") == original_index
    assert git(framework, "diff", tree) == ""


def test_framework_patch_refuses_unrelated_edits(framework):
    setup.apply_patch("fixture", framework)
    (framework / "other.py").write_text("UNCHANGED = False\n")
    with pytest.raises(RuntimeError, match="Unexpected framework edits"):
        setup.apply_patch("fixture", framework)
    assert (framework / "other.py").read_text() == "UNCHANGED = False\n"


def test_runtime_environment_prevents_host_dependency_leakage(tmp_path):
    host = {
        "PYTHONPATH": "/launcher/site-packages",
        "PYTHONHOME": "/another/python",
        "VIRTUAL_ENV": "/launcher/venv",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "HF_HOME": "/immutable/model-cache",
        "TMPDIR": "/tmp/ray-short-path",
        "PATH": "/usr/bin",
    }
    root, src = tmp_path / "runtime", checkout(tmp_path)
    env = launch.runtime_environment(root, source_root=src, environ=host)
    assert env["PYTHONPATH"] == str(src)
    assert "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["VIRTUAL_ENV"] == str(root / "venv")
    assert env["PATH"].split(":")[0] == str(root / "venv" / "bin")
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["HF_HOME"] == host["HF_HOME"]
    assert env["TMPDIR"] == host["TMPDIR"]
    assert host["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


@pytest.mark.parametrize("explicit", [False, True])
def test_runtime_environment_rejects_host_wheel_site_packages(tmp_path, monkeypatch, explicit):
    site = tmp_path / "site-packages"
    monkeypatch.setattr(launch, "__file__", str(site / "nemotron/recipes/parse/runtime/launch.py"))
    with pytest.raises(RuntimeError, match="source checkout"):
        launch.runtime_environment(tmp_path / "runtime", source_root=site if explicit else None)


def test_wheel_dispatch_fails_before_starting_worker(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    monkeypatch.setattr(launch, "__file__", str(site / "nemotron/recipes/parse/runtime/launch.py"))
    python = tmp_path / "runtime/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("must not launch from host wheel"))
    with pytest.raises(RuntimeError, match="source checkout"):
        launch.run_module("nemotron.recipes.parse.stage1_sft.train", runtime_dir=tmp_path / "runtime")


def test_explicit_checkout_does_not_expose_inferred_host_site_packages(tmp_path, monkeypatch):
    src = checkout(tmp_path)
    site = tmp_path / "site-packages"
    monkeypatch.setattr(launch, "__file__", str(site / "nemotron/recipes/parse/runtime/launch.py"))
    env = launch.runtime_environment(tmp_path / "runtime", source_root=src, environ={"PYTHONPATH": str(site)})
    assert env["PYTHONPATH"] == str(src)


def test_wheel_setup_fails_before_creating_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.sys, "platform", "linux")
    site = tmp_path / "site-packages"
    monkeypatch.setattr(launch, "__file__", str(site / "nemotron/recipes/parse/runtime/launch.py"))
    runtime = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="source checkout"):
        setup.install(runtime)
    assert not runtime.exists()


def test_setup_cli_preserves_original_automatic_backend_selection(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup.sys, "argv", ["setup.py", "--runtime-dir", str(tmp_path)])
    monkeypatch.setattr(setup, "install", lambda *a, **kw: calls.append((a, kw)))
    setup.main()
    assert calls == [((tmp_path,), {"torch_backend": "auto", "python_version": "3.13.16"})]


def test_missing_runtime_fails_before_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(FileNotFoundError, match="runtime.setup"):
        launch.run_module("nemotron.recipes.parse.stage1_sft.train", runtime_dir=tmp_path)
    assert calls == []


def test_duplicate_setup_is_rejected(tmp_path):
    with setup.installation_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another Parse runtime installation"):
            with setup.installation_lock(tmp_path):
                pytest.fail("duplicate installer entered")


def test_completed_runtime_rejects_package_drift_without_installing(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.sys, "platform", "linux")
    monkeypatch.setattr(setup, "SOURCES", ())
    request = {"profile": "test"}
    monkeypatch.setattr(setup, "install_request", lambda *args: request)
    (tmp_path / "requested.json").write_text(json.dumps(request))
    (tmp_path / "environment.txt").write_text("torch==2.14.0+cu132\n")
    (tmp_path / "runtime.json").write_text(json.dumps({
        "request": request, "sources": {}, "environment_sha256": setup.sha256(tmp_path / "environment.txt"),
    }))
    commands = []

    def output(*command, **kwargs):
        commands.append(command)
        return "uv 0.13.1" if command == ("uv", "--version") else "torch==2.15.0+cu132"

    monkeypatch.setattr(setup, "output", output)
    monkeypatch.setattr(setup, "run", lambda *a, **kw: pytest.fail("must not mutate a drifted runtime"))
    with pytest.raises(RuntimeError, match="Installed packages differ"):
        setup.install(tmp_path)
    assert len(commands) == 2
