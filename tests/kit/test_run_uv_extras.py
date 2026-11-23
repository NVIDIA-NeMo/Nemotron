"""Container dependency installation preserves the stage's selected extras."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemotron.kit import run_uv


def isolated_container_path(value: str, *, root: Path) -> Path:
    if value == "/opt/nemotron-venv":
        return root / "venv"
    if value == "/tmp/nemotron-run-uv.lock":
        return root / "setup.lock"
    return Path(value)


@pytest.mark.parametrize("extras", ([], ["text"]))
def test_container_sync_selects_declared_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extras: list[str]
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "train.py").write_text("pass\n")
    (stage / "pyproject.toml").write_text(
        '[project]\nname="test-stage"\nversion="0.0.0"\nrequires-python=">=3.12"\n'
        '[tool.nemotron]\nentry-point="train.py"\ncontainer-extras=' + repr(extras) + "\n"
    )
    (tmp_path / "venv").mkdir()
    calls = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(run_uv, "Path", partial(isolated_container_path, root=tmp_path))
    monkeypatch.setattr(run_uv.sys, "path", [])
    monkeypatch.setattr(run_uv.sys, "argv", ["run_uv.py", "--help"])
    monkeypatch.setattr(run_uv.shutil, "which", Mock(return_value="/usr/bin/uv"))
    monkeypatch.setattr(run_uv.shutil, "rmtree", Mock())
    monkeypatch.setattr(run_uv.subprocess, "run", calls)
    monkeypatch.setenv("UV_PYTHON", "/different/python")
    with pytest.raises(SystemExit) as result:
        run_uv.main(stage)
    assert result.value.code == 0
    sync = next(call.args[0] for call in calls.call_args_list if call.args[0][1] == "sync")
    assert sync[sync.index("--python") + 1] == run_uv.sys.executable
    if extras:
        assert sync[-2:] == ["--extra", "text"]
    else:
        assert "--extra" not in sync
    assert (tmp_path / "venv/.nemotron-ready").exists()


def test_shared_entry_point_repairs_incomplete_cache_and_forwards_arguments(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "train.py").write_text("pass\n")
    (stage / "pyproject.toml").write_text(
        '[project]\nname="test"\nversion="0.0.0"\nrequires-python=">=3.12"\n[tool.nemotron]\nentry-point="train.py"\n'
    )
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv/.nemotron-ready").touch()
    calls = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(run_uv, "Path", partial(isolated_container_path, root=tmp_path))
    monkeypatch.setattr(run_uv.sys, "path", [])
    monkeypatch.setattr(run_uv.sys, "argv", ["run_uv.py", "--stage-dir", str(stage), "--config", "run.yaml"])
    monkeypatch.setattr(run_uv.shutil, "which", Mock(return_value="/usr/bin/uv"))
    monkeypatch.setattr(run_uv.shutil, "rmtree", Mock())
    monkeypatch.setattr(run_uv.subprocess, "run", calls)
    with pytest.raises(SystemExit) as result:
        run_uv.main()
    assert result.value.code == 0
    assert any(call.args[0][1] == "sync" for call in calls.call_args_list)
    assert calls.call_args_list[-1].args[0][-2:] == ["--config", "run.yaml"]


def test_isolated_mode_uses_full_lock_without_image_packages(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "train.py").write_text("pass\n")
    (stage / "uv.lock").write_text("fixture\n")
    (stage / "pyproject.toml").write_text(
        '[project]\nname="test"\nversion="0.0.0"\nrequires-python=">=3.12"\n[tool.nemotron]\nentry-point="train.py"\n'
    )
    calls = []
    environments = []

    def run(command, **kwargs):
        calls.append(command)
        environments.append(kwargs.get("env"))
        if command[1] == "venv":
            Path(command[-1]).mkdir(parents=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("UV_OVERRIDE", "/etc/image-overrides.txt")
    monkeypatch.setenv("UV_CONSTRAINT", "/etc/image-constraints.txt")
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/opt/uv/python")
    monkeypatch.setenv("NEMOTRON_UV_MODE", "isolated")
    monkeypatch.setenv("NEMOTRON_VENV_ROOT", str(tmp_path / "envs"))
    monkeypatch.setattr(run_uv.sys, "path", [])
    monkeypatch.setattr(run_uv.sys, "argv", ["run_uv.py", "--stage-dir", str(stage)])
    monkeypatch.setattr(run_uv.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(run_uv.subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        run_uv.main()
    assert result.value.code == 0
    create = next(command for command in calls if command[1] == "venv")
    sync = next(command for command in calls if command[1] == "sync")
    assert "--system-site-packages" not in create
    assert "--python" not in sync
    assert "--locked" in sync and str(stage) in sync
    synced = next(env for command, env in zip(calls, environments) if command[1] == "sync")
    assert "UV_OVERRIDE" not in synced and "UV_CONSTRAINT" not in synced
    assert synced["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "envs/.python")

    # A lock-only update must not reuse packages from a previously ready env.
    (stage / "uv.lock").write_text("updated fixture\n")
    with pytest.raises(SystemExit) as result:
        run_uv.main()
    assert result.value.code == 0
    created = [command[-1] for command in calls if command[1] == "venv"]
    assert len(created) == 2 and created[0] != created[1]


def test_inherited_container_packages_process_editable_paths(tmp_path):
    import subprocess
    import sys

    parent_site = tmp_path / "parent/site-packages"
    child_site = tmp_path / "child/site-packages"
    source = tmp_path / "image-sdk"
    for directory in (parent_site, child_site, source):
        directory.mkdir(parents=True)
    (source / "image_sdk.py").write_text("VALUE = 'image-owned SDK'\n")
    (parent_site / "editable-sdk.pth").write_text(str(source) + "\n")
    run_uv._inherit_container_packages(child_site, [parent_site])
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import site, sys; site.addsitedir(sys.argv[1]); "
            "import image_sdk; assert image_sdk.VALUE == 'image-owned SDK'",
            str(child_site),
        ],
        check=True,
        capture_output=True,
    )


def test_inheritance_failure_does_not_cache_a_ready_environment(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "train.py").write_text("pass\n")
    (stage / "pyproject.toml").write_text(
        '[project]\nname="test"\nversion="0.0.0"\nrequires-python=">=3.12"\n[tool.nemotron]\nentry-point="train.py"\n'
    )
    (tmp_path / "venv").mkdir()
    parent_site = tmp_path / "parent/site-packages"
    parent_site.mkdir(parents=True)
    monkeypatch.setattr(run_uv, "Path", partial(isolated_container_path, root=tmp_path))
    monkeypatch.setattr(run_uv.sys, "path", [str(parent_site)])
    monkeypatch.setattr(run_uv.sys, "argv", ["run_uv.py"])
    monkeypatch.setattr(run_uv.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(run_uv.shutil, "rmtree", Mock())
    monkeypatch.setattr(run_uv.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    monkeypatch.setattr(run_uv, "_inherit_container_packages", Mock(side_effect=OSError("incomplete inheritance")))
    with pytest.raises(OSError, match="incomplete inheritance"):
        run_uv.main(stage)
    assert not (tmp_path / "venv/.nemotron-ready").exists()
