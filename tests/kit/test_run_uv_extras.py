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
    with pytest.raises(SystemExit) as result:
        run_uv.main(stage)
    assert result.value.code == 0
    sync = next(call.args[0] for call in calls.call_args_list if call.args[0][1] == "sync")
    if extras:
        assert sync[-2:] == ["--extra", "text"]
    else:
        assert "--extra" not in sync
    assert (tmp_path / "venv/.nemotron-ready").exists()
