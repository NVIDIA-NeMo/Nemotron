from __future__ import annotations

from pathlib import Path

import pytest
import typer

from nemotron.cli.commands.steps import run_cmd
from nemotron.steps._bootstrap import runtime_payloads


def _write_step_tree(root: Path) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    script_path = root / "src" / "nemotron" / "steps" / "byob" / "step.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")
    return script_path


CURATOR_RUN_COMMAND = (
    "python -m nemotron.steps._bootstrap.curator_runtime --profile byob -- python -m custom.step"
)


def test_curator_runtime_env_vars_for_remote_curator_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_path = _write_step_tree(tmp_path)
    build_calls: list[Path] = []
    encode_calls = []

    def fake_build_runtime_payloads(root: Path):
        build_calls.append(root)
        return [("runtime.json", b'{"version": 1, "profiles": {}}')]

    def fake_encode_runtime_payload_env(payloads):  # noqa: ANN001
        encode_calls.append(payloads)
        return {"NEMOTRON_CURATOR_RUNTIME_CHUNKS": "1"}

    monkeypatch.setattr(runtime_payloads, "build_runtime_payloads", fake_build_runtime_payloads)
    monkeypatch.setattr(runtime_payloads, "encode_runtime_payload_env", fake_encode_runtime_payload_env)

    env_vars = run_cmd._build_curator_runtime_env_vars(  # noqa: SLF001
        script_path=script_path,
        env={"run_command": CURATOR_RUN_COMMAND},
        mode="run",
    )

    assert build_calls == [tmp_path]
    assert encode_calls == [[("runtime.json", b'{"version": 1, "profiles": {}}')]]
    assert env_vars == {"NEMOTRON_CURATOR_RUNTIME_CHUNKS": "1"}
    assert not (tmp_path / "src" / "nemotron" / "steps" / "_bootstrap" / "runtime").exists()


def test_curator_runtime_env_vars_skip_local(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_path = _write_step_tree(tmp_path)
    calls = []

    monkeypatch.setattr(runtime_payloads, "build_runtime_payloads", lambda *args, **kwargs: calls.append(args))

    env_vars = run_cmd._build_curator_runtime_env_vars(  # noqa: SLF001
        script_path=script_path,
        env={"run_command": CURATOR_RUN_COMMAND},
        mode="local",
    )

    assert calls == []
    assert env_vars == {}


def test_curator_runtime_env_vars_uses_packaged_runtime_when_not_source_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "site-packages" / "nemotron" / "steps" / "byob" / "step.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")
    encode_calls = []

    monkeypatch.setattr(
        runtime_payloads,
        "read_runtime_payloads",
        lambda: [("runtime.json", b'{"version": 1, "profiles": {}}')],
    )
    monkeypatch.setattr(
        runtime_payloads,
        "encode_runtime_payload_env",
        lambda payloads: encode_calls.append(payloads) or {"NEMOTRON_CURATOR_RUNTIME_CHUNKS": "1"},
    )

    env_vars = run_cmd._build_curator_runtime_env_vars(  # noqa: SLF001
        script_path=script_path,
        env={"run_command": CURATOR_RUN_COMMAND},
        mode="run",
    )

    assert encode_calls == [[("runtime.json", b'{"version": 1, "profiles": {}}')]]
    assert env_vars == {"NEMOTRON_CURATOR_RUNTIME_CHUNKS": "1"}


def test_curator_runtime_env_vars_fail_fast_without_source_or_packaged_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "site-packages" / "nemotron" / "steps" / "byob" / "step.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime_payloads, "read_runtime_payloads", lambda: [])

    with pytest.raises(typer.Exit) as exc_info:
        run_cmd._build_curator_runtime_env_vars(  # noqa: SLF001
            script_path=script_path,
            env={"run_command": CURATOR_RUN_COMMAND},
            mode="run",
        )

    assert exc_info.value.exit_code == 1


def test_uses_curator_runtime_from_run_command() -> None:
    assert run_cmd._uses_curator_runtime({"run_command": CURATOR_RUN_COMMAND})  # noqa: SLF001
