from __future__ import annotations

from nemo_runspec.env import load_env_file


def test_load_env_file_honors_env_override(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / "custom-env.toml"
    env_file.write_text('[lepton_test]\nexecutor = "lepton"\n', encoding="utf-8")

    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(env_file))

    assert load_env_file()["lepton_test"]["executor"] == "lepton"
