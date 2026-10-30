"""Tests for rerank SDG command helpers."""

from __future__ import annotations

from nemotron.cli.commands.rerank import sdg as sdg_module


def test_sdg_redacts_nvidia_api_key_from_saved_config_surfaces():
    job_config = {
        "nvidia_api_key": "secret-one",
        "run": {
            "cli": {
                "argv": ["nvidia_api_key=secret-one", "--nvidia-api-key", "secret-two"],
                "dotlist": ["nvidia_api_key=secret-one"],
            }
        },
    }
    train_config = {"nvidia_api_key": "secret-one"}

    secret = sdg_module._extract_and_redact_nvidia_api_key(job_config, train_config)

    assert secret == "secret-one"
    assert job_config["nvidia_api_key"] is None
    assert train_config["nvidia_api_key"] is None
    assert job_config["run"]["cli"]["argv"] == [
        "nvidia_api_key=<redacted>",
        "--nvidia-api-key",
        "<redacted>",
    ]
    assert job_config["run"]["cli"]["dotlist"] == ["nvidia_api_key=<redacted>"]
