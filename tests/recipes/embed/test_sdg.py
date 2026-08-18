"""Tests for embed SDG credential handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemotron.cli.commands.embed import sdg as sdg_module


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


@pytest.mark.parametrize(
    ("config_secret", "ambient_secret", "expected_secret"),
    [
        ("configured-test-secret", "ambient-test-secret", "configured-test-secret"),
        (None, "ambient-test-secret", "ambient-test-secret"),
    ],
)
def test_sdg_forwards_key_to_remote_execution_without_persisting_it(
    monkeypatch, tmp_path, config_secret, ambient_secret, expected_secret
):
    job_config = OmegaConf.create(
        {
            "nvidia_api_key": config_secret,
            "run": {"cli": {"argv": [], "dotlist": []}, "env": {}},
        }
    )
    captured: dict[str, object] = {}

    monkeypatch.setenv("NVIDIA_API_KEY", ambient_secret)
    monkeypatch.setattr(sdg_module, "parse_config", lambda *_: OmegaConf.create({}))
    monkeypatch.setattr(sdg_module, "parse_env", lambda *_: {})
    monkeypatch.setattr(sdg_module, "build_job_config", lambda *_args, **_kwargs: job_config)
    monkeypatch.setattr(sdg_module, "display_job_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sdg_module, "generate_job_dir", lambda *_: tmp_path)
    monkeypatch.setattr(
        sdg_module,
        "extract_train_config",
        lambda config, **_kwargs: OmegaConf.create(OmegaConf.to_container(config, resolve=False)),
    )

    def save_configs(job, train, job_dir):
        job_path = job_dir / "job.yaml"
        train_path = job_dir / "train.yaml"
        OmegaConf.save(job, job_path)
        OmegaConf.save(train, train_path)
        return job_path, train_path

    monkeypatch.setattr(sdg_module, "save_configs", save_configs)
    monkeypatch.setattr(sdg_module, "build_env_vars", lambda *_: {})
    monkeypatch.setattr(sdg_module, "display_job_submission", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sdg_module,
        "_execute_remote",
        lambda **kwargs: captured.update(env_vars=kwargs["env_vars"]),
    )

    cfg = SimpleNamespace(
        ctx=object(),
        mode="batch",
        argv=[],
        dry_run=False,
        passthrough=[],
        attached=False,
        force_squash=False,
    )
    sdg_module._execute_sdg(cfg)

    assert captured["env_vars"] == {"NVIDIA_API_KEY": expected_secret}
    job_config_text = (tmp_path / "job.yaml").read_text()
    train_config_text = (tmp_path / "train.yaml").read_text()
    for secret in (config_secret, ambient_secret):
        if secret:
            assert secret not in job_config_text
            assert secret not in train_config_text
