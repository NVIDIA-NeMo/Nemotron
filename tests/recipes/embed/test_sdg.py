"""Tests for embed SDG model selection and credential handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemotron.cli.commands.embed import sdg as sdg_module


@pytest.mark.parametrize("dry_run", [True, False])
def test_multimodal_models_are_resolved_before_preview_and_persistence(monkeypatch, tmp_path, dry_run):
    monkeypatch.delenv("NEMOTRON_3_5_SDG_GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("NEMOTRON_3_5_SDG_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = OmegaConf.create({"sdg_workflow": "multimodal", "source_directory": {"path": "documents"}})
    captured = {}
    monkeypatch.setattr(sdg_module, "parse_config", lambda *_: config)
    monkeypatch.setattr(sdg_module, "parse_env", lambda *_: {})
    monkeypatch.setattr(
        sdg_module,
        "build_job_config",
        lambda train, *_args, **_kwargs: OmegaConf.merge(train, {"run": {"env": {}}}),
    )
    monkeypatch.setattr(
        sdg_module,
        "display_job_config",
        lambda job, **_kwargs: captured.update(preview=OmegaConf.to_container(job, resolve=True)),
    )
    monkeypatch.setattr(sdg_module, "generate_job_dir", lambda *_: tmp_path)
    monkeypatch.setattr(sdg_module, "extract_train_config", lambda job, **_kwargs: job)
    monkeypatch.setattr(sdg_module, "build_env_vars", lambda *_: {})
    monkeypatch.setattr(sdg_module, "display_job_submission", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sdg_module,
        "_execute_uv_local",
        lambda path, _: captured.update(saved=OmegaConf.to_container(OmegaConf.load(path), resolve=True)),
    )
    cfg = SimpleNamespace(
        ctx=object(), mode="local", argv=[], dry_run=dry_run, passthrough=[], attached=False, force_squash=False
    )
    sdg_module._execute_sdg(cfg)

    assert ("saved" in captured) is not dry_run
    for values in captured.values():
        assert values["sdg_generator_model"] == values["sdg_judge_model"] == (
            "nvidia/nemotron-3.5-super-vl-120b-a12b"
        )
        assert "qa_generation_model" not in values
        assert "quality_judge_model" not in values


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
