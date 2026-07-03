"""Regression tests for safe embed NIM deployment."""

from __future__ import annotations

import urllib.request
from types import SimpleNamespace

import pytest

from nemotron.recipes.embed.stage5_deploy import deploy

TEST_NIM_IMAGE = "example.invalid/nim:test"


def _deploy_config(**kwargs) -> deploy.DeployConfig:
    return deploy.DeployConfig(nim_image=TEST_NIM_IMAGE, **kwargs)


def test_docker_command_does_not_contain_ngc_secret(monkeypatch, tmp_path) -> None:
    secret = "secret-that-must-not-appear-in-argv"
    monkeypatch.setenv("CUSTOM_NGC_KEY", secret)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cfg = _deploy_config(
        model_dir=model_dir,
        ngc_api_key_env="CUSTOM_NGC_KEY",
        forward_ngc_api_key=True,
    )
    command = deploy.build_docker_command(cfg)
    docker_env = deploy.build_docker_environment(cfg)

    assert secret not in command
    env_index = command.index("-e")
    assert command[env_index : env_index + 2] == ["-e", "NGC_API_KEY"]
    assert docker_env["NGC_API_KEY"] == secret


def test_nim_model_path_mounts_huggingface_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()

    cfg = _deploy_config(
        nim_model="nvidia/nemotron-3-embed-1b",
        model_dir=model_dir,
        model_path_env="NIM_MODEL_PATH",
        container_model_path="/model",
        container_cache_path="/opt/cache",
        max_seq_len=512,
        pipeline_id="padded-naive-fp16",
    )
    command = deploy.build_docker_command(cfg)

    assert "NIM_MODEL_NAME=nvidia/nemotron-3-embed-1b" in command
    assert "NIM_MODEL_PATH=/model" in command
    assert "NIM_MAX_SEQ_LEN=512" in command
    assert "NIM_PIPELINE_ID=padded-naive-fp16" in command
    assert not any(arg.startswith("NIM_CUSTOM_MODEL=") for arg in command)
    assert "NGC_API_KEY" not in command
    assert f"{model_dir.resolve()}:/model:ro" in command
    assert f"{tmp_path / 'cache' / 'nim'}:/opt/cache" in command


def test_huggingface_checkpoint_artifact_validation(tmp_path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    cfg = _deploy_config(
        model_dir=model_dir,
        model_path_env="NIM_MODEL_PATH",
        expected_model_fingerprint=None,
    )

    assert deploy.model_artifact_errors(cfg) == [
        "missing config.json",
        "missing *.safetensors",
        "missing tokenizer.json or tokenizer_config.json",
    ]

    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").touch()
    (model_dir / "tokenizer_config.json").write_text("{}")

    assert deploy.model_artifact_errors(cfg) == []


def test_huggingface_checkpoint_fingerprint_validation(tmp_path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 16}')
    (model_dir / "model.safetensors").touch()
    (model_dir / "tokenizer.json").write_text("{}")
    cfg = _deploy_config(
        model_dir=model_dir,
        model_path_env="NIM_MODEL_PATH",
        expected_model_fingerprint={"num_hidden_layers": 18},
    )

    assert deploy.model_artifact_errors(cfg) == ["config.json num_hidden_layers=16 (expected 18)"]


def test_health_check_retries_connection_reset(monkeypatch) -> None:
    calls = 0

    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(url, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("NIM is still starting")
        return HealthyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(deploy.time, "sleep", lambda seconds: None)

    cfg = _deploy_config(
        health_check_timeout=10,
        health_check_interval=1,
    )

    assert deploy.wait_for_health(cfg) is True
    assert calls == 2


def test_llama_docker_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("CUSTOM_NGC_KEY", "llama-secret")
    model_dir = tmp_path / "onnx"
    model_dir.mkdir()
    (model_dir / "model.onnx").touch()

    cfg = _deploy_config(
        nim_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        model_dir=model_dir,
        model_path_env="NIM_CUSTOM_MODEL",
        use_onnx=True,
        container_model_path="/opt/nim/custom_model",
        ngc_api_key_env="CUSTOM_NGC_KEY",
        forward_ngc_api_key=True,
    )
    command = deploy.build_docker_command(cfg)

    assert "NIM_CUSTOM_MODEL=/opt/nim/custom_model" in command
    assert not any("NIM_MODEL_PATH=" in argument for argument in command)
    assert "NGC_API_KEY" in command
    assert f"{model_dir.resolve()}:/opt/nim/custom_model:ro" in command
    assert deploy.model_artifact_errors(cfg) == []


def test_detached_health_timeout_raises(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cfg = _deploy_config(model_dir=model_dir, detach=True)

    monkeypatch.setattr(deploy, "check_docker", lambda: True)
    monkeypatch.setattr(deploy, "check_nvidia_docker", lambda: True)
    monkeypatch.setattr(deploy, "model_artifact_errors", lambda cfg: [])
    monkeypatch.setattr(deploy, "stop_existing_container", lambda name: None)
    monkeypatch.setattr(deploy, "build_docker_command", lambda cfg: ["docker", "run"])
    monkeypatch.setattr(deploy, "build_docker_environment", lambda cfg: {})
    monkeypatch.setattr(deploy, "wait_for_health", lambda cfg: False)
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="container-id", stderr=""),
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        deploy.run_deploy(cfg)


def test_detached_health_timeout_can_be_explicitly_allowed(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cfg = _deploy_config(model_dir=model_dir, detach=True, allow_unhealthy_detached=True)

    monkeypatch.setattr(deploy, "check_docker", lambda: True)
    monkeypatch.setattr(deploy, "check_nvidia_docker", lambda: True)
    monkeypatch.setattr(deploy, "model_artifact_errors", lambda cfg: [])
    monkeypatch.setattr(deploy, "stop_existing_container", lambda name: None)
    monkeypatch.setattr(deploy, "build_docker_command", lambda cfg: ["docker", "run"])
    monkeypatch.setattr(deploy, "build_docker_environment", lambda cfg: {})
    monkeypatch.setattr(deploy, "wait_for_health", lambda cfg: False)
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="container-id", stderr=""),
    )

    result = deploy.run_deploy(cfg)

    assert result["container_id"] == "container-id"
