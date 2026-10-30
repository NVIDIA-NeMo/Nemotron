# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemotron.recipes.rerank.stage5_deploy import deploy


def _profile(**overrides):
    path = deploy.STAGE_PATH / "config/nemotron-3.5.yaml"
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return deploy.DeployConfig(**(config | overrides))


@pytest.mark.parametrize("local_checkpoint", [False, True])
def test_vllm_launch_preserves_native_scoring_and_model_selection(tmp_path, monkeypatch, local_checkpoint):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_TOKEN", "secret-test-value")
    cfg = _profile(model_dir=tmp_path / "checkpoint" if local_checkpoint else None, detach=True)

    cmd, env = deploy.build_docker_command(cfg)

    assert cmd[cmd.index("--entrypoint") + 1] == "python3"
    assert cmd[cmd.index("--runner") + 1] == "pooling"
    assert cmd[cmd.index("--served-model-name") + 1] == cfg.vllm_model
    assert json.loads(cmd[cmd.index("--pooler-config") + 1]) == {"seq_pooling_type": "MEAN", "use_activation": False}
    assert "--convert" not in cmd and "--hf-overrides" not in cmd
    assert "--rm" not in cmd  # Detached failures retain logs until cleanup.
    assert "HF_TOKEN" in cmd and "secret-test-value" not in " ".join(cmd)
    assert env["HF_TOKEN"] == "secret-test-value"
    assert f"XDG_CACHE_HOME={cfg.vllm_container_cache_path}" in cmd
    assert f"XDG_CONFIG_HOME={cfg.vllm_container_cache_path}/config" in cmd
    assert f"FLASHINFER_WORKSPACE_BASE={cfg.vllm_container_cache_path}" in cmd
    assert f"TRITON_CACHE_DIR={cfg.vllm_container_cache_path}/triton" in cmd
    assert f"CUDA_CACHE_PATH={cfg.vllm_container_cache_path}/cuda" in cmd
    launch_index = cmd.index(deploy._VLLM_LAUNCH)
    if local_checkpoint:
        assert cmd[launch_index + 1 : launch_index + 3] == ["/model", ""]
        assert f"{cfg.model_dir.resolve()}:/model:ro" in cmd
        assert "--revision" not in cmd
    else:
        assert cmd[launch_index + 1 : launch_index + 3] == [cfg.vllm_model, cfg.model_revision]
        assert cmd[cmd.index("--revision") + 1] == cfg.model_revision
    assert all("NIM_" not in argument for argument in cmd)


def test_container_bootstrap_uses_checkpoint_template_and_structured_argv(tmp_path, monkeypatch):
    expected_template = "{{ messages[0].content }} {{ messages[1].content }}"
    calls = {}

    class Tokenizer:
        @staticmethod
        def from_pretrained(model, *, revision):
            calls["tokenizer"] = (model, revision)
            return types.SimpleNamespace(get_chat_template=lambda: expected_template)

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer))
    monkeypatch.setattr(sys, "argv", ["-c", "model with spaces", "exact-revision", "--served-model-name", "alias"])
    monkeypatch.setattr(deploy.os, "execvp", lambda executable, args: calls.update(executable=executable, args=args))
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    exec(deploy._VLLM_LAUNCH, {})

    assert calls["tokenizer"] == ("model with spaces", "exact-revision")
    assert calls["executable"] == "vllm"
    argv = calls["args"]
    assert argv[:3] == ["vllm", "serve", "model with spaces"]
    template_path = Path(argv[argv.index("--chat-template") + 1])
    try:
        assert template_path.read_text() == expected_template
    finally:
        template_path.unlink()
    assert argv[-2:] == ["--served-model-name", "alias"]


@pytest.mark.parametrize("backend,expected_path", [("nim", "/v1/health/ready"), ("vllm", "/health")])
def test_health_uses_backend_route(monkeypatch, backend, expected_path):
    import urllib.request

    urls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def urlopen(url, **kwargs):
        urls.append(url)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert deploy.wait_for_health(deploy.DeployConfig(backend=backend))
    assert urls == [f"http://127.0.0.1:8000{expected_path}"]


def test_incomplete_checkpoint_fails_before_replacing_service(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "check_docker", lambda: True)
    monkeypatch.setattr(deploy, "check_nvidia_docker", lambda: True)
    monkeypatch.setattr(deploy, "stop_existing_container", lambda *args: pytest.fail("replaced existing service"))

    with pytest.raises(SystemExit) as error:
        deploy.run_deploy(_profile(model_dir=tmp_path))
    assert error.value.code == 1


def test_detached_vllm_reports_classify_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(deploy, "check_docker", lambda: True)
    monkeypatch.setattr(deploy, "check_nvidia_docker", lambda: True)
    monkeypatch.setattr(deploy, "stop_existing_container", lambda *args: None)
    monkeypatch.setattr(deploy, "wait_for_health", lambda cfg: True)
    monkeypatch.setattr(
        deploy.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="id")
    )

    result = deploy.run_deploy(_profile(model_dir=None, detach=True))

    assert result["api_url"] == "http://127.0.0.1:8000/classify"
    output = capsys.readouterr().out
    assert '"add_special_tokens": false' in output
    assert '"use_activation": false' in output
    assert '"role": "query"' in output and '"role": "document"' in output


@pytest.mark.parametrize("model_override", [[], ["model_dir=null"]])
def test_profile_cli_preview_never_starts_process(monkeypatch, model_override):
    from typer.testing import CliRunner

    from nemotron.cli.bin.nemotron import app

    args = ["rerank", "deploy", "-c", "nemotron-3.5", "--dry-run", *model_override]
    monkeypatch.setattr(sys, "argv", ["nemotron", *args])
    monkeypatch.setattr(deploy.subprocess, "run", lambda *args, **kwargs: pytest.fail("preview started a process"))

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "rerank/deploy" in result.output


def test_health_fails_promptly_when_container_exits(monkeypatch):
    import urllib.request

    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError("server exited")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(
        deploy.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="false")
    )
    monkeypatch.setattr(deploy.time, "sleep", lambda _: pytest.fail("waited after container exited"))

    assert not deploy.wait_for_health(deploy.DeployConfig(backend="vllm"))
