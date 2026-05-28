"""Unit tests for rerank recipe config and lightweight helpers."""

from __future__ import annotations

import urllib.request

import pytest
from pydantic import ValidationError

from nemotron.recipes.rerank.stage2_finetune.train import (
    FinetuneConfig,
    _auto_scale_hyperparams,
)
from nemotron.recipes.rerank.stage3_eval import eval as eval_module
from nemotron.recipes.rerank.stage3_eval.eval import EvalConfig
from nemotron.recipes.rerank.stage4_export import export as export_module
from nemotron.recipes.rerank.stage4_export.export import ExportConfig
from nemotron.recipes.rerank.stage5_deploy import deploy as deploy_module
from nemotron.recipes.rerank.stage5_deploy.deploy import DeployConfig, _api_base_url, build_docker_command


def test_finetune_auto_scale_preserves_global_batch_size():
    cfg = FinetuneConfig(global_batch_size=128, checkpoint_every_steps=100, val_every_steps=100)
    global_batch_size, *_ = _auto_scale_hyperparams(cfg, num_examples=500)
    assert global_batch_size == 128


def test_finetune_rejects_untrusted_remote_code_without_opt_in():
    with pytest.raises(ValidationError, match="allow_untrusted_remote_code"):
        FinetuneConfig(base_model="example/custom-reranker")


def test_eval_rejects_untrusted_remote_code_without_opt_in():
    with pytest.raises(ValidationError, match="allow_untrusted_remote_code"):
        EvalConfig(base_model="example/custom-reranker")


def test_eval_allows_untrusted_remote_code_with_explicit_opt_in():
    cfg = EvalConfig(
        base_model="example/custom-reranker",
        retrieval_model="example/custom-embedder",
        allow_untrusted_remote_code=True,
    )
    assert cfg.allow_untrusted_remote_code is True


def test_eval_rejects_metrics_beyond_reranked_top_k():
    with pytest.raises(ValidationError, match="top_k"):
        EvalConfig(top_k=10, k_values=[1, 5, 100])


def test_eval_rejects_custom_prompt_template_for_nim_compare():
    with pytest.raises(ValidationError, match="default NIM prompt template"):
        EvalConfig(eval_nim=True, prompt_template="Q: {query} P: {passage}")


def test_export_defaults_format_reranker_calibration_pairs():
    cfg = ExportConfig()
    text = cfg.prompt_template.format(query=cfg.calibration_query, passage="A passage about GPUs")
    assert "question:" in text
    assert "passage:A passage about GPUs" in text


@pytest.mark.parametrize(
    ("bind_address", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:8000"),
        ("0.0.0.0", "http://localhost:8000"),
        ("::", "http://[::1]:8000"),
        ("::1", "http://[::1]:8000"),
    ],
)
def test_deploy_api_base_url_uses_bind_address(bind_address, expected):
    assert _api_base_url(DeployConfig(bind_address=bind_address)) == expected


def test_finetune_rejects_multi_label_score_head():
    with pytest.raises(ValidationError):
        FinetuneConfig(num_labels=2)


def test_deploy_mounts_custom_model_dir_and_safe_replace(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cfg = DeployConfig(model_dir=tmp_path)

    cmd, _ = build_docker_command(cfg)

    assert ["-v", f"{tmp_path.resolve()}:{cfg.container_model_path}:ro"] == cmd[
        cmd.index("-v") : cmd.index("-v") + 2
    ]
    assert f"NIM_CUSTOM_MODEL={cfg.container_model_path}" in cmd
    assert all("NIM_MANIFEST_PATH" not in item for item in cmd)
    assert cfg.nim_image == "nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0"
    assert cfg.replace_existing is False
    assert cfg.keep_failed_container is False


def test_deploy_health_retries_transient_socket_reset(monkeypatch):
    cfg = DeployConfig(health_check_interval=1)
    calls = {"count": 0}

    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def flaky_urlopen(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionResetError("startup reset")
        return HealthyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(deploy_module.time, "sleep", lambda _: None)

    assert deploy_module.wait_for_health(cfg) is True
    assert calls["count"] == 2


def test_export_failed_onnx_verification_exits_nonzero(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg = ExportConfig(
        model_path=model_path,
        output_dir=tmp_path / "export",
        onnx_export_path=tmp_path / "export" / "onnx",
        export_to_trt=False,
    )

    monkeypatch.setattr(export_module, "load_reranker_model", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(export_module, "export_to_onnx", lambda *args, **kwargs: object())
    monkeypatch.setattr(export_module, "verify_onnx_export", lambda exporter: False)

    with pytest.raises(SystemExit) as exc_info:
        export_module.run_export(cfg)
    assert exc_info.value.code == 1


def test_eval_nim_unreachable_exits_nonzero(tmp_path, monkeypatch):
    cfg = EvalConfig(
        eval_base=False,
        eval_finetuned=False,
        eval_nim=True,
        eval_data_path=tmp_path,
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(eval_module, "_get_first_stage_results", lambda **kwargs: ({}, {}, {}, {}))

    def fail_urlopen(*args, **kwargs):
        raise eval_module.urllib.error.URLError("unreachable")

    monkeypatch.setattr(eval_module.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(SystemExit) as exc_info:
        eval_module.run_eval(cfg)
    assert exc_info.value.code == 1


def test_eval_nim_requires_finetuned_checkpoint_for_comparison(tmp_path, monkeypatch):
    cfg = EvalConfig(
        eval_base=False,
        eval_finetuned=True,
        eval_nim=True,
        eval_data_path=tmp_path,
        finetuned_model_path=tmp_path / "missing-model",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(eval_module, "_get_first_stage_results", lambda **kwargs: ({}, {}, {}, {}))

    with pytest.raises(SystemExit) as exc_info:
        eval_module.run_eval(cfg)
    assert exc_info.value.code == 1


def test_eval_nim_metric_mismatch_exits_nonzero(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg = EvalConfig(
        eval_base=False,
        eval_finetuned=True,
        eval_nim=True,
        eval_data_path=tmp_path,
        finetuned_model_path=model_path,
        output_dir=tmp_path / "out",
        top_k=5,
        k_values=[1],
    )
    metrics = ({"NDCG@1": 0.5}, {"MAP@1": 0.5}, {"Recall@1": 0.5}, {"Precision@1": 0.5})
    nim_metrics = ({"NDCG@1": 0.9}, {"MAP@1": 0.5}, {"Recall@1": 0.9}, {"Precision@1": 0.5})
    monkeypatch.setattr(eval_module, "_get_first_stage_results", lambda **kwargs: ({}, {}, {}, {}))
    monkeypatch.setattr(eval_module, "evaluate_reranker", lambda **kwargs: (metrics, {}))
    monkeypatch.setattr(eval_module, "evaluate_nim_reranker", lambda **kwargs: (nim_metrics, {}))

    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(eval_module.urllib.request, "urlopen", lambda *args, **kwargs: HealthyResponse())

    with pytest.raises(SystemExit) as exc_info:
        eval_module.run_eval(cfg)
    assert exc_info.value.code == 1
