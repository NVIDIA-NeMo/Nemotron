"""Unit tests for rerank recipe config and lightweight helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemotron.recipes.rerank.stage2_finetune.train import (
    FinetuneConfig,
    _auto_scale_hyperparams,
)
from nemotron.recipes.rerank.stage3_eval import eval as eval_module
from nemotron.recipes.rerank.stage3_eval.eval import EvalConfig
from nemotron.recipes.rerank.stage4_export.export import ExportConfig
from nemotron.recipes.rerank.stage5_deploy.deploy import DeployConfig, _api_base_url, _resolve_manifest_path


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


def test_deploy_defaults_to_stage4_manifest_and_safe_replace():
    cfg = DeployConfig()
    assert cfg.model_dir is not None
    assert _resolve_manifest_path(cfg) == cfg.model_dir / "model_manifest.yaml"
    assert cfg.replace_existing is False
    assert cfg.keep_failed_container is False


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
