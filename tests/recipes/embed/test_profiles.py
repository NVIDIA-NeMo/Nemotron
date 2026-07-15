"""Regression coverage for the embed model profiles."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_runspec.config import load_config
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig
from nemotron.recipes.embed.stage2_finetune.train import FinetuneConfig
from nemotron.recipes.embed.stage3_eval.eval import EvalConfig, evaluate_model
from nemotron.recipes.embed.stage4_export.export import ExportConfig
from nemotron.recipes.embed.stage5_deploy.deploy import DeployConfig

from .conftest import REPO_ROOT

BASE_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
TEST_SDG_API_BASE_URL = "https://example.invalid/v1"
NIM_IMAGE = "example.invalid/nim/retriever-embed:test"
AUTOMODEL_IMAGE = "nvcr.io/nvidia/nemo-automodel:26.06"
CONFIG_ROOT = REPO_ROOT / "src/nemotron/recipes/embed"

PROFILE_CONFIGS = [
    ("stage0_sdg", SDGConfig),
    ("stage1_data_prep", DataPrepConfig),
    ("stage2_finetune", FinetuneConfig),
    ("stage3_eval", EvalConfig),
    ("stage4_export", ExportConfig),
    ("stage5_deploy", DeployConfig),
]


def _load_default_config(stage_dir: str, model_cls: type):
    config_path = CONFIG_ROOT / stage_dir / "config/default.yaml"
    config = load_config(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.pop("run", None)
    return model_cls(**config_dict)


@pytest.fixture(autouse=True)
def profile_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_BASE_URL", TEST_SDG_API_BASE_URL)
    monkeypatch.setenv("NEMOTRON3_EMBED_NIM_IMAGE", NIM_IMAGE)
    monkeypatch.delenv("NEMOTRON3_EMBED_NIM_MODEL", raising=False)
    monkeypatch.delenv("NEMOTRON3_EMBED_EXPORT_CHECKPOINT", raising=False)
    monkeypatch.delenv("NEMOTRON3_EMBED_DEPLOY_CHECKPOINT", raising=False)


def test_public_default_uses_optional_endpoint_and_requires_image_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_runspec.config import load_pydantic_config

    monkeypatch.delenv("NVIDIA_API_BASE_URL")
    monkeypatch.delenv("NEMOTRON3_EMBED_NIM_IMAGE")

    sdg = load_pydantic_config(
        CONFIG_ROOT / "stage0_sdg/config/default.yaml",
        [],
        SDGConfig,
    )
    assert sdg.nvidia_api_base_url is None
    with pytest.raises(ValueError, match="NEMOTRON3_EMBED_NIM_IMAGE"):
        load_pydantic_config(
            CONFIG_ROOT / "stage5_deploy/config/default.yaml",
            [],
            DeployConfig,
        )

    vllm = load_pydantic_config(
        CONFIG_ROOT / "stage5_deploy/config/default.yaml",
        ["backend=vllm"],
        DeployConfig,
    )
    assert vllm.backend == "vllm"
    assert vllm.container_image == "nvcr.io/nvidia/vllm:26.06-py3"


@pytest.mark.parametrize("stage_dir,model_cls", PROFILE_CONFIGS)
def test_default_config_inherits_and_validates(stage_dir: str, model_cls: type) -> None:
    config = _load_default_config(stage_dir, model_cls)
    assert config is not None


def test_default_base_model_is_shared_across_model_stages() -> None:
    prep = _load_default_config("stage1_data_prep", DataPrepConfig)
    finetune = _load_default_config("stage2_finetune", FinetuneConfig)
    evaluate = _load_default_config("stage3_eval", EvalConfig)
    export = _load_default_config("stage4_export", ExportConfig)

    assert prep.base_model == BASE_MODEL
    assert prep.trust_remote_code is True
    assert prep.output_dir == Path("output/embed/nemotron-3-1b/stage1_data_prep")
    assert finetune.base_model == BASE_MODEL
    assert finetune.trust_remote_code is True
    assert finetune.flash_adamw_master_weight_bits is None
    assert finetune.train_data_path == Path(
        "output/embed/nemotron-3-1b/stage1_data_prep/train_mined.automodel_unrolled.json"
    )
    assert finetune.checkpoint_dir == Path("output/embed/nemotron-3-1b/stage2_finetune/checkpoints")
    assert finetune.auto_scale_checkpoint_intervals is False
    assert finetune.checkpoint_every_steps == 1000
    assert evaluate.base_model == BASE_MODEL
    assert evaluate.eval_data_path == Path("output/embed/nemotron-3-1b/stage1_data_prep/eval_beir")
    assert evaluate.finetuned_model_path == Path(
        "output/embed/nemotron-3-1b/stage2_finetune/checkpoints/LATEST/model/consolidated"
    )
    assert export.model_path == Path(
        "output/embed/nemotron-3-1b/stage2_finetune/checkpoints/LATEST/model/consolidated"
    )
    assert export.trust_remote_code is True

@pytest.mark.parametrize("profile", ["default", "llama"])
def test_finetune_profiles_use_current_automodel_image(profile: str) -> None:
    config = load_config(CONFIG_ROOT / f"stage2_finetune/config/{profile}.yaml")

    assert config.run.env.container == AUTOMODEL_IMAGE



def test_default_sdg_uses_configured_api() -> None:
    sdg = _load_default_config("stage0_sdg", SDGConfig)

    assert sdg.nvidia_api_base_url == TEST_SDG_API_BASE_URL
    assert sdg.artifact_extraction_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert sdg.qa_generation_model == sdg.artifact_extraction_model
    assert sdg.quality_judge_model == sdg.artifact_extraction_model
    assert sdg.embed_model == "nvidia/llama-nemotron-embed-1b-v2"


def test_default_nim_identity_is_shared_by_eval_and_deploy() -> None:
    evaluate = _load_default_config("stage3_eval", EvalConfig)
    deploy = _load_default_config("stage5_deploy", DeployConfig)

    assert evaluate.nim_model == "nvidia/nemotron-3-embed-1b"
    assert evaluate.nim_invalid_embedding_retries == 32
    assert deploy.nim_model == evaluate.nim_model
    assert deploy.backend == "nim"
    assert deploy.nim_image == NIM_IMAGE
    assert deploy.vllm_image == "nvcr.io/nvidia/vllm:26.06-py3"
    assert deploy.model_dir == Path("output/embed/nemotron-3-1b/stage2_finetune/checkpoints/LATEST/model/consolidated")
    assert deploy.model_path_env == "NIM_ENGINE_MODEL_PATH"
    assert deploy.use_onnx is False
    assert deploy.expected_model_fingerprint == {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "intermediate_size": 6144,
        "vocab_size": 131072,
    }
    assert deploy.container_model_path == "/model"
    assert deploy.container_cache_path == "/opt/cache"
    assert deploy.max_seq_len is None
    assert deploy.pipeline_id is None
    assert deploy.shm_size == "16gb"
    assert deploy.health_check_timeout == 600
    assert deploy.forward_ngc_api_key is False


def test_python_defaults_use_spaced_prompts() -> None:
    prep = DataPrepConfig()
    finetune = FinetuneConfig()
    evaluate = EvalConfig()
    parameters = inspect.signature(evaluate_model).parameters

    assert prep.query_prefix == "query: "
    assert prep.passage_prefix == "passage: "
    assert finetune.query_prefix == "query: "
    assert finetune.passage_prefix == "passage: "
    assert evaluate.batch_size == 4
    assert evaluate.query_prefix == "query: "
    assert evaluate.passage_prefix == "passage: "
    assert parameters["batch_size"].default == 4
    assert parameters["query_prefix"].default == "query: "
    assert parameters["passage_prefix"].default == "passage: "


def _load_profile_config(profile: str, stage_dir: str, model_cls: type):
    config_path = CONFIG_ROOT / stage_dir / f"config/{profile}.yaml"
    config = load_config(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.pop("run", None)
    return model_cls(**config_dict)


@pytest.mark.parametrize("stage_dir,model_cls", PROFILE_CONFIGS)
@pytest.mark.parametrize("profile", ["default", "llama"])
def test_default_and_llama_profiles_validate(profile: str, stage_dir: str, model_cls: type) -> None:
    config = _load_profile_config(profile, stage_dir, model_cls)
    expected_root = Path("output/embed/nemotron-3-1b" if profile == "default" else "output/embed")
    assert config.artifact_root == expected_root


def test_artifact_root_override_rehomes_default_pipeline() -> None:
    from nemo_runspec.config import load_pydantic_config

    artifact_root = Path("/tmp/nemotron-3-8b")
    overrides = [f"artifact_root={artifact_root}"]

    def load(stage: str, model: type):
        return load_pydantic_config(
            CONFIG_ROOT / stage / "config/default.yaml",
            overrides,
            model,
        )

    sdg = load("stage0_sdg", SDGConfig)
    prep = load("stage1_data_prep", DataPrepConfig)
    finetune = load("stage2_finetune", FinetuneConfig)
    evaluate = load("stage3_eval", EvalConfig)
    export = load("stage4_export", ExportConfig)
    deploy = load("stage5_deploy", DeployConfig)

    assert sdg.artifact_root == artifact_root
    assert sdg.output_dir == artifact_root / "stage0_sdg"
    assert sdg.artifact_path == artifact_root / "stage0_sdg/artifacts"
    assert prep.sdg_input_path == sdg.output_dir
    assert prep.output_dir == artifact_root / "stage1_data_prep"
    assert finetune.train_data_path == prep.output_dir / "train_mined.automodel_unrolled.json"
    assert finetune.checkpoint_dir == artifact_root / "stage2_finetune/checkpoints"
    assert evaluate.eval_data_path == prep.output_dir / "eval_beir"
    assert evaluate.finetuned_model_path == finetune.checkpoint_dir / "LATEST/model/consolidated"
    assert evaluate.output_dir == artifact_root / "stage3_eval"
    assert export.model_path == evaluate.finetuned_model_path
    assert export.output_dir == artifact_root / "stage4_export"
    assert deploy.model_dir == evaluate.finetuned_model_path


def test_default_profile_is_ministral_with_direct_checkpoint_deploy() -> None:
    sdg = _load_profile_config("default", "stage0_sdg", SDGConfig)
    prep = _load_profile_config("default", "stage1_data_prep", DataPrepConfig)
    finetune = _load_profile_config("default", "stage2_finetune", FinetuneConfig)
    evaluate = _load_profile_config("default", "stage3_eval", EvalConfig)
    export = _load_profile_config("default", "stage4_export", ExportConfig)
    deploy = _load_profile_config("default", "stage5_deploy", DeployConfig)

    assert sdg.output_dir == Path("output/embed/nemotron-3-1b/stage0_sdg")
    assert sdg.artifact_extraction_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert prep.base_model == BASE_MODEL
    assert prep.sdg_input_path == sdg.output_dir
    assert finetune.base_model == BASE_MODEL
    assert prep.query_prefix == "query: "
    assert prep.passage_prefix == "passage: "
    assert finetune.query_prefix == "query: "
    assert finetune.passage_prefix == "passage: "
    assert finetune.flash_adamw_master_weight_bits is None
    assert finetune.auto_scale_checkpoint_intervals is False
    assert evaluate.base_model == BASE_MODEL
    assert evaluate.batch_size == 4
    assert evaluate.query_prefix == "query: "
    assert evaluate.passage_prefix == "passage: "
    assert evaluate.nim_model == "nvidia/nemotron-3-embed-1b"
    assert export.enabled is False
    assert deploy.nim_image == NIM_IMAGE
    assert deploy.backend == "nim"
    assert deploy.model_path_env == "NIM_ENGINE_MODEL_PATH"
    assert deploy.use_onnx is False
    assert deploy.model_dir == finetune.checkpoint_dir / "LATEST/model/consolidated"


def test_llama_profile_preserves_export_and_nim_contract() -> None:
    sdg = _load_profile_config("llama", "stage0_sdg", SDGConfig)
    prep = _load_profile_config("llama", "stage1_data_prep", DataPrepConfig)
    finetune = _load_profile_config("llama", "stage2_finetune", FinetuneConfig)
    evaluate = _load_profile_config("llama", "stage3_eval", EvalConfig)
    export = _load_profile_config("llama", "stage4_export", ExportConfig)
    deploy = _load_profile_config("llama", "stage5_deploy", DeployConfig)

    assert sdg.output_dir == Path("output/embed/stage0_sdg")
    assert sdg.nvidia_api_base_url is None
    assert sdg.artifact_extraction_model == "nvidia/nemotron-3-nano-30b-a3b"
    assert prep.base_model == "nvidia/llama-nemotron-embed-1b-v2"
    assert prep.output_dir == Path("output/embed/stage1_data_prep")
    assert finetune.base_model == prep.base_model
    assert prep.query_prefix == "query: "
    assert prep.passage_prefix == "passage: "
    assert finetune.query_prefix == "query: "
    assert finetune.passage_prefix == "passage: "
    assert finetune.flash_adamw_master_weight_bits == 32
    assert finetune.auto_scale_checkpoint_intervals is True
    assert evaluate.base_model == prep.base_model
    assert evaluate.batch_size == 128
    assert evaluate.query_prefix == "query: "
    assert evaluate.passage_prefix == "passage: "
    assert evaluate.nim_model == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert evaluate.nim_invalid_embedding_retries == 3
    assert export.enabled is True
    assert export.model_path == Path("output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated")
    assert deploy.nim_image == "nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.1"
    assert deploy.backend == "nim"
    assert deploy.model_path_env == "NIM_CUSTOM_MODEL"
    assert deploy.use_onnx is True
    assert deploy.expected_model_fingerprint is None
    assert deploy.container_model_path == "/opt/nim/custom_model"
    assert deploy.max_seq_len is None
    assert deploy.pipeline_id is None
    assert deploy.forward_ngc_api_key is True
    assert deploy.model_dir == export.onnx_export_path


def test_disabled_export_returns_checkpoint_without_loading_model(tmp_path: Path) -> None:
    from nemotron.recipes.embed.stage4_export.export import run_export

    checkpoint = tmp_path / "checkpoint-does-not-need-to-exist-for-a-skipped-stage"
    result = run_export(ExportConfig(enabled=False, model_path=checkpoint))

    assert result == {"model_path": str(checkpoint), "skipped": True}


@pytest.mark.parametrize("stage_dir,model_cls", PROFILE_CONFIGS)
@pytest.mark.parametrize("profile", ["default", "llama"])
def test_self_contained_profiles_load_in_direct_stage_runner(profile: str, stage_dir: str, model_cls: type) -> None:
    from nemo_runspec.config import load_pydantic_config

    config_path = CONFIG_ROOT / stage_dir / f"config/{profile}.yaml"
    assert load_pydantic_config(config_path, [], model_cls) is not None


def test_enabled_llama_export_branch(monkeypatch, tmp_path: Path) -> None:
    from nemotron.recipes.embed.stage4_export import export as export_module

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls = []
    monkeypatch.setattr(
        export_module,
        "load_embedding_model",
        lambda **kwargs: ("model", "tokenizer"),
    )
    monkeypatch.setattr(
        export_module,
        "export_to_onnx",
        lambda model, tokenizer, cfg: calls.append((model, tokenizer)) or "onnx-exporter",
    )
    monkeypatch.setattr(
        export_module,
        "verify_onnx_export",
        lambda exporter, cfg: calls.append((exporter, cfg.enabled)),
    )
    cfg = ExportConfig(
        enabled=True,
        model_path=checkpoint,
        output_dir=tmp_path / "output",
        onnx_export_path=tmp_path / "onnx",
        trt_model_path=tmp_path / "trt",
        export_to_trt=False,
    )

    result = export_module.run_export(cfg)

    assert result == {"model_path": str(checkpoint), "onnx_path": str(tmp_path / "onnx")}
    assert calls == [("model", "tokenizer"), ("onnx-exporter", True)]
