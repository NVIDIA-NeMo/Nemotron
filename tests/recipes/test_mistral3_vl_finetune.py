# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import tomllib
import yaml
from pydantic import ValidationError

from nemo_runspec.config import load_pydantic_config
from nemotron.kit.artifacts.embed import EmbedModelArtifact
from nemotron.recipes.embed.stage2_finetune import train as embed_train
from nemotron.recipes.embed.stage3_eval.eval import EvalConfig as EmbedEvalConfig
from nemotron.recipes.embed.stage5_deploy.deploy import DeployConfig as EmbedDeployConfig
from nemotron.recipes.rerank.stage2_finetune import train as rerank_train
from nemotron.recipes.rerank.stage3_eval.eval import EvalConfig as RerankEvalConfig
from nemotron.recipes.rerank.stage5_deploy.deploy import DeployConfig as RerankDeployConfig


def _namespace_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        dataset=SimpleNamespace(),
        dataloader=SimpleNamespace(dataset=SimpleNamespace(), collate_fn=SimpleNamespace()),
        step_scheduler=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        lr_scheduler=SimpleNamespace(),
        checkpoint=SimpleNamespace(),
    )


def _install_recipe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recipe_name: str,
    captured: dict[str, Any],
) -> None:
    class Recipe:
        def __init__(self, cfg: Any) -> None:
            captured["config"] = cfg

        def setup(self) -> None:
            pass

        def run_train_validation_loop(self) -> None:
            pass

    root = ModuleType("nemo_automodel")
    components = ModuleType("nemo_automodel.components")
    config = ModuleType("nemo_automodel.components.config")
    loader = ModuleType("nemo_automodel.components.config.loader")
    loader.ConfigNode = object
    recipes = ModuleType("nemo_automodel.recipes")
    retrieval = ModuleType("nemo_automodel.recipes.retrieval")
    setattr(retrieval, recipe_name, Recipe)
    for name, module in {
        "nemo_automodel": root,
        "nemo_automodel.components": components,
        "nemo_automodel.components.config": config,
        "nemo_automodel.components.config.loader": loader,
        "nemo_automodel.recipes": recipes,
        "nemo_automodel.recipes.retrieval": retrieval,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _mined_training_file(tmp_path: Path) -> Path:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {"path": "corpus/train"},
                "data": [
                    {
                        "question_id": "q1",
                        "question": "What is shown?",
                        "pos_doc": [{"id": "p1"}],
                        "neg_doc": [
                            {"id": "n1", "score": 0.3},
                            {"id": "n2", "score": 0.2},
                            {"id": "n3", "score": 0.1},
                        ],
                        "negative_mining_performed": True,
                        "relevance_judgements_complete": False,
                        "unlisted_document_disposition": "unjudged",
                    }
                ],
            }
        )
    )
    return path


def test_embed_vl_base_matches_automodel_mr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed_train, "_can_import_fused_adam", lambda: (True, None))
    monkeypatch.setattr(embed_train, "_can_import_flash_adamw", lambda: (False, "not installed"))
    cfg = embed_train.FinetuneConfig(model_family="mistral3_vl")

    raw, backend = embed_train._load_automodel_config(cfg, lambda value: value)

    assert backend == "fused_adam"
    assert raw["tokenizer"]["_target_"].endswith("Mistral3BiEncoderProcessor.from_pretrained")
    assert "query_prefix" not in raw["tokenizer"]
    assert "passage_prefix" not in raw["tokenizer"]
    assert "image_longest_edge" not in raw["tokenizer"]
    assert raw["dataset"]["_target_"].endswith("DataDesignerRetrievalDatasetConfig")
    assert raw["dataloader"]["collate_fn"]["collator_fn_name"] == "process_queries_documents_biencoder"


@pytest.mark.parametrize("train_module", [embed_train, rerank_train])
def test_vl_flash_adamw_uses_quantized_states_and_full_master_weights(
    monkeypatch: pytest.MonkeyPatch,
    train_module: ModuleType,
) -> None:
    monkeypatch.setattr(train_module, "_can_import_fused_adam", lambda: (False, "not installed"))
    monkeypatch.setattr(train_module, "_can_import_flash_adamw", lambda: (True, None))
    cfg = train_module.FinetuneConfig(model_family="mistral3_vl", optimizer_backend="flash_adamw")

    raw, backend = train_module._load_automodel_config(cfg, lambda value: value)

    assert backend == "flash_adamw"
    assert raw["optimizer"]["quantize"] is True
    assert raw["optimizer"]["compress_state_dict"] is False
    assert raw["optimizer"]["master_weight_bits"] == 32
    assert raw["model"]["torch_dtype"] == "bfloat16"


def test_embed_vl_export_metadata_is_vllm_compatible(tmp_path: Path) -> None:
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
                {"path": "1_Pooling", "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
                {
                    "path": "2_Normalize",
                    "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
                },
            ]
        )
    )
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 2048}}))
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "query: ", "document": "passage: "}})
    )
    cfg = embed_train.FinetuneConfig(
        model_family="mistral3_vl",
        pooling="avg",
        query_prefix="query: ",
        passage_prefix="passage: ",
    )

    embed_train._repair_vllm_sentence_transformers_metadata(tmp_path, cfg)

    modules = json.loads((tmp_path / "modules.json").read_text())
    pooling = json.loads((tmp_path / "1_Pooling/config.json").read_text())
    sentence_transformers = json.loads((tmp_path / "config_sentence_transformers.json").read_text())
    assert modules[1]["type"] == "sentence_transformers.models.Pooling"
    assert pooling["word_embedding_dimension"] == 2048
    assert pooling["pooling_mode_mean_tokens"] is True
    assert pooling["pooling_mode_lasttoken"] is False
    assert sentence_transformers["prompts"] == {"query": "query:", "document": "passage:"}


def test_rerank_vl_base_owns_temperature_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_train, "_can_import_fused_adam", lambda: (True, None))
    monkeypatch.setattr(rerank_train, "_can_import_flash_adamw", lambda: (False, "not installed"))
    cfg = rerank_train.FinetuneConfig(
        model_family="mistral3_vl",
        base_model="mistralai/Ministral-3-3B-Instruct-2512-BF16",
        allow_untrusted_remote_code=True,
    )

    raw, _ = rerank_train._load_automodel_config(cfg, lambda value: value)

    assert raw["temperature"] == 1.0
    assert raw["model"]["temperature"] == 0.02
    assert raw["dataloader"]["collate_fn"]["collator_fn_name"] == "process_queries_documents_crossencoder"


def test_rerank_rejects_two_non_unit_temperatures() -> None:
    with pytest.raises(ValidationError, match="At most one"):
        rerank_train.FinetuneConfig(temperature=0.5, recipe_temperature=0.5)


def test_embed_vl_runtime_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    base = _namespace_config()
    _install_recipe(monkeypatch, recipe_name="TrainBiEncoderRecipe", captured=captured)
    monkeypatch.setattr(embed_train, "_load_automodel_config", lambda cfg, node: (base, "fused_adam"))
    monkeypatch.setattr(embed_train, "_repair_vllm_sentence_transformers_metadata", lambda model_dir, cfg: None)
    cfg = embed_train.FinetuneConfig(
        model_family="mistral3_vl",
        base_model="mistralai/Ministral-3-3B-Instruct-2512-BF16",
        train_data_path=_mined_training_file(tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        train_n_passages=4,
        global_batch_size=8,
        local_batch_size=1,
        num_epochs=1,
        learning_rate=2.0e-6,
        lr_warmup_steps=0,
        attn_implementation="sdpa",
        passage_max_length=4096,
        use_text_in_document=True,
        require_mined_negatives=True,
        do_distributed_inbatch_negative=True,
        detach_distributed_inbatch_negatives=False,
    )

    embed_train.run_finetune(cfg)

    automodel = captured["config"]
    assert automodel.dataset.data_dir_list == [str(cfg.train_data_path)]
    assert automodel.dataset.use_text_in_document is True
    assert automodel.tokenizer.p_max_length == 4096
    assert automodel.tokenizer.query_prefix == "query:"
    assert automodel.model.do_distributed_inbatch_negative is True
    assert automodel.dataloader.collate_fn.__dict__ == {}


def test_embed_vl_runtime_omits_checkpoint_owned_processor_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    base = _namespace_config()
    _install_recipe(monkeypatch, recipe_name="TrainBiEncoderRecipe", captured=captured)
    monkeypatch.setattr(embed_train, "_load_automodel_config", lambda cfg, node: (base, "fused_adam"))
    monkeypatch.setattr(embed_train, "_repair_vllm_sentence_transformers_metadata", lambda model_dir, cfg: None)
    cfg = embed_train.FinetuneConfig(
        model_family="mistral3_vl",
        train_data_path=_mined_training_file(tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        train_n_passages=4,
        global_batch_size=8,
        local_batch_size=1,
        num_epochs=1,
        query_prefix=None,
        passage_prefix=None,
        image_longest_edge=None,
        require_mined_negatives=True,
    )

    embed_train.run_finetune(cfg)

    tokenizer = captured["config"].tokenizer
    assert "query_prefix" not in tokenizer.__dict__
    assert "passage_prefix" not in tokenizer.__dict__
    assert "image_longest_edge" not in tokenizer.__dict__


@pytest.mark.parametrize("profile", ["default", "mistral3-vl"])
def test_embed_exact_step_dotlist_reaches_native_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    monkeypatch.setenv("MISTRAL3_VL_EMBED_MODEL", "nvidia/test-mistral3-vl-embed")
    captured: dict[str, Any] = {}
    saved_artifacts: list[EmbedModelArtifact] = []
    base = _namespace_config()
    _install_recipe(monkeypatch, recipe_name="TrainBiEncoderRecipe", captured=captured)
    monkeypatch.setattr(embed_train, "_load_automodel_config", lambda cfg, node: (base, "fused_adam"))
    monkeypatch.setattr(embed_train, "_repair_vllm_sentence_transformers_metadata", lambda model_dir, cfg: None)
    monkeypatch.setattr(
        EmbedModelArtifact,
        "save",
        lambda self, name=None: saved_artifacts.append(self),
    )
    data = _mined_training_file(tmp_path)
    cfg = embed_train.load_config(
        embed_train.STAGE_PATH / "config" / f"{profile}.yaml",
        [
            "num_epochs=null",
            "max_steps=7",
            f"train_data_path={data}",
            f"checkpoint_dir={tmp_path / 'checkpoints'}",
            "attn_implementation=sdpa",
        ],
        embed_train.FinetuneConfig,
    )

    embed_train.run_finetune(cfg)

    automodel = captured["config"]
    assert cfg.num_epochs is None
    assert cfg.max_steps == 7
    assert automodel.step_scheduler.num_epochs is None
    assert automodel.step_scheduler.max_steps == 7
    assert len(saved_artifacts) == 1
    assert saved_artifacts[0].num_epochs is None
    assert saved_artifacts[0].max_steps == 7
    assert saved_artifacts[0].metadata["max_steps"] == 7


def test_embed_epoch_mode_explicitly_clears_native_max_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    base = _namespace_config()
    base.step_scheduler.max_steps = 999
    _install_recipe(monkeypatch, recipe_name="TrainBiEncoderRecipe", captured=captured)
    monkeypatch.setattr(embed_train, "_load_automodel_config", lambda cfg, node: (base, "fused_adam"))
    cfg = embed_train.FinetuneConfig(
        train_data_path=_mined_training_file(tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        num_epochs=2,
        attn_implementation="sdpa",
    )

    embed_train.run_finetune(cfg)

    automodel = captured["config"]
    assert automodel.step_scheduler.num_epochs == 2
    assert automodel.step_scheduler.max_steps is None


def test_rerank_vl_runtime_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    base = _namespace_config()
    _install_recipe(monkeypatch, recipe_name="TrainCrossEncoderRecipe", captured=captured)
    monkeypatch.setattr(rerank_train, "_load_automodel_config", lambda cfg, node: (base, "fused_adam"))
    cfg = rerank_train.FinetuneConfig(
        model_family="mistral3_vl",
        base_model="mistralai/Ministral-3-3B-Instruct-2512-BF16",
        allow_untrusted_remote_code=True,
        train_data_path=_mined_training_file(tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        train_n_passages=4,
        global_batch_size=8,
        local_batch_size=1,
        num_epochs=1,
        learning_rate=2.0e-6,
        lr_warmup_steps=0,
        attn_implementation="sdpa",
        rerank_max_length=4096,
        temperature=0.02,
        recipe_temperature=1.0,
        use_prompt_template=True,
        export_as_stock_processor=False,
        use_text_in_document=True,
        require_mined_negatives=True,
    )

    rerank_train.run_finetune(cfg)

    automodel = captured["config"]
    assert automodel.temperature == 1.0
    assert automodel.model.temperature == 0.02
    assert automodel.dataset.data_dir_list == [str(cfg.train_data_path)]
    assert automodel.tokenizer.rerank_max_length == 4096
    assert automodel.tokenizer.use_prompt_template is True
    assert automodel.tokenizer.export_as_stock_processor is False


def test_vl_stages_pin_their_selected_automodel_runtime() -> None:
    source = {
        "git": "https://github.com/NVIDIA-NeMo/Automodel.git",
        "rev": "4c50ab3c4c033f9e3f43b494b20bd88fe51b6b7f",
    }
    with (Path(rerank_train.__file__).parent / "pyproject.toml").open("rb") as stream:
        rerank_project = tomllib.load(stream)
    assert "transformers==5.15.1" in rerank_project["project"]["dependencies"]
    assert "torch>=2.10.0,<2.11.0" in rerank_project["project"]["dependencies"]
    assert "torchvision>=0.25.0,<0.26.0" in rerank_project["project"]["dependencies"]
    assert rerank_project["tool"]["uv"]["sources"]["nemo-automodel"] == source

    embed_root = Path(embed_train.__file__).parents[1]
    with (embed_root / "runtimes/native/pyproject.toml").open("rb") as stream:
        native_project = tomllib.load(stream)
    assert "nemo-automodel==0.7.0+aa6245ac" in native_project["project"]["dependencies"]
    assert "transformers==5.15.1" in native_project["project"]["dependencies"]
    assert native_project["tool"]["uv"]["sources"]["nemo-automodel"]["path"].endswith(
        "nemo_automodel-0.7.0+aa6245ac-py3-none-any.whl"
    )

    with (Path(embed_train.__file__).parent / "pyproject.toml").open("rb") as stream:
        legacy_project = tomllib.load(stream)
    assert "transformers==5.12.1" in legacy_project["project"]["dependencies"]
    assert legacy_project["tool"]["uv"]["sources"]["nemo-automodel"] == {
        "url": "https://github.com/NVIDIA-NeMo/Automodel/archive/a9f4423819c513fd08083324fe1f738746ac6e54.tar.gz"
    }


def test_public_profiles_declare_dependency_and_safe_data_contract() -> None:
    root = Path(embed_train.__file__).parents[2]
    paths = [
        root / "embed/stage2_finetune/config/mistral3-vl.yaml",
        root / "rerank/stage2_finetune/config/mistral3-vl.yaml",
    ]

    for path in paths:
        raw = yaml.safe_load(path.read_text())
        assert raw["model_family"] == "mistral3_vl"
        assert raw["require_mined_negatives"] is True
        assert raw["use_text_in_document"] is True
        assert raw["flash_adamw_master_weight_bits"] == 32
        assert "MISTRAL3_VL_TRAIN_DATA" in raw["train_data_path"]

    assert yaml.safe_load(paths[0].read_text())["base_model"] == "${oc.env:MISTRAL3_VL_EMBED_MODEL}"
    assert yaml.safe_load(paths[1].read_text())["base_model"] == "${oc.env:MISTRAL3_VL_RERANK_MODEL}"

    embed_prep = yaml.safe_load((root / "embed/stage1_data_prep/config/mistral3-vl.yaml").read_text())
    assert embed_prep["mining_backend"] == "automodel"
    assert embed_prep["mining_use_images"] is True
    assert embed_prep["hard_negatives_to_mine"] >= 3
    assert embed_prep["sdg_input_path"].endswith("/stage0_sdg/generation_result.json")
    assert embed_prep["retrieval_view"] == "image_and_text"
    assert embed_prep["train_input_file"] is None

    rerank_prep = yaml.safe_load((root / "rerank/stage1_prep/config/mistral3-vl.yaml").read_text())
    assert rerank_prep["mining_backend"] == "vllm"
    assert rerank_prep["mining_use_images"] is True
    assert rerank_prep["hard_negatives_to_mine"] >= 3
    assert "MISTRAL3_VL_SDG_TRAIN" in rerank_prep["train_input_file"]


def test_preview_eval_and_deploy_profiles_are_vllm_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL3_VL_EMBED_MODEL", "nvidia/test-mistral3-vl-embed")
    monkeypatch.setenv("MISTRAL3_VL_RERANK_MODEL", "nvidia/test-mistral3-vl-rerank")
    root = Path(embed_train.__file__).parents[2]
    embed_eval = load_pydantic_config(
        root / "embed/stage3_eval/config/mistral3-vl.yaml",
        [],
        EmbedEvalConfig,
    )
    embed_deploy = load_pydantic_config(
        root / "embed/stage5_deploy/config/mistral3-vl.yaml",
        [],
        EmbedDeployConfig,
    )
    rerank_eval = load_pydantic_config(
        root / "rerank/stage3_eval/config/mistral3-vl.yaml",
        [],
        RerankEvalConfig,
    )
    rerank_deploy = load_pydantic_config(
        root / "rerank/stage5_deploy/config/mistral3-vl.yaml",
        [],
        RerankDeployConfig,
    )

    assert embed_eval.embedding_api_backend == "vllm"
    assert embed_eval.nim_model == "nvidia/test-mistral3-vl-embed"
    assert embed_deploy.backend == "vllm"
    assert embed_deploy.vllm_runner == "pooling"
    assert embed_deploy.vllm_max_model_len == 8192
    assert embed_deploy.vllm_hf_overrides == {"vision_config": {"image_size": 1120}}
    assert rerank_eval.rerank_api_backend == "vllm"
    assert rerank_eval.nim_model == "nvidia/test-mistral3-vl-rerank"
    assert rerank_deploy.backend == "vllm"
    assert rerank_deploy.vllm_trust_remote_code is False
