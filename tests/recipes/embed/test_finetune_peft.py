"""Behavioral obligations from retrieval_training_peft.allium."""

from __future__ import annotations

import json
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemotron.recipes.embed.stage2_finetune import train
from nemotron.recipes.embed.stage2_finetune.peft import merge_peft_checkpoint, validate_peft_base


class NativeConfig(SimpleNamespace):
    """Minimal native configuration boundary without constructing GPU components."""

    def __init__(self, raw: dict) -> None:
        super().__init__(
            **{key: NativeConfig(value) if isinstance(value, dict) else value for key, value in raw.items()}
        )


class FakeRecipe:
    """Write native adapter state at the training engine boundary."""

    def __init__(self, config: NativeConfig) -> None:
        self.config = config

    def setup(self) -> None:
        pass

    def run_train_validation_loop(self) -> None:
        checkpoint = Path(self.config.checkpoint.checkpoint_dir) / "LATEST"
        model = checkpoint / "model"
        model.mkdir(parents=True)
        (model / "adapter_model.safetensors").write_bytes(b"native adapter")
        (model / "adapter_config.json").write_text(
            json.dumps(
                {
                    "task_type": "FEATURE_EXTRACTION",
                    "r": self.config.peft.dim,
                    "lora_alpha": self.config.peft.alpha,
                    "target_modules": ["language_model.q_proj"],
                    "base_model_name_or_path": self.config.model.pretrained_model_name_or_path,
                }
            )
        )
        (checkpoint / "optimizer-state").write_bytes(b"resumable optimizer")


class MergedModel:
    """Small stand-in for the library-owned merge and full-weight serialization."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.conflict: str | None = None
        self.fail = False
        self.omit_weights = False
        self.config_change: dict = {}

    def merge_and_unload(self) -> MergedModel:
        return self

    def save_pretrained(self, output: str | Path, safe_serialization: bool) -> None:
        assert safe_serialization
        if self.fail:
            raise RuntimeError("merge failure")
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        config = json.loads((self.base / "config.json").read_text())
        config["dtype"] = "bfloat16"
        config.update(self.config_change)
        (destination / "config.json").write_text(json.dumps(config))
        if not self.omit_weights:
            (destination / "model.safetensors").write_bytes(b"merged full model")
        if self.conflict:
            (destination / self.conflict).write_bytes(b"changed metadata")


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Local self-contained text encoder; image metadata is added only by image tests."""
    root = tmp_path / "base"
    root.mkdir()
    values = {
        "config.json": {"architectures": ["BertModel"], "model_type": "bert", "hidden_size": 8},
        "tokenizer.json": {"model": {}},
        "tokenizer_config.json": {"tokenizer_class": "ExampleTokenizer"},
        "modules.json": [
            {"path": "", "type": "sentence_transformers.models.Transformer"},
            {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ],
        "1_Pooling/config.json": {"pooling_mode_mean_tokens": True, "include_prompt": True},
        "sentence_bert_config.json": {"max_seq_length": 4096},
        "config_sentence_transformers.json": {"prompts": {"query": "query:", "document": "passage:"}},
    }
    for name, value in values.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    (root / "model.safetensors").write_bytes(b"base full model")
    return root


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, base: Path) -> tuple[MagicMock, MergedModel]:
    """Replace only expensive native training and public model-library boundaries."""
    monkeypatch.setattr(train, "_can_import_fused_adam", lambda: (True, None))
    monkeypatch.setattr(train, "_can_import_flash_adamw", lambda: (False, None))
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.components.config.loader", SimpleNamespace(ConfigNode=NativeConfig)
    )
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.recipes.retrieval", SimpleNamespace(TrainBiEncoderRecipe=FakeRecipe)
    )
    model = MergedModel(base)
    auto = MagicMock()
    auto.from_pretrained.return_value = model
    peft = MagicMock()
    peft.from_pretrained.return_value = model
    from transformers import AutoConfig

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=auto, AutoConfig=AutoConfig))
    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(PeftModel=peft, LoraModel=MagicMock(), __version__="0.18.1")
    )
    return auto, model


def settings(tmp_path: Path, base: Path) -> train.FinetuneConfig:
    data = tmp_path / "train.json"
    data.write_text(json.dumps({"data": [{"query": "standalone query", "pos": ["positive"], "neg": ["negative"]}]}))
    return train.FinetuneConfig(
        base_model=str(base),
        train_data_path=data,
        checkpoint_dir=tmp_path / "checkpoints",
        train_n_passages=2,
        global_batch_size=2,
        num_epochs=1,
        lr_warmup_steps=0,
        peft={"dim": 16, "alpha": 32, "target_modules": ["*.q_proj"]},
    )


@pytest.mark.parametrize("model_family", ["text", "mistral3_vl"])
def test_full_finetune_configuration_unchanged(engine: tuple, model_family: str) -> None:
    cfg = train.FinetuneConfig(model_family=model_family)
    raw, _ = train._load_automodel_config(cfg, dict)
    assert cfg.peft is None
    assert "peft" not in raw
    assert raw["checkpoint"]["save_consolidated"] is True


@pytest.mark.parametrize("peft", [None, {"dim": 16, "alpha": 32, "target_modules": ["*.q_proj"]}])
@pytest.mark.parametrize("model_family", ["text", "mistral3_vl"])
def test_training_retains_one_recent_checkpoint(engine: tuple, peft: dict | None, model_family: str) -> None:
    """Both templates carry native retention into full and adapter training."""
    cfg = train.FinetuneConfig(model_family=model_family, peft=peft)
    raw, _ = train._load_automodel_config(cfg, dict)
    assert raw["checkpoint"]["max_recent_checkpoints"] == 1


def test_selected_native_peft_settings_reach_trainer(tmp_path: Path, base: Path, engine: tuple) -> None:
    cfg = settings(tmp_path, base)
    raw, _ = train._load_automodel_config(cfg, dict)
    assert raw["peft"] == {
        "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
        "dim": 16,
        "alpha": 32,
        "target_modules": ["*.q_proj"],
        "use_memory_efficient_lora": False,
        "use_triton": False,
    }
    assert raw["checkpoint"]["enabled"] is True
    assert raw["checkpoint"]["save_consolidated"] is False
    assert raw["seed"] == 42


@pytest.mark.parametrize(
    "field,value",
    [
        ("dim", 0),
        ("dim", True),
        ("dim", 1.5),
        ("alpha", -1),
        ("target_modules", []),
        ("target_modules", [" "]),
        ("extra", 1),
    ],
)
def test_invalid_adapter_settings_rejected(field: str, value: object) -> None:
    options = {"dim": 16, "alpha": 32, "target_modules": ["*.q_proj"], field: value}
    with pytest.raises(ValidationError) as error:
        train.FinetuneConfig(peft=options)
    assert error.value.errors()[0]["loc"] == ("peft", field)


def test_existing_export_is_never_overwritten(tmp_path: Path, base: Path, engine: tuple) -> None:
    cfg = settings(tmp_path, base)
    existing = cfg.checkpoint_dir / "LATEST/model/consolidated"
    existing.mkdir(parents=True)
    marker = existing / "partial-output"
    marker.write_bytes(b"retain failed export")
    with pytest.raises((ValueError, FileExistsError), match="exist"):
        train.run_finetune(cfg)
    assert marker.read_bytes() == b"retain failed export"
    engine[0].from_pretrained.assert_not_called()


def test_non_coordinator_rank_does_not_merge(
    tmp_path: Path, base: Path, engine: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = settings(tmp_path, base)
    monkeypatch.setenv("RANK", "1")
    train.run_finetune(cfg)
    engine[0].from_pretrained.assert_not_called()


def test_training_returns_full_model_and_preserves_native_state(tmp_path: Path, base: Path, engine: tuple) -> None:
    cfg = settings(tmp_path, base)
    original = {str(path.relative_to(base)): path.read_bytes() for path in base.rglob("*") if path.is_file()}
    output = train.run_finetune(cfg)
    assert output == cfg.checkpoint_dir / "LATEST/model/consolidated"
    assert (output / "model.safetensors").read_bytes() == b"merged full model"
    assert (output.parent / "adapter_model.safetensors").read_bytes() == b"native adapter"
    assert (output.parent.parent / "optimizer-state").read_bytes() == b"resumable optimizer"
    for name, content in original.items():
        assert (base / name).read_bytes() == content
        if name not in {"model.safetensors", "config.json"}:
            assert (output / name).read_bytes() == content
    assert engine[0].from_pretrained.call_args.kwargs["device_map"] == "cpu"
    assert engine[0].from_pretrained.call_args.kwargs["local_files_only"] is True


@pytest.mark.parametrize("missing", ["tokenizer.json", "1_Pooling/config.json", "modules.json"])
def test_missing_source_context_fails_before_training(tmp_path: Path, base: Path, engine: tuple, missing: str) -> None:
    cfg = settings(tmp_path, base)
    (base / missing).unlink()
    with pytest.raises(ValueError, match="metadata"):
        train.run_finetune(cfg)
    assert not cfg.checkpoint_dir.exists()


def test_vision_source_requires_processor_but_text_does_not(tmp_path: Path, base: Path, engine: tuple) -> None:
    cfg = settings(tmp_path, base)
    config = json.loads((base / "config.json").read_text())
    config["vision_config"] = {"image_size": 224}
    (base / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="processor"):
        train.run_finetune(cfg)
    assert not cfg.checkpoint_dir.exists()


@pytest.mark.parametrize("override", [{"pooling": "cls"}, {"l2_normalize": False}, {"query_prefix": "other: "}])
def test_conflicting_embedding_contract_fails_before_training(
    tmp_path: Path, base: Path, engine: tuple, override: dict
) -> None:
    cfg = settings(tmp_path, base).model_copy(update=override)
    with pytest.raises(ValueError, match="metadata"):
        train.run_finetune(cfg)
    assert not cfg.checkpoint_dir.exists()


@pytest.mark.parametrize("failure", ["conflict", "merge", "weights"])
def test_failed_export_never_returns_success(tmp_path: Path, base: Path, engine: tuple, failure: str) -> None:
    cfg = settings(tmp_path, base)
    _, model = engine
    model.conflict = "tokenizer.json" if failure == "conflict" else None
    model.fail = failure == "merge"
    model.omit_weights = failure == "weights"
    with pytest.raises((ValueError, RuntimeError), match="metadata|merge|weights"):
        train.run_finetune(cfg)
    assert (cfg.checkpoint_dir / "LATEST/model/adapter_model.safetensors").read_bytes() == b"native adapter"


def test_model_config_serialization_defaults_are_not_semantic_changes(
    tmp_path: Path, base: Path, engine: tuple
) -> None:
    cfg = settings(tmp_path, base)
    engine[1].config_change = {"text_config": {"model_type": "mistral", "is_causal": False, "return_dict": True}}
    source = json.loads((base / "config.json").read_text())
    source.update(
        {
            "model_type": "mistral3",
            "architectures": ["Mistral3Model"],
            "text_config": {"model_type": "mistral", "is_causal": False},
            "vision_config": {"model_type": "pixtral"},
        }
    )
    (base / "config.json").write_text(json.dumps(source))
    (base / "processor_config.json").write_text("{}")
    output = train.run_finetune(cfg)
    assert (output / "model.safetensors").is_file()


@pytest.mark.parametrize("change", [{"architectures": ["WrongModel"]}, {"hidden_size": 16}, {"dtype": "float32"}])
def test_semantic_model_config_change_rejected(tmp_path: Path, base: Path, engine: tuple, change: dict) -> None:
    cfg = settings(tmp_path, base)
    engine[1].config_change = change
    with pytest.raises(ValueError, match="metadata"):
        train.run_finetune(cfg)


def test_public_dotlist_adapter_block_reaches_native_config(engine: tuple) -> None:
    cfg = train.load_config(
        train.STAGE_PATH / "config/default.yaml",
        ['peft={"dim":16,"alpha":32,"target_modules":["*.q_proj","*.v_proj"]}'],
        train.FinetuneConfig,
    )
    raw, _ = train._load_automodel_config(cfg, dict)
    assert raw["peft"]["target_modules"] == ["*.q_proj", "*.v_proj"]


def test_text_tokenizer_can_bypass_model_registry(engine: tuple) -> None:
    """The opt-in HF wrapper selection reaches the native text tokenizer config."""
    cfg = train.FinetuneConfig(tokenizer_force_default=True)
    raw, _ = train._load_automodel_config(cfg, dict)
    assert raw["tokenizer"]["force_default"] is True


@pytest.mark.parametrize(
    "filename,value", [("modules.json", {}), ("1_Pooling/config.json", []), ("config_sentence_transformers.json", [])]
)
def test_malformed_source_context_is_actionable(
    tmp_path: Path, base: Path, engine: tuple, filename: str, value: object
) -> None:
    cfg = settings(tmp_path, base)
    (base / filename).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata"):
        train.run_finetune(cfg)
    assert not cfg.checkpoint_dir.exists()


def test_missing_pooling_module_rejected_before_training(tmp_path: Path, base: Path, engine: tuple) -> None:
    cfg = settings(tmp_path, base)
    modules = json.loads((base / "modules.json").read_text())
    (base / "modules.json").write_text(json.dumps([module for module in modules if module["path"] != "1_Pooling"]))
    with pytest.raises(ValueError, match="metadata"):
        train.run_finetune(cfg)
    assert not cfg.checkpoint_dir.exists()


@pytest.mark.parametrize(
    "dependency", [None, SimpleNamespace(__version__="0.1.0"), SimpleNamespace(__version__="0.18.1")]
)
def test_missing_or_unsupported_merge_dependency_fails_before_training(
    tmp_path: Path, base: Path, engine: tuple, monkeypatch: pytest.MonkeyPatch, dependency: object
) -> None:
    cfg = settings(tmp_path, base)
    monkeypatch.setitem(sys.modules, "peft", dependency)
    with pytest.raises(RuntimeError, match="peft.*0.18.1"):
        train.run_finetune(cfg)
    assert not cfg.checkpoint_dir.exists()


def test_real_cpu_float32_base_exports_intentional_bfloat16(
    tmp_path: Path, base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from transformers import AutoModel, BertConfig, BertModel

    cfg = settings(tmp_path, base)
    model = BertModel(
        BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    )
    model.save_pretrained(base)
    original = (base / "model.safetensors").read_bytes()
    adapter = cfg.checkpoint_dir / "LATEST/model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter loading boundary")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "task_type": "FEATURE_EXTRACTION",
                "r": 16,
                "lora_alpha": 32,
                "base_model_name_or_path": str(base),
            }
        )
    )
    # Only adapter application is mocked: real HF weights/config are loaded,
    # converted, serialized and reloaded on CPU through the public export helper.
    monkeypatch.setattr(BertModel, "merge_and_unload", lambda self: self, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            __spec__=ModuleSpec("peft", loader=None),
            PeftModel=SimpleNamespace(from_pretrained=lambda model, *args, **kwargs: model),
        ),
    )
    output = adapter / "consolidated"
    merge_peft_checkpoint(cfg, output)
    reloaded = AutoModel.from_pretrained(output, local_files_only=True, torch_dtype="auto")
    assert reloaded.dtype == torch.bfloat16
    assert next(reloaded.parameters()).device.type == "cpu"
    assert (base / "model.safetensors").read_bytes() == original


@pytest.mark.parametrize("prefixes", [(None, None), (None, "passage:"), ("query:", None)])
def test_peft_inherits_checkpoint_prompts(tmp_path: Path, base: Path, prefixes: tuple) -> None:
    cfg = settings(tmp_path, base).model_copy(
        update={"model_family": "mistral3_vl", "query_prefix": prefixes[0], "passage_prefix": prefixes[1]}
    )
    assert "config_sentence_transformers.json" in validate_peft_base(cfg)


def test_peft_still_rejects_explicit_prompt_override(tmp_path: Path, base: Path) -> None:
    cfg = settings(tmp_path, base).model_copy(update={"query_prefix": "different:", "passage_prefix": None})
    with pytest.raises(ValueError, match="query prompt metadata conflicts"):
        validate_peft_base(cfg)
