# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional native LoRA configuration and portable full-model export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from nemotron.recipes.embed.stage2_finetune.train import FinetuneConfig


class PeftConfig(BaseModel):
    """Explicit native LoRA settings; omitted by full fine-tuning runs."""

    model_config = ConfigDict(extra="forbid")
    dim: int = Field(gt=0, strict=True, description="Native LoRA adapter dimension.")
    alpha: int = Field(gt=0, strict=True, description="Native LoRA scaling numerator.")
    target_modules: list[str] = Field(
        min_length=1, description="Explicit native module selectors, including wildcards."
    )

    @field_validator("target_modules")
    @classmethod
    def validate_targets(cls, value: list[str]) -> list[str]:
        """Reject blank selectors rather than silently falling back to native defaults."""
        if any(not target.strip() for target in value):
            raise ValueError("target_modules must contain nonblank selectors")
        return value


_REQUIRED_AUXILIARY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
)
_OPTIONAL_AUXILIARY = (
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
)


def _read_metadata(path: Path) -> Any:
    """Read JSON metadata with an actionable error for incomplete local checkpoints."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Missing or invalid PEFT base metadata: {path}") from error


def validate_peft_runtime() -> None:
    """Fail before training when the optional CPU merge dependency is unavailable.

    Raises:
        RuntimeError: HF PEFT is missing, too old, or lacks the supported APIs.
    """
    try:
        from packaging.version import Version
        from peft import LoraModel, PeftModel, __version__

        if Version(__version__) < Version("0.18.1") or not all(
            callable(method) for method in (PeftModel.from_pretrained, LoraModel.merge_and_unload)
        ):
            raise ValueError("Unsupported PEFT merge capabilities")
    except (ImportError, AttributeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "Optional LoRA export requires a compatible HF peft>=0.18.1 installation in the training "
            "environment. Install it before retrying, for example: python -m pip install 'peft>=0.18.1'."
        ) from error


def validate_peft_base(cfg: FinetuneConfig) -> tuple[str, ...]:
    """Validate local source context before training or merging adapters.

    Args:
        cfg: Stage2 settings whose embedding semantics must match the base.

    Returns:
        Relative auxiliary file names to preserve in the exported model.

    Raises:
        ValueError: The local checkpoint or its embedding metadata is incomplete
            or conflicts with the requested training settings.
    """
    base = Path(cfg.base_model)
    config = _read_metadata(base / "config.json")
    if not isinstance(config, dict) or not config.get("architectures"):
        raise ValueError("PEFT requires local model architecture metadata")
    if not any(base.glob("model*.safetensors")):
        raise ValueError("PEFT requires local base model weights and metadata")
    for name in _REQUIRED_AUXILIARY:
        if not (base / name).is_file():
            raise ValueError(f"Missing required PEFT base metadata: {name}")
    if config.get("vision_config") and not any(
        (base / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")
    ):
        raise ValueError("Image-capable PEFT base requires processor metadata")
    pooling = _read_metadata(base / "1_Pooling/config.json")
    if not isinstance(pooling, dict):
        raise ValueError("PEFT pooling metadata must be a JSON object")
    mode = {"avg": "pooling_mode_mean_tokens", "cls": "pooling_mode_cls_token", "last": "pooling_mode_lasttoken"}[
        cfg.pooling
    ]
    enabled = {name for name, value in pooling.items() if name.startswith("pooling_mode_") and value is True}
    if enabled != {mode} or pooling.get("include_prompt", True) is not True:
        raise ValueError("PEFT pooling metadata conflicts with training settings")
    modules = _read_metadata(base / "modules.json")
    if not isinstance(modules, list) or not modules or not all(isinstance(module, dict) for module in modules):
        raise ValueError("PEFT module metadata must be a nonempty list of objects")
    expected_modules = [
        ("", "sentence_transformers.models.Transformer"),
        ("1_Pooling", "sentence_transformers.models.Pooling"),
    ]
    if cfg.l2_normalize:
        expected_modules.append(("2_Normalize", "sentence_transformers.models.Normalize"))
    if [(module.get("path"), module.get("type")) for module in modules] != expected_modules:
        raise ValueError("PEFT module metadata must match the configured transformer, pooling and normalization")
    sentence_config = _read_metadata(base / "config_sentence_transformers.json")
    if not isinstance(sentence_config, dict) or not isinstance(sentence_config.get("prompts"), dict):
        raise ValueError("PEFT prompt metadata must contain a prompts object")
    prompts = sentence_config["prompts"]
    for name, prefix in (("query", cfg.query_prefix), ("document", cfg.passage_prefix)):
        if prompts.get(name) not in {prefix, prefix.removesuffix(" ")}:
            raise ValueError(f"PEFT {name} prompt metadata conflicts with training settings")
    text_config = config.get("text_config", config)
    if "is_causal" in text_config and text_config["is_causal"] != cfg.is_causal:
        raise ValueError("PEFT attention metadata conflicts with training settings")
    return _REQUIRED_AUXILIARY + tuple(name for name in _OPTIONAL_AUXILIARY if (base / name).is_file())


def merge_peft_checkpoint(cfg: FinetuneConfig, output: Path) -> None:
    """Merge the selected native adapter on CPU without changing source metadata.

    Uses the supported HF/PEFT operations also used by AutoModel's source-only
    merge utility; that utility is not part of the installed AutoModel package.
    Partial failed exports are retained and cannot be silently overwritten.

    Args:
        cfg: Stage2 settings containing the immutable local base path.
        output: New consolidated-model destination next to native adapter files.

    Raises:
        FileExistsError: The output already exists.
        ValueError: Adapter, model weights, or auxiliary metadata are incompatible.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModel

    auxiliary = validate_peft_base(cfg)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"PEFT export already exists: {output}")
    adapter = output.parent
    adapter_config = _read_metadata(adapter / "adapter_config.json")
    if (
        adapter_config.get("task_type") != "FEATURE_EXTRACTION"
        or not (adapter / "adapter_model.safetensors").is_file()
    ):
        raise ValueError("Expected a native retrieval adapter checkpoint with FEATURE_EXTRACTION metadata")
    if cfg.peft is None or (adapter_config.get("r"), adapter_config.get("lora_alpha")) != (
        cfg.peft.dim,
        cfg.peft.alpha,
    ):
        raise ValueError("PEFT adapter metadata conflicts with selected dimension or alpha")
    if Path(adapter_config.get("base_model_name_or_path", "")).resolve() != Path(cfg.base_model).resolve():
        raise ValueError("PEFT adapter metadata refers to a different base model")
    output.mkdir(parents=True, exist_ok=False)
    model = AutoModel.from_pretrained(
        cfg.base_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=cfg.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, adapter, autocast_adapter_dtype=True)
    model = model.merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    if not any(path.is_file() and path.stat().st_size for path in output.glob("model*.safetensors")):
        raise ValueError("PEFT export did not produce full model weights")
    # Config classes restore serialization-omitted defaults; raw JSON equality
    # incorrectly rejects valid exports across Transformers config serializers.
    source_config_object = AutoConfig.from_pretrained(
        cfg.base_model,
        local_files_only=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    # Native training and this export deliberately use BF16. Keep this one
    # expected precision conversion explicit; every other config field is checked.
    source_config_object.torch_dtype = torch.bfloat16
    source_config = source_config_object.to_dict()
    merged_config = AutoConfig.from_pretrained(
        output,
        local_files_only=True,
        trust_remote_code=cfg.trust_remote_code,
    ).to_dict()
    # Loading from the new directory changes only the location annotation.
    source_config.pop("_name_or_path", None)
    merged_config.pop("_name_or_path", None)
    if merged_config != source_config:
        raise ValueError("PEFT merged model metadata conflicts with source model configuration")
    for name in auxiliary:
        source, target = Path(cfg.base_model) / name, output / name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"PEFT export metadata conflicts with source: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
