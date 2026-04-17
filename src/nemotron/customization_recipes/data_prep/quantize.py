# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model quantization: FP8, INT4-AWQ, INT8-SQ via TensorRT Model Optimizer.

Uses lazy imports for heavy quantization libraries (modelopt, tensorrt_llm)
so the module can be safely imported without those packages installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

log = logging.getLogger(__name__)

_MODELOPT_MSG = (
    "nvidia-modelopt is required for quantization. "
    "Install with: pip install nvidia-modelopt"
)

_TRTLLM_MSG = (
    "tensorrt-llm is required for TRT-LLM engine export. "
    "Install with: pip install tensorrt-llm"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QuantizeConfig:
    """Configuration for model quantization."""

    model_path: str = ""
    """Path to the HuggingFace model checkpoint to quantize."""

    output_dir: str = "quantized_model"
    """Directory to write the quantized model."""

    method: str = "fp8"
    """Quantization method: fp8 | int4_awq | int8_sq"""

    calibration_data_path: Optional[str] = None
    """Path to calibration data (JSONL with 'text' field)."""

    calibration_num_samples: int = 512
    """Number of calibration samples."""

    calibration_max_length: int = 4096
    """Max sequence length for calibration."""

    calibration_batch_size: int = 1
    """Batch size for calibration forward passes."""

    # AWQ-specific
    awq_group_size: int = 128
    """Group size for AWQ quantization."""

    awq_zero_point: bool = True
    """Use zero-point quantization for AWQ."""

    # TRT-LLM engine export
    build_trt_engine: bool = False
    """Whether to build a TensorRT-LLM engine after quantization."""

    trt_tp_size: int = 1
    """Tensor parallelism size for TRT-LLM engine."""

    trt_max_batch_size: int = 32
    """Max batch size for TRT-LLM engine."""

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "QuantizeConfig":
        """Build a QuantizeConfig from an OmegaConf DictConfig.

        Handles the nested YAML structure used by stage6_quantization configs::

            model:
              name_or_path: ...
            quantization:
              method: fp8
              output_dir: ...
              calibration:
                num_samples: 512
                max_length: 2048
              int4_awq:
                group_size: 128

        These nested keys are mapped to the flat dataclass fields
        (``model_path``, ``method``, ``calibration_num_samples``, etc.).
        Flat configs (where keys match dataclass fields directly) are
        also supported for backward compatibility.
        """
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)

        # If the config has nested structure, flatten it to match dataclass fields.
        flat: dict = {}

        # --- model section ---
        model_section = raw.get("model", {})
        if isinstance(model_section, dict):
            if "name_or_path" in model_section:
                flat["model_path"] = model_section["name_or_path"]

        # --- quantization section ---
        quant_section = raw.get("quantization", {})
        if isinstance(quant_section, dict):
            if "method" in quant_section:
                flat["method"] = quant_section["method"]
            if "output_dir" in quant_section:
                flat["output_dir"] = quant_section["output_dir"]

            # calibration sub-section
            calib = quant_section.get("calibration", {})
            if isinstance(calib, dict):
                if "dataset" in calib or "data_path" in calib:
                    flat["calibration_data_path"] = calib.get("data_path") or calib.get("dataset")
                if "num_samples" in calib:
                    flat["calibration_num_samples"] = calib["num_samples"]
                if "max_length" in calib:
                    flat["calibration_max_length"] = calib["max_length"]
                if "batch_size" in calib:
                    flat["calibration_batch_size"] = calib["batch_size"]

            # fp8 sub-section (reserved for future fp8-specific settings)
            # Currently no flat dataclass fields map to fp8 sub-keys.

            # int4_awq sub-section
            awq = quant_section.get("int4_awq", {})
            if isinstance(awq, dict):
                if "group_size" in awq:
                    flat["awq_group_size"] = awq["group_size"]
                if "zero_point" in awq:
                    flat["awq_zero_point"] = awq["zero_point"]

        # --- export section ---
        export_section = raw.get("export", {})
        if isinstance(export_section, dict):
            if export_section.get("format") == "trt_llm":
                flat["build_trt_engine"] = True

        # If no nested sections were found, assume the config is already flat
        # (keys match dataclass field names directly).
        if not flat:
            schema = OmegaConf.structured(QuantizeConfig)
            merged = OmegaConf.merge(schema, cfg)
            return QuantizeConfig(**OmegaConf.to_container(merged, resolve=True))

        # Build from extracted flat keys only; dataclass defaults handle
        # any field not present in the YAML.
        return QuantizeConfig(**flat)


# ---------------------------------------------------------------------------
# Lazy dependency helpers
# ---------------------------------------------------------------------------


def _require_modelopt():
    try:
        import modelopt  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MODELOPT_MSG) from exc


def _require_trtllm():
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError as exc:
        raise ImportError(_TRTLLM_MSG) from exc


# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(model_path: str):
    """Load a HuggingFace model and tokenizer with lazy import."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
    )
    return model, tokenizer


def _load_calibration_data(cfg: QuantizeConfig, tokenizer):
    """Load and tokenize calibration data."""
    import json

    texts = []
    if cfg.calibration_data_path and Path(cfg.calibration_data_path).exists():
        with open(cfg.calibration_data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= cfg.calibration_num_samples:
                    break
                record = json.loads(line)
                texts.append(record.get("text", ""))
    else:
        log.warning(
            "No calibration data at %s; using dummy calibration. "
            "Results may be suboptimal.",
            cfg.calibration_data_path,
        )
        texts = ["The quick brown fox jumps over the lazy dog."] * min(
            64, cfg.calibration_num_samples
        )

    log.info("Loaded %d calibration samples", len(texts))
    return texts


def _quantize_fp8(model, tokenizer, cfg: QuantizeConfig):
    """Apply FP8 quantization via modelopt."""
    _require_modelopt()
    import modelopt.torch.quantization as mtq

    log.info("Applying FP8 quantization")
    calib_texts = _load_calibration_data(cfg, tokenizer)

    def _calib_forward(model):
        for text in calib_texts[: cfg.calibration_num_samples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=cfg.calibration_max_length,
                truncation=True,
            ).to(model.device)
            model(**inputs)

    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=_calib_forward)
    return model


def _quantize_int4_awq(model, tokenizer, cfg: QuantizeConfig):
    """Apply INT4 AWQ quantization via modelopt."""
    _require_modelopt()
    import modelopt.torch.quantization as mtq

    log.info("Applying INT4 AWQ quantization (group_size=%d)", cfg.awq_group_size)
    calib_texts = _load_calibration_data(cfg, tokenizer)

    def _calib_forward(model):
        for text in calib_texts[: cfg.calibration_num_samples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=cfg.calibration_max_length,
                truncation=True,
            ).to(model.device)
            model(**inputs)

    quant_cfg = mtq.INT4_AWQ_CFG.copy()
    quant_cfg["quant_cfg"]["*weight_quantizer"]["group_size"] = cfg.awq_group_size
    mtq.quantize(model, quant_cfg, forward_loop=_calib_forward)
    return model


def _quantize_int8_sq(model, tokenizer, cfg: QuantizeConfig):
    """Apply INT8 SmoothQuant quantization via modelopt."""
    _require_modelopt()
    import modelopt.torch.quantization as mtq

    log.info("Applying INT8 SmoothQuant quantization")
    calib_texts = _load_calibration_data(cfg, tokenizer)

    def _calib_forward(model):
        for text in calib_texts[: cfg.calibration_num_samples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=cfg.calibration_max_length,
                truncation=True,
            ).to(model.device)
            model(**inputs)

    mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop=_calib_forward)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METHOD_MAP = {
    "fp8": _quantize_fp8,
    "int4_awq": _quantize_int4_awq,
    "int8_sq": _quantize_int8_sq,
    "int8": _quantize_int8_sq,  # alias
}


def quantize_model(cfg: "DictConfig") -> dict:
    """Quantize a model end-to-end from an OmegaConf config.

    Loads the model, applies the specified quantization method, and saves
    the quantized checkpoint.

    Args:
        cfg: OmegaConf DictConfig with quantization parameters.

    Returns:
        Dict with ``output_dir`` and ``method``.
    """
    qcfg = QuantizeConfig.from_omegaconf(cfg)

    if qcfg.method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown quantization method '{qcfg.method}'. "
            f"Supported: {list(_METHOD_MAP.keys())}"
        )

    model, tokenizer = _load_model_and_tokenizer(qcfg.model_path)
    quantize_fn = _METHOD_MAP[qcfg.method]
    model = quantize_fn(model, tokenizer, qcfg)

    output_path = Path(qcfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Saving quantized model to %s", output_path)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    if qcfg.build_trt_engine:
        _require_trtllm()
        log.info(
            "TRT-LLM engine build requested (tp=%d). "
            "Use trtllm-build CLI or TRT-LLM Python API to convert.",
            qcfg.trt_tp_size,
        )

    log.info("Quantization complete: method=%s, output=%s", qcfg.method, output_path)
    return {"output_dir": str(output_path), "method": qcfg.method}
