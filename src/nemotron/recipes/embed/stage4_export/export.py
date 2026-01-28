#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export script for embedding models to ONNX and TensorRT.

Exports fine-tuned embedding models to ONNX format and optionally converts
to TensorRT for optimized inference. Based on the NeMo Export-Deploy tutorial:
https://github.com/NVIDIA-NeMo/Export-Deploy/blob/main/tutorials/onnx_tensorrt/embedding/llama_embedding.ipynb

Usage:
    # With default config
    nemotron embed export -c default

    # Export to ONNX only
    nemotron embed export -c default export_to_trt=false

    # With custom model path
    nemotron embed export -c default model_path=/path/to/model
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from nemotron.kit.config_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class ExportConfig(RecipeSettings):
    """Export configuration for embedding models to ONNX/TensorRT."""

    model_config = ConfigDict(extra="forbid")

    # Model path
    model_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated", description="Path to fine-tuned HuggingFace model checkpoint.")

    # Model settings
    pooling_mode: Literal["avg", "cls", "last"] = Field(default="avg", description="Pooling method in the embedding model (avg, cls, last).")
    normalize: bool = Field(default=True, description="Whether to L2 normalize embeddings in the model.")
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = Field(default="eager", description="Attention implementation: 'eager', 'sdpa', or 'flash_attention_2'.")
    use_dimension_arg: bool = Field(default=True, description="Whether dimension argument is used in the model forward function.")

    # Quantization settings
    quant_cfg: Literal["fp8", "int8_sq"] | None = Field(default=None, description="Quantization config: 'fp8', 'int8_sq', or None (no quantization).")
    calibration_batch_size: int = Field(default=64, gt=0, description="Batch size for quantization calibration.")

    # ONNX export settings
    onnx_export_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage4_export/onnx", description="Output path for ONNX model.")
    opset: int = Field(default=17, gt=0, description="ONNX opset version.")
    export_dtype: Literal["fp32", "fp16"] = Field(default="fp32", description="ONNX export data precision (fp32, fp16).")

    # TensorRT settings
    export_to_trt: bool = Field(default=False, description="Whether to export ONNX model to TensorRT.")
    trt_model_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage4_export/tensorrt", description="Output path for TensorRT .plan file.")
    override_layernorm_precision_to_fp32: bool = Field(default=True, description="Whether to override LayerNorm precision to fp32 for stability.")
    override_layers_to_fp32: list[str] = Field(default_factory=lambda: ["/model/norm/", "/pooling_module", "/ReduceL2", "/Div"], description="Layer patterns to override precision to fp32.")
    profiling_verbosity: str = Field(default="layer_names_only", description="TensorRT profiling verbosity level.")

    # TensorRT input profiles (min, opt, max shapes)
    trt_min_batch: int = Field(default=1, gt=0, description="Minimum batch size for TensorRT optimization.")
    trt_opt_batch: int = Field(default=16, gt=0, description="Optimal batch size for TensorRT optimization.")
    trt_max_batch: int = Field(default=64, gt=0, description="Maximum batch size for TensorRT optimization.")
    trt_min_seq_len: int = Field(default=3, gt=0, description="Minimum sequence length for TensorRT optimization.")
    trt_opt_seq_len: int = Field(default=128, gt=0, description="Optimal sequence length for TensorRT optimization.")
    trt_max_seq_len: int = Field(default=256, gt=0, description="Maximum sequence length for TensorRT optimization.")

    @model_validator(mode="after")
    def _check_trt_profile_order(self):
        if self.trt_min_batch > self.trt_opt_batch:
            raise ValueError(f"trt_min_batch ({self.trt_min_batch}) must be <= trt_opt_batch ({self.trt_opt_batch})")
        if self.trt_opt_batch > self.trt_max_batch:
            raise ValueError(f"trt_opt_batch ({self.trt_opt_batch}) must be <= trt_max_batch ({self.trt_max_batch})")
        if self.trt_min_seq_len > self.trt_opt_seq_len:
            raise ValueError(f"trt_min_seq_len ({self.trt_min_seq_len}) must be <= trt_opt_seq_len ({self.trt_opt_seq_len})")
        if self.trt_opt_seq_len > self.trt_max_seq_len:
            raise ValueError(f"trt_opt_seq_len ({self.trt_opt_seq_len}) must be <= trt_max_seq_len ({self.trt_max_seq_len})")
        return self

    # Output settings
    output_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage4_export", description="Base output directory for export artifacts.")


def load_embedding_model(
    model_path: str | Path,
    pooling_mode: str = "avg",
    normalize: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load a Llama-based embedding model with bidirectional attention.

    Args:
        model_path: Path to the HuggingFace model.
        pooling_mode: Pooling strategy (avg, cls, last).
        normalize: Whether to normalize embeddings.
        attn_implementation: Attention implementation to use. Use "eager" for
            ONNX export compatibility (SDPA/GQA not supported by TorchScript exporter).

    Returns:
        Tuple of (model, tokenizer).
    """
    from nemo_export.model_adapters.embedding.embedding_adapter import Pooling, LlamaBidirectionalHFAdapter
    from transformers import AutoTokenizer, AutoModel


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    pooling_mode = pooling_mode or "avg"
    if pooling_mode == "last" and tokenizer.padding_side == "right":
        pooling_mode = "last__right"  # type: ignore
    if pooling_mode == "cls" and tokenizer.padding_side == "left":
        pooling_mode = "cls__left"  # type: ignore

    # load the model
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
    ).eval()

    # configure pooling
    pooling_module = Pooling(pooling_mode=pooling_mode)

    # NV-Embed-v1 model has seperate embedding model and a built-in pooling module
    if (
        model.__class__.__name__ == "NVEmbedModel"
        and hasattr(model, "latent_attention_model")
        and hasattr(model, "embedding_model")
    ):
        pooling_module = model.latent_attention_model
        model = model.embedding_model

    adapted_model = LlamaBidirectionalHFAdapter(model=model, normalize=normalize, pooling_module=pooling_module)
    return adapted_model, tokenizer


def export_to_onnx(
    model: Any,
    tokenizer: Any,
    cfg: ExportConfig,
) -> Any:
    """Export model to ONNX format.

    Args:
        model: The embedding model.
        tokenizer: The tokenizer.
        cfg: Export configuration.

    Returns:
        OnnxLLMExporter instance.
    """
    try:
        from nemo_export.onnx_llm_exporter import OnnxLLMExporter
    except ImportError:
        print(
            "Error: nemo_export is required. Please install from NeMo Export-Deploy:\n"
            "  cd /opt/Export-Deploy && uv sync --inexact --link-mode symlink --locked --extra trt-onnx"
        )
        sys.exit(1)

    # Configure input/output names based on dimension argument usage
    if cfg.use_dimension_arg:
        input_names = ["input_ids", "attention_mask", "dimensions"]
        dynamic_axes_input = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "dimensions": {0: "batch_size"},
        }
    else:
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes_input = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
        }

    output_names = ["embeddings"]
    dynamic_axes_output = {"embeddings": {0: "batch_size", 1: "embedding_dim"}}

    # Create exporter
    onnx_exporter = OnnxLLMExporter(
        onnx_model_dir=str(cfg.onnx_export_path),
        model=model,
        tokenizer=tokenizer,
    )

    # Apply quantization if configured
    if cfg.quant_cfg is not None:
        print(f"  Applying quantization: {cfg.quant_cfg}")
        _apply_quantization(onnx_exporter, cfg)

    # Disable dynamo export
    import torch.onnx
    from functools import partial

    # Save the original export function
    original_export = torch.onnx.export

    # Define a wrapper that always sets dynamo=False
    def forced_legacy_export(*args, **kwargs):
        kwargs['dynamo'] = False
        return original_export(*args, **kwargs)

    # Overwrite the function in the torch.onnx namespace
    torch.onnx.export = forced_legacy_export

    # Export to ONNX
    print(f"  Exporting to ONNX (opset {cfg.opset}, dtype {cfg.export_dtype})...")
    onnx_exporter.export(
        input_names=input_names,
        output_names=output_names,
        opset=cfg.opset,
        dynamic_axes_input=dynamic_axes_input,
        dynamic_axes_output=dynamic_axes_output,
        export_dtype=cfg.export_dtype,
    )

    return onnx_exporter


def _apply_quantization(onnx_exporter: Any, cfg: ExportConfig) -> None:
    """Apply quantization to the model before ONNX export.

    Args:
        onnx_exporter: The ONNX exporter instance.
        cfg: Export configuration.
    """
    from functools import partial

    import torch
    from nemo.collections.llm.modelopt.quantization.quantizer import get_calib_data_iter
    from tqdm import tqdm

    def forward_loop(model: Any, data: Any, tokenizer: Any) -> None:
        for inputs in tqdm(data, desc="Calibration"):
            batch = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch)

    data = get_calib_data_iter(batch_size=cfg.calibration_batch_size)
    forward_loop_fn = partial(forward_loop, data=data, tokenizer=onnx_exporter.tokenizer)

    onnx_exporter.quantize(quant_cfg=cfg.quant_cfg, forward_loop=forward_loop_fn)


def export_onnx_to_tensorrt(onnx_exporter: Any, cfg: ExportConfig) -> None:
    """Convert ONNX model to TensorRT.

    Args:
        onnx_exporter: The ONNX exporter instance.
        cfg: Export configuration.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT is required for TRT export. Please install tensorrt.")
        sys.exit(1)

    # Build input profiles for dynamic shapes
    if cfg.use_dimension_arg:
        input_profiles = [
            {
                "input_ids": [
                    [cfg.trt_min_batch, cfg.trt_min_seq_len],
                    [cfg.trt_opt_batch, cfg.trt_opt_seq_len],
                    [cfg.trt_max_batch, cfg.trt_max_seq_len],
                ],
                "attention_mask": [
                    [cfg.trt_min_batch, cfg.trt_min_seq_len],
                    [cfg.trt_opt_batch, cfg.trt_opt_seq_len],
                    [cfg.trt_max_batch, cfg.trt_max_seq_len],
                ],
                "dimensions": [
                    [cfg.trt_min_batch],
                    [cfg.trt_opt_batch],
                    [cfg.trt_max_batch],
                ],
            }
        ]
    else:
        input_profiles = [
            {
                "input_ids": [
                    [cfg.trt_min_batch, cfg.trt_min_seq_len],
                    [cfg.trt_opt_batch, cfg.trt_opt_seq_len],
                    [cfg.trt_max_batch, cfg.trt_max_seq_len],
                ],
                "attention_mask": [
                    [cfg.trt_min_batch, cfg.trt_min_seq_len],
                    [cfg.trt_opt_batch, cfg.trt_opt_seq_len],
                    [cfg.trt_max_batch, cfg.trt_max_seq_len],
                ],
            }
        ]

    print(f"  Converting to TensorRT...")
    print(f"    Batch sizes: min={cfg.trt_min_batch}, opt={cfg.trt_opt_batch}, max={cfg.trt_max_batch}")
    print(f"    Seq lengths: min={cfg.trt_min_seq_len}, opt={cfg.trt_opt_seq_len}, max={cfg.trt_max_seq_len}")

    onnx_exporter.export_onnx_to_trt(
        trt_model_dir=str(cfg.trt_model_path),
        profiles=input_profiles,
        override_layernorm_precision_to_fp32=cfg.override_layernorm_precision_to_fp32,
        override_layers_to_fp32=cfg.override_layers_to_fp32,
        profiling_verbosity=cfg.profiling_verbosity,
        trt_builder_flags=[trt.BuilderFlag.VERSION_COMPATIBLE],
    )


def verify_onnx_export(onnx_exporter: Any, cfg: ExportConfig) -> None:
    """Verify ONNX export with a simple forward pass.

    Args:
        onnx_exporter: The ONNX exporter instance.
        cfg: Export configuration.
    """
    print("  Verifying ONNX export...")
    prompt = ["hello", "world"]
    dimensions = [2, 4] if cfg.use_dimension_arg else None

    try:
        result = onnx_exporter.forward(prompt, dimensions)
        print(f"    Test embeddings shape: {result.shape if hasattr(result, 'shape') else 'OK'}")
    except Exception as e:
        print(f"    Warning: Verification failed: {e}")


def run_export(cfg: ExportConfig) -> dict:
    """Run embedding model export to ONNX/TensorRT.

    Args:
        cfg: Export configuration.

    Returns:
        Dictionary with export paths.
    """
    print(f"🚀 Embedding Model Export to ONNX/TensorRT")
    print(f"=" * 60)
    print(f"Model path:      {cfg.model_path}")
    print(f"Pooling mode:    {cfg.pooling_mode}")
    print(f"Normalize:       {cfg.normalize}")
    print(f"Attention impl:  {cfg.attn_implementation}")
    print(f"Quantization:    {cfg.quant_cfg or 'None'}")
    print(f"ONNX output:     {cfg.onnx_export_path}")
    print(f"Export to TRT:   {cfg.export_to_trt}")
    if cfg.export_to_trt:
        print(f"TRT output:      {cfg.trt_model_path}")
    print(f"=" * 60)
    print()

    # Validate model path exists
    if not cfg.model_path.exists():
        print(f"Error: Model not found at {cfg.model_path}")
        print("       Please run stage2_finetune first or specify a valid model_path.")
        sys.exit(1)

    # Create output directories
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.onnx_export_path.mkdir(parents=True, exist_ok=True)
    if cfg.export_to_trt:
        cfg.trt_model_path.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": str(cfg.model_path),
        "onnx_path": str(cfg.onnx_export_path),
    }

    # Step 1: Load the embedding model
    print(f"📦 Loading embedding model from: {cfg.model_path}")
    model, tokenizer = load_embedding_model(
        model_path=cfg.model_path,
        pooling_mode=cfg.pooling_mode,
        normalize=cfg.normalize,
        attn_implementation=cfg.attn_implementation,
    )
    print(f"   Model loaded successfully")
    print()

    # Step 2: Export to ONNX
    print(f"📤 Exporting to ONNX...")
    onnx_exporter = export_to_onnx(model, tokenizer, cfg)
    print(f"   ONNX model saved to: {cfg.onnx_export_path}")
    print()

    # Step 3: Verify ONNX export
    verify_onnx_export(onnx_exporter, cfg)
    print()

    # Step 4: Export to TensorRT (optional)
    if cfg.export_to_trt:
        print(f"⚡ Exporting to TensorRT...")
        export_onnx_to_tensorrt(onnx_exporter, cfg)
        results["trt_path"] = str(cfg.trt_model_path)
        print(f"   TensorRT engine saved to: {cfg.trt_model_path}")
        print()

    print(f"✅ Export complete!")
    print(f"   ONNX model:     {cfg.onnx_export_path}")
    if cfg.export_to_trt:
        print(f"   TensorRT model: {cfg.trt_model_path}")

    return results


def main(cfg: ExportConfig | None = None) -> dict:
    """Entry point for export.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Dictionary with export paths.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, ExportConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_export(cfg)


if __name__ == "__main__":
    main()
