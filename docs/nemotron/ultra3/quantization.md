# Quantization

This stage applies post-training quantization (PTQ) to the aligned Nemotron 3 Ultra model for efficient deployment on Blackwell GPUs.

## Overview

Quantization improves inference efficiency in several ways: quantized GEMMs increase compute throughput, quantized weights reduce model memory footprint, and quantized caches accelerate memory-bound workloads such as decoding.

One quantized checkpoint is released for Nemotron 3 Ultra:

| Checkpoint | Target Hardware | Format | Key Benefit |
|------------|-----------------|--------|-------------|
| **NVFP4 + FP8 hybrid** | Blackwell (B200) | NVFP4 for routed experts (W4A4), FP8 per-tensor for shared experts and Mamba mixer linears, FP8 KV cache | ~3.3× weight memory reduction (1.1 TB → 331 GB measured); fits on 1 node (4× B200) vs 2 nodes (8× B200) for BF16 |

The checkpoint is produced using [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) PTQ with [Megatron-Bridge](../nvidia-stack.md#megatron-bridge), driven by the [`super-nvfp4-max-calib.yaml`](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/models/Nemotron-3-Super-120B-A12B/super-nvfp4-max-calib.yaml) recipe.

## NVFP4 + FP8 Hybrid Checkpoint

The recipe quantizes the largest compute-bound GEMMs to NVFP4 (the routed experts, which dominate FLOPs at high concurrency) while keeping the more numerically sensitive components in FP8 or BF16. NVFP4 W4A4 offers roughly 1.5×–2.2× higher GEMM FLOPS than FP8 on Blackwell GPUs and reduces those weights ~4× relative to BF16. For Nemotron 3 Ultra (≈560B parameters, 108 layers, 512 routed experts), this is what enables single-node deployment on 4× B200.

### Precision Settings

Per the recipe:

| Configuration | NVFP4 + FP8 Checkpoint | BF16 Baseline |
|---------------|------------------------|---------------|
| Embedding | BF16 | BF16 |
| Attention GEMM (`{q,k,v,o}_proj`) | BF16 | BF16 |
| KV Cache + Attention BMM1 | FP8 | FP8 |
| Attention BMM2 | BF16 | BF16 |
| MoE Routed Experts (`experts.*.{up,down}_proj`) | **NVFP4 W4A4, group_size 16, e4m3 scale** | BF16 |
| MoE Shared Experts (`shared_experts.{up,down}_proj`) | **FP8 per-tensor** | BF16 |
| Latent MoE (`fc{1,2}_latent_proj`) | BF16 | BF16 |
| MoE Router (`gate.weight`) | BF16 weights, FP32 softmax compute | BF16 / FP32 |
| Mamba `mixer.in_proj` / `mixer.out_proj` | **FP8 per-tensor** | BF16 |
| Mamba SSM Kernel | FP32 | FP32 |
| Mamba 1D Conv | BF16 | BF16 |
| MTP head, lm_head, output | BF16 | BF16 |

### Mamba State Quantization

The Mamba SSM cache is kept in FP32 (`--mamba_ssm_cache_dtype float32`) for numerical stability during decode. Per-token recurrent quantization in this cache accumulates errors across every token, and the cost of keeping it in FP32 is small relative to the routed-expert GEMM speedups from NVFP4.

## Commands

### Quantize

```bash
export HF_MODEL=<HF_MODEL_PATH>
export MEGATRON_SAVE_PATH=<MEGATRON_QUANTIED_MODEL_PATH>
export RECIPE_YAML=/opt/venv/lib/python3.12/site-packages/modelopt_recipes/models/Nemotron-3-Super-120B-A12B/super-nvfp4-max-calib.yaml

python examples/quantization/quantize.py \
    --hf-model-id $HF_MODEL \
    --export-quant-cfg $RECIPE_YAML \
    --megatron-save-path $MEGATRON_SAVE_PATH \
    --pp 9 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

> **Note (parallelism)**: The 108-layer Ultra model requires `--pp 9` (12 layers/stage) to fit calibration state in memory. Smaller `--pp` values OOM the first or last pipeline stage on H100. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` for additional allocator headroom.

> **Note (recipe path)**: The path above assumes the `modelopt_recipes` package is mounted into the container at the standard site-packages location. If running outside the container, point `--export-quant-cfg` at the on-disk YAML directly.

### Resume Quantized Megatron Checkpoint and Generate

```bash
python examples/quantization/ptq_generate.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_SAVE_PATH \
    --pp 9 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

This sanity-checks the quantized Megatron checkpoint with sample prompts before exporting. Same parallelism as `quantize.py`. A working checkpoint typically produces self-identification output such as:

```
Prompt 1: Hello!
Generated: " I'm a language model developed by researchers from NVIDIA.
</think>My name is Nemotron 3 Ultra. I am created by NVIDIA researchers."
```

### Export Quantized Megatron Checkpoint to HuggingFace

After quantization, export the Megatron checkpoint back to HuggingFace format:

```bash
export EXPORT_DIR=<HF_QUANTIED_MODEL_PATH>

python examples/quantization/export.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_SAVE_PATH \
    --export-dir $EXPORT_DIR \
    --pp 9 \
    --tp 1 \
    --ep 8 \
    --dtype bfloat16 \
    --trust-remote-code
```

---

## Infrastructure

This stage uses the following components from the [NVIDIA AI Stack](../nvidia-stack.md):

| Component | Role | Documentation |
|-----------|------|---------------|
| [Megatron-Core](../nvidia-stack.md#megatron-core) | Distributed training primitives (TP, PP, EP) | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| [Megatron-Bridge](../nvidia-stack.md#megatron-bridge) | PTQ quantization, checkpoint export | [Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/) |
| [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | Quantization algorithms (FP8, NVFP4) | [GitHub](https://github.com/NVIDIA/TensorRT-Model-Optimizer) |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | NVFP4/FP8 GEMM kernels | [GitHub](https://github.com/NVIDIA/TransformerEngine) |

---

## Reference

- [Nemotron 3 Ultra Tech Report](https://research.nvidia.com/labs/nemotron/) — Quantization methodology (§4)
- [Megatron-Bridge Nemotron 3 Ultra](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — MB documentation and examples (`ultra-v3` branch)
- [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) — PTQ quantization
- [NVIDIA AI Stack](../nvidia-stack.md) — Megatron-Core, Megatron-Bridge, Transformer Engine
- [Stage 1: SFT](./sft.md) — SFT alignment (input to quantization)
- **Recipe Source**: `src/nemotron/recipes/ultra3/` — Implementation details
- [Back to Overview](./README.md)
