# Quantization

This stage applies post-training quantization (PTQ) to the aligned Nemotron 3 Ultra model for efficient deployment on Blackwell GPUs.

---

## Overview

Quantization improves inference efficiency in several ways: quantized GEMMs increase compute throughput, quantized weights reduce model memory footprint, and quantized caches accelerate memory-bound workloads such as decoding.

One quantized checkpoint is released for Nemotron 3 Ultra:

| Checkpoint | Target Hardware | Format | Key Benefit |
|------------|-----------------|--------|-------------|
| **NVFP4** (W4A4) | Blackwell (B200) | NVFP4 weights and activations | ~4x weight memory reduction; fits on 1 node (4× B200) vs 2 nodes (8× B200) for BF16 |

The checkpoint is produced using [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq) PTQ with [Megatron-Bridge](../nvidia-stack.md#megatron-bridge).

---

## NVFP4 Checkpoint

FP4 is attractive for efficient inference because NVFP4 offers roughly 1.5x--2.2x higher GEMM FLOPS than FP8 on Blackwell GPUs, while also reducing model memory footprint by about 4x relative to BF16. For Nemotron 3 Ultra (≈561B parameters, 108 layers, 512 routed experts), this is what enables single-node deployment on 4× B200.

### Precision Settings

| Configuration | NVFP4 Checkpoint | BF16 Baseline |
|---------------|------------------|---------------|
| Embedding | BF16 | BF16 |
| Attention GEMM (QKV and Out Projection) | NVFP4 | BF16 |
| KV Cache + Attention BMM1 | FP8 | FP8 |
| Attention BMM2 | BF16 | BF16 |
| MoE GEMM (Routed Experts) | NVFP4 | BF16 |
| MoE Router | FP32 | FP32 |
| Mamba GEMM | NVFP4 | BF16 |
| Mamba SSM Kernel | FP32 | FP32 |
| Mamba 1D Conv | BF16 | BF16 |
| Output Layers | BF16 | BF16 |

### Mamba State Quantization

The Mamba SSM cache is kept in FP32 (`--mamba_ssm_cache_dtype float32`) for numerical stability during decode. Per-token recurrent quantization in this cache accumulates errors across every token, and the cost of keeping it in FP32 is small relative to the MoE GEMM speedups from NVFP4.

---

## Quantization Configurations

Nemotron 3 Ultra supports four quantization configurations tailored for the Mamba-MoE architecture:

| Config Name | Format | Description |
|---|---|---|
| `mamba_moe_fp8_aggressive` | FP8 | Aggressive FP8 quantization for Mamba-MoE |
| `mamba_moe_fp8_conservative` | FP8 | Conservative FP8 quantization for Mamba-MoE |
| `mamba_moe_nvfp4_aggressive` | NVFP4 | Aggressive NVFP4 quantization for Mamba-MoE |
| `mamba_moe_nvfp4_conservative` | NVFP4 | Conservative NVFP4 quantization for Mamba-MoE |

Pass the desired config name via `--export-quant-cfg` to `quantize.py`. The conservative NVFP4 config is the released recipe for Ultra.

---

## Recipe Execution

### Direct Script Execution (Megatron-Bridge)

For direct execution, use the scripts in the [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) repository:

```bash
# Clone the repository and checkout the ultra-v3 branch
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout ultra-v3
```

### Quantize

```bash
export HF_MODEL=/models/nemotron-ultra-rl-050826/
export MEGATRON_SAVE_PATH=/models/nemotron-ultra-rl-050826-NVFP4-CONSERV-MLM

python examples/quantization/quantize.py \
    --hf-model-id $HF_MODEL \
    --export-quant-cfg mamba_moe_nvfp4_conservative \
    --megatron-save-path $MEGATRON_SAVE_PATH \
    --pp 9 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

> **Note**: The 108-layer Ultra model requires `--pp 9` (12 layers/stage) to fit calibration state in memory. Smaller `--pp` values OOM the first or last pipeline stage on B200. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` for additional allocator headroom.

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

This sanity-checks the quantized Megatron checkpoint with sample prompts before exporting. Same parallelism as `quantize.py`.

### Export Quantized Megatron Checkpoint to HuggingFace

After quantization, export the Megatron checkpoint back to HuggingFace format:

```bash
export EXPORT_DIR=/models/nemotron-ultra-rl-050826-NVFP4-CONSERV-MLM_hf

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

### Parallelism Configuration

| Parallelism | Quantize / Generate | Export | Flag |
|-------------|---------------------|--------|------|
| Tensor (TP) | 8 | 1 | `--tp` |
| Pipeline (PP) | 9 | 9 | `--pp` |
| Expert (EP) | 8 | 8 | `--ep` |

**Minimum resources:** 1 node with 8× B200 GPUs. The 108-layer model requires `--pp 9` (12 layers/stage) so calibration state fits in memory.

---

## Reference

- [Nemotron 3 Ultra Tech Report](https://research.nvidia.com/labs/nemotron/) — Quantization methodology (§4)
- [Megatron-Bridge Nemotron 3 Ultra](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — MB documentation and examples (`ultra-v3` branch)
- [Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) — PTQ quantization
- [NVIDIA AI Stack](../nvidia-stack.md) — Megatron-Core, Megatron-Bridge, Transformer Engine
- [Stage 1: SFT](./sft.md) — SFT alignment (input to quantization)
- **Recipe Source**: `src/nemotron/recipes/ultra3/` — Implementation details
- [Back to Overview](./README.md)
