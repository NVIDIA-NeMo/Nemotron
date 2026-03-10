# Stage 3: Quantization

This stage applies post-training quantization (PTQ) to the aligned Nemotron 3 Super model for efficient deployment across GPU generations.

---

## Overview

Quantization improves inference efficiency in several ways: quantized GEMMs increase compute throughput, quantized weights reduce model memory footprint, and quantized caches accelerate memory-bound workloads such as decoding.

Two quantized checkpoints are released:

| Checkpoint | Target Hardware | Format | Key Benefit |
|------------|-----------------|--------|-------------|
| **FP8** (W8A8) | Hopper (H100) | FP8 weights and activations | Balanced accuracy/throughput |
| **NVFP4** (W4A4) | Blackwell (B200) | NVFP4 weights and activations | 1.5--2.2x higher GEMM FLOPS than FP8 |

Both checkpoints are produced using [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq) PTQ with [Megatron-Bridge](../nvidia-stack.md#megatron-bridge).

---

## FP8 Checkpoint

The FP8 checkpoint quantizes MoE GEMMs and Mamba GEMMs to FP8, with FP8 KV Cache quantization. The Mamba state cache is quantized to FP16 (from FP32) for speedup. Calibration used 256 samples from the post-training SFT dataset.

### Precision Settings

| Configuration | FP8 Checkpoint | BF16 Baseline |
|---------------|----------------|---------------|
| Embedding | BF16 | BF16 |
| Attention GEMM (QKV and Out Projection) | BF16 | BF16 |
| KV Cache + Attention BMM1 | FP8 | FP8 |
| Attention BMM2 | BF16 | BF16 |
| MoE GEMM (Sparse Experts and Shared Experts) | FP8 | BF16 |
| MoE Latent Projection GEMM | BF16 | BF16 |
| Router | FP32 | FP32 |
| Mamba GEMM | FP8 | BF16 |
| Mamba SSM Kernel | FP16 | FP32 |
| Mamba 1D Conv | BF16 | BF16 |
| Output Layers | BF16 | BF16 |

---

## NVFP4 Checkpoint

FP4 is attractive for efficient inference because NVFP4 offers roughly 1.5x--2.2x higher GEMM FLOPS than FP8 on Blackwell GPUs, while also reducing model memory footprint by about 2x. This makes FP4 especially appealing for prefill-heavy workloads, such as coding-agent deployments, where MoE GEMMs dominate latency.

### FP4 PTQ Recipe

The best results were obtained with a **hybrid FP4 recipe**:

| Component | Scaling Method | Rationale |
|-----------|---------------|-----------|
| **Weight per-block scales** | Minimizing weight MSE | Calibrated offline; supports scale search |
| **Activation per-block scales** | Max-based scaling | Must be computed efficiently at runtime |

Despite these improvements, the PTQ recipe still left a median accuracy gap of more than 1% relative to BF16. To recover this loss, **AutoQuantize** is used to automatically assign each layer to FP4, FP8, or BF16 based on both sensitivity and performance cost.

AutoQuantize is a mixed-precision quantization algorithm that casts format assignment as a neural architecture search (NAS) problem under a deployment-cost budget. It estimates operator sensitivity using a second-order Taylor approximation (inspired by Optimal Brain Surgeon), models performance cost, and solves for the allocation that minimizes total sensitivity subject to the cost constraint.

**Result:** The mixed-precision PTQ process completes in less than 2 hours on a single B200 node (8 GPUs) using 512 samples. The resulting model achieves **99.8% median accuracy** relative to BF16 while retaining near-FP4 performance.

### FP4 QAD (Quantization-Aware Distillation)

QAD uses the BF16 checkpoint as teacher and the NVFP4 checkpoint as student:

| Parameter | Value |
|-----------|-------|
| **Calibration** | 2K samples, 131K context from post-training reasoning SFT |
| **Loss Function** | Logit-based loss (best of logit, logit+LM, hidden-cosine) |
| **Learning Rate** | 1e-5 |
| **Data Blend** | SFT + RL on-policy rollouts (60:40 ratio) |
| **Training Budget** | 5B tokens |

### Mamba State Quantization

The Mamba SSM cache presents a unique quantization challenge: during training the cache is computed via chunked SSD without per-token quantization boundaries, but during inference per-token recurrent quantization accumulates errors across every token.

**Selected Recipe:** FP16 with Stochastic Rounding (Philox<5>)

| Reason | Detail |
|--------|--------|
| No block scales required | Simpler implementation |
| Blackwell hardware support | Dedicated PTX instruction for FP16 conversion with stochastic rounding |
| cuRAND support | Philox PRNG on Blackwell via cuRAND |
| Accuracy | Maintains accuracy and verbosity with Philox round count of 5 |

---

## Recipe Execution

### Quick Start

<div class="termy">

```console
// Quantize model to FP8
$ uv run nemotron super3 quantize --format fp8 --run YOUR-CLUSTER

// Quantize model to NVFP4
$ uv run nemotron super3 quantize --format nvfp4 --run YOUR-CLUSTER
```

</div>

### Megatron-Bridge PTQ Commands

For direct execution using [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer):

```bash
# Quantize to FP8
python -m modelopt.torch.quantize \
    --model-path /path/to/rl/checkpoint \
    --output-path /path/to/fp8/checkpoint \
    --format fp8 \
    --calib-size 256

# Quantize to NVFP4 with AutoQuantize
python -m modelopt.torch.quantize \
    --model-path /path/to/rl/checkpoint \
    --output-path /path/to/nvfp4/checkpoint \
    --format nvfp4 \
    --autoquantize \
    --calib-size 512

# Export for TensorRT-LLM deployment
python -m modelopt.torch.export \
    --model-path /path/to/quantized/checkpoint \
    --output-path /path/to/trt-llm/engine
```

### Configuration

| File | Purpose |
|------|---------|
| `config/quantize/fp8.yaml` | FP8 quantization settings |
| `config/quantize/nvfp4.yaml` | NVFP4 quantization settings |

---

## Quantized Model Evaluation

Comparison of BF16, FP8, and NVFP4 checkpoints:

| Benchmark | N-3-Super (BF16) | N-3-Super FP8 | N-3-Super NVFP4 |
|-----------|-------------------|---------------|-----------------|
| **General Knowledge** | | | |
| MMLU-Pro | 83.57 | 83.78 | 83.41 |
| **Reasoning** | | | |
| GPQA (no tools) | 79.29 | 79.67 | 79.23 |
| LiveCodeBench (v6) | 78.25 | 78.80 | 78.57 |
| SciCode (subtask) | 40.64 | 39.87 | 39.94 |
| HLE (no tools) | 18.02 | 17.70 | 17.33 |
| **Agentic** | | | |
| Terminal Bench (hard) | 25.78 | 26.82 | 25.78 |
| TauBench V2 Airline | 57.00 | 55.00 | 55.25 |
| TauBench V2 Retail | 65.13 | 62.17 | 63.71 |
| TauBench V2 Telecom | 60.96 | 62.39 | 60.63 |
| **Chat & IF** | | | |
| IFBench (prompt) | 72.91 | 71.25 | 72.79 |
| Multi-Challenge | 52.31 | 54.55 | 51.70 |
| Arena-Hard-V2 | 75.19 | 74.83 | 75.50 |
| **Long Context** | | | |
| AA-LCR | 57.63 | 58.13 | 59.25 |
| **Multilingual** | | | |
| MMLU-ProX (avg) | 80.00 | 78.97 | 79.36 |

---

## Infrastructure

| Component | Role | Documentation |
|-----------|------|---------------|
| [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | PTQ, AutoQuantize, QAD | [GitHub](https://github.com/NVIDIA/Model-Optimizer) |
| [Megatron-Bridge](../nvidia-stack.md#megatron-bridge) | Checkpoint management | [Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/) |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | NVFP4/FP8 GEMM kernels | [GitHub](https://github.com/NVIDIA/TransformerEngine) |

### Parallelism Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| Tensor (TP) | 4 | Tensor parallelism for quantization |
| Expert (EP) | 8 | Expert parallelism for MoE layers |

---

## Reference

- [Nemotron 3 Super Tech Report](TBD) — Quantization methodology
- [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) — PTQ and AutoQuantize
- [NVIDIA AI Stack](../nvidia-stack.md) — Megatron-Bridge, Transformer Engine
- [Stage 2: RL](./rl.md) — RL alignment (input to quantization)
- [Stage 4: Evaluation](./evaluate.md) — Benchmark evaluation
- **Recipe Source**: `src/nemotron/recipes/super3/` — Implementation details
- [Back to Overview](./README.md)
