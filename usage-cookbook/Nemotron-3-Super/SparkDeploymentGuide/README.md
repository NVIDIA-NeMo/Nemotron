# Nemotron 3 Super — DGX Spark Deployment Guide

DGX Spark ships a single Grace-Blackwell GPU with 128 GB of unified memory. This guide covers serving Nemotron 3 Super on a single DGX Spark using vLLM (nightly) and TensorRT-LLM.

## Architecture Refresher

Three properties of Nemotron 3 Super that directly affect inference configuration:

**LatentMoE** — Expert computation happens in a compressed latent dimension (`d=4096 → ℓ=1024`). All-to-all routing traffic is reduced ~4× vs a standard MoE. On a single-GPU system like DGX Spark, expert parallelism is not applicable — set `--tensor-parallel-size 1`.

**MTP (Multi-Token Prediction)** — One MTP layer is baked into the checkpoint. This layer functions as a tail augmented draft model (similar to Eagle or other MTP heads) for speculative decoding. Unlike external draft models, additional KV cache and latency overhead is minimal as there is only a single layer called per predicted token.

**Mamba-2 Hybrid** — SSM state cache (`mamba_ssm_cache`) is distinct from the KV cache. Use `float32` for all checkpoint precisions (unless throughput-optimizing with stochastic rounding — see TRT-LLM section).

---

## Reasoning Parser

Both vLLM and TRT-LLM require the Nemotron 3 Super reasoning parser. Download it before starting the server:

```bash
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
```

---

## vLLM

### Image

```
vllm/vllm-openai:cu130-nightly
```

MTP + NVFP4 on DGX Spark requires a vLLM nightly build (cu130). The pinned release `0.17.1` does not support this combination on a single-GPU Spark configuration.

### Serve Command

```bash
VLLM_NVFP4_GEMM_BACKEND=marlin \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
VLLM_USE_FLASHINFER_MOE_FP4=0 \
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --served-model-name nemotron-3-super \
  --host 0.0.0.0 \
  --port 8000 \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --max-model-len 1000000 \
  --moe-backend marlin \
  --mamba_ssm_cache_dtype float32 \
  --quantization fp4 \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --reasoning-parser super_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Env Vars

| Variable | Value | Notes |
|---|---|---|
| `VLLM_NVFP4_GEMM_BACKEND` | `marlin` | Use Marlin GEMM backend for NVFP4 on Spark. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Required to allow `--max-model-len 1000000` on a single GPU. |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `trtllm` | Fixes allreduce on single-GPU topology. Fixed upstream in [#35793](https://github.com/vllm-project/vllm/pull/35793). |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Disabled — FlashInfer FP4 MoE kernels are Blackwell multi-GPU only. Marlin backend handles FP4 on Spark. |

### Key Flag Rationale

| Flag | Value | Rationale |
|---|---|---|
| `--kv-cache-dtype fp8` | FP8 KV cache | Reduces memory footprint to fit 1M context in 128 GB unified memory. |
| `--max-num-seqs 4` | 4 | Conservative concurrency for single-GPU memory headroom. |
| `--max-model-len 1000000` | 1M tokens | Full context window. Requires `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. |
| `--moe-backend marlin` | Marlin | Marlin MoE backend for single-GPU NVFP4 inference. |
| `--quantization fp4` | FP4 | NVFP4 quantized checkpoint. |
| `--speculative_config` | MTP, 3 draft tokens | Lightweight speculative decoding using the baked-in MTP head. Triton MoE backend for speculative path. |
| `--async-scheduling` | — | Enables async scheduling for improved single-GPU throughput. |

---

## TensorRT-LLM

### Image

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9
```

### Config — NVFP4, 1× DGX Spark

**`extra-llm-api-config.yml`**

```yaml
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
moe_config:
   backend: CUTLASS
cuda_graph_config:
    enable_padding: true
    max_batch_size: 8
enable_attention_dp: false
enable_chunked_prefill: true
stream_interval: 1
print_iter_log: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
  allow_advanced_sampling: true
```

**Serve command**

```bash
TLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
trtllm-serve <nvfp4_ckpt> \
  --host 0.0.0.0 \
  --port 8123 \
  --max_batch_size 8 \
  --tp_size 1 --ep_size 1 \
  --max_num_tokens 8192 \
  --trust_remote_code \
  --reasoning_parser nano-v3 \
  --tool_parser qwen3_coder \
  --extra_llm_api_options extra-llm-api-config.yml \
  --max_seq_len 1048576
```

### Config Rationale

| Setting | Rationale |
|---|---|
| `kv_cache_config.dtype: fp8` | FP8 KV cache to maximize context length in 128 GB unified memory. |
| `mamba_ssm_cache_dtype: float16` | FP16 SSM cache with stochastic rounding — trades negligible accuracy for significant memory savings on a single GPU. |
| `mamba_ssm_stochastic_rounding: true` | Compensates for FP16 SSM cache precision loss via stochastic rounding with 5 Philox rounds. |
| `enable_block_reuse: false` | Mamba recurrent state is not prefix-cacheable; block reuse has no benefit. |
| `free_gpu_memory_fraction: 0.9` | Aggressive — single-GPU has no inter-GPU communication overhead. |
| `moe_config.backend: CUTLASS` | CUTLASS MoE backend for single-GPU NVFP4. |
| `max_batch_size: 8` | Conservative for single-GPU memory. |
| `num_nextn_predict_layers: 3` | MTP with 3 speculative draft steps. |
| `max_seq_len: 1048576` | Full 1M token context window. |

---

## Contributors

The configurations in this document were created by Izzy Putterman, Nave Assaf, Joyjit Daw, and many other talented NVIDIA engineers.
