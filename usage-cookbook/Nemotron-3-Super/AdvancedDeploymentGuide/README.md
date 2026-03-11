# Nemotron 3 Super — Advanced Deployment Guide

## Architecture Considerations

Three properties of Nemotron 3 Super that directly affect inference configuration:

**LatentMoE** — Expert computation happens in a compressed latent dimension (`d=4096 → ℓ=1024`). All-to-all routing traffic is reduced ~4× vs a standard MoE, which matters significantly for EP across NVLinks. Expert parallelism (`--enable-expert-parallel` / `--ep`) is strongly preferred over pure TP for this architecture.

**MTP (Multi-Token Prediction)** — One MTP layer is baked into the checkpoint. This layer functions as a tail augmented draft model (similar to Eagle or other MTP heads) for speculative decoding. Unlike external draft models, additional KV cache and latency overhead is minimal as there is only a single layer called per predicted token.

**Mamba-2 Hybrid** — SSM state cache (`mamba_ssm_cache`) is distinct from the KV cache. Use `float32` for all checkpoint precisions.

## Pinned Versions

| Framework | Pinned Version / Image |
|---|---|
| vLLM | `0.17.1` |
| SGLang | `lmsysorg/sglang:v0.5.9` |
| TRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5` |

## vLLM

### Install

```bash
pip install vllm==0.17.1
```

### Baseline Serve Command (4× GB200, 8k/64k)

```bash
VLLM_FLASHINFER_MOE_BACKEND=latency \
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
vllm serve $MODEL_CKPT \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --max-cudagraph-capture-size 512 \
  --reasoning-parser-plugin super_v3_reasoning_parser.py \
  --reasoning-parser super_v3
```

### Env Vars

| Variable | Value | Effect |
|---|---|---|
| `VLLM_FLASHINFER_MOE_BACKEND` | `latency` | TRT-LLM Gen kernels — optimal for online/latency-bound serving. Use `throughput` (CUTLASS) for offline batch jobs. |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `1` | Enables NVFP4 MoE kernels. Blackwell only. |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `1` | Enables FP8 MoE kernels. |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `trtllm` | Fixes allreduce on certain topologies. Fixed upstream in [#35793](https://github.com/vllm-project/vllm/pull/35793). |

### MTP (Speculative Decoding)

Add to the base command:

```bash
  --speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}'
```

### Optional Flags Reference

```bash
# Triton attention backend — required on some configurations. Fixed upstream in vllm#35219.
--attention-backend TRITON_ATTN

# Reduce CUDA graph memory if headroom is tight (default 512)
--max-cudagraph-capture-size 256

# For unconstrained memory: prefer chunked prefill off
--no-enable-chunked-prefill        # required for vLLM <= 0.15.0 due to accuracy bug

# SSM cache precision — float32 for all checkpoint precisions
--mamba-ssm-cache-dtype float32

# Cap context length for fair benchmarking or memory control
--max-model-len 65536
```

## SGLang

### Docker Pull

```bash
docker pull lmsysorg/sglang:v0.5.9
```

### Baseline Serve Command (4× GB200, 8k/64k)

```bash
python3 -m sglang.launch_server \
  --model nvidia/NVIDIA-Nemotron-3-Super \
  --trust-remote-code \
  --tp 4 \
  --ep 4 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3 \
  --chunked-prefill-size 8192
```

### MTP (Speculative Decoding)

```bash
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5
```

> **NVFP4/FP8 + MTP on v0.5.9**: MTP does not work with NVFP4/FP8 on the `v0.5.9` image. The fix is merged into SGLang main. Use the nightly image `nightly-dev-20260310-0fd9a57d` and add `--disable-radix-cache`.

### SSM Cache

```bash
--mamba-ssm-dtype float32    # use for all checkpoint precisions
```

Not required if baked into the checkpoint (which it is for released Nemotron 3 Super checkpoints).

---

## TensorRT-LLM

### Docker Pull

> **Requires branch build**: These configs depend on changes not yet merged into the `1.3.0rc7` release [image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc7). Build TRT-LLM from `main` [branch](https://github.com/NVIDIA/TensorRT-LLM) before using these configs.

TRT-LLM requires an `extra_llm_api_options` YAML for MoE backend, KV cache, and CUDA graph settings that can't be passed as CLI flags.

### Config A — NVFP4, 2× B200 (TEP2, Latency-Optimized)

Optimal for a 2-GPU B200 node running NVFP4 with MTP enabled.

**`y.yaml`**

```yaml
trust_remote_code: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
  mamba_ssm_cache_dtype: float32
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
  allow_advanced_sampling: true
cuda_graph_config:
  max_batch_size: 16
  enable_padding: true
moe_config:
  backend: TRTLLM
stream_interval: 1
enable_chunked_prefill: true
```

**Serve command**

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe \
  trtllm-serve /data/super_fp4/ \
    --host 0.0.0.0 \
    --port 8000 \
    --max_batch_size 16 \
    --tp_size 2 \
    --ep_size 2 \
    --max_num_tokens 8192 \
    --max_seq_len 262144 \
    --extra_llm_api_options y.yaml
```

**Config rationale**

- `max_batch_size: 16` — Conservative for 2 GPUs. Balances MTP draft acceptance overhead vs. throughput.
- `tp_size 2 / ep_size 2` — Full EP across both GPUs. On LatentMoE, EP reduces all-to-all by ~4× vs TP at the same GPU count.
- `mamba_ssm_cache_dtype: float32` — Use float32 for all checkpoint precisions.
- `enable_block_reuse: false` — Mamba recurrent state is not prefix-cacheable; block reuse has no benefit here.
- `num_nextn_predict_layers: 3` — Drives MTP with 3 speculative draft steps. Average acceptance length is 3.45 on SPEED-Bench at draft length 7.
- `allow_advanced_sampling: true` — Required for MTP sampler compatibility.
- `enable_chunked_prefill: true` — Reduces inter-token latency on long prompts by interleaving prefill and decode steps.

### Config B — NVFP4, 8× B200 (DEP8, Throughput-Optimized)

Optimal for a full 8-GPU B200 node (DGX B200) serving NVFP4.

**`y.yaml`**

```yaml
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
moe_config:
   backend: TRTLLM
cuda_graph_config:
    enable_padding: true
    max_batch_size: 256
enable_attention_dp: true
num_postprocess_workers: 4
enable_chunked_prefill: true
stream_interval: 10
```

Additionally, TRT-LLM supports using a quantized mamba cache with stochastic rounding to improve throughput. Extend the `kv_cache_config` with the following info.

```
kv_cache_config:
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
```

**Serve command**

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe \
  trtllm-serve /data/super_fp4/ \
  --host 0.0.0.0 \
  --port 8000 \
  --max_batch_size 256 \
  --tp_size 8 --ep_size 8 \
  --max_num_tokens 8192 \
  --trust_remote_code \
  --reasoning_parser nano-v3 \
  --tool_parser qwen3_coder \
  --extra_llm_api_options y.yaml
```

### Config C — NVFP4, DGX Spark

Config to deploy the model on 1x DGX Spark.

```
kv_cache_config:
  enable_block_reuse: false
cuda_graph_config:
  max_batch_size: 32
  enable_padding: true
moe_config:
  backend: CUTLASS
EOF
```

**Serve command**

```bash
trtllm-serve /data/super_fp4/ \
  --host 0.0.0.0 \
  --port 8000 \
  --max_batch_size 4 \
  --trust_remote_code \
  --reasoning_parser nano-v3 \
  --tool_parser qwen3_coder \
  --extra_llm_api_options y.yaml
```

### Updated reasoning parser

To use the `force_nonempty_content` kwarg in the chat template, build TRT-LLM from `main`. Alternatively, the changes from [PR-12061](https://github.com/NVIDIA/TensorRT-LLM/pull/12061) can be manually cherry-picked into the release container to enable it.

### Contributors: 

The configurations in this document were created by:

Izzy Putterman, Nave Assaf, Joyjit Daw, and many other talented NVIDIA engineers.