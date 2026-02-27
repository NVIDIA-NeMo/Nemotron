# LoRA T-Shirt Sizing: Nemotron 3 Nano 30B-A3B

> Minimum GPU configurations for LoRA fine-tuning of
> [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
> across three NeMo training pathways: **Megatron-Bridge**, **Automodel**, and
> **NeMo RL**. All configs assume H100 80 GB GPUs and BF16 precision.

> **WARNING — Mamba-2 LoRA constraint:** Do NOT apply LoRA to `out_proj` or
> `conv1d` in Mamba-2 layers. Fused CUDA kernels bypass `forward()`, causing
> zero gradients and `merge_and_unload()` failures. All frameworks exclude these
> modules — verify your config does too. See [details below](#mamba-2-fused-kernel-constraint-on-lora-targets).

---

## T-Shirt Sizing: How Many GPUs Do I Need?

### At a Glance

| Framework | Min LoRA | Min Full SFT | Recommended LoRA | Long Context (>8K) |
|-----------|:--------:|:------------:|:----------------:|:------------------:|
| **Megatron-Bridge** | 8 GPUs (1 node) | 16 GPUs (2 nodes) | 16 GPUs (2 nodes) | 16–32 GPUs |
| **Automodel** | 4 GPUs | 8 GPUs | 8 GPUs (1 node) | 8+ GPUs |
| **NeMo RL** (SFT) | 8 GPUs (2x4) | 16 GPUs (2x8) | 16 GPUs (2x8) | — |
| **NeMo RL** (GRPO) | 16 GPUs (2x8) | — | 16 GPUs (2x8) | — |

### Megatron-Bridge

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Max MBS | Status |
|------|-----:|------:|-------------|----------:|--------:|--------:|--------|
| **Min LoRA** | **8** | **1** | EP=8, TP=1, PP=1 | 32 | 2048 | 8 | Supported (matches Qwen3-30B-A3B recipe) |
| **Min Full SFT** | **16** | **2** | EP=8, TP=1, PP=1 | — | 2048 | 4 | Official recommendation |
| Recommended LoRA | 16 | 2 | EP=8, TP=2, PP=1 | 32 | 2048 | 16 | Tested in SLURM examples |
| Long context LoRA | 16 | 2 | EP=8, TP=2, PP=1 | 32 | 8192 | 2 | TP=2 needed for activation headroom |
| Long context LoRA | 32 | 4 | EP=8, TP=4, PP=1 | 32 | 32768 | 1 | TP=4 required for 32K |

Container: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`

```bash
# Minimum LoRA config (1 node, 8 GPUs, seq_len=2048)
torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_nano.py \
  --peft lora \
  train.global_batch_size=128 \
  train.train_iters=100 \
  scheduler.lr_warmup_iters=10 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt

# Long context LoRA (2 nodes, 16 GPUs, seq_len=8192)
torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_nano.py \
  --peft lora \
  train.global_batch_size=64 \
  train.train_iters=100 \
  model.tensor_model_parallel_size=2 \
  dataset.seq_length=8192 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt
```

### Automodel (FSDP2)

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Max MBS | Status |
|------|-----:|------:|-------------|----------:|--------:|--------:|--------|
| **Min LoRA** | **4** | **< 1 node** | EP=4, FSDP2 | 8 | 2048 | 8 | Config exists, needs empirical validation |
| Recommended LoRA | 8 | 1 | EP=8, FSDP2 | 8 | 2048 | 8 | ~14 GB/GPU, ample headroom |
| Long context LoRA | 8 | 1 | EP=8, FSDP2 | 8 | 8192 | 2 | Activation-bound |
| Long context LoRA | 8 | 1 | EP=8, FSDP2 | 8 | 16384 | 1 | Near capacity |

```bash
# Minimum LoRA config (4 GPUs, seq_len=2048)
torchrun --nproc-per-node=4 examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml
```

### NeMo RL (FSDP2 / DTensor)

| Size | GPUs | Nodes | LoRA Rank | Seq Len | Workflow | Recipe |
|------|-----:|------:|----------:|--------:|----------|--------|
| Min SFT LoRA | 8 | 2x4 | 256 | 2048 | SFT | `sft-nanov3-30BA3B-2n4g-fsdp2-lora` |
| SFT LoRA | 16 | 2x8 | 256 | 2048 | SFT | `sft-nanov3-30BA3B-2n8g-fsdp2-lora` |
| Full SFT | 16 | 2x8 | — | 2048 | SFT | `sft-nanov3-30BA3B-2n8g-fsdp2` |
| GRPO LoRA | 16 | 2x8 | 128 | 2048 | GRPO | `grpo-nanov3-30BA3B-2n8g-fsdp2-lora` |

> NeMo RL GRPO uses vLLM with TP=4 for generation alongside FSDP2 for training.
> The high LoRA ranks (128–256) require additional GPUs for optimizer state memory.

```bash
# NeMo RL SFT LoRA (2 nodes, 8 GPUs/node)
uv run examples/run_sft.py \
  --config examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2-lora.yaml

# NeMo RL SFT LoRA (2 nodes, 4 GPUs/node — minimum)
uv run examples/run_sft.py \
  --config examples/configs/recipes/llm/sft-nanov3-30BA3B-2n4g-fsdp2-lora.yaml

# NeMo RL GRPO LoRA (2 nodes, 8 GPUs/node)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml
```

---

## Sequence Length Scaling

Weight memory is fixed regardless of sequence length. **Activation memory scales
linearly** and becomes the dominant consumer beyond ~8K tokens.

| Regime | Seq Length | Bottleneck | What to tune |
|--------|-----------|------------|--------------|
| **Weight-bound** | < 8K | Frozen expert weights | EP (more GPUs = fewer experts/GPU) |
| **Activation-bound** | > 8K | Activations from forward pass | TP, CP, MBS, gradient checkpointing, FP8 KV cache |

### Activation memory vs. sequence length

For 8x H100 (Megatron-Bridge, EP=8, TP=1, LoRA rank=32), static memory ~15.8 GB/GPU:

| Seq Length | Activation/GPU | Total/GPU | Fits 80 GB? | Max MBS |
|-----------:|---------------:|----------:|:-----------:|--------:|
| 1,024 | 3.1 GB | 18.9 GB | Yes | 16 |
| 2,048 | 6.1 GB | 22.0 GB | Yes | 8 |
| 4,096 | 12.2 GB | 28.1 GB | Yes | 4 |
| 8,192 | 24.5 GB | 40.3 GB | Yes | 2 |
| **16,384** | **49.0 GB** | **64.8 GB** | **Yes (tight)** | **1** |
| 32,768 | 98.0 GB | 113.8 GB | **OOM** | — |

**Key takeaway:** On a single 8-GPU node with EP=8, the maximum finetuning
sequence length is approximately **16K tokens** (MBS=1). Beyond that, add TP or CP.

### Scaling to longer sequences

| Seq Length | TP | CP | Min GPUs | Nodes | Act/GPU | Total/GPU | Fits? |
|-----------:|---:|---:|---------:|------:|--------:|----------:|:-----:|
| 16,384 | 1 | 1 | 8 | 1 | 49.0 GB | 64.8 GB | Yes |
| 32,768 | 2 | 1 | 16 | 2 | 49.2 GB | 62.0 GB | Yes |
| 32,768 | 4 | 1 | 32 | 4 | 24.8 GB | 36.1 GB | Yes |
| 65,536 | 4 | 1 | 32 | 4 | 49.5 GB | 60.9 GB | Yes |
| 65,536 | 4 | 2 | 64 | 8 | ~25 GB | ~36 GB | Yes |

### FP8 KV cache for extended context

FP8 quantization of the KV cache can approximately **halve activation memory**
for the attention components with minimal accuracy degradation. While attention
layers are only 6 of the 52 layers, FP8 KV cache is most impactful for longer
sequences and larger micro-batch sizes. Available in both Megatron-Bridge (via
Transformer Engine) and Automodel.

---

## Open Items

1. **4-GPU Automodel run**: Config exists (`ep_size=4`), memory math says it
   fits at ~22 GB/GPU. Needs an empirical run to confirm.
2. **Megatron-Bridge EP=4**: Not in existing recipes — would need recipe
   modification to test whether 4 GPUs are viable.
3. **`out_proj` LoRA target in Megatron-Bridge**: The default target list includes
   `out_proj` for all layer types. Automodel and NeMo RL explicitly exclude it for
   Mamba-2 layers. Verify whether Megatron-Bridge correctly filters Mamba-2
   `out_proj` or if this is a latent bug.
4. **Throughput benchmarks**: Neither framework has published LoRA finetuning
   performance numbers for this model. The Automodel perf page shows pretraining
   only (328 TFLOPs/sec/GPU on 8x H100).
5. **FP8 KV cache validation**: Quantify the actual activation memory savings
   and accuracy impact for this model at various sequence lengths.
6. **Long-context LoRA recipes**: No framework currently provides tested recipes
   for seq_len > 2048 on this model. The scaling tables above are theoretical.

---

# Appendix: Technical Details

The sections below explain *why* the sizing tables above look the way they do.

---

## Model Architecture Summary

| Property | Value |
|----------|-------|
| Total parameters | 31.6B |
| Active parameters per token | 3.5B |
| Precision | BF16 |
| Architecture | Hybrid Mamba-2 + MoE + Attention |
| Layers | 52 (23 Mamba-2 + 23 MoE + 6 Attention) |
| Hidden size | 2688 |
| Vocab size | 131,072 |
| Experts per MoE layer | 128 routed + 1 shared |
| Experts activated per token | 6 |
| Expert FFN hidden size | 1856 (shared expert: 3712) |
| Default finetuning seq length | 2048 |
| Max supported context | 262,144 (256K) |

---

## Mamba-2 Fused Kernel Constraint on LoRA Targets

The Mamba-2 implementation passes raw weights directly into fused CUDA kernels
(e.g., `mamba_split_conv1d_scan_combined`). When LoRA wraps these modules, the
standard `forward()` methods of the LoRA-wrapped modules are **never called** —
the kernel reads the `.weight` tensor directly. This means:

1. **LoRA adapters on these modules produce zero gradients** during training
   (no learning occurs).
2. **`merge_and_unload()` will fail** with shape mismatch errors when merging
   adapters back into the base model.

### Safe LoRA target modules

| Module | Layer Types | Safe? |
|--------|-------------|:-----:|
| `linear_qkv` | Attention | Yes |
| `linear_proj` | Attention | Yes |
| `linear_fc1` | MoE experts, shared expert | Yes |
| `linear_fc2` | MoE experts, shared expert | Yes |
| `in_proj` | Mamba-2 | Yes |
| `out_proj` | Attention: Yes, **Mamba-2: NO** | Use `exclude_modules` to filter Mamba-2 |
| `conv1d` | Mamba-2 | **NO** — fused kernel |

All NeMo frameworks explicitly exclude these modules:
- Automodel: `exclude_modules: ["*.out_proj"]`
- NeMo RL: `exclude_modules: ['*out_proj*']`

If using Megatron-Bridge, verify that the default target list
`["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]`
does not apply LoRA to Mamba-2 `out_proj` layers.

---

## Parameter Breakdown

| Component | Params | BF16 Memory | % of Total |
|-----------|-------:|------------:|-----------:|
| Routed experts (128 per layer x 23 MoE layers) | 29.37B | 58.75 GB | 93.0% |
| Mamba-2 layers (23) | 0.89B | 1.78 GB | 2.8% |
| Embeddings + LM head | 0.70B | 1.41 GB | 2.2% |
| Shared experts (23 layers) | 0.46B | 0.92 GB | 1.5% |
| Attention layers (6) | 0.14B | 0.28 GB | 0.4% |
| Routers + norms | 0.01B | 0.02 GB | <0.1% |
| **Total** | **31.58B** | **63.16 GB** | |

**Key insight:** 93% of the model weight lives in the 128 routed experts.
Expert Parallelism (EP) is the dominant factor for fitting in GPU memory.

### Non-expert vs. expert split

| Category | Params | BF16 Memory |
|----------|-------:|------------:|
| Non-expert (shared across all GPUs) | 2.20B | 4.41 GB |
| Routed experts (split by EP) | 29.37B | 58.75 GB |

---

## LoRA Adapter Overhead

| LoRA Rank | Trainable Params | BF16 Memory | % of Base | Used in |
|----------:|----------------:|------------:|----------:|---------|
| 8 | 220M | 0.44 GB | 0.70% | Automodel default |
| 32 | 883M | 1.77 GB | 2.80% | Megatron-Bridge default |
| 128 | ~3.5B | ~7.1 GB | ~11.1% | NeMo RL GRPO recipe |
| 256 | ~7.1B | ~14.1 GB | ~22.2% | NeMo RL SFT recipe |

> Higher LoRA ranks (128, 256) substantially increase optimizer state memory.
> The NeMo RL recipes use these higher ranks on 16 GPUs for a reason.

---

## Per-GPU Memory Estimation

Memory components during LoRA training:

1. **Base model weights** (frozen, BF16) — 2 bytes/param
2. **LoRA adapter weights** (trainable, BF16) — 2 bytes/param
3. **Optimizer states** (Adam FP32 momentum + variance) — 8 bytes/trainable param
4. **Gradients** (BF16) — 2 bytes/trainable param
5. **Activations** — depends on seq length, micro-batch size, recomputation
6. **Framework overhead** (CUDA context, NCCL buffers) — ~2–4 GB

### Megatron-Bridge (EP-based, no FSDP)

Non-expert weights are **replicated** across data-parallel ranks. Expert weights
are split by EP.

Assumptions: LoRA rank=32, seq_len=2048, MBS=1.

| Config | GPUs | EP | TP | PP | Base Wt/GPU | Total/GPU | Fits? | Notes |
|--------|-----:|---:|---:|---:|------------:|----------:|:-----:|-------|
| 1 GPU | 1 | 1 | 1 | 1 | 63.2 GB | ~79.7 GB | No | 0.3 GB headroom — not practical; EP=1 unsupported |
| 4 GPU | 4 | 4 | 1 | 1 | 19.1 GB | ~27.9 GB | Memory fits | EP=4 not in current recipe (hardcodes EP=8) |
| **8 GPU (1 node)** | **8** | **8** | **1** | **1** | **11.8 GB** | **~19.3 GB** | **Yes** | **Minimum LoRA config** |
| 16 GPU (2 nodes) | 16 | 8 | 2 | 1 | 9.6 GB | ~15.2 GB | Yes | Recommended; tested in SLURM script |

> **Full SFT requires 16 GPUs (2 nodes) minimum.** The official Megatron-Bridge
> docs state: *"Running this recipe requires at least 2 H100 nodes (16 GPUs)"*
> for full parameter fine-tuning (TP=1, EP=8). The 8-GPU minimum applies to
> **LoRA only**, where optimizer state memory is dramatically reduced.

### Automodel (FSDP2)

Frozen weights are **sharded** across data-parallel ranks via FSDP2, in addition
to expert weights being split by EP. This allows fewer GPUs.

Assumptions: LoRA rank=8, seq_len=2048, MBS=1.

| Config | GPUs | EP | DP | Base Wt/GPU | Total/GPU | Fits? | Notes |
|--------|-----:|---:|---:|------------:|----------:|:-----:|-------|
| **4 GPU** | **4** | **4** | **4** | **15.8 GB** | **~22.4 GB** | **Yes** | **Minimum viable** — config exists in repo |
| 8 GPU (1 node) | 8 | 4 | 8 | 15.2 GB | ~21.8 GB | Yes | More DP replicas, better throughput |
| 8 GPU (1 node) | 8 | 8 | 8 | 7.9 GB | ~14.2 GB | Yes | Most headroom — room for larger batch/seq |

---

## Why MoE LoRA Sizing Differs from Dense Models

For a dense 30B model, LoRA typically reduces the minimum GPU requirement
dramatically (e.g., from 8 GPUs to 1). For MoE models like Nemotron 3 Nano,
the story is different:

1. **Expert weights dominate** — 93% of the 63 GB is in the 128 routed experts.
   Even though they're frozen during LoRA, they still must reside in GPU memory.
2. **EP is the binding constraint** — expert weights can only be distributed via
   Expert Parallelism. With 128 experts and EP=8, each GPU holds 16 experts
   (~7.3 GB). With EP=4, each holds 32 experts (~14.7 GB).
3. **LoRA savings are on the optimizer side** — instead of Adam states for 31.6B
   params (full SFT), you only need them for ~220M params (rank 8). This saves
   ~250 GB of optimizer memory across the cluster, but the base weight footprint
   remains unchanged.
4. **Sequence length scales activation memory linearly** — at 2048, activations
   are ~6 GB/GPU. At 16K they consume ~49 GB/GPU, becoming the bottleneck
   regardless of LoRA vs. full SFT.

### Memory comparison: Full SFT vs. LoRA (16x H100, EP=8)

| Component | Full SFT (16 GPU) | LoRA rank=32 (8 GPU) | LoRA rank=256 (16 GPU) |
|-----------|:-----------------:|:--------------------:|:---------------------:|
| Base weights/GPU | 11.8 GB | 11.8 GB | 11.8 GB |
| Optimizer/GPU | ~23.6 GB | ~1.1 GB | ~8.9 GB |
| Gradients/GPU | ~11.8 GB | ~0.3 GB | ~1.8 GB |
| Activations/GPU (seq=2048) | ~6.1 GB | ~6.1 GB | ~6.1 GB |
| **Total/GPU** | **~56 GB** | **~22 GB** | **~31 GB** |

---

## How FSDP2 Enables Fewer GPUs (Automodel / NeMo RL)

Both Automodel and NeMo RL use PyTorch-native FSDP2 rather than relying solely
on Expert Parallelism.

### Memory distribution comparison

```
Megatron-Bridge (EP-based, no FSDP):
  GPU 0: [all non-expert weights] + [experts 0-15]
  GPU 1: [all non-expert weights] + [experts 16-31]   <-- 4.4 GB replicated
  ...
  (8 GPUs, EP=8: each holds 4.4 GB shared + 7.3 GB experts = 11.7 GB)

Automodel / NeMo RL (FSDP2):
  GPU 0: [1/4 non-expert weights] + [experts 0-31]
  GPU 1: [1/4 non-expert weights] + [experts 32-63]   <-- 1.1 GB each
  ...
  (4 GPUs, EP=4: each holds 1.1 GB shared + 14.7 GB experts = 15.8 GB)
```

FSDP2 shards **all** frozen weights (not just experts) across data-parallel
ranks. The constraint becomes:

```
min_gpus >= ep_size   (where dp_size * cp_size must be divisible by ep_size)
```

The tradeoff: FSDP2 adds communication overhead for all-gathering frozen weights
during forward/backward passes.

### Key implementation details

- **Device mesh**: `(pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)`
- **Expert weights**: sharded across EP dimension via `distribute_tensor(Shard(0))`
- **Non-expert frozen weights**: sharded across DP ranks via `fully_shard()`
- **LoRA adapters**: replicated on all ranks (tiny, <1% of model)
- **Gradient sync**: only LoRA gradients synced across DP; frozen param gradients = 0

---

## Activation Memory Breakdown by Layer Type

At seq_len=8192, MBS=1 (the crossover point where activations match weight memory):

| Layer Type | Activation/GPU | % of Total Activations |
|------------|---------------:|-----------------------:|
| MoE layers (23) | 16.2 GB | 66% |
| Mamba-2 layers (23) | 7.1 GB | 29% |
| Attention layers (6) | 1.1 GB | 4% |
| Embeddings + other | 0.1 GB | 1% |

MoE layers dominate activations because each token activates 6 experts, and the
intermediate activations for all active experts must be stored for the backward
pass (or recomputed).

---

## Existing Memory Calculator Limitations

The Megatron-Bridge utility at `training/utils/theoretical_memory_utils.py` has
critical gaps for this use case:

| Gap | Impact | Severity |
|-----|--------|----------|
| No Expert Parallelism (EP) | Assumes expert weights replicated — overestimates by ~8x for EP=8 | Critical |
| No LoRA/PEFT awareness | Assumes all 31.6B params are trainable — overestimates optimizer memory by ~35x | Critical |
| No FSDP2 sharding | Cannot model the Automodel/NeMo RL memory distribution | Critical |
| No FP8 KV cache | Cannot model the activation memory savings from FP8 quantization | Medium |

A corrected calculator would need to:

- Split parameters into `expert_params` and `non_expert_params`
- Divide expert params by `expert_model_parallel_size`
- For LoRA: allocate 2 bytes/param (BF16) for frozen base, 12 bytes/param
  (BF16 weight + FP32 optimizer + BF16 gradient) only for adapter parameters
- Account for FSDP sharding of frozen weights across DP (Automodel/NeMo RL path)
- Model activation memory as a function of sequence length, MBS, and precision

---

## References

### Documentation
- Nemotron 3 docs (Megatron-Bridge): [nemotron3.html](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
- HuggingFace model card: [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- NeMo RL SFT guide: `RL/docs/guides/sft.md`
- NeMo RL GRPO guide: `RL/docs/guides/grpo.md`

### Recipes and Configs
- Megatron-Bridge recipe: `Megatron-Bridge/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py`
- Megatron-Bridge SLURM example: `Megatron-Bridge/examples/models/nemotron_3/slurm_peft.sh`
- Automodel PEFT config: `Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml`
- NeMo RL SFT LoRA (2x8): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`
- NeMo RL SFT LoRA (2x4): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n4g-fsdp2-lora.yaml`
- NeMo RL GRPO LoRA: `RL/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`
- NeMo RL Full SFT: `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml`

### Performance Data
- Megatron-Bridge performance: `Megatron-Bridge/docs/performance-summary.md`
- Automodel performance: `Automodel/docs/performance-summary.md`

### Internal Utilities
- Memory utility: `Megatron-Bridge/src/megatron/bridge/training/utils/theoretical_memory_utils.py`
- FSDP2 mesh logic: `Automodel/nemo_automodel/components/distributed/mesh_utils.py`
- MoE parallelizer: `Automodel/nemo_automodel/components/moe/parallelizer.py`
