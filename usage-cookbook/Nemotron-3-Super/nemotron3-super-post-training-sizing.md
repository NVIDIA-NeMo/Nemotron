# Post-Training T-Shirt Sizing: Nemotron 3 Super 120B-A12B

> Minimum GPU configurations for LoRA fine-tuning, full SFT, and GRPO of
> [NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
> across three NeMo training pathways: **Megatron-Bridge**, **Automodel**, and
> **NeMo RL**.

> **Target hardware: NVIDIA H100 80 GiB SXM.** All GPU counts, memory estimates,
> and "fits" conclusions in this document assume **H100 SXM with 80 GiB HBM3**
> and **BF16 precision**. Other GPUs will have different limits — for example,
> a B200 (192 GiB HBM3e) could hold the model at much lower EP values, while an
> A100 40 GiB would require substantially more GPUs. All memory figures use
> **binary GiB** (1 GiB = 2^30 bytes), matching `nvidia-smi` output.

> **WARNING — EP=1 recipe default does not fit on H100.** The Megatron-Bridge
> PEFT recipe defaults to `expert_model_parallel_size=1`. With ~225 GiB of BF16
> weights, the full model **cannot fit on any single H100 GPU** — unlike the
> smaller Nano model. You **must** override EP via CLI (e.g.
> `model.expert_model_parallel_size=8`). See [details below](#megatron-bridge).

> **WARNING — Mamba-2 LoRA constraint:** Do NOT apply LoRA to `out_proj` or
> `conv1d` in Mamba-2 layers. Fused CUDA kernels bypass `forward()`, causing
> zero gradients and `merge_and_unload()` failures. All frameworks exclude these
> modules — verify your config does too. See [details below](#mamba-2-fused-kernel-constraint-on-lora-targets).

---

## T-Shirt Sizing: How Many GPUs Do I Need?

### At a Glance (H100 80 GiB)

| Framework | Min LoRA (tested) | Min LoRA (memory) | Min Full SFT | Recommended LoRA |
|-----------|:-----------------:|:-----------------:|:------------:|:----------------:|
| **Megatron-Bridge** | 64 GPUs (8 nodes)† | 8 GPUs (1 node)†† | 64 GPUs (8 nodes) | 64 GPUs (8 nodes) |
| **Automodel** | 8 GPUs (1 node) | 8 GPUs (1 node) | 32 GPUs (4 nodes) | 8 GPUs (1 node) |
| **NeMo RL** (GRPO) | **128 GPUs (16 nodes)** | — | — | 256 GPUs (32 nodes, production) |

> † The Megatron-Bridge SLURM scripts test two configs on 64 GPUs:
> `TP=8,EP=64` and `TP=4,EP=8,CP=2`. Smaller configs are memory-viable but
> untested in official recipes.
>
> †† The PEFT recipe (`nemotron_3_super.py`) defaults to EP=1, which **does not
> fit** on H100. Override `model.expert_model_parallel_size` to at least 8. With
> EP=8 on 8 GPUs, each GPU holds ~41 GiB of base weights — memory-viable with
> ~56 GiB total for LoRA at seq=2048, but this is a **memory extrapolation**
> with no tested recipe or SLURM script. All tested configs use 64 GPUs.

### Megatron-Bridge

The recipe Python file (`nemotron_3_super.py`) defaults to TP=1/EP=1 for PEFT
and TP=1/EP=8 for SFT (re-verified 2026-07 — both defaults persist). However,
the SLURM scripts and docs use much larger parallelism (TP=8,EP=64 on 64
GPUs; docs recommend TP=4,EP=8 for full FT and TP=4,CP=2 for LoRA). The
recipe defaults **must be overridden** for H100 — the full model is ~225 GiB
in BF16, so neither EP=1 PEFT nor TP=1 full SFT can fit verbatim. A user
running the recipe defaults unmodified will OOM. Both EP and TP are
overridable via Hydra-style CLI. Valid EP values are divisors of 512
(EP=1, 2, 4, 8, 16, 32, 64, 128, 256, 512).

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Max MBS | Status |
|------|-----:|------:|-------------|----------:|--------:|--------:|--------|
| Min LoRA (memory only) | 8 | 1 | EP=8, TP=1, PP=1 | 32 | 2048 | 2 | **Untested** — memory extrapolation; no recipe or SLURM script |
| Min LoRA (memory only) | 16 | 2 | EP=8, TP=2, PP=1 | 32 | 2048 | 4 | **Untested** — memory extrapolation |
| Min LoRA (memory only) | 32 | 4 | EP=8, TP=4, PP=1 | 32 | 4096 | 2 | **Untested** — memory extrapolation |
| **Recommended LoRA** | **64** | **8** | **EP=64, TP=8, PP=1** | **32** | **4096** | **4** | **Tested — SLURM script (`slurm_peft.sh`)** |
| Recommended LoRA | 64 | 8 | EP=8, TP=4, CP=2 | 32 | 4096 | 2 | Tested — SLURM script (`slurm_peft.sh`) |
| **Min Full SFT** | **64** | **8** | **EP=64, TP=8, PP=1** | — | 4096 | 2 | **Tested — SLURM script (`slurm_sft.sh`)** |
| Full SFT | 64 | 8 | EP=64, TP=4, CP=2 | — | 4096 | 1 | Tested — SLURM script (`slurm_sft.sh`) |

Container: `nvcr.io/nvidia/nemo:26.02.nemotron_3_super`

> All commands replace the recipe's unsafe LoRA target list (see the Mamba-2
> section). The 64-GPU configs are **multi-node**: a bare
> `torchrun --nproc-per-node=8` creates only an 8-rank local world — use the
> SLURM scripts (which set up rendezvous across 8 nodes) or add
> `--nnodes=8 --node-rank=$NODE_RANK --rdzv-endpoint=$MASTER_ADDR:29500` on
> every node.

```bash
SAFE_TARGETS='peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]'

# LoRA, 8 nodes x 8 GPUs (TP=8, EP=64) — preferred launch: sbatch slurm_peft.sh
# Manual torchrun equivalent (run on EACH of the 8 nodes):
torchrun --nproc-per-node=8 --nnodes=8 --node-rank=$NODE_RANK \
  --rdzv-endpoint=$MASTER_ADDR:29500 \
  examples/models/nemotron/nemotron_3/super/finetune_nemotron_3_super.py \
  --peft lora "$SAFE_TARGETS" \
  train.global_batch_size=16 \
  train.micro_batch_size=1 \
  train.train_iters=200 \
  model.tensor_model_parallel_size=8 \
  model.expert_model_parallel_size=64 \
  model.sequence_parallel=True \
  scheduler.lr_warmup_iters=10 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt

# LoRA with context parallelism (8 nodes, TP=4, EP=8, CP=2): as above but
#   model.tensor_model_parallel_size=4 model.expert_model_parallel_size=8 \
#   model.context_parallel_size=2

# Fewer GPUs via EP override (1 node, 8 GPUs, EP=8 — memory extrapolation, untested)
torchrun --nproc-per-node=8 examples/models/nemotron/nemotron_3/super/finetune_nemotron_3_super.py \
  --peft lora "$SAFE_TARGETS" \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  train.train_iters=200 \
  model.tensor_model_parallel_size=1 \
  model.expert_model_parallel_size=8 \
  model.sequence_parallel=True \
  scheduler.lr_warmup_iters=10 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt
```

> Configs below 64 GPUs require overriding `model.expert_model_parallel_size` on
> the command line and have not been validated in official recipes. They are
> memory-viable but may surface untested code paths in the MoE token dispatcher.
>
> **Full SFT uses 64 GPUs (8 nodes) in the SLURM scripts.** The tested configs
> are `TP=8,EP=64` and `TP=4,EP=64,CP=2`.

### Automodel (FSDP2)

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Max MBS | Status |
|------|-----:|------:|-------------|----------:|--------:|--------:|--------|
| **Min LoRA** | **8** | **1** | EP=8, FSDP2 | 8 | 2048–4096 | 1–8 | Config exists; requires 8×80 GiB |
| Recommended LoRA | 16 | 2 | EP=8, FSDP2 | 8 | 4096 | 4 | More headroom |
| Long context LoRA | 16 | 2 | EP=8, FSDP2 | 8 | 8192 | 1 | Activation-bound |
| **Full SFT** | **32** | **4** | EP=32, FSDP2 | — | 2048 | 1 | Config exists |

```bash
# Minimum LoRA config (1 node, 8 GPUs, EP=8, seq_len=4096)
torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag_peft.yaml

# Full SFT (4 nodes, 32 GPUs, EP=32)
torchrun --nproc-per-node=8 --nnodes=4 examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag.yaml
```

> The LoRA cookbook config (`base-peft-config-cookbook.yaml`) uses
> `activation_checkpointing: true` to fit on 8 GPUs at seq_len=4096. The
> standard hellaswag PEFT config omits this setting and uses shorter sequences.
> Enable activation checkpointing for longer sequences to avoid OOM.
>
> **FSDP2 sharding semantics (verified in code, 2026-07):** Automodel
> FSDP-shards ALL frozen weights — expert weights are FSDP-sharded across the
> `ep_shard` mesh on top of their EP assignment
> (`components/moe/parallelizer.py`). **Resident base weight ≈ 225 GiB / N
> GPUs regardless of EP**; EP determines expert placement and transient
> all-gather peaks during compute, which come on top of the resident figure.

### NeMo RL

> **No standalone SFT recipes** exist for NeMo RL with the Super 120B model
> (the only "super" SFT recipe in the RL repo is for the unrelated
> Llama-3.3-Nemotron-Super-49B, and it is disabled). For SFT, use
> Megatron-Bridge or Automodel instead.

New since April 2026: the standalone RL repo now ships **two 128-GPU GRPO
recipes** for Super 120B — the new minimum — alongside the larger
production-scale configs in the Nemotron repo:

| Size | GPUs | Nodes | Parallelism | Seq Len | Workflow | Source / Status |
|------|-----:|------:|-------------|--------:|----------|-----------------|
| **GRPO (min, DTensor)** | **128** | **16x8** | FSDP2/automodel, EP=8; vLLM TP=8 | 8,192 | GRPO | RL repo recipe |
| **GRPO (min, Megatron)** | **128** | **16x8** | TP=4, EP=32; vLLM TP=8; `mtp_num_layers: 5` | 8,192 | GRPO | RL repo recipe |
| GRPO (default) | 256 | 32x8 | TP=2, EP=8, PP=2, CP=4; vLLM TP=4 colocated | 49,152 | GRPO | Nemotron repo — production |
| GRPO (RLVR) | 872 | 109x8 | TP=4, EP=8, CP=8; vLLM TP=4 **non-colocated** (72 gen nodes) | 65,536 | RLVR | Nemotron repo — production |
| RLVR / SWE / RLHF pipeline | 512–1464 | 64–183 nodes/stage | per-stage configs | — | multi-stage | `RL/docs/guides/nemotron-3-super.md` |

Containers: `nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super` (Nemotron-repo
stage recipes) or `nvcr.io/nvidia/nemo-rl:v0.7.0` (RL-repo guide and recipes).

```bash
# Minimum GRPO (16 nodes x 8 GPUs, DTensor/automodel backend)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.yaml

# Minimum GRPO (16 nodes x 8 GPUs, Megatron backend, TP=4/EP=32)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-16n8g-megatron.yaml
```

> **MTP during RL varies by recipe** — check before assuming: the RLVR config
> disables it (`mtp_num_layers: 0`), the GRPO production default sets no MTP
> key, and the new RL-repo Megatron recipe sets `mtp_num_layers: 5`. MTP
> layers add weight and activation memory on the hosting pipeline stage.
>
> **GPU counts are total cluster** (training + generation). Generation
> topology also varies: the 256-GPU GRPO default colocates vLLM
> (`gpu_memory_utilization: 0.5`) on the training GPUs; the 872-GPU RLVR
> config dedicates 72 of 109 nodes to generation (non-colocated).
>
> The Nemotron repo also has a full-SFT recipe (`super3/stage1_sft`) that
> delegates to the Megatron-Bridge SFT config — note it inherits the Bridge
> defaults (TP=1, EP=8), which do NOT fit on H100 verbatim. The **shipped
> recipe contains no parallelism overrides**; TP=4 appears only in the docs'
> example commands as a documented manual override. Pass it yourself.

---

## Sequence Length Scaling

Weight memory is fixed regardless of sequence length. **Activation memory scales
linearly** and becomes the dominant consumer beyond ~4K tokens.

| Regime | Seq Length | Bottleneck | What to tune |
|--------|-----------|------------|--------------|
| **Weight-bound** | < 4K | Frozen expert weights | EP (more GPUs = fewer experts/GPU) |
| **Activation-bound** | > 4K | Activations from forward pass | TP, CP, MBS, activation recomputation |

### Activation memory vs. sequence length

For 8x H100 (Megatron-Bridge, EP=8, TP=1, LoRA rank=32), static memory ~44 GiB/GPU
(base weights ~41 GiB + adapters <0.2 GiB + ~3 GiB framework):

> Activation estimates below are approximate and assume activation checkpointing
> with MBS=1. Actual values vary with recomputation strategy and MoE dispatch.

| Seq Length | Activation/GPU | Total/GPU | Fits 80 GiB? | Max MBS |
|-----------:|---------------:|----------:|:------------:|--------:|
| 1,024 | ~5 GiB | ~50 GiB | Yes | 4 |
| 2,048 | ~10 GiB | ~55 GiB | Yes | 2 |
| 4,096 | ~20 GiB | ~65 GiB | Yes | 1 |
| **8,192** | **~40 GiB** | **~85 GiB** | **OOM** | **—** |

> **Max MBS** = largest power-of-2 micro-batch size fitting in 80 GiB with a
> 2 GiB safety margin. Use gradient accumulation to reach the desired global
> batch size when MBS is constrained.

**Key takeaway:** On a single 8-GPU node with EP=8, the maximum finetuning
sequence length is approximately **4K tokens** (MBS=1). This is substantially
lower than the Nano model (~16K) because the base weight footprint per GPU is
~41 GiB (EP=8) vs ~11 GiB (Nano EP=8). Beyond 4K, add TP or CP.

### Scaling to longer sequences

| Seq Length | TP | CP | EP | Min GPUs | Nodes | Act/GPU | Total/GPU | Fits? |
|-----------:|---:|---:|---:|---------:|------:|--------:|----------:|:-----:|
| 4,096 | 1 | 1 | 8 | 8 | 1 | ~20 GiB | ~65 GiB | Yes |
| 8,192 | 2 | 1 | 8 | 16 | 2 | ~20 GiB | ~54 GiB | Yes |
| 8,192 | 4 | 1 | 8 | 32 | 4 | ~10 GiB | ~38 GiB | Yes |
| 32,768 | 4 | 2 | 8 | 64 | 8 | ~20 GiB | ~48 GiB | Yes |
| 65,536 | 4 | 4 | 8 | 128 | 16 | ~20 GiB | ~48 GiB | Yes |
| 65,536 | 4 | 8 | 8 | 256 | 32 | ~10 GiB | ~38 GiB | Yes |

> TP splits both activations and non-expert weights across GPUs. Expert weights
> remain split by EP only. CP splits activation memory for long sequences. The
> SLURM scripts test configs like `TP=8,EP=64` and `TP=4,EP=8,CP=2`.

### Levers for longer sequences (SFT)

Teacher-forced SFT maintains **no KV cache**, so KV-cache quantization is not
an SFT lever (it applies to inference and RL rollout generation). For SFT
activation pressure, in order of impact: activation recomputation (selective
→ full/uniform), TP and CP, MBS=1 + gradient accumulation. Note only 8 of 88
layers are attention — Mamba-2 and MoE activations dominate here regardless.

---

# Appendix: Technical Details

The sections below explain *why* the sizing tables above look the way they do.

---

## Model Architecture Summary

| Property | Value |
|----------|-------|
| Total parameters | 120.7B backbone (123.6B on-disk incl. MTP block) |
| Active parameters per token | 12.7B |
| Precision | BF16 |
| Architecture | Hybrid Mamba-2 + LatentMoE + Attention |
| Layers | 88 (40 Mamba-2 + 40 MoE + 8 Attention) |
| Hidden size | 4096 |
| Vocab size | 131,072 |
| Experts per MoE layer | 512 routed + 1 shared |
| Experts activated per token | 22 |
| MoE latent dimension | 1024 |
| Expert FFN hidden size | 2688 (shared expert: 5376) |
| Activation function (experts) | Squared ReLU |
| MTP layers | 2 (shared weight) |
| Attention heads (Q / KV) | 32 / 2 (GQA) |
| Head dimension | 128 |
| Mamba state dimension | 128 |
| Mamba heads / head dim | 128 / 64 |
| Mamba groups | 8 |
| Default finetuning seq length | 2048–4096 |
| Max supported context | 1,048,576 (1M) |

---

## LatentMoE: Impact on Memory and Parallelism

Unlike the Nano model's standard MoE (where experts operate in the full hidden
dimension), Super uses **LatentMoE** — a hardware-aware sparse design that
reduces communication and memory:

1. **Latent projection**: Tokens are projected from hidden=4096 to latent=1024
   *before* routing and expert computation. This is a shared (non-expert)
   projection per MoE layer.
2. **Smaller experts**: Each expert operates in 1024-dim latent space instead
   of 4096-dim hidden space, making each expert ~4x smaller than a standard MoE
   expert would be.
3. **Reduced EP communication**: All-to-all traffic between GPUs during expert
   dispatch is in 1024-dim space — approximately **4x less communication** than
   standard MoE.
4. **More experts per GPU**: At EP=8, each GPU holds 64 experts at ~26 GiB total
   (vs ~52 GiB if experts used the full 4096-dim hidden space).
5. **Latent up-projection**: After expert computation, results are projected
   back from 1024-dim to 4096-dim hidden space. This is also a shared projection.

The net effect: despite having 4x more experts (512 vs 128) and higher topk
(22 vs 6), the per-expert weight footprint is manageable due to the latent
space design.

---

## Multi-Token Prediction (MTP) Layers

The Super checkpoint contains **one physical MTP block** (an attention + MoE
layer pair, ~2.94B params / ~5.5 GiB BF16) whose weights are *reused across
prediction depths* (`mtp_num_layers=2` with `mtp_use_repeated_layer=true`).
This is separate physical weight on top of the 120.7B backbone — the on-disk
checkpoint is 123.6B params / 230.2 GiB. It is **not negligible** for memory
planning, though its MoE experts do shard by EP like any other MoE layer:

- **During SFT**: MTP is enabled with `mtp_loss_scaling_factor=0.3`. The MTP
  block adds its weights (on the last PP stage) plus activation memory for
  each prediction depth during forward/backward.
- **During RL**: varies by recipe — the RLVR config disables MTP
  (`mtp_num_layers=0`, `mtp_loss_scaling_factor=0.0`), the GRPO production
  default sets no MTP key, and the RL-repo 128-GPU Megatron recipe sets
  `mtp_num_layers=5`. Check your config; enabled MTP adds activation memory.
- **During inference**: MTP enables native speculative decoding for faster
  generation.

The per-GPU weight tables above are backbone-only; add the MTP block's share
(~2.94B / (EP × PP) for its experts, small non-expert remainder on the last
stage) when MTP is enabled — e.g. the TP=8/EP=64 figure rises from ~5.1 to
~5.24 GiB/GPU.

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

Automodel explicitly excludes these modules, and the Megatron-Bridge LoRA
cookbook sidesteps them:
- Automodel: `exclude_modules: ["*.out_proj"]`
- Megatron-Bridge LoRA cookbook: targets only `["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]` (conservative — omits `in_proj` and `out_proj` entirely)

**The Megatron-Bridge recipe default remains unsafe** (re-verified 2026-07):
it still includes `out_proj` in the target list
`["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]`
with no Mamba-2 filtering anywhere in `peft/lora.py` or
`peft/module_matcher.py`. Manually exclude Mamba-2 `out_proj` (or use the
conservative cookbook target list) and validate adapter merge on a small run.

---

## Parameter Breakdown

| Component | Params | BF16 Memory | % of Backbone |
|-----------|-------:|------------:|-----------:|
| Routed experts (512 × 40 MoE layers) | 112.74B | 210.0 GiB | 93.4% |
| Mamba-2 layers (40) | 4.39B | 8.2 GiB | 3.6% |
| Shared experts (40 layers) | 1.76B | 3.3 GiB | 1.5% |
| Embeddings + LM head | 1.07B | 2.0 GiB | 0.9% |
| Latent projections (40 layers) | 0.34B | 0.6 GiB | 0.3% |
| Attention layers (8) | 0.29B | 0.5 GiB | 0.2% |
| Routers + norms | 0.08B | 0.2 GiB | <0.1% |
| **Backbone total** | **120.67B** | **224.8 GiB** | |
| MTP block (1 physical attn+MoE block, reused across depths) | +2.94B | +5.5 GiB | → **123.61B / 230.2 GiB on disk** |

> Each routed expert has exactly 2 × 1024 × 2688 = 5,505,024 parameters
> (latent-space up/down projections, squared ReLU). Layer split 40/40/8 and
> component totals validated against the HF safetensors metadata
> (123.61B on-disk total). Earlier revisions misattributed Mamba vs attention
> params (2.8B/0.9B) — the errors cancelled in the total.

**Key insight:** 93% of the model weight lives in the 512 routed experts per
MoE layer — the same proportion as the Nano model. Expert Parallelism (EP) is
the dominant factor for fitting in GPU memory. However, unlike Nano where EP=1
fits on a single H100, Super requires **at least EP=4** (and realistically
EP=8) due to the 4x larger model.

### Non-expert vs. expert split

| Category | Params | BF16 Memory |
|----------|-------:|------------:|
| Non-expert (shared across all GPUs) | 7.93B | 14.8 GiB |
| Routed experts (split by EP) | 112.74B | 210.0 GiB |

---

## LoRA Adapter Overhead

**What actually gets adapted depends on the framework** — earlier revisions
assumed one adapter per routed expert, which no shipped config does:

- **Megatron-Bridge**: `share_expert_adapters=True` by default — the 512
  grouped experts share ONE adapter per grouped linear per EP rank, not 512
  individual adapters.
- **Automodel**: with only `exclude_modules` set (the shipped configs), the
  fallback matcher adapts **`nn.Linear` modules only** — grouped experts are
  never adapted (they'd need an explicit target like `*.experts`).

| Config (rank) | Trainable Params | Adapter BF16 | Notes |
|---------------|----------------:|------------:|-------|
| Automodel default (r=8) | ~20M | ~37 MiB | non-expert linears only |
| Bridge recipe (r=32, incl. unsafe out_proj) | ~83M | ~158 MiB | shared expert adapters |
| Bridge safe targets (r=32) | ~38M | ~73 MiB | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` |
| *Per-expert adapters (r=8)* | *~1.22B* | *~2.3 GiB* | *only if experts explicitly targeted — not shipped* |
| *Per-expert adapters (r=32)* | *~4.87B* | *~9.1 GiB* | *not a shipped configuration* |

> Counts are logical single-copy numbers (Bridge expert adapters are per EP
> rank). Practical takeaway: **shipped LoRA configs train 20-85M params —
> adapter and optimizer memory are negligible**; Super's GPU minimums are set
> entirely by frozen weights and activations.

---

## Per-GPU Memory Estimation

Memory components during LoRA training:

1. **Base model weights** (frozen, BF16) — 2 bytes/param
2. **LoRA adapter weights** (trainable, BF16) — 2 bytes/param
3. **Optimizer states** (Adam FP32 momentum + variance + master copy) — 12 bytes/trainable param, sharded across DP ranks with distributed optimizer
4. **Gradients** (BF16) — 2 bytes/trainable param
5. **Activations** — depends on seq length, micro-batch size, recomputation (~10 GiB at seq=2048, MBS=1 with activation checkpointing)
6. **Framework overhead** (CUDA context, NCCL buffers) — ~3 GiB

> Totals below include base weights + LoRA overhead (with distributed optimizer)
> + estimated activations at seq=2048 MBS=1. Framework overhead (~3 GiB) is not
> included in the table totals but is accounted for in the Max MBS calculations
> via a 2 GiB safety margin.

### Megatron-Bridge (EP-based, no FSDP)

Non-expert weights are **replicated** across data-parallel ranks. Expert weights
are split by EP. The PEFT recipe defaults to EP=1, but this **must be
overridden** — any value where `512 % EP == 0` works.

Assumptions: LoRA rank=32, seq_len=2048, MBS=1, activation checkpointing.

| Config | GPUs | EP | TP | PP | Base Wt/GPU | Total/GPU | Fits? | Notes |
|--------|-----:|---:|---:|---:|------------:|----------:|:-----:|-------|
| 4 GPU | 4 | 4 | 1 | 1 | 67.1 GiB | ~82 GiB | **OOM** | Expert memory alone dominates |
| **8 GPU (1 node)** | **8** | **8** | **1** | **1** | **40.8 GiB** | **~56 GiB** | **Yes** | **Untested — memory extrapolation** |
| 16 GPU (2 nodes) | 16 | 8 | 2 | 1 | 33.4 GiB | ~47 GiB | Yes | Untested — memory extrapolation |
| 32 GPU (4 nodes) | 32 | 8 | 4 | 1 | 29.7 GiB | ~42 GiB | Yes | Untested — memory extrapolation |
| **64 GPU (8 nodes)** | **64** | **64** | **8** | **1** | **5.1 GiB** | **~18 GiB** | **Yes** | **Tested — SLURM script (`slurm_peft.sh`)** |
| 64 GPU (8 nodes) | 64 | 8 | 4 | 1 | 29.7 GiB | ~42 GiB | Yes | Tested — SLURM script (`slurm_peft.sh`, CP=2) |

> The TP=8/EP=64 row divides non-expert weights by TP (14.6/8 ≈ 1.8 GiB) and
> expert weights by EP (209.8/64 ≈ 3.3 GiB), assuming ETP=1. An earlier
> revision of this table showed 17.9 GiB for this row (non-expert weights
> not divided by TP) — that figure only applies to a TP=1, EP=64 layout.

> Configs below 64 GPUs require overriding `model.expert_model_parallel_size` on
> the command line and have not been validated in official recipes. They are
> memory-viable but may surface untested code paths in the MoE token dispatcher.
>
> **Full SFT requires 64 GPUs (8 nodes) in the tested SLURM scripts.** The
> tested configs use TP=8,EP=64 and TP=4,EP=64,CP=2. Optimizer states for
> 120.6B params (12 bytes/param with distributed optimizer divided across DP
> ranks) push memory much higher than LoRA.

### Automodel (FSDP2)

Frozen weights are **sharded** across data-parallel ranks via FSDP2, in addition
to expert weights being split by EP. This allows fewer GPUs than Megatron-Bridge.

Assumptions: LoRA rank=8, seq_len=4096, MBS=1, activation checkpointing.

| Config | GPUs | EP | Resident Base Wt/GPU | Total/GPU | Fits? | Notes |
|--------|-----:|---:|------------:|----------:|:-----:|-------|
| **8 GPU (1 node)** | **8** | **8** | **~28 GiB** | **~53 GiB** | **Yes** | **Cookbook config — validated on 8×80 GiB** |
| 16 GPU (2 nodes) | 16 | 8 | ~14 GiB | ~35 GiB | Yes | Resident = 225/16; more DP, better throughput |
| 32 GPU (4 nodes, Full SFT) | 32 | 32 | ~7 GiB | varies | Yes | Full SFT config |

> Resident = 225 GiB / N GPUs (see sharding semantics above); transient
> all-gather peaks during expert compute come on top.

---

## Why MoE LoRA Sizing Differs from Dense Models

For a dense 120B model, LoRA typically reduces the minimum GPU requirement
dramatically. For MoE models like Nemotron 3 Super, LoRA helps with optimizer
memory but the base weight footprint remains the binding constraint:

1. **Expert weights dominate** — 93% of the ~225 GiB is in the 512 routed
   experts. Even frozen during LoRA, they must reside in GPU memory. **Unlike
   the Nano model, EP=1 is impossible on H100** — 225 GiB >> 80 GiB.
2. **EP trades memory for GPUs** — with EP=8, each GPU holds 64 experts
   (~26 GiB). With EP=64, each holds 8 experts (~3.3 GiB). The minimum viable
   EP for H100 is 4 (tight) or 8 (recommended).
3. **LoRA savings are on the optimizer side** — instead of Adam states for
   120.7B params (full SFT), shipped configs train only ~20-85M adapter params
   (see LoRA Adapter Overhead), making optimizer memory essentially free. The
   base weight footprint remains unchanged.
4. **Sequence length scales activation memory linearly** — at 2048, activations
   are ~10 GiB/GPU. At 8K they consume ~40 GiB/GPU, making the 8-GPU EP=8
   config OOM.
5. **LatentMoE mitigates expert size** — each expert is ~4x smaller than it
   would be with standard 4096-dim MoE, enabling 512 experts with manageable
   per-GPU memory. Without LatentMoE, this model would require significantly
   higher EP.

### Memory comparison: Full SFT vs. LoRA (H100 80 GiB, EP=8)

| Component | Full SFT (64 GPU) | LoRA rank=32 (8 GPU) | LoRA rank=8 (8 GPU, FSDP2) |
|-----------|:-----------------:|:--------------------:|:---------------------------:|
| Base weights/GPU | ~41 GiB | ~41 GiB | ~28 GiB |
| Optimizer/GPU | ~24 GiB | ~0.1 GiB | <0.1 GiB |
| Gradients/GPU | ~12 GiB | <0.1 GiB | <0.1 GiB |
| Activations/GPU (seq=2048) | ~10 GiB | ~10 GiB | ~10 GiB |
| **Total/GPU** | **~87 GiB** | **~51 GiB** | **~38 GiB** |

> LoRA optimizer/gradient figures reflect shipped adapter behavior (~83M
> trainable at Bridge r=32, ~20M at Automodel r=8 — see LoRA Adapter
> Overhead). For LoRA, per-GPU memory ≈ frozen weights + activations.

> Full SFT at EP=8 does not fit on 8 GPUs — optimizer + gradient memory pushes
> beyond 80 GiB. The tested SFT configs use 64 GPUs (EP=64 or EP=8 with higher
> TP/DP) to distribute the optimizer load.

---

## How FSDP2 Enables Fewer GPUs (Automodel)

Automodel uses PyTorch-native FSDP2 rather than relying solely on Expert
Parallelism.

### Memory distribution comparison

```
Megatron-Bridge (EP-based, no FSDP) — example with EP=8:
  GPU 0: [all non-expert weights] + [experts 0-63]
  GPU 1: [all non-expert weights] + [experts 64-127]   <-- 14.6 GiB replicated
  ...
  (8 GPUs, EP=8: each holds 14.6 GiB shared + 26.2 GiB experts = 40.8 GiB)

Megatron-Bridge with EP override — example with EP=64:
  GPU 0: [all non-expert weights] + [experts 0-7]
  GPU 63: [all non-expert weights] + [experts 504-511]  <-- 14.6 GiB replicated
  (64 GPUs, EP=64: each holds 14.6 GiB shared + 3.3 GiB experts = 17.9 GiB)

Automodel (FSDP2) — example with 8 GPUs, EP=8:
  GPU 0: [1/8 non-expert weights] + [1/8 of all expert shards]
  GPU 1: [1/8 non-expert weights] + [1/8 of all expert shards]
  ...
  (8 GPUs: resident = 225/8 = ~28 GiB each — experts are FSDP-sharded across
   the ep_shard mesh on top of EP, so resident weight is total/N for any EP)
```

FSDP2 shards **all** frozen weights (not just experts); EP sets expert
placement and transient materialization peaks. The constraint becomes:

```
min_gpus >= ep_size   (where dp_size * cp_size must be divisible by ep_size)
```

The tradeoff: FSDP2 adds communication overhead for all-gathering frozen weights
during forward/backward passes.

### Key implementation details

- **Device mesh**: `(pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)`
- **Expert weights**: sharded across EP dimension via `distribute_tensor(Shard(0))`
- **Non-expert frozen weights**: sharded across DP ranks via `fully_shard()`
- **LoRA adapters**: replicated on all ranks (tiny, ~1% of model)
- **Gradient sync**: only LoRA gradients synced across DP; frozen param gradients = 0

---

## Activation Memory Breakdown by Layer Type

At seq_len=4096, MBS=1 (where activations become significant):

| Layer Type | Activation/GPU | % of Total Activations |
|------------|---------------:|-----------------------:|
| MoE layers (40) | ~13 GiB | ~65% |
| Mamba-2 layers (40) | ~5 GiB | ~25% |
| Attention layers (8) | ~1 GiB | ~5% |
| MTP block (during SFT) | ~0.5 GiB | ~2.5% |
| Embeddings + other | ~0.5 GiB | ~2.5% |

> These are estimates with activation checkpointing. MoE layers dominate
> activations because each token activates 22 experts in LatentMoE, and the
> intermediate activations for all active experts must be stored for the backward
> pass (or recomputed). The LatentMoE design reduces per-expert activation size
> (1024-dim vs 4096-dim) but the high topk=22 partially offsets this savings.

---

## References

### Documentation
- Megatron-Bridge docs: `Megatron-Bridge/docs/models/nemotron/nemotron3-super.md`
  (full FT: TP=4, EP=8, SP; LoRA example: TP=4, CP=2)
- HuggingFace model card: [NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
- Nemotron training docs: `docs/nemotron/super3/{sft,pretrain,quantization}.md`, `docs/nemotron/super3/rl/`
- NeMo RL Super pipeline guide: `RL/docs/guides/nemotron-3-super.md`
  (per-stage production node counts: RLVR 183, SWE 64, RLHF 100 H100 nodes)

### Recipes and Configs (paths current as of 2026-07)
- Megatron-Bridge recipe: `Megatron-Bridge/src/megatron/bridge/recipes/nemotronh/nemotron_3_super.py`
- Megatron-Bridge SLURM (LoRA): `Megatron-Bridge/examples/models/nemotron/nemotron_3/super/slurm_peft.sh`
- Megatron-Bridge SLURM (SFT): `Megatron-Bridge/examples/models/nemotron/nemotron_3/super/slurm_sft.sh`
- Automodel PEFT config: `Automodel/examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag_peft.yaml`
- Automodel Full SFT config: `Automodel/examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag.yaml`
- Automodel benchmark configs (EP=64 + activation checkpointing): `Automodel/examples/llm_benchmark/nemotron/nemotron_super_v3_{lora,te_deepep}.yaml`
- Automodel LoRA cookbook: `usage-cookbook/Nemotron-3-Super/lora-text2sql/nemo-automodel/base-peft-config-cookbook.yaml`
- Megatron-Bridge LoRA cookbook: `usage-cookbook/Nemotron-3-Super/lora-text2sql/nemo-megatron-bridge/train.py`
- NeMo RL GRPO 128-GPU (DTensor): `RL/examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.yaml`
- NeMo RL GRPO 128-GPU (Megatron): `RL/examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-16n8g-megatron.yaml`
- Nemotron GRPO production default: `src/nemotron/recipes/super3/stage2_rl/config/default.yaml`
- Nemotron RLVR default: `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/config/default.yaml`
- NeMo RL GRPO cookbook: `usage-cookbook/Nemotron-3-Super/grpo-dapo/grpo_training_cookbook.ipynb`

### Performance Data
- Automodel performance: `Automodel/docs/performance-summary.mdx` (120B pretraining: 334 TFLOPs/sec/GPU on 64x H100)
