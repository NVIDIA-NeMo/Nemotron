# Post-Training T-Shirt Sizing: Nemotron 3 Nano 30B-A3B

> Minimum GPU configurations for LoRA fine-tuning, full SFT, and GRPO of
> [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
> across three NeMo training pathways: **Megatron-Bridge**, **Automodel**, and
> **NeMo RL**.

> **Target hardware: NVIDIA H100 80 GiB SXM.** All GPU counts, memory estimates,
> and "fits" conclusions in this document assume **H100 SXM with 80 GiB HBM3**
> and **BF16 precision**. Other GPUs will have different limits — for example,
> a single B200 (192 GiB HBM3e) could hold the full model at EP=1, while an
> A100 40 GiB would require more GPUs. All memory figures use **binary GiB**
> (1 GiB = 2^30 bytes), matching `nvidia-smi` output.

> **WARNING — Mamba-2 LoRA constraint:** Do NOT apply LoRA to `out_proj` or
> `conv1d` in Mamba-2 layers. Fused CUDA kernels bypass `forward()`, causing
> zero gradients and `merge_and_unload()` failures. Automodel and NeMo RL
> exclude these modules by default; **Megatron-Bridge recipes do NOT** — pass
> `peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]` on the
> command line. See [details below](#mamba-2-fused-kernel-constraint-on-lora-targets).

---

## T-Shirt Sizing: How Many GPUs Do I Need?

### At a Glance (H100 80 GiB)

All rows H100 80 GiB unless marked GB200. NeMo RL nightly CI runs H100 at
8 GPUs/node and GB200 at 4 GPUs/node — recipe names like "2n4g" are GB200.

| Framework | Min LoRA (H100) | Min LoRA (memory) | Min Full SFT (H100) | Recommended LoRA | GB200 |
|-----------|:-----------------:|:-----------------:|:------------:|:----------------:|:------------------:|
| **Megatron-Bridge** | 8 (recipe default) / 16 (launcher-tested) | 1 GPU† | 16 GPUs (2 nodes) | 16 GPUs (2 nodes) | — |
| **Automodel** | **1 GPU**†† (smoke config) | 1 GPU | 8 GPUs (1 node, config) | 8 GPUs (1 node) | — |
| **NeMo RL** (SFT) | 16 GPUs (2x8, nightly smoke) | — | **8 GPUs (1x8, nightly smoke)** | 16 GPUs (2x8) | 8 GPUs (2x4) |
| **NeMo RL** (GRPO) | 16 GPUs (2x8, nightly smoke) | — | **8 GPUs (1x8, full-param, nightly smoke)** | 16 GPUs (2x8) | — |
| **NeMo RL** (DPO) | 8 GPUs (1x8 @4K, nightly smoke) | — | — | 8 GPUs (1x8) | 4 GPUs (1x4) |

> † The Megatron-Bridge recipe defaults to EP=8, but `expert_model_parallel_size`
> is overridable via CLI (see below). With lower EP values, the model fits on
> fewer GPUs — even 1 GPU (~65 GiB) is memory-viable. Configs below 8 GPUs
> are untested.
>
> †† New since 26.02: Automodel ships an official single-GPU LoRA config
> (`nemotron_nano_v3_singlegpu_lora.yaml`) — EP=1, all 128 experts local,
> activation checkpointing + memory-efficient LoRA (rank 8, alpha 16).

### Megatron-Bridge

The recipe defaults to `expert_model_parallel_size=8`, but this is overridable
via Hydra-style CLI. Lower EP values reduce the GPU minimum — any EP where
`128 % EP == 0` is valid (EP=1, 2, 4, 8, 16, 32, 64, 128).

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Max MBS | Status |
|------|-----:|------:|-------------|----------:|--------:|--------:|--------|
| Min LoRA (memory) | 1 | — | EP=1, TP=1, PP=1 | 32 | 2048 | 2 | Untested — override EP via CLI |
| Min LoRA (memory) | 2 | — | EP=2, TP=1, PP=1 | 32 | 2048 | 8 | Untested — override EP via CLI |
| Min LoRA (memory) | 4 | < 1 | EP=4, TP=1, PP=1 | 32 | 2048 | 8 | Untested — override EP via CLI |
| **Min LoRA (tested)** | **8** | **1** | **EP=8, TP=1, PP=1** | **32** | **2048** | **8** | **Recipe default; matches Qwen3-30B-A3B recipe** |
| **Min Full SFT** | **16** | **2** | EP=8, TP=1, PP=1 | — | 2048 | 4 | Official recommendation |
| Recommended LoRA | 16 | 2 | EP=8, TP=2, PP=1 | 32 | 2048 | 16 | Tested in SLURM examples |
| Long context LoRA | 16 | 2 | EP=8, TP=2, PP=1 | 32 | 8192 | 2 | TP=2 needed for activation headroom |
| Long context LoRA | 32 | 4 | EP=8, TP=4, PP=1 | 32 | 32768 | 1 | TP=4 required for 32K |

Container: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`

> All commands below replace the recipe's default LoRA target list with the
> Mamba-safe conservative list (the default includes Mamba-2 `out_proj`,
> which silently trains nothing and breaks adapter merges — see the Mamba-2
> section). Multi-node runs additionally need `--nnodes`, `--node-rank`, and
> `--rdzv-endpoint` on each node (or use `slurm_peft.sh`, which handles this).

```bash
SAFE_TARGETS='peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]'

# Default LoRA config (1 node, 8 GPUs, EP=8)
torchrun --nproc-per-node=8 examples/models/nemotron/nemotron_3/nano/finetune_nemotron_3_nano.py \
  --peft lora "$SAFE_TARGETS" \
  train.global_batch_size=128 \
  train.train_iters=100 \
  scheduler.lr_warmup_iters=10 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt

# Fewer GPUs via EP override (e.g. 2 GPUs, EP=2 — untested)
torchrun --nproc-per-node=2 examples/models/nemotron/nemotron_3/nano/finetune_nemotron_3_nano.py \
  --peft lora "$SAFE_TARGETS" \
  model.expert_model_parallel_size=2 \
  train.global_batch_size=32 \
  train.train_iters=100 \
  scheduler.lr_warmup_iters=10 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt

# Long context LoRA (2 nodes, 16 GPUs, seq_len=8192; --seq-length sets the
# model context, dataset.seq_length the data pipeline — set both)
torchrun --nproc-per-node=8 --nnodes=2 --node-rank=$NODE_RANK \
  --rdzv-endpoint=$MASTER_ADDR:29500 \
  examples/models/nemotron/nemotron_3/nano/finetune_nemotron_3_nano.py \
  --peft lora "$SAFE_TARGETS" \
  --seq-length 8192 \
  train.global_batch_size=64 \
  train.train_iters=100 \
  model.tensor_model_parallel_size=2 \
  dataset.seq_length=8192 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt
```

### Automodel (FSDP2)

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Max MBS | Status |
|------|-----:|------:|-------------|----------:|--------:|--------:|--------|
| **Min LoRA** | **1** | **< 1 node** | EP=1, FSDP2 | 8 (α=16) | 2048 | 1 | **Official config** — activation ckpt + memory-efficient LoRA |
| Small LoRA | 4 | < 1 node | EP=4, FSDP2 | 8 | 2048 | 8 | Config exists, needs empirical validation |
| Recommended LoRA | 8 | 1 | EP=8, FSDP2 | 8 | 2048 | 8 | ~14 GiB/GPU, ample headroom |
| Long context LoRA | 8 | 1 | EP=8, FSDP2 | 8 | 8192 | 2 | Activation-bound |
| Long context LoRA | 8 | 1 | EP=8, FSDP2 | 8 | 16384 | 1 | Near capacity |

```bash
# Minimum LoRA config (1 GPU, EP=1, seq_len=2048)
automodel examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml \
  --nproc-per-node 1

# 4-GPU LoRA config (EP=4, seq_len=2048)
torchrun --nproc-per-node=4 examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml
```

### NeMo RL (FSDP2 / DTensor)

Hardware per the nightly CI manifests (`RL/tests/test_suites/`): H100 =
8 GPUs/node, GB200 = 4 GPUs/node. The test scripts' `NUM_NODES`/`GPUS_PER_NODE`
override what recipe filenames suggest.

| Size | GPUs | Hardware | LoRA Rank | Seq Len | Workflow | Recipe | Evidence |
|------|-----:|:--------:|----------:|--------:|----------|--------|----------|
| **Min Full SFT** | **8 (1x8)** | H100 | — | 2048 | SFT | `sft-nanov3-30BA3B-2n8g-fsdp2` (script runs 1 node) | nightly smoke |
| Min SFT LoRA | 16 (2x8) | H100 | 256 | 2048 | SFT | `sft-nanov3-30BA3B-2n8g-fsdp2-lora` | nightly smoke |
| SFT (either) | 8 (2x4) | GB200 | 256 / — | 2048 | SFT | `sft-nanov3-30BA3B-2n4g-fsdp2[-lora]` | GB200 nightly smoke |
| **Min GRPO (full-param)** | **8 (1x8)** | H100 | — | 2048 | GRPO | `grpo-nanov3-30BA3B-2n8g-fsdp2` (script runs 1 node) | nightly smoke |
| GRPO LoRA | 16 (2x8) | H100 | 128 | 2048 | GRPO | `grpo-nanov3-30BA3B-2n8g-fsdp2-lora` | nightly smoke |
| GRPO LoRA (Megatron backend) | 16 (2x8) | H100 | 128 | 2048 | GRPO | `grpo-nanov3-30BA3B-2n8g-megatron-lora` (EP=8) | nightly smoke |
| Min DPO | 8 (1x8) | H100 | — | 4096 | DPO | `dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel` | nightly smoke |
| DPO | 4 (1x4) | GB200 | — | 4096 | DPO | `dpo-nanov3-30B3AB-1n4g-fsdp4ep4-automodel` | GB200 nightly smoke |

> NeMo RL GRPO uses vLLM with TP=4 for generation alongside FSDP2 for training.
> Note the "high LoRA rank" caveat from earlier revisions is retired: NeMo RL's
> Automodel-backend LoRA only adapts non-expert linear layers (see the LoRA
> Adapter Overhead section), so even rank 256 is only ~0.2B trainable params.
> New since April 2026: a Megatron-backend GRPO path, context-parallel +
> sequence-packing GRPO variants (`grpo-...-megatron-pack-cp`), DPO recipes,
> and the 8-GPU full-param SFT/GRPO smoke configs. All are ~10-20-step CI
> smoke tests — they prove the topology fits in memory, not convergence.

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
| **Activation-bound** | > 8K | Activations from forward pass | TP, CP, MBS, activation recomputation |

### Activation memory vs. sequence length

For 8x H100 (Megatron-Bridge, EP=8, TP=1, LoRA rank=32), static memory ~14 GiB/GPU
(base weights + LoRA overhead + ~3 GiB framework):

> Activation estimates below are approximate and assume selective recomputation
> with MBS=1. Actual values vary with recomputation strategy and MoE dispatch.

| Seq Length | Activation/GPU | Total/GPU | Fits 80 GiB? | Max MBS |
|-----------:|---------------:|----------:|:------------:|--------:|
| 1,024 | ~3 GiB | ~17 GiB | Yes | 16 |
| 2,048 | ~6 GiB | ~20 GiB | Yes | 8 |
| 4,096 | ~12 GiB | ~26 GiB | Yes | 4 |
| 8,192 | ~24 GiB | ~38 GiB | Yes | 2 |
| **16,384** | **~48 GiB** | **~62 GiB** | **Yes (tight)** | **1** |
| 32,768 | ~96 GiB | ~110 GiB | **OOM** | — |

> **Max MBS** = largest power-of-2 micro-batch size fitting in 80 GiB with a
> 2 GiB safety margin. Use gradient accumulation to reach the desired global
> batch size when MBS is constrained.

**Key takeaway:** On a single 8-GPU node with EP=8, the maximum finetuning
sequence length is approximately **16K tokens** (MBS=1). Beyond that, add TP or CP.

### Scaling to longer sequences

| Seq Length | TP | CP | Min GPUs | Nodes | Act/GPU | Total/GPU | Fits? |
|-----------:|---:|---:|---------:|------:|--------:|----------:|:-----:|
| 16,384 | 1 | 1 | 8 | 1 | ~48 GiB | ~63 GiB | Yes |
| 32,768 | 2 | 1 | 16 | 2 | ~48 GiB | ~61 GiB | Yes |
| 32,768 | 4 | 1 | 32 | 4 | ~24 GiB | ~35 GiB | Yes |
| 65,536 | 4 | 1 | 32 | 4 | ~48 GiB | ~59 GiB | Yes |
| 65,536 | 4 | 2 | 64 | 8 | ~24 GiB | ~35 GiB | Yes |

> TP splits both activations and non-expert weights across GPUs. Expert weights
> remain split by EP only. The SLURM script (2 nodes, 16 GPUs) sweeps
> `TP=4,PP=1,EP=8,CP=1`, `TP=2,PP=2,EP=8,CP=1`, and `TP=2,PP=1,EP=8,CP=2`
> for this reason. Configs beyond 16 GPUs are extrapolations.

### Levers for longer sequences (SFT)

Teacher-forced SFT maintains **no KV cache**, so KV-cache quantization is not
an SFT memory lever (it applies to inference and NeMo RL rollout generation,
where vLLM manages a cache). For SFT activation pressure, in order of impact:

1. **Activation recomputation** (selective → full/uniform)
2. **TP and CP** (both divide activations)
3. **MBS=1 + gradient accumulation** to hold the global batch size
4. Activation precision options where the framework exposes them

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

Automodel and NeMo RL explicitly exclude these modules (with the comment
"NemotronHMamba2Mixer uses cuda_kernels_forward, out_proj LoRA has no gradient"):
- Automodel: `exclude_modules: ["*.out_proj"]`
- NeMo RL: `exclude_modules: ['*out_proj*']` (verified across all LoRA-enabled
  nanov3 recipes, including the Megatron-backend GRPO variant)

**Megatron-Bridge has NOT adopted this exclusion** (verified 2026-07): the
recipes pass a target list
`["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]`
that still includes `out_proj`, with no Mamba-2 filtering in `peft/lora.py`
or `peft/module_matcher.py`. Note that **`exclude_modules` cannot be
combined with a target list** (the module matcher asserts it is empty when
`target_modules` is set), so the fix is to **replace** the targets on the
command line:

```
peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]
```

(the `LoRA` class's own default — the recipes are what extend it unsafely).
Validate adapter merge on a small run first.

---

## Parameter Breakdown

| Component | Params | BF16 Memory | % of Total |
|-----------|-------:|------------:|-----------:|
| Routed experts (128 per layer x 23 MoE layers) | 29.37B | 54.7 GiB | 93.0% |
| Mamba-2 layers (23) | 0.89B | 1.7 GiB | 2.8% |
| Embeddings + LM head | 0.70B | 1.3 GiB | 2.2% |
| Shared experts (23 layers) | 0.46B | 0.9 GiB | 1.5% |
| Attention layers (6) | 0.14B | 0.3 GiB | 0.4% |
| Routers + norms | 0.01B | 0.02 GiB | <0.1% |
| **Total** | **31.58B** | **58.8 GiB** | |

**Key insight:** 93% of the model weight lives in the 128 routed experts.
Expert Parallelism (EP) is the dominant factor for fitting in GPU memory.

### Non-expert vs. expert split

| Category | Params | BF16 Memory |
|----------|-------:|------------:|
| Non-expert (shared across all GPUs) | 2.20B | 4.1 GiB |
| Routed experts (split by EP) | 29.37B | 54.7 GiB |

---

## LoRA Adapter Overhead

**What actually gets adapted depends on the framework** — earlier revisions
of this guide assumed one adapter per routed expert, which no shipped config
does:

- **Megatron-Bridge**: `share_expert_adapters=True` by default — grouped
  experts share ONE adapter per grouped linear, so the 128 experts contribute
  2 adapters per MoE layer, not 256.
- **Automodel / NeMo RL (Automodel backend)**: with only `exclude_modules`
  set (the shipped configs), the fallback matcher adapts **`nn.Linear`
  modules only** — grouped experts are never adapted at all.

| Config (rank) | Trainable Params | Adapter BF16 | Optimizer (12 B/param) | Notes |
|---------------|----------------:|------------:|----------------------:|-------|
| Automodel default (r=8) | ~6.2M | ~12 MiB | ~70 MiB | non-expert linears only |
| Bridge recipe (r=32, incl. unsafe out_proj) | ~33M | ~64 MiB | ~0.4 GiB | shared expert adapters |
| Bridge safe targets (r=32) | ~19M | ~36 MiB | ~0.2 GiB | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` |
| NeMo RL SFT (r=256) | ~198M | ~0.4 GiB | ~2.4 GiB | Automodel-backend matcher, non-expert only |
| *Per-expert adapters (r=8)* | *~214M* | *~0.4 GiB* | *~2.6 GiB* | *only if you explicitly target experts (e.g. `*.experts`)* |
| *Per-expert adapters (r=32)* | *~856M* | *~1.6 GiB* | *~10 GiB* | *not a shipped configuration* |

> Counts are logical single-copy numbers (Bridge expert adapters exist per EP
> rank; those are distinct params, so cluster-wide totals scale with EP).
> The practical takeaway: **shipped LoRA configs train 6M-200M params — LoRA
> optimizer memory is negligible for this model**, and GPU minimums are set
> by frozen weights + activations, not adapter rank.

---

## Per-GPU Memory Estimation

Memory components during LoRA training:

1. **Base model weights** (frozen, BF16) — 2 bytes/param
2. **LoRA adapter weights** (trainable, BF16) — 2 bytes/param
3. **Optimizer states** (Adam FP32 momentum + variance + master copy) — 12 bytes/trainable param, sharded across DP ranks with distributed optimizer
4. **Gradients** (BF16) — 2 bytes/trainable param
5. **Activations** — depends on seq length, micro-batch size, recomputation (~6 GiB at seq=2048, MBS=1)
6. **Framework overhead** (CUDA context, NCCL buffers) — ~3 GiB

> Totals below include base weights + LoRA overhead (with distributed optimizer)
> + estimated activations at seq=2048 MBS=1. Framework overhead (~3 GiB) is not
> included in the table totals but is accounted for in the Max MBS calculations
> via a 2 GiB safety margin.

### Megatron-Bridge (EP-based, no FSDP)

Non-expert weights are **replicated** across data-parallel ranks. Expert weights
are split by EP. The recipe defaults to EP=8, but `expert_model_parallel_size`
is overridable via CLI — any value where `128 % EP == 0` works.

Assumptions: LoRA rank=32, seq_len=2048, MBS=1.

| Config | GPUs | EP | TP | PP | Base Wt/GPU | Total/GPU | Fits? | Notes |
|--------|-----:|---:|---:|---:|------------:|----------:|:-----:|-------|
| 1 GPU | 1 | 1 | 1 | 1 | 58.8 GiB | ~65 GiB | Yes | Untested; ~12 GiB headroom after framework |
| 2 GPU | 2 | 2 | 1 | 1 | 31.5 GiB | ~38 GiB | Yes | Untested; ample headroom |
| 4 GPU | 4 | 4 | 1 | 1 | 17.8 GiB | ~24 GiB | Yes | Untested; override EP via CLI |
| **8 GPU (1 node)** | **8** | **8** | **1** | **1** | **10.9 GiB** | **~17 GiB** | **Yes** | **Recipe default — tested** |
| 16 GPU (2 nodes) | 16 | 8 | 2 | 1 | 8.9 GiB | ~15 GiB | Yes | Recommended; tested in SLURM script |

> Configs below 8 GPUs require overriding `model.expert_model_parallel_size` on
> the command line and have not been validated in official recipes. They are
> memory-viable but may surface untested code paths in the MoE token dispatcher.
>
> **Full SFT requires 16 GPUs (2 nodes) minimum.** The official Megatron-Bridge
> docs state: *"Running this recipe requires at least 2 H100 nodes (16 GPUs)"*
> for full parameter fine-tuning (TP=1, EP=8).

### Automodel (FSDP2)

Automodel FSDP-shards **all** frozen weights — expert weights are FSDP-sharded
across the `ep_shard` mesh *in addition* to their EP split (see
`components/moe/parallelizer.py`), so **resident base weight ≈ 58.8 GiB / N
GPUs regardless of EP**. EP determines expert placement and the size of
transient all-gathers during forward/backward (peak memory), not the resident
footprint.

Assumptions: LoRA rank=8, seq_len=2048, MBS=1. "Total" = resident + estimated
activations; peak transient all-gather memory comes on top.

| Config | GPUs | EP | Resident Base Wt/GPU | Total/GPU | Fits? | Notes |
|--------|-----:|---:|------------:|----------:|:-----:|-------|
| **1 GPU** | **1** | **1** | **58.8 GiB** | **~66 GiB** | **Yes** | **Official smoke config** — activation ckpt + memory-efficient LoRA |
| 4 GPU | 4 | 4 | 14.7 GiB | ~21 GiB | Yes | Config exists in repo |
| 8 GPU (1 node) | 8 | 4 or 8 | 7.4 GiB | ~14 GiB | Yes | Resident identical either way; EP changes peak/comm |

---

## Why MoE LoRA Sizing Differs from Dense Models

For a dense 30B model, LoRA typically reduces the minimum GPU requirement
dramatically (e.g., from 8 GPUs to 1). For MoE models like Nemotron 3 Nano,
LoRA *also* enables single-GPU training — but the dynamics are different:

1. **Expert weights dominate** — 93% of the ~59 GiB is in the 128 routed experts.
   Even though they're frozen during LoRA, they still must reside in GPU memory.
   At EP=1 (1 GPU), all 128 experts fit in ~55 GiB, leaving room for LoRA
   overhead and activations on an 80 GiB GPU.
2. **EP trades memory for GPUs** — expert weights can be distributed via Expert
   Parallelism. With EP=8, each GPU holds only 16 experts (~6.8 GiB), freeing
   memory for larger batches or longer sequences. With EP=1, all experts stay
   on one GPU but it still fits.
3. **LoRA savings are on the optimizer side** — instead of Adam states for 31.6B
   params (full SFT), shipped configs train only ~6-33M adapter params (see
   LoRA Adapter Overhead), making optimizer memory essentially free. The base
   weight footprint remains unchanged.
4. **Sequence length scales activation memory linearly** — at 2048, activations
   are ~6 GiB/GPU. At 16K they consume ~48 GiB/GPU, becoming the bottleneck
   regardless of LoRA vs. full SFT.

### Memory comparison: Full SFT vs. LoRA (H100 80 GiB, EP=8)

| Component | Full SFT (16 GPU) | LoRA rank=32 (8 GPU) | LoRA rank=256 (16 GPU) |
|-----------|:-----------------:|:--------------------:|:---------------------:|
| Base weights/GPU | ~11 GiB | ~11 GiB | ~11 GiB |
| Optimizer/GPU | ~22 GiB | <0.1 GiB | ~0.2 GiB |
| Gradients/GPU | ~11 GiB | <0.1 GiB | <0.1 GiB |
| Activations/GPU (seq=2048) | ~6 GiB | ~6 GiB | ~6 GiB |
| **Total/GPU** | **~50 GiB** | **~17 GiB** | **~17 GiB** |

> LoRA optimizer/gradient figures reflect shipped adapter behavior (~33M
> trainable at Bridge r=32, ~198M at RL r=256 — see LoRA Adapter Overhead).
> For LoRA, per-GPU memory is essentially frozen weights + activations.

> Full SFT optimizer and gradient figures assume distributed optimizer. The exact
> sharding depends on DP degree and framework implementation.

---

## How FSDP2 Enables Fewer GPUs (Automodel / NeMo RL)

Both Automodel and NeMo RL use PyTorch-native FSDP2 rather than relying solely
on Expert Parallelism.

### Memory distribution comparison

```
Megatron-Bridge (EP-based, no FSDP) — example with EP=8:
  GPU 0: [all non-expert weights] + [experts 0-15]
  GPU 1: [all non-expert weights] + [experts 16-31]   <-- 4.1 GiB replicated
  ...
  (8 GPUs, EP=8: each holds 4.1 GiB shared + 6.8 GiB experts = 10.9 GiB)

Megatron-Bridge with EP override — example with EP=2:
  GPU 0: [all non-expert weights] + [experts 0-63]
  GPU 1: [all non-expert weights] + [experts 64-127]  <-- 4.1 GiB replicated
  (2 GPUs, EP=2: each holds 4.1 GiB shared + 27.4 GiB experts = 31.5 GiB)

Automodel / NeMo RL (FSDP2) — example with 4 GPUs, EP=4:
  GPU 0: [1/4 non-expert weights] + [1/4 of expert shards]
  GPU 1: [1/4 non-expert weights] + [1/4 of expert shards]
  ...
  (4 GPUs: resident = 58.8/4 = 14.7 GiB each — experts are FSDP-sharded
   across the ep_shard mesh on top of their EP assignment, so resident
   weight is total/N for ANY valid EP)
```

FSDP2 shards **all** frozen weights (not just experts). EP still matters:
it sets which experts a rank *materializes* during compute (transient
all-gather peaks) and the dispatch communication pattern. The constraint:

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
| MoE layers (23) | ~16 GiB | ~66% |
| Mamba-2 layers (23) | ~7 GiB | ~29% |
| Attention layers (6) | ~1 GiB | ~4% |
| Embeddings + other | ~0.1 GiB | ~1% |

> These are estimates. MoE layers dominate activations because each token
> activates 6 experts, and the intermediate activations for all active experts
> must be stored for the backward pass (or recomputed).

---

## References

### Documentation
- Nemotron 3 Nano docs (Megatron-Bridge): `Megatron-Bridge/docs/models/nemotron/nemotron3-nano.md`
- HuggingFace model card: [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
  (an FP8 Day-0 variant `...-FP8` also exists)
- NeMo RL SFT guide: `RL/docs/guides/sft.md`
- NeMo RL GRPO guide: `RL/docs/guides/grpo.md`

### Recipes and Configs (paths current as of 2026-07)
- Megatron-Bridge recipe: `Megatron-Bridge/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py`
- Megatron-Bridge SLURM example: `Megatron-Bridge/examples/models/nemotron/nemotron_3/nano/slurm_peft.sh`
- Automodel single-GPU LoRA: `Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml`
- Automodel PEFT config: `Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml`
- NeMo RL SFT LoRA (2x8): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`
- NeMo RL SFT LoRA (2x4): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n4g-fsdp2-lora.yaml`
- NeMo RL GRPO LoRA: `RL/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`
- NeMo RL Full SFT (2x4 min): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n4g-fsdp2.yaml`
- NeMo RL DPO (1x4 min): `RL/examples/configs/recipes/llm/dpo-nanov3-30B3AB-1n4g-fsdp4ep4-automodel.yaml`

### Performance Data
- Megatron-Bridge performance: `Megatron-Bridge/docs/performance-summary.md`
- Automodel performance: `Automodel/docs/performance-summary.mdx`
  (Nano pretraining: 328 TFLOPs/s/GPU on 8x H100; LoRA table exists but has no Nano row yet)

### Memory Calculator Status (2026-07)
`theoretical_memory_utils.py` is now **EP/ETP-aware** (routed-expert params
divided by `PP × EP × ETP`) — the April 2026 over-count is fixed. Remaining
limits: it is **LoRA-unaware** (assumes all params trainable), applies
attention formulas to every layer (treats Mamba layers as dense/attention),
models MTP as copies of the final layer, and knows nothing of FSDP2. Even
for full SFT it over-counts this hybrid model by ~0.5B params (~0.9 GiB).
Treat it as a **coarse planning tool**, not an exact estimator.
