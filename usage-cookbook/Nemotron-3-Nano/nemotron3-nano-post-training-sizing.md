# Post-Training GPU Memory Estimation: Nemotron 3 Nano 30B-A3B

> GPU sizing for LoRA fine-tuning, full SFT, and GRPO of
> [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
> across three NeMo training pathways: **Megatron-Bridge**, **Automodel**, and
> **NeMo RL**.

---

## Part 1 - Quick Sizing Rules - At a Glance

### Heuristic

A few rules of thumb cover most estimates about minimum training configurations -

`NOTE`: Numbers below use H100 80 GiB as a reference — the per-parameter math applies
to any GPU; just divide by your device's memory capacity to get GPU counts.

- **Full SFT** needs **~16 bytes per parameter** (weights + gradients + Adam optimizer states).
  For 31.6B total params in Nemotron 3 Nano → **~471 GiB** (plus activation memory). For this MoE model the per-GPU
  memory is dominated by how the experts are distributed (Expert Parallelism). With FSDP2
  (Automodel), Full SFT fits on **1 node (8 GPUs)** at EP=8 (~65 GiB/GPU). Without FSDP2
  (Megatron-Bridge), you need DP≥2 to shard optimizer states, so the floor is **16 GPUs**
  (EP=8, DP=2). Recommended recipes ship with 16 GPUs (2 nodes) for headroom
  (ex: [Megatron-Bridge Nemotron 3 Nano SFT recipe](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)).

- **LoRA** memory is dominated by **frozen model weights** (~59 GiB in BF16 for 31.6B params).
  Training overhead (adapter, optimizer, gradients, activations) adds ~15–20% on top,
  bringing the single-GPU total to **~68 GiB** at LoRA rank 8 — much less than full SFT because
  only the small adapter is trained. Fits on 1 H100 80 GiB at
  [EP](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-reference/distributed_checkpointing.html#expert-parallelism)=1
  (tight), comfortably on 2 GPUs at EP=2, and with ample headroom on 4 or 8 GPU nodes with larger EP settings.

- **GRPO / RL**: the training side follows the SFT or LoRA rule above, **plus** additional
  GPUs for inference/generation if training and generation are not colocated.

> These are **memory-floor estimates** at seq_len=2048 and BF16 precision. Longer
> sequences, larger micro-batch sizes, higher LoRA ranks, or mixed-precision
> choices will shift the numbers — see [Activation Memory & Sequence Length](#part-3--activation-memory--sequence-length) for details.
> [EP](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-reference/distributed_checkpointing.html#expert-parallelism) (Expert Parallelism) distributes the model's 128 routed experts across
> GPUs.

### Memory Floor

Minimum GPUs where the model fits in memory. Lower EP values are achievable by
overriding the recipe's default EP via config/CLI but may be untested.

> Reference GPU: **H100 80 GiB**. Scale GPU counts proportionally for other devices.

| Framework | Training Mode | GPUs | EP | TP | PP | DP | Seq Len |
|-----------|---------------|:----:|:--:|:--:|:--:|:--:|--------:|
| **Megatron-Bridge** | LoRA | 1 | 1 | 1 | 1 | 1 | 2048 |
| **Megatron-Bridge** | Full SFT | 16 | 8 | 1 | 1 | 2 | 2048 |
| **Automodel** | LoRA | 1 | 1 | — | — | 1 | 2048 |
| **Automodel** | Full SFT | 8 | 8 | — | — | 8 | 2048 |
| **NeMo RL** | Full SFT | 16 | 8 | — | — | 16 | 2048 |
| **NeMo RL** | GRPO† | 16 | 8 | 4* | — | 16 | 2048 |

> Automodel and NeMo RL use DTensor (FSDP2) for weight sharding. NeMo RL also
> supports Megatron-Bridge as an alternative training backend (pick one, not both).
> †NeMo RL GRPO recipe uses LoRA rank 128 for the training side, which increases optimizer memory; GRPO also
> needs GPUs for both training and generation (vLLM) simultaneously, driving the floor to 16.
> *TP=4 in GRPO is for the vLLM generation engine, not the training side.

### Tested / Recommended Recipes

Shipped tested recipe defaults with headroom.

> Reference GPU: **H100 80 GiB**.

| Framework | Training Mode | GPUs | EP | TP | PP | DP | Seq Len | Recipe Config |
|-----------|---------------|:----:|:--:|:--:|:--:|:--:|--------:|--------|
| [**Megatron-Bridge**](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html#lora-fine-tuning) | LoRA | 8 | 8 | 1 | 1 | 1 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py) |
| [**Megatron-Bridge**](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html) | Full SFT | 16 | 8 | 1 | 1 | 2 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py) |
| [**Automodel**](https://docs.nvidia.com/nemo/automodel/latest/guides/llm-finetuning.html) | LoRA | 8 | 8 | — | — | 8 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml) |
| [**Automodel**](https://docs.nvidia.com/nemo/automodel/latest/guides/llm-finetuning.html) | Full SFT | 8 | 8 | — | — | 8 | 2048 | [Config](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml) |
| [**NeMo RL**](https://docs.nvidia.com/nemo/rl/latest/guides/sft.html) | Full SFT | 16 (2×8) | 8 | — | — | 16 | 2048 | [Config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml) |
| [**NeMo RL**](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html) | GRPO | 16 (2×8) | 8 | 4* | — | 16 | 2048 | [Full](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml), [LoRA](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-lora.yaml) |

> *TP=4 in GRPO is for the vLLM generation engine, not the training side.

### Notes

1. **Mamba-2 LoRA constraint:** Do NOT apply LoRA to `out_proj` or `conv1d` in
   Mamba-2 layers. Fused CUDA kernels bypass `forward()`, causing zero gradients
   and `merge_and_unload()` failures. See [Mamba-2 details](#mamba-2-specifics).

2. **Target hardware:** The data above assumes **H100 80 GiB SXM** with **BF16 precision**.
   A B200 (192 GiB) could hold the full model at EP=1; an A100 40 GiB would need
   more GPUs. All memory figures use binary GiB (1 GiB = 2^30), matching `nvidia-smi`.

---

## Part 2 — Training Memory Estimation Details

### Model Architecture Summary

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
| Default finetuning seq length | 2048 |
| Max supported context | 1,048,576 (1M); default config: 262,144 (256K) |


### Where the 16× Comes From (Full SFT)

The **16 bytes/param** is the minimum **static** memory — fixed regardless of
sequence length or batch size:

| Item | Bytes/Param | Notes |
|------|:-----------:|-------|
| Weight (BF16) | 2 | The model itself |
| Gradient (BF16) | 2 | Backward pass output |
| Optimizer (Adam; FP32) | 12 | Full-precision weights, first and second moments |
| **Static per-param total** | **16** | |

On top of that, **activation memory** (dynamic) must be stored for the backward
pass. This depends on sequence length and micro-batch size, and scales roughly
linearly for this model — see [Activation Memory & Sequence Length](#part-3--activation-memory--sequence-length).

With a distributed optimizer the 12 bytes of Adam state are sharded across DP ranks,
giving **6 + 12/DP_size bytes/param** per GPU (plus activations).

### Why LoRA Is ~1.15× Model Size (Not 2×)

For dense models, a common heuristic is "LoRA ≈ 2× model size." For this MoE model
the multiplier is much lower because only 3.5B of 31.6B params are active per token,
keeping activations small relative to total model size.

| Item | Size | Notes |
|------|------|-------|
| Frozen weights (BF16) | 2 bytes/param → 58.8 GiB | Dominant cost; unchanged by LoRA |
| LoRA adapter weights | Rank-dependent | Rank 8 → 0.4 GiB, rank 32 → 1.6 GiB |
| Optimizer states | 12 bytes/trainable param | Only over adapter params — small |
| Activations (seq=2048) | ~6 GiB/GPU | Same as full SFT |
| **Total** | **~68 GiB** | ~1.15× model BF16 size |

The base model footprint doesn't change — LoRA only saves on the optimizer side.

### Expert Parallelism — Why It Matters Here

93% of this model's 31.6B parameters live in 128 routed experts (29.4B params,
54.7 GiB in BF16). Expert Parallelism (EP) is therefore the **first knob to tune**:

- **EP distributes whole experts across GPUs** — at EP=8, each GPU holds 16 of
  128 experts (~6.8 GiB) instead of all 128 (~54.7 GiB).
- **TP splits individual weight matrices** across GPUs. With only 3.5B active
  params per token, TP>1 adds inter-GPU communication overhead without much
  benefit at short sequences. Use TP when you need to split activations for
  longer sequences (>8K tokens) or when dense layers are too large.
- **PP splits layers across GPUs** — rarely needed here; PP=1 in all shipped recipes.

**Rule of thumb:** set EP first to fit expert weights, then add TP only for long
sequences. Lower EP values (1, 2, 4) work if you have fewer GPUs — EP=8 is the
recipe default because it gives the most headroom.


---

## Part 3 — Activation Memory & Sequence Length

### Activation Memory Scaling

Weight memory is fixed regardless of sequence length. Activation memory scales
approximately **linearly** with sequence length and becomes the dominant consumer
beyond ~8K tokens. The linear approximation holds further than for dense
transformers because:

- **23 Mamba-2 layers** have O(1) memory scaling for sequence length (recurrent state, not KV cache).
- **23 MoE layers** scale linearly (activations per expert per token).
- **Only 6 attention layers** are quadratic, but Flash Attention reduces them to linear memory.

| Regime | Seq Length | Bottleneck | What to Tune |
|--------|-----------|------------|--------------|
| **Weight-bound** | < 8K | Frozen expert weights | EP (more GPUs = fewer experts/GPU) |
| **Activation-bound** | > 8K | Forward-pass activations | TP, CP, MBS, gradient checkpointing |

#### Activation memory vs. sequence length

For 8× H100, Megatron-Bridge, EP=8, TP=1, LoRA rank=32 — static memory ~15 GiB/GPU.
Estimates assume selective recomputation with MBS=1.

| Seq Length | Activation/GPU | Total/GPU | Fits 80 GiB? | Max MBS |
|-----------:|---------------:|----------:|:------------:|--------:|
| 1,024 | ~3 GiB | ~18 GiB | Yes | 16 |
| 2,048 | ~6 GiB | ~21 GiB | Yes | 8 |
| 4,096 | ~12 GiB | ~27 GiB | Yes | 4 |
| 8,192 | ~24 GiB | ~39 GiB | Yes | 2 |
| 16,384 | ~48 GiB | ~63 GiB | Yes (tight) | 1 |
| 32,768 | ~96 GiB | ~111 GiB | **OOM** | — |

> **Max MBS** = largest power-of-2 micro-batch size fitting in 80 GiB with a
> 2 GiB safety margin. Use gradient accumulation for the desired global batch size.

**Key takeaway:** on a single 8-GPU node with EP=8, the max finetuning sequence
length is ~**16K tokens** (MBS=1). Beyond that, add TP or CP.

#### Scaling to longer sequences

| Seq Length | TP | CP | Min GPUs | Nodes | Act/GPU | Total/GPU | Fits? |
|-----------:|---:|---:|---------:|------:|--------:|----------:|:-----:|
| 16,384 | 1 | 1 | 8 | 1 | ~48 GiB | ~63 GiB | Yes |
| 32,768 | 2 | 1 | 16 | 2 | ~48 GiB | ~61 GiB | Yes |
| 32,768 | 4 | 1 | 32 | 4 | ~24 GiB | ~35 GiB | Yes |
| 65,536 | 4 | 1 | 32 | 4 | ~48 GiB | ~59 GiB | Yes |
| 65,536 | 4 | 2 | 64 | 8 | ~24 GiB | ~35 GiB | Yes |

> TP splits both activations and non-expert weights across GPUs. Expert weights
> remain split by EP only.

#### FP8 KV cache for extended context

FP8 quantization of the KV cache can approximately **halve activation memory**
for attention components with minimal accuracy degradation. While attention
layers are only 6 of 52, FP8 KV cache is most impactful for longer sequences
and larger micro-batch sizes. Available in both Megatron-Bridge (via Transformer
Engine) and Automodel.

### Mamba-2 Specifics

23 of 52 layers are Mamba-2 with **O(1) memory scaling** for sequence length —
they use a fixed-size recurrent state instead of a KV cache that grows with
sequence length. This is why the linear activation heuristic holds much further
than for dense transformers.

#### Fused kernel constraint on LoRA targets

The Mamba-2 implementation passes raw weight tensors directly into fused CUDA
kernels (e.g., `mamba_split_conv1d_scan_combined`). When LoRA wraps these modules,
the standard `forward()` is **never called** — the kernel reads `.weight` directly:

1. **LoRA adapters on these modules produce zero gradients** (no learning).
2. **`merge_and_unload()` will fail** with shape mismatch errors.

**Safe LoRA targets:**

| Module | Layer Types | Safe? |
|--------|-------------|:-----:|
| `linear_qkv` | Attention | Yes |
| `linear_proj` | Attention | Yes |
| `linear_fc1` | MoE experts, shared expert | Yes |
| `linear_fc2` | MoE experts, shared expert | Yes |
| `in_proj` | Mamba-2 | Yes |
| `out_proj` | Attention: Yes, **Mamba-2: NO** | Use `exclude_modules` to filter Mamba-2 |
| `conv1d` | Mamba-2 | **NO** — fused kernel |

Framework exclusion settings:
- **Automodel:** `exclude_modules: ["*.out_proj"]`
- **NeMo RL:** `exclude_modules: ['*out_proj*']`
- **Megatron-Bridge:** the PEFT recipe includes `"out_proj"` in `target_modules`
  without an `exclude_modules` filter — verify that the framework filters by layer
  type internally, or add an explicit exclusion.

---

## Part 4 — Framework-Specific Details

### Per-GPU Memory Estimation

Memory components during LoRA training:

1. **Base model weights** (frozen, BF16) — 2 bytes/param
2. **LoRA adapter weights** (trainable, BF16) — 2 bytes/param
3. **Optimizer states** (Adam FP32) — 12 bytes/trainable param, sharded across DP ranks
4. **Gradients** (BF16) — 2 bytes/trainable param
5. **Activations** — ~6 GiB at seq=2048 MBS=1
6. **Framework overhead** (CUDA context, NCCL buffers) — ~3 GiB

#### Megatron-Bridge (EP-based, no FSDP)

Non-expert weights are **replicated** across data-parallel ranks. Expert weights
are split by EP. Assumptions: LoRA rank=32, seq_len=2048, MBS=1.

| Config | GPUs | EP | TP | Base Wt/GPU | Total/GPU | Fits? |
|--------|-----:|---:|---:|------------:|----------:|:-----:|
| 1 GPU | 1 | 1 | 1 | 58.8 GiB | ~68 GiB | Tight |
| 2 GPU | 2 | 2 | 1 | 31.5 GiB | ~40 GiB | Yes |
| 4 GPU | 4 | 4 | 1 | 17.8 GiB | ~26 GiB | Yes |
| **8 GPU (1 node)** | **8** | **8** | **1** | **10.9 GiB** | **~18 GiB** | **Yes** |
| 16 GPU (2 nodes) | 16 | 8 | 2 | 8.9 GiB | ~16 GiB | Yes |

> 8 GPUs is the recipe default (EP=8), not a hard constraint — EP can be overridden.
> Full SFT requires 16 GPUs (2 nodes) minimum per official docs.

#### Automodel (FSDP2)

Frozen weights are **sharded** across DP ranks via FSDP2, in addition to expert
weights being split by EP. Assumptions: LoRA rank=8, seq_len=2048, MBS=1.

| Config | GPUs | EP | DP | Base Wt/GPU | Total/GPU | Fits? |
|--------|-----:|---:|---:|------------:|----------:|:-----:|
| **4 GPU** | **4** | **4** | **4** | **14.7 GiB** | **~21 GiB** | **Yes** |
| 8 GPU (EP=4) | 8 | 4 | 8 | 14.2 GiB | ~20 GiB | Yes |
| 8 GPU (EP=8) | 8 | 8 | 8 | 7.4 GiB | ~14 GiB | Yes |

FSDP2 memory distribution (simplified):

```
Megatron-Bridge (EP-based, no FSDP) — 8 GPUs:
  Each GPU: [all non-expert weights: 4.1 GiB] + [16 experts: 6.8 GiB] = 10.9 GiB

Automodel (FSDP2) — 4 GPUs:
  Each GPU: [1/4 non-expert weights: 1.0 GiB] + [32 experts: 13.7 GiB] = 14.7 GiB

NeMo RL (FSDP2) — 16 GPUs:
  Each GPU: [1/16 non-expert weights: 0.26 GiB] + [16 experts: 6.8 GiB] = 7.1 GiB
```

### LoRA Adapter Overhead

| LoRA Rank | Trainable Params | BF16 Memory | % of Base | Used In |
|----------:|----------------:|------------:|----------:|---------|
| 8 | 219M | 0.41 GiB | 0.69% | Automodel default |
| 32 | 878M | 1.6 GiB | 2.78% | Megatron-Bridge default |
| 128 | ~3.5B | ~6.5 GiB | ~11.1% | NeMo RL GRPO recipe |
| 256 | ~7.0B | ~13.1 GiB | ~22.2% | NeMo RL SFT recipe |

> Higher LoRA ranks (128, 256) substantially increase optimizer state memory.
> NeMo RL recipes use these ranks on 16 GPUs for a reason.

### Parameter Breakdown

| Component | Params | BF16 Memory | % of Total |
|-----------|-------:|------------:|-----------:|
| Routed experts (128/layer × 23 MoE layers) | 29.37B | 54.7 GiB | 93.0% |
| Mamba-2 layers (23) | 0.89B | 1.7 GiB | 2.8% |
| Embeddings + LM head | 0.70B | 1.3 GiB | 2.2% |
| Shared experts (23 layers) | 0.46B | 0.9 GiB | 1.5% |
| Attention layers (6) | 0.14B | 0.3 GiB | 0.4% |
| Routers + norms | 0.01B | 0.02 GiB | <0.1% |
| **Total** | **31.58B** | **58.8 GiB** | |

Non-expert vs. expert split:

| Category | Params | BF16 Memory |
|----------|-------:|------------:|
| Non-expert (shared across all GPUs) | 2.20B | 4.1 GiB |
| Routed experts (split by EP) | 29.37B | 54.7 GiB |

### References

#### Documentation
- Nemotron 3 docs (Megatron-Bridge): [nemotron3.html](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
- HuggingFace model card: [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- NeMo RL SFT guide: `RL/docs/guides/sft.md`
- NeMo RL GRPO guide: `RL/docs/guides/grpo.md`

#### Recipes and Configs
- Megatron-Bridge recipe: `Megatron-Bridge/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py`
- Megatron-Bridge SLURM example: `Megatron-Bridge/examples/models/nemotron_3/slurm_peft.sh`
- Automodel PEFT config: `Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml`
- NeMo RL SFT LoRA (2×8): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`
- NeMo RL SFT LoRA (2×4): `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n4g-fsdp2-lora.yaml`
- NeMo RL GRPO LoRA: `RL/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`
- NeMo RL Full SFT: `RL/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml`

#### Performance Data
- Megatron-Bridge performance: `Megatron-Bridge/docs/performance-summary.md`
- Automodel performance: `Automodel/docs/performance-summary.md`
