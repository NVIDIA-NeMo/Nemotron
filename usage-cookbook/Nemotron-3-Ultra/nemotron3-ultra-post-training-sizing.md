# Post-Training T-Shirt Sizing: Nemotron 3 Ultra 550B-A55B

> Minimum GPU configurations for LoRA fine-tuning, full SFT, and GRPO of
> [NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
> across three NeMo training pathways: **Megatron-Bridge**, **Automodel**, and
> **NeMo RL**.

> **Target hardware: NVIDIA H100 80 GiB SXM**, with GB200 (186 GiB) noted where
> official recipes exist — for Ultra, several pathways are GB200-first. All
> memory figures use **binary GiB** (1 GiB = 2^30 bytes), matching `nvidia-smi`.

> **WARNING — EP-only sharding is impractical on H100.** Ultra has ~63 GiB of
> BF16 *non-expert* weights (~64 GiB with the MTP block) that Expert
> Parallelism does not shard. At EP=512 the weights alone would technically
> squeeze in (~66 GiB/GPU) but leave no room for activations at hidden size
> 8192 — and **no published recipe uses EP-only**. In practice, **TP and/or PP
> (Megatron-Bridge) or FSDP2 sharding (Automodel) is required** for any
> useful configuration — a fundamental difference from Nano (fits at EP=1)
> and Super (fits at EP=8).

> **WARNING — Mamba-2 LoRA constraint:** Do NOT apply LoRA to `out_proj` or
> `conv1d` in Mamba-2 layers. Fused CUDA kernels bypass `forward()`, causing
> zero gradients and `merge_and_unload()` failures. See
> [details below](#mamba-2-fused-kernel-constraint-on-lora-targets).

> **Storage warning:** The BF16 HF checkpoint is ~1.1 TB, and the imported
> Megatron checkpoint is another ~1.1 TB. Reserve **≥ 2.5 TB** of fast storage
> before conversion; full-SFT training checkpoints can be several TB each.

---

## T-Shirt Sizing: How Many GPUs Do I Need?

### At a Glance

| Framework | Min LoRA (tested) | Min Full SFT (tested) | GRPO | GB200 alternative |
|-----------|:-----------------:|:---------------------:|:----:|:-----------------:|
| **Megatron-Bridge** | **32× H100 (4 nodes)** | 384× H100 (48 nodes) | — | LoRA: 16× GB200 (4 nodes) |
| **Automodel** | **32× H100 (4 nodes)** | 256× H100 (32 nodes) | — | LoRA: 16× GB200 (4 nodes) |
| **NeMo RL** | — | — | **72× GB200 (18×4)** | Published GRPO recipes are GB200/GB300; H100 untested |

> Unlike Nano and Super — where the tested recipes sat far above the memory
> floor — Ultra's tested 32-GPU LoRA configs sit **at** the practical H100
> memory floor. There is no meaningful "fewer GPUs via CLI override" story:
> 16 H100s puts base weights at ~64-68 GiB/GPU before activations, which is
> not workable at hidden size 8192.
>
> **GRPO has no H100 recipe.** All published RL configs target GB200/GB300
> (4 GPUs/node). See the NeMo RL section.

### Megatron-Bridge

Official recipes exist in `src/megatron/bridge/recipes/nemotronh/nemotron_3_ultra.py`
with SLURM launchers under `examples/models/nemotron/nemotron_3/ultra/`.
All Ultra recipes set `expert_tensor_parallel_size=1` (ETP=1), so the minimum
GPU count follows:

```
min GPUs = PP × max(TP × CP, EP × ETP)
```

Keep TP within a node-local NVLink domain. Additional GPUs beyond the minimum
add data parallelism.

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Recompute | Status |
|------|-----:|------:|-------------|----------:|--------:|-----------|--------|
| **Min LoRA** | **32** | **4×8** | **TP=2, PP=4, EP=8, ETP=1** | 32 | 4096 | selective | **Tested — `slurm_peft.sh` + library recipe** |
| Min LoRA (memory only) | 16 | 2×8 | TP=4, PP=4, EP=4 | 32 | 2048 | selective | Not viable in practice — ~69 GiB/GPU on the busiest PP stage before activations |
| **Min Full SFT** | **384** | **48×8** | **TP=2, PP=12, EP=16** | — | 4096 | full/uniform | **Tested — `slurm_sft.sh`** |
| Full SFT (library recipe) | 192 | 24×8 | TP=2, PP=6, EP=32 | — | 4096 | selective | Recipe default — smaller than SLURM config |
| LoRA (GB200) | 16 | 4×4 | TP=2, PP=1, EP=16 | 32 | 4096 | selective | Tested — README hardware table |
| Pretrain (reference) | 384 | 48×8 | TP=4, PP=12, EP=16 | — | 4096 | full/uniform | `slurm_pretrain.sh` |

Container: build from `nvcr.io/nvidia/nemo:26.04.01` (the SLURM scripts require
you to set `CONTAINER_IMAGE`).

```bash
# LoRA fine-tuning (4 nodes × 8 H100, TP=2, PP=4, EP=8)
sbatch examples/models/nemotron/nemotron_3/ultra/slurm_peft.sh

# Full SFT (48 nodes × 8 H100, TP=2, PP=12, EP=16)
sbatch examples/models/nemotron/nemotron_3/ultra/slurm_sft.sh
```

The library recipe functions (`nemotron_3_ultra_peft_openmathinstruct2_packed_config`,
`nemotron_3_ultra_sft_openmathinstruct2_packed_config`) use packed
OpenMathInstruct-2 data with GBS=128, MBS=1, and enable MTP
(`mtp_num_layers=2`, `mtp_use_repeated_layer=true`, `mtp_loss_scaling_factor=0.3`).
The MoE token dispatcher is `flex` with the `hybridep` backend.

> **LoRA targets:** the Megatron-Bridge default target list is
> `["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]`
> (rank=32, alpha=32). This *includes* Mamba-2 `out_proj` — verify your version
> filters Mamba-2 layers before merging adapters (see the Mamba-2 constraint
> section). Automodel excludes `*.out_proj` entirely.

### Automodel (FSDP2)

FSDP2 shards **all** frozen weights across data-parallel ranks, so per-GPU
weight memory is roughly `1023 GiB / N GPUs`. That puts the floor at 32× H100
(~32 GiB/GPU) — 16× H100 (~64 GiB/GPU) does not leave room for activations.
Ultra configs also introduce pipeline parallelism for full SFT.

| Size | GPUs | Nodes | Parallelism | LoRA Rank | Seq Len | Status |
|------|-----:|------:|-------------|----------:|--------:|--------|
| **Min LoRA** | **32** | **4×8 H100** | EP=32, FSDP2 | 32 | 2048 | **Tested — hellaswag + text2sql cookbook configs** |
| LoRA (GB200) | 16 | 4×4 GB200 | EP=16, FSDP2 | 32 | 2048 | Tested — loss curve validated |
| **Min Full SFT** | **256** | **32×8 H100** | EP=64, PP=4, FSDP2 | — | 4096 | Config exists (`nemotron_ultra_v3_squad.yaml`) |
| Pretrain benchmark | 128 | 16×8 H100 | EP=64, FSDP=128 | — | 4096 | 293 TFLOPs/s/GPU, 815 tok/s/GPU |

Container: `nvcr.io/nvidia/nemo-automodel:26.04.00`

```bash
# Minimum LoRA (4 nodes × 8 H100, EP=32) — run on EACH node with its rank:
torchrun --nproc-per-node=8 --nnodes=4 --node-rank=$NODE_RANK \
  --rdzv-endpoint=$MASTER_ADDR:29500 \
  examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/nemotron/nemotron_ultra_v3_hellaswag_peft.yaml

# LoRA on GB200 (4 nodes × 4 GPUs, EP=16): same pattern with
#   --nproc-per-node=4 and ..._peft_gb200.yaml
```

(Or launch through the Automodel SLURM/SkyPilot launcher, which sets up the
rendezvous automatically.)

> Ultra's Automodel LoRA uses **rank=32** (alpha=32, Triton kernels,
> `exclude_modules: ["*.out_proj"]`) — higher than Nano's rank-8 default.
> All configs enable `activation_checkpointing: true` and MTP
> (`num_nextn_predict_layers=2`). The H100 configs use the `deepep` or
> `hybridep` token dispatcher; the full-SFT config sets
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and interleaved-1F1B
> pipeline scheduling.

### NeMo RL (GRPO — published recipes target GB200/GB300)

**There is no H100 GRPO recipe for Ultra** — published RL configurations
target GB200/GB300 nodes (4 GPUs per node, aarch64 containers). H100 is
untested for Ultra GRPO, not proven impossible. Note the starter recipe and
the `ultra3` stage recipes live on the **Nemotron repo's ultra/newer-main
branches** (`usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo/` pinning the NeMo
RL `ultra-v3` branch) — they are *not* present on this checkout's working
branch. Treat the numbers below as published starter topologies.

| Size | GPUs | Nodes | Training mesh | Generation | Workflow | Status |
|------|-----:|------:|---------------|-----------|----------|--------|
| **GRPO starter (DAPO)** | **72× GB200** | 18×4 | TP=8, ETP=1, EP=64, PP=1, CP=8 (64 GPUs) | vLLM TP=8, 2 nodes | GRPO/DAPO | Starter recipe |
| GRPO pipecleaner | 192× GB300 | 48×4 | 32 train + 10 gen + 6 judge nodes | vLLM | GRPO | k8s infra example |
| GRPO pipecleaner | 256× GB300 | 64×4 | 32 train + 26 gen + 6 judge nodes | vLLM | GRPO | k8s infra example |

```bash
# GRPO/DAPO starter (18 GB200 nodes: 16 training + 2 generation)
# NOTE: config lives on the Nemotron repo's ultra branches (not this
# checkout) and requires the NeMo RL `ultra-v3` branch + aarch64 container.
uv run examples/run_grpo.py \
  --config usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo/dapo_ultra_starter.yaml
```

> Training config: GBS=1024 (64 prompts × 16 generations), MBS=1,
> `max_total_sequence_length=4096`, BF16 with `first_last_layers_bf16=true`,
> activation checkpointing, distributed optimizer. Generation is
> **non-colocated** vLLM (TP=8). Note the RL configs set `mtp_num_layers: 5`
> (vs 2 in SFT recipes) — an RL-specific draft-horizon setting.
>
> The TP=8 group spans two 4-GPU GB200 nodes (NVLink domain). Extrapolating
> this mesh to H100 (8 GPUs/node) is untested; expect the 64-GPU training mesh
> to be the floor, plus generation nodes.

---

## Sequence Length Scaling

Weight memory is fixed; **activation memory scales linearly with sequence
length** and is severe at hidden size 8192 across 108 layers. All tested Ultra
recipes cap fine-tuning at seq 4096 on H100 and lean on recomputation:

| Regime | Config | Recompute strategy |
|--------|--------|--------------------|
| LoRA, seq ≤ 4096 | 32× H100 | Selective: `[moe, layernorm, core_attn, moe_act, mlp, shared_experts]` |
| Full SFT, seq 4096 | 384× H100 | **Full/uniform** (`recompute_num_layers=1`) — activations dominate |
| Long-context SFT (294,912 packed tokens) | 3072× H100 (Nemotron recipe) | TP=2, PP=6, EP=32, **CP=8** (36,864 tokens/rank), full recompute |
| 1M-context extension (reference) | GB200 | TP=8, PP=2, EP=128, **CP=32** |

Rules of thumb for Ultra on H100:

- **Beyond seq 4096, add CP.** The production recipes scale context via CP
  (CP=8 for ~37K tokens/rank), not via larger per-rank activation budgets.
- Approximate activation cost at seq=4096, MBS=1 with the LoRA selective
  recompute set: ~20-25 GiB/GPU at TP=2 — which is why the tested 32-GPU
  config (~38 GiB/GPU base weights) works and 16 GPUs (~64 GiB/GPU) does not.
- FP8 KV-cache is not an SFT lever (teacher-forced SFT has no KV cache; it
  applies to inference and RL rollouts) — and only 12 of 108 layers are
  attention anyway. Recompute + TP/CP are the levers here.

---

# Appendix: Technical Details

---

## Model Architecture Summary

From the HF `config.json` (BF16 checkpoint):

| Property | Value |
|----------|-------|
| Total parameters | ~550B (549.3B computed; 561B on disk incl. MTP block) |
| Active parameters per token | ~55B (56.1B computed) |
| Precision | BF16 (select MXFP8/BF16 layers; pretrained in NVFP4) |
| Architecture | Hybrid Mamba-2 + LatentMoE + Attention |
| Layers | 108 (48 Mamba-2 + 48 MoE + 12 Attention) |
| Hidden size | 8192 |
| Vocab size | 131,072 |
| Experts per MoE layer | 512 routed + 1 shared |
| Experts activated per token | 22 |
| MoE latent dimension | 2048 |
| Expert FFN hidden size | 5120 (shared expert: 10240, in hidden space) |
| Activation function (experts) | Squared ReLU |
| MTP | 1 next-token block = [attention, moe] (`num_nextn_predict_layers=1`; trained with `mtp_num_layers=2`, repeated-layer) |
| Attention heads (Q / KV) | 64 / 2 (GQA), head dim 128 |
| Mamba (state / groups / heads / head dim) | 128 / 8 / 256 / 64 |
| Config max context | 262,144 (256K); paper claims 1M via long-context extension |
| Default finetuning seq length | 4096 |

---

## Parameter Breakdown

Derived from the HF config; matches the published 550B/55B totals to within
rounding.

| Component | Params | BF16 Memory | % of Total |
|-----------|-------:|------------:|-----------:|
| Routed experts (512 × 48 MoE layers) | 515.4B | 960.0 GiB | 93.8% |
| Mamba-2 layers (48) | 20.2B | 37.7 GiB | 3.7% |
| Shared experts (48 layers, hidden-space) | 8.1B | 15.0 GiB | 1.5% |
| Embeddings + LM head (untied) | 2.1B | 4.0 GiB | 0.4% |
| Attention layers (12) | 1.7B | 3.1 GiB | 0.3% |
| Latent projections (48 layers) | 1.6B | 3.0 GiB | 0.3% |
| Routers + norms | 0.2B | 0.4 GiB | <0.1% |
| **Total** | **549.3B** | **1023.2 GiB** | |
| MTP block (attention + MoE, unshared on disk) | +11.1B | +20.6 GiB | → 560.4B ≈ "561B on disk" |

### Non-expert vs. expert split — why EP-only is impractical

| Category | Params | BF16 Memory |
|----------|-------:|------------:|
| Non-expert backbone (NOT sharded by EP) | 33.9B | **63.2 GiB** (~64.1 GiB with MTP block) |
| Routed experts (split by EP × ETP × PP) | 515.4B | 960.0 GiB (~980 GiB with MTP block's experts) |

The non-expert block is 15× larger than Super's (4.1 GiB Nano → 14.8 GiB Super
→ 63.2 GiB Ultra). EP=512 would technically fit the weights (~66 GiB/GPU) but
is unusable in practice: no activation headroom at hidden 8192, and a 512-GPU
mesh with zero data parallelism. On H100, split the non-expert block with
TP and/or PP (Megatron-Bridge) or FSDP2 (Automodel). This is the single most
important sizing fact for Ultra.

### Per-GPU base weights (Megatron-Bridge, ETP=1)

Simple average: `base/GPU ≈ 63.2/(TP×PP) + 960/(EP×PP)` GiB. But PP stages
are **not balanced**: embeddings sit on the first/last stages and the MTP
block (~11.2B incl. a full 512-expert MoE layer) on the last, so the busiest
stage carries several GiB more than the average. Both are shown below —
**budget against the max-stage column**:

| Config | GPUs | Avg Wt/GPU | Max-stage Wt/GPU (incl. MTP) | Fits 80 GiB? | Notes |
|--------|-----:|-----------:|------------------:|:------------:|-------|
| TP=2, PP=4, EP=8 | 32 | ~37.9 GiB | ~41.3 GiB | **Yes** | **Tested LoRA config (launcher)** |
| TP=4, PP=4, EP=4 | 16 | ~63.9 GiB | ~69.4 GiB | **No** (with activations) | Untested; memory estimate |
| TP=4, PP=2, EP=8 | 16 | ~67.9 GiB | ~70.6 GiB | **No** (with activations) | Untested; memory estimate |
| TP=2, PP=12, EP=16 | 384 | ~7.6 GiB | ~11.5 GiB | Yes | Tested full-SFT config (headroom for optimizer + full recompute) |
| TP=1, EP=8, PP=1 | 8 | ~183 GiB | ~183 GiB | **OOM** | The Nano/Super-style single-node config is impossible |

---

## LoRA Adapter Overhead

**What actually gets adapted depends on the framework** — earlier revisions
assumed one adapter per routed expert, which no shipped config does:

- **Megatron-Bridge**: `share_expert_adapters=True` by default — the 512
  grouped experts share ONE adapter per grouped linear per EP rank.
- **Automodel**: with only `exclude_modules: ["*.out_proj"]` set (the shipped
  Ultra configs), the fallback matcher adapts **`nn.Linear` modules only** —
  grouped experts are never adapted.

| Config (rank 32) | Trainable Params | Adapter BF16 | Notes |
|------------------|----------------:|------------:|-------|
| Automodel default | ~187M | ~0.35 GiB | non-expert linears only |
| Bridge recipe (incl. unsafe out_proj) | ~196M | ~0.36 GiB | shared expert adapters |
| Bridge safe targets | ~91M | ~0.17 GiB | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` |
| *Per-expert adapters* | *~11.3B* | *~21 GiB* | *48 × 512 × 32 × (2048+5120) × 2 — only if experts explicitly targeted; not shipped* |

> Counts are logical single-copy numbers (Bridge expert adapters are per EP
> rank). Practical takeaway: **shipped Ultra LoRA configs train ~90-200M
> params — adapter and optimizer memory are negligible**; the GPU minimums
> are set entirely by the ~1 TiB of frozen weights plus activations.

---

## Mamba-2 Fused Kernel Constraint on LoRA Targets

Identical constraint to Nano and Super: the Mamba-2 implementation passes raw
weights into fused CUDA kernels (`mamba_split_conv1d_scan_combined`), so
LoRA-wrapped `out_proj`/`conv1d` in Mamba-2 layers get zero gradients and
break `merge_and_unload()`.

| Module | Layer Types | Safe? |
|--------|-------------|:-----:|
| `linear_qkv`, `linear_proj` | Attention | Yes |
| `linear_fc1`, `linear_fc2` | MoE experts, shared expert | Yes |
| `in_proj` | Mamba-2 | Yes |
| `out_proj` | Attention: Yes; **Mamba-2: NO** | Exclude Mamba-2 |
| `conv1d` | Mamba-2 | **NO** — fused kernel |

- Automodel Ultra configs: `exclude_modules: ["*.out_proj"]` ✓
- Megatron-Bridge default targets include `out_proj` — verify Mamba-2
  filtering in your version before merging adapters.

---

## Why Ultra Sizing Is a Different Regime

1. **Non-expert weights (63 GiB) ≈ one whole H100.** For Nano and Super, EP
   was the only mandatory parallelism; for Ultra, TP/PP/FSDP2 sharding of the
   non-expert block is equally mandatory. The min-GPU formula becomes
   `PP × max(TP×CP, EP×ETP)`.
2. **The tested configs ARE the floor.** Nano's 8-GPU recipe sat 8× above its
   1-GPU memory floor; Ultra's 32-GPU LoRA recipes sit essentially at the
   floor. Don't expect CLI overrides to go lower on H100.
3. **MTP adds real memory during SFT.** Training recipes enable a 2-layer MTP
   block (`mtp_loss_scaling_factor=0.3` SFT / 0.1 pretrain); its MoE layer
   adds ~11B params (~20.6 GiB spread over EP×PP) plus activations. GRPO
   configs vary (`mtp_num_layers: 5` in RL recipes).
4. **Recomputation is always on.** Even LoRA at seq 4096 uses selective
   recompute over six module classes; full SFT uses full/uniform recompute.
   Budget compute accordingly — expect noticeably lower TFLOPs efficiency
   than the 293 TFLOPs/s/GPU pretraining benchmark.
5. **GB200 halves the node count.** 186 GiB HBM per GPU absorbs the non-expert
   block easily: LoRA drops from 32× H100 to 16× GB200; NVIDIA's inference
   minimum is a single 8× B200 node (~1.5 TB aggregate).

---

## References

### Recipes and Configs (repo paths as of 2026-07)
- Megatron-Bridge recipe: `Megatron-Bridge/src/megatron/bridge/recipes/nemotronh/nemotron_3_ultra.py`
- Megatron-Bridge SLURM + hardware README: `Megatron-Bridge/examples/models/nemotron/nemotron_3/ultra/`
- Automodel LoRA: `Automodel/examples/llm_finetune/nemotron/nemotron_ultra_v3_hellaswag_peft.yaml` (+ `_gb200` variant)
- Automodel full SFT: `Automodel/examples/llm_finetune/nemotron/nemotron_ultra_v3_squad.yaml`
- Automodel guide: `Automodel/docs/guides/llm/nemotron-3-ultra.md`

**On Nemotron remote branches only** (`origin/ultra` / `origin/ultra-release` /
newer `origin/main` — NOT on this checkout's working branch):
- GRPO/DAPO starter: `usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo/dapo_ultra_starter.yaml`
- ultra3 stage recipes: `src/nemotron/recipes/ultra3/` (pretrain 768 GPUs NVFP4; SFT 3072 GPUs at 294K packed seq)
- LoRA text2sql cookbooks (Automodel + Megatron-Bridge): `usage-cookbook/Nemotron-3-Ultra/lora-text2sql/`

### Performance Data
- Automodel pretraining benchmark (only published Ultra training perf):
  128× H100, EP=64, seq 4096 — 10.05 s/step, 293 TFLOPs/s/GPU, 815 tokens/s/GPU
  (`Automodel/docs/performance-summary.mdx`)

### Model
- HF model card: [NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
  (released 2026-06-04, post-trained; OpenMDW v1.1 license; NeMo 26.04.01 runtime)
