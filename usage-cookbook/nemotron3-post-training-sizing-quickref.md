# Nemotron 3 Post-Training: How Many GPUs Do I Need?

**One-page quick reference** for fine-tuning the Nemotron 3 family on
**H100 80 GiB** (GB200 rows marked). Numbers are the smallest configurations
backed by a checked-in config or launcher as of 2026-07-05 (repo state:
Automodel `3a9b011`, Megatron-Bridge `99bdc5d9`, RL `133cd68`). Every row
carries an evidence badge — see Fine Print. Detailed math, commands, and
caveats live in the per-model guides linked at the bottom.

## The Answer Table

| | **Nano 30B-A3B** | **Super 120B-A12B** | **Ultra 550B-A55B** |
|---|:---:|:---:|:---:|
| BF16 weights (backbone / checkpoint) | 59 / 59 GiB | 225 / 230 GiB | 1,023 / 1,044 GiB |
| **LoRA** — smallest official config | **1 GPU** @2K (Automodel, smoke config) | **8 GPUs** @2K, 1 node (Automodel, config) | **32 GPUs** @2–4K, 4 nodes (both frameworks, config+launcher) |
| **LoRA** — Megatron-Bridge | 8 @2K (recipe default) / 16 @2K (launcher) | 64 @4K, 8 nodes (launcher) | 32 @4K, 4 nodes (launcher) |
| **Full SFT** — smallest | **8** @2K (1×8, RL nightly smoke; Automodel config) | 32 @2K, 4 nodes (Automodel config) | 256 @4K, 32 nodes (Automodel config) |
| **GRPO** (NeMo RL) | **8** @2K full-param (1×8) / 16 LoRA (2×8), nightly smoke | 128 @8K, 16×8 (nightly smoke) | GB200/GB300 recipes only — 72× GB200 @4K starter |
| **DPO** (NeMo RL) | 8 @4K (1×8, nightly smoke) | — | — |
| GB200 alternatives | SFT/LoRA 8 (2×4), DPO 4 (1×4) | — | LoRA 16 (4×4) |

## Five Rules That Explain Every Number

1. **~93% of every Nemotron 3 model is frozen routed-expert weight.** Expert
   Parallelism (EP) splits it; that's the first knob. LoRA does NOT shrink it —
   LoRA only shrinks optimizer/gradient memory (12+2 bytes per *trainable*
   param vs 2 bytes per frozen param).
2. **The non-expert remainder is what separates the models.** It is replicated
   on every GPU under plain EP: Nano 4 GiB (irrelevant), Super 15 GiB
   (annoying), **Ultra 64 GiB (EP-only is impractical — no published recipe
   uses it; TP/PP or FSDP2 sharding is required for any useful config)**.
3. **Megatron-Bridge minimum:** `GPUs = PP × max(TP × CP, EP × ETP)`.
   **Automodel (FSDP2) resident weights ≈ total ÷ N GPUs** — FSDP2 shards
   *everything* (experts included, beyond their EP split), which is why its
   minimums are the smallest. Peak memory adds transient all-gathers on top.
4. **Sequence length is the second budget.** Activations scale linearly with
   tokens and dominate past ~8K (Nano), ~4K (Super/Ultra). Fix with activation
   recomputation, TP/CP, and MBS=1 + gradient accumulation — not more EP.
   (FP8 KV-cache is an inference/rollout lever — SFT has no KV cache.)
5. **Never LoRA a Mamba-2 `out_proj`/`conv1d`.** Fused kernels bypass
   `forward()`: zero gradients, broken adapter merges. Automodel and NeMo RL
   exclude it by default; **Megatron-Bridge recipes still target it (2026-07)**
   — and `exclude_modules` cannot be combined with a target list, so the fix
   is to **replace** the targets:
   `peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]`.

## Recipe Entry Points

These are *starting points*, not turnkey training runs — check dataset,
steps, checkpointing, and (for multi-node) your launcher's rendezvous setup.

| I want… | Start from |
|---|---|
| Nano LoRA, 1 GPU (smoke: 20 steps, ckpt off) | `automodel examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml --nproc-per-node 1` |
| Nano LoRA, 1 node (Bridge) | `torchrun --nproc-per-node=8 examples/models/nemotron/nemotron_3/nano/finetune_nemotron_3_nano.py --peft lora 'peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2]' checkpoint.pretrained_checkpoint=/path/to/ckpt` |
| Super LoRA, 1 node | `torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag_peft.yaml` |
| Ultra LoRA, 4 nodes | `sbatch examples/models/nemotron/nemotron_3/ultra/slurm_peft.sh` (Bridge; SLURM handles topology) or Automodel `nemotron_ultra_v3_hellaswag_peft.yaml` via multi-node torchrun |
| GRPO at minimum scale | Nano: `grpo-nanov3-30BA3B-2n8g-fsdp2.yaml` (full-param, 1×8) · Super: `grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.yaml` · Ultra: `dapo_ultra_starter.yaml` (GB200; on the Nemotron repo's ultra branches, not this checkout) |

Multi-node `torchrun` additionally needs `--nnodes`, `--node-rank`, and
`--rdzv-endpoint` (or use the SLURM scripts, which set these up).

## Containers

| | Megatron-Bridge | Automodel | NeMo RL |
|---|---|---|---|
| Nano | `nemo:25.11.nemotron_3_nano` | `nemo-automodel:26.04.00` | `nemo-rl:v0.7.0` |
| Super | `nemo:26.02.nemotron_3_super` | `nemo-automodel:26.04.00` | `nemo-rl:v0.5.0.nemotron_3_super` (stage recipes) / `v0.7.0` (RL-repo recipes) |
| Ultra | `nemo:26.04.01` (base) | `nemo-automodel:26.04.00` | nightly ultra image, **aarch64/GB200** — custom build |

All `nvcr.io/nvidia/...`. **Storage:** checkpoint sizes ≈ 59 GiB Nano,
230 GiB Super, **1,044 GiB Ultra — reserve ≥2.5 TB** before HF→Megatron
conversion (two copies + headroom).

## Fine Print

- **Evidence badges**: `launcher` = checked-in SLURM/launch script proves the
  topology; `config` = checked-in YAML exists (may be CI-validated, not
  convergence-tested); `nightly smoke` = runs ~10-20 steps in CI on the stated
  hardware; `smoke config` = short-run config with checkpointing disabled;
  `memory estimate` = arithmetic only. None of these guarantee convergence.
- **Hardware matters for RL rows**: NeMo RL nightly manifests separate H100
  (8 GPUs/node) from GB200 (4 GPUs/node); recipe filenames like "2n4g" refer
  to GB200. Script topology overrides recipe names (the "2n8g" full-param
  GRPO script actually runs 1 node × 8).
- Megatron-Bridge **recipe defaults for Super do not fit on H100 verbatim**
  (PEFT defaults EP=1, SFT defaults TP=1) — always pass the documented
  parallelism overrides.
- LoRA trainable-param counts depend on framework behavior (Bridge shares
  expert adapters; Automodel's default matcher skips grouped experts
  entirely) — see the LoRA sections of the detailed guides before assuming
  expert adapters exist.
- GRPO adds a vLLM generation footprint (colocated or dedicated nodes) on top
  of the training mesh — "GPUs" above are total cluster.
- Minimums are sequence-length-specific (the @2K/@4K/@8K annotations above);
  longer sequences need more GPUs or CP.

## Detailed Guides

- [Nano 30B-A3B sizing](Nemotron-3-Nano/nemotron3-nano-post-training-sizing.md)
- [Super 120B-A12B sizing](Nemotron-3-Super/nemotron3-super-post-training-sizing.md)
- [Ultra 550B-A55B sizing](Nemotron-3-Ultra/nemotron3-ultra-post-training-sizing.md)
