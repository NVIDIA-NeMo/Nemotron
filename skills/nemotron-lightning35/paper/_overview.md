---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "overview"
paper_sections: ["abstract", "1", "2", "3", "3.6", "3.7", "4", "4.7", "4.8", "6"]
title: "Nemotron 3.5 Lightning Overview: Identity, Lineage, Stage Order, Released Checkpoints, Report vs Repo Docs"
summary: |
  Nemotron 3.5 Lightning is a 30B-total / 3B-active text-only Mixture-of-Experts hybrid Mamba-Transformer
  model with a native MTP head, initialized from Nemotron 3 Nano Base. The joint Super/Lightning tech
  report describes its base-model recipe: an MTP head built from Nemotron 3 Super Base's head by projection
  fusion and pruning, a 100B-token MTP warm-up with the backbone frozen, 1T tokens of distillation from
  Super 3 Base, then long-context training. No Indic CPT and no agentic mid-training. Post-training is
  lighter than Super's: single-stage text-only SFT, then RLVR, with no MOPD, followed by 48k steps of MTP
  boosting. The repo docs now follow the report and cite it. The Base model card describes the base
  differently (>20T tokens with an NVFP4 recipe, MTP continued pretraining, no distillation); the docs
  follow the report and note the card.
key_facts:
  - "30B total / 3B active parameters; text-only; 'optimized for efficient deployment' (abstract, §1)."
  - "Initialized from Nano 3 Base (NVIDIA, 2025c) (§1, §2.2). Super 3.5 is initialized from Super 3 instead."
  - "MTP head initialized from Nemotron 3 Super Base's MTP head: latent MoE projections fused into the expert up/down projections, then pruned (hidden 4096 -> 2688, MoE intermediate 2688 -> 1856, routed experts 512 -> 128) (§1, §2.1)."
  - "100B-token MTP warm-up with the backbone frozen, using standard 2-step recursive MTP training. Hidden-state mixing is Super-only (§2.1)."
  - "Distillation: 1T tokens of Ultra 3 Phase 2 data, top-128 logits from Super 3 Base, forward KL + sink token, KL/CE weights 0.9/0.1, LR 1e-5 -> 1e-4 over 200B -> linear decay to 1e-5 over 800B, MTP loss scale 0.1, MoE sequence load-balancing coefficient 1e-4 (§2.2)."
  - "After distillation 'Lightning undergoes long-context training' (§1). The report gives no Lightning-specific long-context hyperparameters. The §2.4 numbers are Super's."
  - "The released Lightning did NOT include Indic CPT. Table 3 Indic-share ablations (200B tokens) are separate research runs (§2.3)."
  - "Post-training: single-stage text-only SFT + RLVR (§1, §3, §3.1). No MOPD stage (§3.6). MTP boosting only, no warm start: 48k steps against the final backbone, frozen backbone, LK loss (§3.7)."
  - "Quantization: Lightning deploys W4A16 regardless of GPU SKU (§4). The MTP module is not quantized (§4.7). NVFP4 vs BF16: 99.24% median accuracy recovery, -3.63% ratio-of-totals verbosity change (§4.8, Table 21)."
  - "Released checkpoints (§1): Nemotron 3.5 Lightning 30B-A3B Base BF16, BF16 (post-trained), NVFP4 (post-trained + quantized)."
  - "Repo docs / model card: 52 layers, hidden 2688, 128 routed experts (top-6) + 1 shared expert, GQA 32 heads / 2 KV groups, OpenMDW-1.1 license, BF16 66 GB -> NVFP4 22 GB (HF safetensors sizes); DSpark and DFlash draft models ship alongside."
  - "Conflict: the Base model card says the base was 'pre-trained with over 20T tokens' using 'an NVFP4 recipe', with a 'Continued Pre-Training' phase for the MTP layers and 'no supervised fine-tuning, reinforcement learning, or distillation'. The report says Nano 3 Base + 100B-token MTP warm-up + 1T-token distillation from Super 3 Base. The repo docs follow the report and note the card."
related_steps:
  - "sft/megatron_bridge"
currency: "frozen"
---

# Scope
Answers:
- What is Nemotron 3.5 Lightning (size, modality, lineage), and how does it differ from Super 3.5?
- In what order were its base-model stages run, and what does the report say about each?
- What post-training did it get, and which Super stages did it skip?
- Which checkpoints were released?
- Where do the report and this repository's docs (derived from the HF model cards) disagree?

Source labels used in this chunk set: **report** = the Nemotron 3.5 Super and Lightning tech report. **model card** = the three Hugging Face model cards (Base-BF16, BF16, NVFP4). **repo docs** = `docs/nemotron/lightning35/*.md` and `src/nemotron/recipes/lightning35/**/README.md`. The repo docs were corrected to follow the report and cite its sections; they separate the released model (report, model cards) from what the open-data recipes run.

# Headline Specs

| Item | Lightning 3.5 | Source | Super 3.5 (for disambiguation only) |
|---|---|---|---|
| Total / active parameters | 30B / 3B | report abstract, §1 | 120B / 12B |
| Modality | Text only ("a smaller ... text model") | abstract, §3.1 | Text, image, video |
| Initialized from | Nano 3 Base (NVIDIA, 2025c) | §1, §2.2 | Super 3 Base |
| Architecture family | MoE hybrid Mamba-Transformer with native MTP | abstract, §1, §2.1 | same family, LatentMoE |
| MTP head origin | Super 3 Base's MTP head, projection-fused and pruned | §2.1 | first of Super 3 Base's two heads |
| Distillation teacher (1T tokens) | Super 3 Base | §1, §2.2 | Ultra 3 Base |
| Extra base stages | long-context training only | §1 | Indic CPT, 1M LC, agentic mid-training, LC recovery |
| Post-training | single-stage text-only SFT + RLVR, then MTP boosting | §1, §3, §3.1, §3.7 | 2-stage SFT, RLVR + RLHF, specialist teachers, MOPD, MTP boosting |
| NVFP4 deployment | W4A16 on every GPU SKU | §4 | W4A4 on Blackwell, W4A16 elsewhere |
| Layers / hidden | 52 / 2688 | repo docs (README "Model Specifications") | not in report |
| Experts | 128 routed (top-6) + 1 shared | repo docs | 512 routed (§4.2) |

The report states the design goal: Lightning "is designed to excel in local, agentic use cases as well as datacenter deployments" (§4).

# Base-Model Stage Order (report)

| # | Stage | Tokens | Section | Chunk |
|---|---|---|---|---|
| 0 | Start from Nano 3 Base | — | §1, §2.2 | architecture.md |
| 1 | MTP head initialized from Super 3 Base's head (fuse latent projections, prune) | — | §2.1 | architecture.md |
| 2 | MTP warm-up, backbone frozen, standard 2-step recursive MTP training | 100B | §1, §2.1 | pretraining.md |
| 3 | Distillation from Super 3 Base on Ultra 3 Phase 2 data | 1T | §1, §2.2 | pretraining.md |
| 4 | Long-context training | not reported for Lightning | §1, §6 | pretraining.md |

Not applied to the released Lightning: Indic CPT (§2.3 says so explicitly), agentic SWE mid-training, and long-context recovery. These are described only for Super (§1, §2 intro).

Base-model results: Table 6 (§2.6), see evaluation-base.md.

# Post-training (report, pointers only)
- "Nemotron 3.5 Lightning follows a lighter post-training pipeline consisting of single-stage text-only SFT and RLVR" (§1). §3 repeats: "Lightning goes through a lighter pipeline, with text-only SFT, and RLVR."
- §3.1: "Nemotron 3.5 Lightning is trained with single SFT stage with the text only SFT data."
- §3.6: "Nemotron 3.5 Lightning, however did not have an MOPD stage."
- §3.7: "For Nemotron 3.5 Lightning, we skip the warm start and apply only MTP boosting." The backbone is frozen and only MTP-head parameters are updated. Lightning's boosting ran **48k steps** against the final backbone (the report misspells it "Lightining"). It uses the LK loss (Samarin et al., 2026) instead of Ultra's temperature-scaled forward KL. The report gives no acceptance-length numbers for Lightning; Table 8 and Table 9 are Super-only.
- Evaluation of the post-trained model: §4.8 uses NeMo Gym; Lightning-specific settings are in the NeMo Gym recipes folder (`nemotron_recipes/lightning-3.5/reproducibility.md`).

# Quantization (report, pointers only)
- PTQ + QAD with ModelOpt (§4). Lightning is quantized "to minimize weight fetch and raise activation precision to maintain accuracy" and deploys **W4A16 regardless of GPU SKU**.
- §4.8 W4A16 NVFP4 checkpoint:
  - Routed experts, shared experts, and the LM output head use 4-bit NVFP4 weights with 16-bit activations.
  - Mamba input/output projections use FP8 weights and activations. Attention projections stay BF16.
  - Served with FP8 KV cache and FP32 Mamba SSM cache. The SSM cache stays FP32 "as it was not a limiter on DGX Spark for low batch (< 4) use cases".
- Results: median accuracy recovery **99.24%**; ratio-of-totals verbosity change **-3.63%** (Table 21).
- §4.7: "the MTP module of Nemotron 3.5 Lightning is not quantized."
- Appendix (long-context quantization analysis): the disambiguation loss from quantization is "also observed on Nemotron 3.5 Lightning". On Lightning, moving the KV cache from FP8 to BF16 recovers MRCR-128k by +1.43 pp (2.4 sigma). Moving the projections from FP8 to BF16 recovers +2.97 pp (2.8 sigma).

# Released Checkpoints

| Checkpoint (report §1 name) | HF repo id (repo docs) | Description |
|---|---|---|
| Nemotron 3.5 Lightning 30B-A3B Base BF16 | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 | base model |
| Nemotron 3.5 Lightning 30B-A3B BF16 | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | post-trained |
| Nemotron 3.5 Lightning 30B-A3B NVFP4 | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | post-trained + NVFP4 (W4A16) |

Repo docs (README) only, not in the report:
- Weights, data, and recipes are released under **OpenMDW-1.1**.
- **DSpark and DFlash draft models** ship with the checkpoints for speculative decoding. MTP-based speculation suits medium/high concurrency; DSpark suits DGX Spark and low-concurrency serving.
- The NVFP4 checkpoint is 22 GB vs 66 GB for BF16 (`docs/nemotron/lightning35/quantization.md`; matches the HF safetensors sizes). Neither the report nor the model cards give a Lightning throughput multiplier.
- Open RL data: Nemotron-3.5-Lightning-RL-Training-Blend, plus "Nemotron-RL Agentic Terminal Pivot" (`docs/nemotron/lightning35/rl.md`).

# Report vs Model Cards vs Repo Recipe
1. **How the base was built (report vs Base model card).**
   - Report (§1, §2.1–§2.2): initialized from Nano 3 Base, then MTP warm-up (100B) + distillation from Super 3 Base (1T) + long context.
   - Base model card: "pre-trained with over 20T tokens", "pre-trained using an NVFP4 recipe", a "Continued Pre-Training" stage for the MTP layers, and "no supervised fine-tuning, reinforcement learning, or distillation has been applied".
   - The two are not reconciled. A plausible reading is that the card's ">20T tokens" counts Nano 3 Base's own pretraining, but neither source says so. Give both and attribute each. The repo docs (pretrain.md, README) follow the report and add a note about the card.
   - The repo's `lightning35 pretrain` recipe trains the Lightning architecture from random init on an open blend; it reproduces neither account.
2. **MTP loss scaling.** Report §2.2: 0.1 during distillation. Recipe: 0.3 in pretraining (Megatron-Bridge `nemotron_3_5_lightning_pretrain_config`) and 0.3 in RL. The docs pretrain.md states the difference explicitly; the report gives no RL-stage value.
3. **MoE load balancing.** Report §2.2: MoE sequence load-balancing loss coefficient 1e-4. Recipe: `seq_aux_loss` with coefficient 1e-4 plus router expert bias. They agree.
4. **Post-training scope.** Report: single-stage text-only SFT + RLVR, no MOPD, then MTP boosting (48k steps). BF16 model card: multi-environment GRPO in NeMo RL / NeMo Gym with an asynchronous architecture that uses MTP to speed rollouts. Recipe: one synchronous GRPO run over 6 NeMo Gym environments, no RLHF/GenRM/DPO, no MTP boosting.
5. **NVFP4 Mamba projections.**
   - Report §4.8 and the NVFP4 model card: Mamba in/out projections are FP8 (the card: "FP8 per-tensor dynamic scales on mamba in_proj/out_proj and KV cache").
   - docs quantization.md: the ModelOpt recipe it documents is `four_over_six` "with W4A16 Mamba linears"; FP8 Mamba appears only in the `mamba_fp8_max` alternative. The docs flag this as unreconciled.
6. **NVFP4 calibration.** The report names no Lightning calibration method ("the same PTQ/QAD procedure" as Super). The NVFP4 model card names "Four Over Six NVFP4 (a variant of static MSE calibration) W4A16 on routed and shared experts", calibrated on 1000 samples of a Nemotron Ultra validation subset at 32K tokens. The docs' commands use four-over-six on 1,000 samples of the public `cnn_nemotron_v2_mix`.

Earlier versions of the repo docs said Lightning had "no separate technical report" and contained text copied from the Nemotron 3 Nano docs (a from-scratch 23.5T + 1.5T curriculum, GenRM/DPO tables, "7 reward environments", "up to 3.3x" throughput, "15+ languages"). Those passages are gone; if a user quotes them, they describe Nano 3, not Lightning.

# Caveats
- Many numbers in the report's abstract and §1 are Super-only: 9.82-point Indic gain, 51.78% SWE-bench Pro, 4,632 tok/s/GPU, 99.52% NVFP4 recovery, MTP boosting acceptance lengths. Never attribute them to Lightning. Lightning's NVFP4 recovery is 99.24% (§4.8).
- "Both models combine large-scale distillation, long-context training, and native speculative decoding" (abstract, §6). The long-context *details* in §2.4 (35B tokens, LR 4.5e-6, CP16/TP8/EP64 on GB300) are not attributed to Lightning. The §2 intro lists the 1M extension under "For Super".
- The report never states Lightning's context length, although Table 21 evaluates the post-trained model on RULER-v1 (1M) (91.52 BF16 / 90.83 NVFP4). The explicit 1M figure comes from the model cards ("Maximum context length up to 1M tokens") and the repo docs.
- The repo recipes train only on open data and do not reproduce the released model (no distillation, no MTP boosting).
