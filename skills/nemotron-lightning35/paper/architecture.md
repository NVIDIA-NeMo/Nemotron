---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "architecture"
paper_sections: ["abstract", "1", "2.1", "2.2", "4", "4.7", "4.8"]
title: "Nemotron 3.5 Lightning Architecture: Nano 3 Hybrid Mamba-Transformer MoE Backbone, MTP Head Built by Projection Fusion and Pruning"
summary: |
  Lightning 3.5 is a 30B-total / 3B-active text-only hybrid Mamba-Transformer MoE initialized from Nemotron
  3 Nano Base, with a native MTP drafting head. The report has no dimensions table. The only Lightning
  dims it states come from the MTP-initialization sentence (§2.1): the Super 3 Base MTP head is pruned to
  hidden size 2688, MoE intermediate size 1856, and 128 routed experts "to match Lightning". Before pruning,
  the head's latent MoE projections are fused into the experts' up/down projections. Layer count (52), top-6
  routing, the shared expert, and GQA 32/2 come from the repo docs / model card. The MTP module is trained
  with 2-step recursion and is left unquantized in the NVFP4 checkpoint.
key_facts:
  - "30B total / 3B active; text-only; MoE hybrid Mamba-Transformer (report abstract, §1)."
  - "Backbone initialized from Nano 3 Base (§1, §2.2). The report does not restate Nano 3's dims."
  - "Lightning dims stated in the report (§2.1, MTP pruning targets): hidden size 2688, MoE intermediate size 1856, 128 routed experts."
  - "MTP init: start from Nemotron 3 Super Base's MTP head, fuse its latent MoE projections into the experts' up- and down-projection matrices, then prune hidden 4096 -> 2688, MoE intermediate 2688 -> 1856, routed experts 512 -> 128 (§2.1)."
  - "MTP uses the recursively shared formulation of the Nemotron 3 Super report; Lightning was warmed up with standard 2-step recursive MTP training, without hidden-state mixing (§2.1)."
  - "Module families named for the NVFP4 checkpoint (§4.8): routed experts, shared experts, LM output head, Mamba input/output projections, attention projections, KV cache, Mamba SSM cache."
  - "Lightning's MTP module is not quantized (§4.7)."
  - "Repo docs / model card: 52 layers, hidden 2688, 128 routed experts with top-6, 1 shared expert (intermediate size 3712), GQA with 32 heads / 2 KV groups, repeated-layer MTP module (depth 2 in pretraining, 5 in RL; MTP layer pattern '*E')."
  - "Repo docs: BF16 checkpoint 66 GB; NVFP4 22 GB."
related_steps:
  - "sft/megatron_bridge"
currency: "frozen"
---

# Scope
Answers:
- What is Lightning's backbone, and what dimensions does the report itself give?
- How was the MTP head created from Super 3 Base's head (projection fusion and pruning)?
- How is MTP structured and trained for Lightning, and how does that differ from Super?
- Which architectural numbers come only from repo docs / model cards, and where do sources disagree?

# Backbone (report)
- "Lightning is a smaller 30B-parameter, 3B-active text model optimized for efficient deployment" (abstract). It is "a 30 billion total and 3 billion active parameter model initialized from Nano 3 Base (NVIDIA, 2025c)" (§1).
- Family: "efficient Mixture-of-Experts Hybrid Mamba-Transformer models" (§1, §6).
- §2.2: "All other hyperparameters and settings match Nano 3 Base, except for the added MTP layers." The backbone shape is Nano 3 Base's plus MTP. The report does not restate Nano 3's dims.
- Text only: SFT uses "the text only SFT data" (§3.1). No vision encoder.

**The report has no Lightning architecture table.** The only Lightning dims it states are the targets of the MTP pruning in §2.1, which prunes "to match Lightning":

| Dim | Value (report §2.1) |
|---|---|
| Hidden size | 2688 |
| MoE intermediate size (per routed expert) | 1856 |
| Number of routed experts | 128 |

Module families named in §4.8 for the W4A16 NVFP4 checkpoint: routed experts, shared experts, language-model output head, Mamba input and output projections, attention projections, KV cache, and Mamba SSM cache. So Lightning has shared experts, attention layers, and Mamba layers.

# Dimensions from Repo Docs / Model Card (not in report)

| Component | Value | Source |
|---|---|---|
| Layers | 52 (hybrid Mamba / attention / MoE pattern) | docs README, recipe README, docs pretrain.md |
| Hidden size | 2688 (matches report §2.1) | docs README, recipe README |
| Routed experts | 128 (matches report §2.1) | docs README, recipe README |
| Activated experts | 6 (top-k) | docs README, recipe README, pretrain.md |
| Shared experts | 1, intermediate size 3712 | recipe README; pretrain.md says "1 shared expert" |
| Attention | GQA, 32 heads / 2 KV groups | recipe README |
| MTP | "Repeated-layer MTP module", depth 2 in pretraining, 5 in RL; `mtp_hybrid_override_pattern: "*E"` (attention + MoE) | recipe README, docs pretrain.md, docs rl.md |
| Layer mix | 23 Mamba-2, 6 attention, 23 MoE; attention at six positions spread through the stack | docs pretrain.md (layer pattern from the Megatron-Bridge config, inherited from Nano 3) |
| MoE router | sigmoid router score, expert bias enabled, `seq_aux_loss` | docs pretrain.md hyperparameter table (Megatron-Bridge recipe) |
| Checkpoint size | 66 GB BF16, 22 GB NVFP4 | docs quantization.md |
| Serving | vLLM, `nemotron_v3` reasoning parser, `qwen3_coder` tool parser, fp32 Mamba SSM cache, optional MTP speculative decoding | docs evaluate.md |
| Context length | up to 1M tokens | model cards, docs README / recipe README; **not stated in the report**, though Table 21 evaluates the post-trained model on RULER-v1 (1M) |

For the full Nano 3 layer pattern and dims (Q/KV heads 32/2, head dim 128, Mamba state 128, 64 Mamba heads, expert dim 1856, 2 shared experts), see `skills/nemotron-nano3/paper/architecture.md`. These apply to Lightning's initialization; the report does not say any were changed.

# MTP Head (report §2.1)
- Both 3.5 models "feature native speculative-decoding support via a Multi-Token Prediction (MTP) head, adopting the recursively shared MTP formulation detailed in NVIDIA (2026b)" (the Nemotron 3 Super report).
- The head is an integrated internal drafter. It predicts several future tokens from the backbone's hidden representations, and candidates are verified in one backbone forward pass using rejection sampling or exact token matching.

## Initialization by projection fusion and pruning
Verbatim mechanism (§2.1): "We initialize Lightning's MTP head from that of Nemotron 3 Super Base. We first fuse the latent MoE projections into the experts' up- and down-projection matrices, then prune the hidden size from 4096 to 2688, MoE intermediate size from 2688 to 1856, and number of routed experts from 512 to 128 to match Lightning."

| Step | Super 3 Base MTP head | Lightning MTP head |
|---|---|---|
| 1. Fuse latent projections | LatentMoE: experts operate in a latent space with down/up latent projections | latent projections folded into each expert's up- and down-projection matrices |
| 2. Prune hidden size | 4096 | 2688 |
| 3. Prune MoE intermediate size | 2688 | 1856 |
| 4. Prune routed experts | 512 | 128 |

- The intro (§1) calls this "architecture-aware projection fusion and pruning".
- The report does not say *which* channels or experts are kept, which pruning criterion is used, or whether the attention part of the MTP block is pruned too. It does not say whether Super 3 Base's first or second MTP head is the source.
- Inference, not stated in the report: fusing the latent projections into the experts implies Lightning's MoE has no latent projection, i.e. it is a standard (non-latent) MoE, as in Nano 3.

## Training
- Warm-up: 100B tokens, backbone frozen, only the MTP head optimized, "the standard 2-step recursive MTP training" (§2.1).
- The report says this naive training "induces a pronounced train-inference conditioning mismatch". The hidden-state-mixing fix is applied **to Super only**.
- During 1T-token distillation the MTP layers are trained with an MTP loss scaling factor of 0.1 (§2.2).
- After post-training: MTP boosting only (no warm start), 48k steps, frozen backbone, LK loss (§3.7).
- NVFP4: "the MTP module of Nemotron 3.5 Lightning is not quantized" (§4.7).
- §4.7's description of the MTP module (one attention layer and one MoE layer, and a single lm_head shared between verification and drafting) is written about **Super**. It is consistent with the repo docs' `"*E"` MTP pattern for Lightning, but the report does not state it for Lightning.

# Comparison with Super 3.5

| Aspect | Lightning 3.5 | Super 3.5 |
|---|---|---|
| Total / active | 30B / 3B | 120B / 12B |
| Init | Nano 3 Base | Super 3 Base |
| MoE type | not named in report; latent projections fused away when building MTP head | LatentMoE, 512 routed experts |
| Hidden size | 2688 (§2.1) | 4096 (Super 3 Base, §2.1) |
| MTP init | Super 3 Base head, fused + pruned | first of Super 3 Base's two heads |
| MTP warm-up | standard 2-step recursion | hidden-state mixing |
| MTP quantized in NVFP4 | no | yes (experts NVFP4 W4A4) |
| Vision | none | C-RADIO encoder + projector |

# Caveats
- **Shared-expert count.** Repo docs say 1 shared expert, intermediate size 3712. The Nemotron 3 Nano report (Lightning's initialization) lists 2 shared experts with expert dim 1856. Since 3712 = 2 x 1856, the two descriptions may be the same capacity represented differently, but no source says so. Record both.
- Earlier versions of docs pretrain.md had a layer table saying "8 experts per layer" and attention "every ~8 layers" (copied from the Nano 3 docs). The corrected page says 128 routed experts (top-6) plus 1 shared per MoE layer and attention at six positions, and labels the Megatron-Bridge link as the Nemotron 3 Nano page that the Lightning config extends.
- 30B / 3B are the report's rounded figures. The docs say "~3B active ... ~30B total". The report gives no exact parameter count for Lightning (it gives one only for Super, §4.1).
- MTP depth 5 in RL comes from repo docs (`mtp_num_layers: 5` in rl.md). The report does not discuss Lightning's MTP depth beyond the 2-step warm-up. The MTP boosting draft length for Lightning is not reported (Table 8's draft length 7 is Super).
