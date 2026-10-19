---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "architecture"
paper_sections: ["1", "2.1", "3.1", "3.7", "4.1", "4.2"]
title: "Nemotron 3.5 Super Architecture: Inherited Hybrid Mamba-Transformer LatentMoE, Recursive Shared MTP Head, Vision Stack"
summary: |
  Nemotron 3.5 Super VL keeps the hybrid Mamba-Transformer LatentMoE language backbone of Nemotron 3 Super
  (120B total / 12B active, 512 routed experts per MoE layer). The report gives no architecture dimensions
  table of its own. What it changes is the MTP drafter: one recursively shared MTP head, initialized from
  the first of Super 3 Base's two heads and trained with hidden-state mixing. Vision is added in post-training
  through a C-RADIO vision encoder and a vision-language projector/connector (from the Nemotron 3 Nano Omni
  stack). There is no audio encoder.
key_facts:
  - "120B total / 12B active parameters; mixture-of-experts vision-language model (§1, §4.2)."
  - "Language backbone: hybrid Mamba/attention with LatentMoE and 512 routed experts per MoE layer (§4.2, Table 12)."
  - "Initialized from Nemotron 3 Super; the report has no dimensions table, so dims come from the Nemotron 3 Super report / HF config."
  - "Super 3 Base dims stated in passing (§2.1, Lightning MTP pruning): hidden size 4096, MoE intermediate size 2688, 512 routed experts."
  - "Single recursively shared MTP head, initialized from the first of Nemotron 3 Super Base's two MTP heads (pre-long-context checkpoint); the second head is discarded."
  - "Hidden-state mixing (Super only): at recursive step k, input hidden states are sampled from those of earlier MTP steps 1..k-1, not only step k-1."
  - "Vision path: C-RADIO vision encoder + VL connector/projector, both kept BF16 in every NVFP4 recipe tier (Table 12)."
  - "Base model is text-only; the vision projector is warmed up before SFT, then language and vision components are jointly fine-tuned (§3.1)."
  - "Parameter count including vision, MTP, and unquantized modules: N = 124.453 billion (§4.1)."
  - "No audio encoder is described; modalities are text, images, video."
related_steps:
  - "stage0_pretrain"
  - "stage1_sft"
currency: "frozen"
---

# Scope
Answers:
- What backbone does Super 3.5 use, and what does the report itself say about its dimensions?
- How is the MTP head structured, initialized, and trained (single recursive head, hidden-state mixing)?
- What are the vision-language components, and where does the report describe them?
- How many parameters are there once vision and MTP are counted?

# Backbone: Inherited from Nemotron 3 Super (§1, §2.2, §4.2)
- Super 3.5 is "a 120 billion total and 12 billion active parameter model initialized from Nemotron 3 Super" (§1). §2.2: "We initialized Super 3.5 Base from Super 3 Base."
- §4.2 summarizes the model as "a 120B-parameter, 12B-active mixture-of-experts vision-language model comprising a C-RADIO vision encoder, a hybrid Mamba/attention language backbone with 512 routed experts per MoE layer, and a multi-token prediction (MTP) head."
- Table 12 (NVFP4 precision per module) lists the module families: Vision path (C-RADIO encoder, VL connector), Latent MoE (MoE + shared-expert GEMM, latent down/up projection, router/gate), Attention (QKV/out projections, KV cache), Mamba (Mamba linear, Mamba SSM cache, Mamba convolution), Embedding, lm_head. This confirms LatentMoE with shared experts, attention, and Mamba layers, and untied embedding / lm_head entries.
- §5 calls it a "hybrid Mamba-Attention model" and builds reusable Mamba-2 recurrent-state checkpoints (§5.5).

**The report has no architecture dimensions table** (no layer count, model dimension, head counts, latent size). The only dims it states are Super 3 Base values mentioned while describing Lightning's MTP initialization (§2.1): Lightning prunes "the hidden size from 4096 to 2688, MoE intermediate size from 2688 to 1856, and number of routed experts from 512 to 128". So for Super 3 Base: hidden size 4096, MoE intermediate size 2688, 512 routed experts.

For the full layer pattern and dimensions (88 layers, 32 Q-heads / 2 KV-heads, top-k 22, latent size 1024, etc.) see `skills/nemotron-super3/paper/architecture.md` (Nemotron 3 Super report) and the released Hugging Face `config.json` of NVIDIA-Nemotron-3.5-Super-VL-120B-A12B. The Super 3.5 report does not restate them, and nothing in it says they changed.

# Multi-Token Prediction: One Recursively Shared Head (§2.1)
- Native speculative-decoding support via an MTP head "adopting the recursively shared MTP formulation detailed in NVIDIA (2026b)" (the Nemotron 3 Super report). The head is an internal drafter that predicts several future tokens from the backbone's hidden representations. Candidates are verified in a single backbone forward pass using rejection sampling or exact token matching.
- **Initialization (Super):** MTP weights come from the Nemotron 3 Super Base checkpoint **prior to long-context extension**, which had two distinct MTP heads. To consolidate drafting into a single, recursively shared head "while preserving the underlying representation space", Super takes the MTP parameters **only from the first head and discards the second**.
- (For Lightning, by contrast, the MTP head is derived from Super 3 Base's by fusing latent MoE projections into expert up/down matrices and pruning; this is not part of Super.)
- **Warm-up:** 100B tokens, backbone frozen, only the MTP head optimized (details in pretraining.md).

## Hidden-state mixing (Super only)
- Problem: under standard recursive MTP training, step k conditions on the full shifted sequence of ground-truth hidden states from step k-1. At inference, the head conditions on a growing mix of backbone states and synthetic states from earlier MTP steps. This train-inference conditioning mismatch compounds with draft depth and lowers acceptance at later positions. Lightning used the standard 2-step recursive training.
- Fix (similar to NVIDIA 2026c): at training step k, input hidden states are **sampled from the states produced by earlier MTP steps 1, ..., k-1**, rather than only step k-1.
- Claimed effect: more robust deeper draft steps and "significant acceptance rate gains" without the multi-step simulation cost of test-time training (TTT).
- Hidden-state mixing is applied again in post-training MTP warm start and MTP boosting (§3.7), which also switch from cross-entropy to knowledge distillation with the frozen backbone as teacher. Boosting results: SPEED-Bench average acceptance length 4.287 -> 4.718 (greedy), 3.787 -> 4.612 (probabilistic sampling) (§1). Those details belong to the post-training/inference chunks.

# Vision-Language Components (§3.1, §3.4, §4.1, §4.2)
The report does **not** have a dedicated vision-architecture section. What it states:

| Component | What the report says | Where |
|---|---|---|
| Vision encoder | "C-RADIO vision encoder"; kept BF16 in all recipe tiers; frozen during QAD; replicated (not expert-split) during calibration; held outside the AutoQuant search | §4.2, Table 12, §4.3, §4.4 |
| Projector / connector | "vision projector", warmed up before SFT "(NVIDIA, 2026a)" (the Nemotron 3 Nano Omni report); called "VL connector" in Table 12 (BF16 in all tiers) | §3.1, Table 12 |
| Training of vision parts | Base is text-only; after projector warm-up, SFT "jointly fine-tune[s] the language and vision components"; vision modules (and the MoE router) are frozen in the video teacher's RL | §3.1, §3.4.1 |
| Video input | Video-reasoning data is "processed as a single video"; SA-V tracking frames are "processed as ordered multi-image inputs"; video teacher evaluated under a 256-frame reasoning protocol | §3.4.1 |
| Visual tokens | Calibration passes images through the vision encoder "so that the language model observes ... the projected visual tokens" | §4.2 |
| Audio | None described; abstract lists text, images, video, tool environments | abstract |

**Temporal video embedder:** the report does not name one. The repository's docs (`docs/nemotron/super35/README.md`) and the Megatron-Bridge Super-VL model provider list a temporal video embedder alongside C-RADIO and the projector. Treat it as coming from the released model/provider, not the report.

Vision training data comes from the Nemotron 3 Nano Omni corpus plus newly curated data (§3.1). That belongs to the SFT chunk.

# Parameter Accounting (§4.1)
- Headline: 120B total / 12B active.
- For BPE computation, N = **124.453 billion** = "the original parameter count, including vision, MTP, and unquantized modules". Released NVFP4 payload S = 74.987 GB gives 4.82 bits per element.
- The report does not split the ~4.45B difference between the vision encoder, projector, and MTP head.

# Caveats
- Do not invent Super 3.5 dims. The report gives none apart from the three Super 3 Base values quoted in the Lightning MTP-pruning sentence. Defer to the Super 3 report and the HF config.
- **MTP wording inconsistencies:**
  - The intro (§1) says Super "consolidates two non-shared MTP heads into a single recursively shared head".
  - §2.1 says the Super 3 Base checkpoint "originally contained two distinct MTP heads".
  - The Nemotron 3 Super report (see `skills/nemotron-super3/paper/architecture.md`) describes "two shared-weight MTP layers".
  - §3.1 SFT config says Super 3.5 SFT "retain[s] two shared-weight MTP layers with an auxiliary-loss scaling factor of 0.3". The most plausible reading is one shared head applied for two recursive steps, but the report does not reconcile these phrasings.
- "C-RADIO" is the name used in §4; §3.1 says only "vision projector". The report gives no encoder version, resolution, tiling, or patch details.
- `related_steps` refer to this repository's `super35` stages (VL image-caption pretraining and SFT), which exercise the full VL stack. They are not the report's text-only pretraining.
