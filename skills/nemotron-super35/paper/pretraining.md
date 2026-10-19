---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "pretraining"
paper_sections: ["2", "2.1", "2.2", "2.4", "2.5"]
title: "Nemotron 3.5 Super Pretraining: MTP Warm-up, 1T-Token Distillation from Ultra 3, 1M Context, Agentic Mid-training, LC Recovery"
summary: |
  Super 3.5 Base is built from Nemotron 3 Super Base through a text-only sequence: a 100B-token MTP
  warm-up with the backbone frozen and hidden-state mixing; 1T tokens of distillation from Ultra 3 Base
  (Skew Reverse KL, alpha 0.1, plus cross-entropy, LR 4.5e-6 -> 4.5e-5 -> 4.5e-6); 100B tokens of Indic
  CPT (see data.md); a 35B-token 1M-context phase (constant LR 4.5e-6, CP16/TP8/EP64 on GB300); 65.4B
  tokens of agentic software-engineering mid-training at 256K (LR reset, warm-up to 1e-5, then constant);
  and a 9B-token long-context recovery that brings RULER-1M from 33.5 back to 81.0.
key_facts:
  - "MTP warm-up: 100B tokens, backbone frozen, only the MTP head optimized; Super uses hidden-state mixing (§2.1)."
  - "Distillation: 1T tokens; teacher Ultra 3 Base; loss Skew Reverse KL (Ko et al., 2024) with alpha = 0.1 (§2.2)."
  - "Shared recipe (stated for Lightning, reused by Super): top-128 teacher logits + an extra sink token for residual mass; KL and cross-entropy weighted 0.9 / 0.1; MTP loss scaling factor 0.1."
  - "Super distillation LR: minimum 4.5e-6, maximum 4.5e-5 (anchored on Super 3 Base LRs); schedule shape: ramp up over 200B tokens, linear decay back over the remaining 800B (stated for Lightning; Super uses 'the same way')."
  - "Super MoE sequence load-balancing loss coefficient 1e-5 (vs 1e-4 for Lightning)."
  - "LC-Phase: CPT at 1M-token context, constant LR 4.5e-6, 16-way CP, 8-way TP, 64-way EP on GB300; data blend same as Nemotron 3 Ultra; 35B tokens (§2.4)."
  - "Mid-training: 65.4B tokens at 256K sequence length; LR reset, linear warm-up to 1e-5 (about twice the final pretraining LR), then constant (§2.5)."
  - "Mid-training dropped RULER-1M from 79.9 to 33.5; a 9B-token LC recovery with the same LC recipe restored it to 81.0."
  - "Mid-training effect (same experimental ~33B-token / 2K-step SFT on both): DeepSWE 6.4 -> 10.4; non-OpenHands harnesses improve by 8.6 points on average on SWE-bench Verified and 6.8 on Terminal-Bench 2.1 (Table 5)."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- In what order were Super 3.5 Base's pretraining stages run, and how many tokens did each use?
- How were the MTP head initialized and warmed up?
- What distillation teacher, loss, and LR schedule were used?
- How was the 1M context added (LR, parallelism, data, tokens)?
- What did agentic mid-training do (LR, context, effect), and why was a long-context recovery needed?

Indic continued pretraining (§2.3) and the mid-training data blend (Table 4) are covered in data.md. Base-model evaluation (§2.6, Table 6) belongs to the evaluation chunk.

# Stage Order (§1, §2 intro)

| # | Stage | Tokens | Context | LR |
|---|---|---|---|---|
| 1 | MTP warm-up (§2.1) | 100B | not stated | not stated |
| 2 | Distillation from Ultra 3 Base (§2.2) | 1T | not stated | 4.5e-6 -> 4.5e-5 -> 4.5e-6 |
| 3 | Indic CPT (§2.3; data.md) | 100B nominal | not stated | not stated |
| 4 | Long-context phase (§2.4) | 35B | 1M | constant 4.5e-6 |
| 5 | Agentic SWE mid-training (§2.5) | 65.4B | 256K | warm-up to 1e-5, constant |
| 6 | Long-context recovery (§2.4) | 9B ("about 9B" in §2.5) | 1M (same recipe as stage 4) | same recipe as stage 4 |

All stages are text-only; vision enters in SFT (§3.1).

# §2.1 MTP Initialization and Warm-up
- Super takes MTP weights from **Nemotron 3 Super Base before long-context extension**. That checkpoint had two distinct MTP heads. Super keeps only the **first** head and discards the second, to form a single, recursively shared head (see architecture.md).
- **Warm-up: 100B tokens**, backbone parameters frozen, only the MTP head optimized.
- **Hidden-state mixing (Super):** at recursive training step k, input hidden states are sampled from those produced by earlier MTP steps 1..k-1 rather than only step k-1. This reduces the train-inference conditioning mismatch that otherwise lowers acceptance at deeper draft positions. The report says it achieves significant acceptance-rate gains without the multi-step simulation cost of test-time training (TTT). Lightning used standard 2-step recursive MTP training instead.
- The report gives no LR, batch size, data, or acceptance-length numbers for the warm-up.

# §2.2 Distillation (1T tokens)

The report describes the Lightning recipe in full and then says Super uses "the same distillation recipe ... but with" specific changes.

| Setting | Super 3.5 Base | Source |
|---|---|---|
| Initialization | Super 3 Base | §2.2 |
| Token budget | 1 trillion | §1, §2.2 |
| Teacher | Ultra 3 Base | §2.2 |
| Loss | Skew Reverse KL divergence (Ko et al., 2024), alpha = 0.1 | §2.2 (Super-specific) |
| Teacher logits | top-128, with an extra sink token for residual probability mass | stated for Lightning; inherited via "same recipe" |
| Loss mixing | distillation 0.9 + cross-entropy 0.1 (for rare-token signal) | stated for Lightning (with forward KL); inherited via "same recipe" |
| Data | 1T tokens of Ultra 3 Phase 2 data | stated for Lightning only; not restated for Super |
| MoE sequence load-balancing coefficient | 1e-5 (Lightning: 1e-4) | §2.2 (Super-specific) |
| LR min / max | 4.5e-6 / 4.5e-5, anchored on Super 3 Base LRs | §2.2 (Super-specific) |
| LR schedule shape | start at min, ramp slowly to max over 200B tokens, then decay linearly back to min over the remaining 800B | stated for Lightning; Super chose LRs "the same way" |
| MTP loss scaling factor | 0.1 | stated for Lightning; inherited via "same recipe" |

Rationale stated in the report: start from the minimum LR and ramp slowly "to limit early drift from the initialization".

# §2.4 Long-Context Phase (LC-Phase) and Recovery

| Aspect | Setting |
|---|---|
| Method | Continuous pretraining (CPT) at 1M-token context, "similar to Nemotron 3 series" |
| Learning rate | Constant 4.5 x 10^-6 |
| Context parallelism | 16-way |
| Tensor parallelism | 8-way |
| Expert parallelism | 64-way |
| Hardware | GB300 GPUs |
| Data blend | Same as Nemotron 3 Ultra's (NVIDIA, 2026c) |
| Tokens | 35B |

**Recovery after mid-training:** after 256K mid-training, RULER-1M dropped from **79.9 to 33.5**. A short recovery phase of **9B tokens with the same recipe** brought it back to **81.0** (the intro phrases this as "restoring RULER-1M ... from 33.5 to 81.0").

# §2.5 Agentic Software-Engineering Mid-training

**Why:** pretraining runs at short context and sees almost no agent interactions. Natural SWE data (whole repositories, issue-linked PRs, commit histories, build logs) is often too long for it. SFT is mostly synthetic, and very long SFT stages can reduce later RL gains. Mid-training sits between the two: long-context training on natural SWE data plus a small share of agent traces, with behavior shaping left to post-training.

| Setting | Value |
|---|---|
| Starting checkpoint | The long-context (LC-Phase) checkpoint |
| Tokens | 65.4B |
| Sequence length | 256K |
| LR | Reset; linear warm-up to max 1 x 10^-5 ("about twice the final pretraining learning rate"), then constant for the rest of the run |
| Data format | Mostly natural data in original format, no chat templates; agent traces are the only SFT-style data, with tool responses masked from the loss |
| Hygiene | Every source decontaminated against SWE-bench and Terminal-Bench; secrets removed |
| Blend | Table 4 (see data.md): Issue-PR 41%, Repositories 20%, Agentic traces 10%, Commit diffs 3%, Build/CI logs 1%, Pretraining replay 20%, Long-context data 5% |

Warm-up length for the mid-training LR is not given.

## Table 5: Effect of mid-training
Measured by running **identical experimental SFT** (agentic blend of about 33B tokens, 2K steps, not the §3 SFT) on the base and the mid-trained checkpoint. pass@1, %. SWE-bench Multilingual and Pro use OpenHands. DeepSWE is averaged over 4 runs. OpenHands results come from a separate evaluation from the other harnesses.

| Benchmark | Base | Mid-trained |
|---|---|---|
| SWE-bench Multilingual | 65.9 | 65.6 |
| SWE-bench Pro | 52.1 | 51.7 |
| DeepSWE | 6.4 | 10.4 |
| BrowseComp (subset) | 64.2 | 65.0 |
| SWE-bench Verified, OpenHands | 71.3 | 71.9 |
| SWE-bench Verified, Pi | 57.4 | 67.8 |
| SWE-bench Verified, OpenCode | 54.4 | 65.8 |
| SWE-bench Verified, Hermes | 59.8 | 63.2 |
| SWE-bench Verified, Codex | 48.4 | 61.8 |
| SWE-bench Verified, Claude Code | 53.8 | 58.0 |
| Terminal-Bench 2.1, Pi | 30.3 | 35.2 |
| Terminal-Bench 2.1, OpenCode | 35.2 | 38.2 |
| Terminal-Bench 2.1, Hermes | 44.2 | 53.9 |
| Terminal-Bench 2.1, Codex | 27.0 | 39.7 |
| Terminal-Bench 2.1, Claude Code | 39.7 | 43.5 |

Report's reading:
- The clearest standard-benchmark gain is DeepSWE (the hardest long-horizon benchmark). BrowseComp improves slightly.
- SWE-bench Verified / Multilingual / Pro under OpenHands stay within run-to-run variation (about one point).
- The larger effect is harness robustness. Under OpenHands (the standard evaluation harness) the base already scores well and mid-training changes little. Under the five other harnesses the mid-trained model is better on every one: +8.6 points on average on SWE-bench Verified and +6.8 on Terminal-Bench 2.1. The largest gain is on Codex, where the base is weakest.

# Not Reproducible Here
This repository's `nemotron super35 pretrain` runs a Megatron-Bridge **vision-language image-caption** recipe, not this text pretraining. The report's pipeline needs Ultra 3 Base teacher logits and non-public mixtures (see `docs/nemotron/super35/pretrain.md`).

# Caveats
- Super's distillation data, the 200B/800B schedule split, top-128 logits, sink token, 0.9/0.1 weights, and MTP loss scale 0.1 are stated explicitly for Lightning only. They apply to Super through "same distillation recipe" / "chose learning rates the same way". The Super-specific changes are the teacher, Skew Reverse KL with alpha 0.1, the 1e-5 load-balancing coefficient, and the 4.5e-6 / 4.5e-5 LRs.
- §2.2 also says Lightning ramps to "10^-4, 10% of the Nano 3 Base's peak LR". That is a Lightning number; do not apply it to Super.
- Recovery length: "9B tokens" in §2.4 and the intro, "about 9B tokens" in §2.5. The RULER-1M "before mid-training" value (79.9) appears only in §2.4.
- Mid-training "about twice the final pretraining learning rate": 1e-5 vs the 4.5e-6 LC/minimum LR is about 2.2x. Consistent, but the report does not say which LR it means.
- Table 5 numbers come from an experimental SFT probe, not the released post-trained model. Table 5's SWE-bench Pro (OpenHands) is not comparable to the released model's 51.78% (OpenCode harness, Table 10).
- The parallelism here (CP16/TP8/EP64 on GB300) is Super's LC-Phase. Only the data blend is stated to match Nemotron 3 Ultra's LC-Phase; do not assume Ultra's parallelism or LR settings apply.
