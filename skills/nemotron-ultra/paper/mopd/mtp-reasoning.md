---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "mtp-reasoning"
paper_sections: ["3.4", "3.5"]
title: "MTP Boosting (speculative-decoding drafter) + Reasoning Efficiency and Control"
summary: |
  Covers §3.4 MTP Boosting, a head-only continued-training stage that aligns the
  Multi-Token Prediction (MTP) drafter to the frozen backbone's next-token
  distribution via temperature-scaled forward-KL distillation, improving
  speculative-decoding acceptance lengths. Also covers §3.5 reasoning efficiency
  and control: the three reasoning modes (reasoning-off, regular, medium-effort)
  and inference-time budget control, where medium-effort uses ~2.5x fewer tokens
  at ~7% accuracy cost.
key_facts:
  - "Nemotron 3 Ultra ships native speculative-decoding support via a Multi-Token Prediction (MTP) head trained throughout all training stages, following Nemotron 3 Super."
  - "MTP uses a shared MTP-head formulation applied recursively for several MTP steps, so the draft horizon grows without additional parameters (same as Super3)."
  - "MTP Boosting continues training the MTP starting from the MOPD checkpoint (§3.3); the backbone is frozen and only the MTP head receives gradient updates."
  - "MTP Boosting loss: temperature-scaled forward-KL against the backbone's logits; the standard cross-entropy term against the gold token is disabled (head matches full backbone distribution, not the one-hot label)."
  - "Distillation temperature T = 2; number of MTP steps N_mtp = 7; objective scaled by T^2 (following Hinton et al., 2015)."
  - "At boosting time the MTP forward pass is modified so that the hidden states input to MTP step k are sampled from the set of hidden states produced at MTP steps 1..k-1 (rather than just the previous step's generated hidden states), exposing the head to inference-like noise."
  - "MTP Boosting data: on-policy rollouts from the MOPD checkpoint, seed prompts from Nemotron-Post-Training-Dataset-v2 (general) and Nemotron-RL-Super-Training-Blends (agentic); rollouts sampled with temp = 1."
  - "MTP head trained on these rollouts for 12K steps at global batch size 64, sequences capped at 8K tokens; loss accumulated over the assistant response in each sample."
  - "MTP accuracy evaluated with SPEED-Bench, measuring per-sample acceptance lengths (ALs) across the qualitative data split; Table 6 uses draft length 7."
  - "Boosted MTP raises average acceptance length on SPEED-Bench qualitative split from 4.387 (greedy) to 4.584 (greedy), and 4.165 to 4.331 (temp=1 sampling)."
  - "MTP-Boosting yields relative speculative-decoding speedup improvements from 3.15% (summarization) to 5.82% (coding)."
  - "Three reasoning modes: reasoning-off, regular, and medium-effort; regular and medium-effort can be combined with inference-time budget control."
  - "Medium-effort reasoning mode introduced during SFT, later optimized during RLVR; ~2.5% of RLVR training prompts are in medium-effort mode, covering math, STEM and coding, with length-based adjustments applied to RL rewards."
  - "Medium-effort mode uses on average ~2.5x fewer tokens than regular mode at the cost of approximately 7% drop in accuracy (Figure 10, Artificial Analysis Intelligence Index V4)."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What is MTP Boosting and why is it needed (train-inference mismatch)?
- How is the MTP head trained during boosting (frozen backbone, KL loss, data, hyperparameters)?
- What speculative-decoding gains does MTP Boosting deliver (SPEED-Bench acceptance lengths, speedups)?
- What reasoning modes does Nemotron 3 Ultra support and how is the accuracy/efficiency trade-off controlled?

# §3.4 MTP Boosting

## Motivation: train-inference mismatch
Even with a shared MTP head, naive teacher-forced MTP training does not match autoregressive MTP inference. At inference, later MTP steps condition on an increasingly noisy mixture of target-model (backbone) and MTP-generated hidden states. This distribution differs from the teacher-forced training distribution and degrades acceptance at deeper draft positions. MTP Boosting aims to make the MTP head match the backbone's next-token distribution under the input conditions/noise it encounters at inference.

## Training procedure
- Continue training the MTP starting from the MOPD checkpoint (§3.3).
- Backbone fixed for the entire phase; only the MTP head receives gradients (no risk of regressing backbone quality; reduced optimizer-state and activation memory per step).
- Modified MTP forward pass: hidden states input to MTP step k are sampled from the set of hidden states produced at MTP steps 1..k-1 (not simply the previous MTP step's generated hidden states), exposing the head to inference-like noise.

## Data
| Item | Value |
|---|---|
| Rollout source | On-policy rollouts from the MOPD checkpoint (§3.3) |
| Seed prompts (general) | Nemotron-Post-Training-Dataset-v2 |
| Seed prompts (agentic) | Nemotron-RL-Super-Training-Blends |
| Rollout sampling temp | 1 |
| Training steps | 12K |
| Global batch size | 64 |
| Sequence cap | 8K tokens |
| Loss positions | accumulated over assistant response |

## Loss (Eq. 4)
- Temperature-scaled forward-KL against backbone logits; cross-entropy gold-token term disabled.
- Distillation temperature T = 2.
- Number of MTP steps N_mtp = 7.
- Objective scaled by T^2 (Hinton et al., 2015).

## Results — SPEED-Bench acceptance lengths (Table 6, draft length 7)
Main values = greedy decoding; values in parentheses = temperature sampling (temp = 1).

| Category | Nemotron 3 Ultra | + MTP-Boosting | Qwen3.5-397B-A17B | DeepSeek-V4-Flash |
|---|---|---|---|---|
| Coding | 5.152 (4.739) | 5.452 (4.872) | 5.550 (5.307) | 2.835 (2.584) |
| Humanities | 3.950 (3.738) | 4.102 (3.857) | 4.095 (3.875) | 2.611 (2.402) |
| Math | 5.127 (4.661) | 5.343 (4.733) | 5.002 (4.646) | 2.934 (2.747) |
| Multilingual | 5.141 (4.937) | 5.382 (5.179) | 4.951 (4.836) | 2.811 (2.666) |
| QA | 4.251 (4.089) | 4.469 (4.331) | 4.371 (4.219) | 2.586 (2.419) |
| RAG | 5.207 (5.098) | 5.380 (5.367) | 5.093 (5.051) | 2.868 (2.768) |
| Reasoning | 4.672 (4.315) | 4.896 (4.502) | 4.572 (4.466) | 2.782 (2.631) |
| Roleplay | 2.801 (2.767) | 2.940 (2.856) | 3.972 (3.807) | 2.149 (2.056) |
| STEM | 4.258 (3.935) | 4.435 (4.094) | 4.323 (4.062) | 2.703 (2.462) |
| Summarization | 4.413 (4.321) | 4.552 (4.468) | 4.787 (4.785) | 2.642 (2.560) |
| Writing | 3.285 (3.209) | 3.476 (3.385) | 3.667 (3.605) | 2.419 (2.201) |
| Average | 4.387 (4.165) | 4.584 (4.331) | 4.580 (4.423) | 2.667 (2.500) |

- Boosted MTP consistently increases acceptance length over the base MTP, most pronounced at deep draft positions.
- Relative speculative-decoding speedup improvements range from 3.15% (summarization) to 5.82% (coding).

## Relation to Super3
Same shared, recursively-applied MTP-head formulation as Nemotron 3 Super; MTP head trained throughout all stages following Super. MTP Boosting (head-only KL distillation) is the added post-training refinement described here.

# §3.5 Reasoning Efficiency and Control

- Trained for three reasoning modes: reasoning-off, regular, and medium-effort.
- Regular and medium-effort modes can be combined with inference-time budget control, covering the accuracy-efficiency trade-off spectrum and complementing task-level controls (e.g. turn-limits in agentic applications).
- Medium-effort mode introduced during SFT, later optimized during RLVR.
- ~2.5% of RLVR training prompts are in medium-effort mode, covering math, STEM and coding; length-based adjustments applied on the RL rewards for them.
- Same recipe trained both Ultra and Super; optimization effects generalize beyond math/STEM/coding; effort mode calibrated by adjusting hyperparameters.

## Accuracy-efficiency trade-off (Figure 10)
- Figure 10 plots Artificial Analysis Intelligence Index V4 (y-axis) vs a relative verbosity measure (x-axis), using Qwen 3.5 397B's average token usage per task as reference, averaged over the 10 tasks in AA Index V4.
- Claim: Ultra's medium-effort mode uses on average ~2.5x fewer tokens than the regular mode at the cost of approximately 7% drop in accuracy.

# Caveats
- The ~2.5x token reduction and ~7% accuracy drop are paper claims tied specifically to Figure 10 (AA Intelligence Index V4, 10 tasks), not a general guarantee.
- SPEED-Bench acceptance lengths in Table 6 are at draft length 7; greedy vs temp=1 numbers must not be conflated.
- The 3.15%-5.82% figures are relative speculative-decoding speedup improvements (boosted vs base MTP), not absolute throughput.
- MTP Boosting freezes the backbone; do not claim it changes backbone quality.
- Do not infer end-to-end inference speedup numbers from acceptance lengths alone; the paper reports relative improvements only.
