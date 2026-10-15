---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "rl-mtp-boosting"
paper_sections: ["1", "2.1", "3.6", "3.7", "4.7", "Table 8", "Table 9", "Table 20", "Figure 6"]
title: "Nemotron 3.5 Super MTP Boosting: Head-Only LK-Loss Distillation, Hidden-State Mixing, and Acceptance Lengths"
summary: |
  Covers §3.7 MTP boosting for Nemotron 3.5 Super: a backbone-frozen, MTP-head-only recipe applied
  twice - a 4k-step warm start on the SFT checkpoint and 25k boosting steps on the final post-trained
  backbone - combining hidden-state mixing (from §2.1) with a knowledge-distillation loss against the
  frozen backbone (the LK loss, replacing Ultra's temperature-scaled forward KL). Records data,
  procedure, Table 8 (warm-start transfer), Table 9 (per-category SPEED-Bench acceptance lengths vs
  other models), Figure 6 (per-position acceptance), and how the headline 4.612 relates to the
  4.592 that appears in the MTP-quantization section (§4.7 / Table 20).
key_facts:
  - "§2.1: Super 3.5 uses a single recursively shared MTP head, initialized from the first of Nemotron 3 Super Base's two MTP heads (second discarded); 100B-token MTP warm-up with frozen backbone during pretraining, with hidden-state mixing: at step k, input hidden states are sampled from those produced by earlier MTP steps 1..k-1 rather than only step k-1."
  - "§3.7: a dedicated MTP-only recipe is applied at two points - a short warm start from the Super 3.5 SFT checkpoint, and again after all backbone post-training (= 'MTP boosting'). Backbone frozen; only MTP head parameters updated."
  - "§3.7 components: (1) continue hidden-state mixing (§2.1); (2) knowledge-distillation loss with the frozen backbone as teacher instead of the cross-entropy used in preceding MTP stages - 'particularly beneficial under non-greedy sampling'. Loss = LK loss (Samarin et al., 2026), replacing Ultra's temperature-scaled forward KL."
  - "Data (§3.7): on-policy rollouts from the corresponding backbone checkpoint; seed prompts from Nemotron-Post-Training-Dataset-v2 (general) and Nemotron-RL-Super-Training-Blends (agentic); temp = 1; sequences capped at 8K tokens; loss only on assistant-response tokens."
  - "Procedure (§3.7): 4k MTP-only steps from the SFT checkpoint (plateaus much sooner than the final run); that head initializes final boosting, trained 25k more steps against the final Super 3.5 backbone. Lightning: no warm start, 48k boosting steps."
  - "Table 8 (avg AL, draft length 7, temp=0 / temp=1): SFT checkpoint 4.589 / 4.048; +4k MTP-only 5.080 / 4.730; Super 3.5 Final (native MTP) 4.287 / 3.787; + MTP head from SFT-stage run 4.528 / 4.436; + 25k MTP boosting 4.718 / 4.612."
  - "Table 9 (SPEED-Bench qualitative split, draft length 7, reasoning on): Super 3.5 average Native 4.288 (3.787) -> Boosted 4.718 (4.612); greedy outside, temp=1 probabilistic draft sampling in parentheses."
  - "§1 / abstract: boosting improves average AL from 4.287 to 4.718 (greedy) and 3.787 to 4.612 (probabilistic draft sampling) - 'up to 21.8%'. §3.7 text rounds this to 'from 3.78 to 4.61' at T = 1."
  - "Figure 6: boosted MTP improves joint acceptance at every draft position (1-7), with the largest gains at deeper positions, for both T=0 and T=1."
  - "4.612 vs 4.592: 4.612 is the BF16 model's boosted-MTP AL (Tables 8, 9, 20 row 1). 4.592 is Table 20's NVFP4 backbone + BF16 MTP. §4.7 prose instead says the NVFP4 backbone with BF16 MTP has AL 4.612 and quantizing MTP experts reduces it to 4.592, which disagrees with Table 20 (NVFP4 experts -> 4.518). The selected (shaded) Table 20 config is 4.520 at DL=7."
related_steps: []
currency: "frozen"
---

# Scope

- What is MTP boosting for Nemotron 3.5 Super and how does it differ from Ultra's?
- What data, loss, steps and initialization are used?
- What acceptance lengths does it achieve, per category and vs other models?
- Which acceptance-length number to quote (4.612 vs 4.592 vs 4.520)?

# Background: MTP in Super 3.5 (§2.1)

- Native speculative decoding via a recursively shared MTP head (formulation from Nemotron 3 Super).
- Init: Nemotron 3 Super Base (before long-context extension) had two MTP heads; Super 3.5 takes the first
  and discards the second, forming one recursively shared head.
- 100B-token MTP warm-up in pretraining with the backbone frozen.
- **Hidden-state mixing** (Super only; Lightning used standard 2-step recursive MTP training): at training
  step k, input hidden states are sampled from those produced by earlier MTP steps 1..k-1 instead of only
  step k-1, matching the mixed backbone/MTP states seen during autoregressive drafting, without the cost of
  test-time-training-style multi-step simulation.
- §3.1: SFT keeps two shared-weight MTP layers at aux-loss 0.3; §3.6: MTP heads trained jointly during MOPD.

# The recipe (§3.7)

| Item | Value |
| --- | --- |
| When | (a) warm start from the Super 3.5 SFT checkpoint; (b) after all backbone post-training (= MTP boosting) |
| Trainable | MTP head only; backbone frozen |
| Component 1 | hidden-state mixing (§2.1) |
| Component 2 | knowledge distillation from the frozen backbone (vs cross-entropy in earlier MTP stages) |
| Loss | LK loss (Samarin et al., 2026); Ultra used temperature-scaled forward KL |
| Rollouts | on-policy from the corresponding backbone checkpoint |
| Seed prompts | Nemotron-Post-Training-Dataset-v2 (general), Nemotron-RL-Super-Training-Blends (agentic) |
| Sampling temp | 1 |
| Sequence cap | 8K tokens |
| Loss positions | assistant-response tokens only |
| Warm-start steps | 4k (from SFT checkpoint) |
| Boosting steps | 25k more, against final backbone, head initialized from warm-start run |
| Lightning | no warm start; 48k boosting steps |
| Not reported | batch size, LR, number of MTP steps used in training, LK-loss hyperparameters |

Evaluation: SPEED-Bench qualitative split, per-sample acceptance length (AL), draft length 7.

# Table 8: warm start and transfer (avg AL, draft length 7)

| Model | temp = 0 | temp = 1 |
| --- | --- | --- |
| Super 3.5 (SFT checkpoint) | 4.589 | 4.048 |
| + 4k MTP-only training iterations | 5.080 | 4.730 |
| Super 3.5 (Final), native MTP | 4.287 | 3.787 |
| + MTP head from the SFT-stage run | 4.528 | 4.436 |
| + 25k MTP Boosting iterations | 4.718 | 4.612 |

Reading: the SFT-stage head transferred to the final backbone retains "significant part" of its gain
(4.287 -> 4.528 greedy; 3.787 -> 4.436 sampled); boosting adds the rest. temp = 0 is greedy; temp = 1 is
probabilistic draft sampling.

# Table 9: SPEED-Bench per category (draft length 7, reasoning on)

Main value = temp 0 (greedy); parentheses = temp 1 with probabilistic draft sampling.

| Category | Super 3.5 Native | Super 3.5 Boosted | Gemma-4 31B | Qwen3.5 122B-A10B | Qwen3.8-Flash-Next low | medium | xhigh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coding | 4.966 (4.343) | 5.563 (5.441) | 5.792 (5.459) | 5.630 (5.221) | 5.506 (5.310) | 5.355 (5.124) | 3.975 (3.714) |
| Humanities | 3.703 (3.245) | 3.975 (3.817) | 4.488 (4.039) | 4.584 (3.956) | 4.122 (3.763) | 4.073 (3.628) | 3.873 (3.005) |
| Math | 4.533 (4.093) | 4.841 (4.825) | 5.676 (5.321) | 5.149 (4.826) | 5.207 (4.982) | 5.169 (4.815) | 3.942 (3.672) |
| Multilingual | 4.945 (4.266) | 5.548 (5.358) | 5.703 (5.435) | 4.995 (4.613) | 5.056 (4.911) | 4.942 (4.678) | 4.218 (3.926) |
| QA | 4.083 (3.624) | 4.526 (4.378) | 4.366 (4.092) | 4.513 (4.111) | 3.988 (3.913) | 4.096 (3.748) | 3.626 (3.322) |
| RAG | 5.054 (4.424) | 5.538 (5.496) | 5.532 (5.338) | 5.233 (4.986) | 5.453 (5.146) | 5.469 (5.185) | 4.424 (3.981) |
| Reasoning | 4.314 (3.919) | 4.791 (4.751) | 5.086 (4.715) | 4.747 (4.427) | 4.545 (4.358) | 4.463 (4.213) | 3.797 (3.619) |
| Roleplay | 3.444 (3.046) | 3.781 (3.703) | 3.475 (3.348) | 4.614 (3.331) | 2.657 (2.523) | 2.605 (2.457) | 2.881 (2.584) |
| STEM | 3.761 (3.406) | 4.102 (4.045) | 4.858 (4.464) | 4.640 (4.210) | 4.556 (4.135) | 4.414 (4.038) | 3.614 (3.182) |
| Summarization | 4.716 (4.110) | 5.235 (5.041) | 4.536 (4.438) | 4.893 (4.416) | 4.248 (4.135) | 4.288 (4.113) | 3.627 (3.517) |
| Writing | 3.646 (3.183) | 4.001 (3.873) | 3.697 (3.566) | 3.787 (3.549) | 3.340 (3.162) | 3.255 (3.045) | 3.424 (2.968) |
| Average | 4.288 (3.787) | 4.718 (4.612) | 4.837 (4.565) | 4.799 (4.331) | 4.425 (4.213) | 4.375 (4.095) | 3.764 (3.408) |

- Boosting raises AL in every category, at both temperatures.
- At temp = 1, boosted Super 3.5 has the highest average in the table (4.612); at temp = 0, Gemma-4 31B
  (4.837) and Qwen3.5 122B-A10B (4.799) average higher than boosted Super 3.5 (4.718).
- Qwen3.8-Flash-Next sub-columns are reasoning-effort settings.

# Figure 6: per-position acceptance

![Joint MTP acceptance rate per draft position (report Figure 6)](../../../../docs/assets/super35/mtp-acceptance.png)

Joint acceptance rate at draft positions 1-7 (SPEED-Bench qualitative, draft length 7) for original vs boosted
MTP at T=0 and T=1. Boosted is better at every position, with the largest gains at deeper positions, where
inference conditions on the head's own hidden states. The report gives no numeric per-position values; read
the plot only qualitatively.

# Headline numbers and where they appear

| Value | Meaning | Where |
| --- | --- | --- |
| 4.287 -> 4.718 | avg AL, greedy, native -> boosted | §1, Table 8 |
| 4.288 -> 4.718 | same, per Table 9 average | Table 9 (native 4.288, not 4.287) |
| 3.787 -> 4.612 | avg AL, temp = 1, native -> boosted | §1, Tables 8, 9 |
| "3.78 to 4.61" | same, rounded | §3.7 text |
| "up to 21.8%" | 4.612 / 3.787 = +21.8% | abstract |
| 4.612 (DL=7), 4.115 (DL=5) | BF16 backbone + BF16 MTP, temp 1 | Table 20 row 1 |
| 4.592 (DL=7), 4.109 (DL=5) | NVFP4 backbone + BF16 MTP | Table 20 row 2 |
| 4.518 (DL=7) | NVFP4 backbone + NVFP4 MTP experts, BF16 heads | Table 20 row 3 |
| 4.520 (DL=7), 4.063 (DL=5), 1.127x projected speedup | **selected (shaded)**: NVFP4 backbone, FP8 LM head, NVFP4 MTP experts, FP8 MTP LM head | Table 20 |

**Contradiction (§4.7 vs Table 20).** §4.7 prose: "Against the NVFP4 backbone with a BF16 MTP (AL = 4.612 at
draft length 7), quantizing the MTP experts reduces AL to AL = 4.592 (Table 20)." Table 20 instead gives 4.612
for the **BF16** backbone with BF16 MTP, 4.592 for the NVFP4 backbone with BF16 MTP, and 4.518 after quantizing
the MTP experts. Quote 4.612 as the BF16 boosted-MTP AL, and give both readings if asked about NVFP4 MTP.

# Repo cross-reference

The repo's RL recipe ends at MOPD; MTP boosting is not included (`docs/nemotron/super35/rl/index.md`,
`rl/mopd.md`). MTP quantization is discussed in `docs/nemotron/super35/quantization.md`.

# Caveats

- AL numbers are SPEED-Bench qualitative split, draft length 7 unless noted; greedy and temp-1 values must not be conflated.
- 4.287 (§1, Table 8) vs 4.288 (Table 9) for native greedy AL is a 0.001 inconsistency in the report.
- Table 9 and the abstract compare *acceptance length*, not end-to-end throughput; the only speedups given are Table 20's projected DL=5 speedups for quantization choices.
- Boosting freezes the backbone; it does not change backbone accuracy.
- Lightning numbers (48k steps, no warm start) are separate; never attribute them to Super.
