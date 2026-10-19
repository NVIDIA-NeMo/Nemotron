---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "rl-mopd"
paper_sections: ["3.6", "Appendix C.1", "Table 27"]
title: "Nemotron 3.5 Super MOPD: Two-Stage Multi-Teacher On-Policy Distillation with Self-Anchor and Proximal Teacher"
summary: |
  Covers §3.6, the multi-teacher on-policy distillation (MOPD) stage that merges the specialist
  teachers into one Nemotron 3.5 Super student. The student (RL step 120) samples responses and
  minimizes per-token reverse KL to the teacher owning the prompt's domain, using a stop-gradient
  dense advantage from one teacher forward pass, with stale-rollout importance weights masked outside
  [0.2, 5]. Training runs in two stages with different teacher sets and prompt shares; un-owned
  environments go to a frozen self-anchor; stage 2 smooths the target with a proximal teacher
  (alpha = 0.2). Records all reported hyperparameters, infrastructure and the (limited) results:
  pre-MOPD vs MOPD output length / truncation (Table 27) and runaway-loop counts (Appendix C.1).
key_facts:
  - "§3.6: MOPD runs after RL, distilling the specialized teachers into one student, similar to Nemotron 3 Ultra; MTP boosting (§3.7) runs on the MOPD output. Lightning had no MOPD stage."
  - "Student = the RL checkpoint of §3.2 after 120 steps, the same checkpoint the teachers were branched from."
  - "Objective: student minimizes reverse KL D_KL(pi_S(.|c_t) || pi_T(.|c_t)) at every position of its own responses using the token-local update (Eq. 1): Delta theta proportional to E_{y ~ pi_S_bar}[ sum_t A_hat_t grad log pi_S(y_t|c_t) ] at theta = theta_bar."
  - "Dense advantage (Eq. 2): A_hat_t = sg[ log pi_T(y_t|c_t) - log pi_S_bar(y_t|c_t) ], pi_S_bar = frozen pre-update snapshot. One teacher forward pass per token; no verifier and no teacher sampling; zero where student and teacher agree."
  - "Asynchrony: responses may be one step stale (from pi_S_bar_{-1}); following Ultra, tokens importance-weighted by pi_S_bar / pi_S_bar_{-1} and masked if weight outside [0.2, 5] (Team et al., 2025). Only student-generated tokens enter the loss. The report notes context-distribution mismatch and masking introduce bias."
  - "Two stages on 128 GB200 nodes, because benchmarks improve at different rates and some peak within a few steps then decline: stage 1 weighted to fast-improving/regress-later domains, stage 2 to slower ones."
  - "Routing: prompts come from each teacher's training environments and go to that teacher; environments no specialist owns go to a self-anchor (frozen copy of the student's starting checkpoint), which provides no signal while the student matches it and regularizes departures."
  - "Stage 1: student and self-anchor = RL checkpoint; teachers agentic 35%, reasoning 25%, alignment (RLHF) 20%, vision tool use 15%, self-anchor 5%; 512 prompts x 1 rollout; generation capped at 48K in a 128K context; LR 2e-6; checkpoint after 30 steps; alpha = 1."
  - "Stage 2: targets chat quality (e.g. LMArena-style) and vision; student and self-anchor = stage-1 model; agentic and reasoning teachers retired (their environments folded into self-anchor); alignment 30%, vision tool use 20%, vision reasoning 20%, video reasoning and tracking 20%, self-anchor 10%; 1,024 prompts; full 128K generation; LR 3e-6; 40 steps = one pass over its prompts."
  - "Stage 2 proximal teacher (Xie et al., 2026; see also Xing et al., 2026): teacher in Eq. 2 replaced by frozen mixture alpha*pi_T + (1 - alpha)*pi_S_bar with alpha = 0.2, bounding the advantage below by log(1 - alpha) ~= -0.22."
  - "Infra: teachers run as independent, non-colocated inference fleets (TP 8, EP 8) sized per domain share and sequence length, scoring trajectories as they arrive; rollout, scoring and optimization pipelined in NeMo-RL. MTP heads trained jointly with the LM throughout."
  - "Table 27 (BF16, pre-MOPD -> MOPD): output tokens per rollout and truncation fall on vision/agentic benchmarks, e.g. BabyVision 42.3k -> 11.2k (0.27x), truncation 39.0% -> 1.9%; Tau3 Banking 35.2k -> 13.7k (0.39x), 20.9% -> 0.3%; AA-LCR unchanged 4.7k -> 4.7k."
  - "Appendix C.1: on tau3-banking, pre-MOPD had 19.7% runaway-loop failures (153/776 rollouts); after MOPD only one runaway loop failure was captured."
related_steps:
  - "stage2_rl/stage3_mopd"
currency: "frozen"
---

# Scope

- How does MOPD work for Nemotron 3.5 Super (objective, advantage, staleness correction)?
- Why two stages, and what are the teachers, shares, batch sizes, generation caps and LRs of each?
- What is the self-anchor and the proximal teacher?
- How is MOPD served/pipelined?
- What results does the report attribute to MOPD?

# Algorithm (§3.6, Eq. 1-2)

- Student pi_S (params theta), domain teacher pi_T, context c_t of token y_t in the student's response y.
- Student minimizes reverse KL D_KL(pi_S(.|c_t) || pi_T(.|c_t)) at every position of its own responses,
  using the token-local update (Lu & Lab, 2025; Yang et al., 2026; Xiao et al., 2026), treating sampled
  contexts as fixed during each update:

```
Delta theta  ∝  E_{y ~ pi_S_bar(.|q)} [ sum_t  A_hat_t * grad_theta log pi_S(y_t | c_t) ]  |_{theta = theta_bar}   (Eq. 1)
A_hat_t      =  sg[ log pi_T(y_t | c_t) - log pi_S_bar(y_t | c_t) ]                                          (Eq. 2)
```

- pi_S_bar = frozen snapshot before the update; sg = stop-gradient.
- Every generated token gets a dense advantage from **one teacher forward pass**, with **no verifier and no
  sampling from the teacher**. Direct contribution is zero where student and teacher agree.
- Asynchrony: responses may come from the previous snapshot (one step stale). Following the Ultra recipe,
  tokens are importance-weighted by pi_S_bar / pi_S_bar_{-1} and **masked outside [0.2, 5]** (Team et al., 2025).
  The report notes this corrects action probabilities at sampled contexts, while context-distribution
  mismatch and masking introduce bias.
- Only student-generated tokens enter the loss.

Difference from Ultra's write-up: the Ultra report presents a PPO-style clipped surrogate around a proximal
policy (Eq. 3 there). The Super 3.5 report presents the token-local update above with importance weighting
and masking; it does not mention a clipped ratio for MOPD.

# Routing and self-anchor (§3.6)

- Prompts come from the environments used to train each teacher and are routed to that teacher.
- Environments no specialist owns go to the **self-anchor**: a frozen copy of the student's starting
  checkpoint. It gives no signal while the student still matches it and regularizes departures.

# Two stages (§3.6)

Motivation: "benchmarks improve at different rates under distillation, and ... some peak within a few steps
and then decline with further training." Stage 1 favors domains that improve quickly and tend to regress
later; stage 2 favors slower domains.

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Student | RL checkpoint (step 120) | stage-1 model |
| Self-anchor | RL checkpoint | stage-1 model |
| Agentic teacher | 35% | retired (env -> self-anchor) |
| Reasoning teacher | 25% | retired (env -> self-anchor) |
| Alignment (RLHF) teacher | 20% | 30% |
| Vision tool-use teacher | 15% | 20% |
| Vision-reasoning teacher | — | 20% |
| Video reasoning & tracking teacher | — | 20% |
| Self-anchor share | 5% | 10% |
| Prompts per batch | 512 | 1,024 |
| Rollouts per prompt | 1 | not restated (1 in stage 1) |
| Generation | capped at 48K in 128K context | full 128K context |
| LR | 2e-6 | 3e-6 |
| Length | checkpoint after 30 steps | 40 steps (one pass over its prompts) |
| Proximal-teacher alpha | 1 (plain teacher) | 0.2 |
| Hardware | 128 GB200 nodes (both stages) | 128 GB200 nodes |

Stage 2 targets chat quality (e.g. LMArena-style evaluations) and the vision domains, which improved more
slowly in stage 1 than agentic and reasoning benchmarks.

# Proximal teacher (stage 2)

Following trust-region policy distillation (Xie et al., 2026; see also Xing et al., 2026), the teacher in
Eq. 2 is replaced by the frozen mixture `alpha * pi_T + (1 - alpha) * pi_S_bar` with **alpha = 0.2**. This
bounds the advantage below by log(1 - alpha) ~= -0.22, so tokens the teacher finds very unlikely get a
bounded negative signal; it "stabilizes training once the student has already absorbed one round of
teachers". Stage 1 uses alpha = 1.

# Infrastructure (§3.6)

- Teachers: independent, non-colocated inference fleets, tensor parallel 8, expert parallel 8, sized to
  each domain's share and sequence length.
- Teachers score trajectories as they arrive; rollout generation, teacher scoring and student optimization
  form a pipeline in NeMo-RL.
- MTP heads are trained jointly with the language model throughout (§3.7 then boosts them).

# Results attributed to MOPD

The report has no MOPD ablation table on the main benchmarks. What it does report:

## Table 27: output length per rollout and truncation, pre-MOPD -> MOPD (both BF16)

| Benchmark | Modality | Rollouts | OSL (k) / rollout | Trunc. % |
| --- | --- | --- | --- | --- |
| Baby Vision | vision | 388 x 3 | 42.3 -> 11.2 (0.27x) | 39.0 -> 1.9 |
| CharXiv (RQ) | vision | 1,000 x 3 | 11.8 -> 3.3 (0.28x) | 9.2 -> 0.6 |
| MMLongBench-Doc | vision | 1,091 x 3 | 10.8 -> 3.0 (0.28x) | 9.9 -> 0.4 |
| OCR Reasoning | vision | 1,069 x 3 | 16.8 -> 5.2 (0.31x) | 13.7 -> 0.6 |
| OSWorld | vision/agentic | 361 x 3 | 13.6 -> 5.1 (0.38x) | 1.2 -> 0.0 |
| Tau3 Banking | text/agentic | 97 x 8 | 35.2 -> 13.7 (0.39x) | 20.9 -> 0.3 |
| AA-Omniscience | text | 600 x 10 | 10.0 -> 5.6 (0.56x) | 1.9 -> 0.3 |
| SciCode | text | 65 x 8 | 7.2 -> 6.5 (0.90x) | 0.0 -> 0.0 |
| AA-LCR | text | 100 x 16 | 4.7 -> 4.7 (1.00x) | 0.0 -> 0.2 |
| Tau2-AA | text/agentic | 278 x 8 | 5.0 -> 5.1 (1.03x) | 0.0 -> 0.0 |

Caption: "Post-training RL removed the non-terminating tail, not baseline verbosity."

## Appendix C.1-C.2: runaway loops

- tau3-banking pre-MOPD: 20.9% truncation, 35,173 output tokens/rollout, 19.7% runaway-loop failures (153/776).
- After MOPD: 0.3% truncation, 13,669 output tokens/rollout, one runaway-loop failure.
- MOPD "fortifies the latent support for stopping" enough to withstand FP4 expert quantization noise (3
  runaway cases in FP4 under MOPD); see report Appendix C.

Final post-trained model results (after MTP boosting, which does not change the backbone) are in Table 10
(e.g. SWE-bench Pro 51.78).

# Repo cross-reference

`docs/nemotron/super35/rl/mopd.md` maps the two phases onto `stage2_rl/stage3_mopd` configs. Per those docs,
the reference run selected phase-2 step 30 as the released backbone, whereas the report says stage 2 "runs for
40 steps, one pass over its prompts"; that step-30 selection is a repo/reference-run statement, not the report's.

# Caveats

- "128 GB200 nodes" is the report's unit for MOPD (vs "128 GPUs" for some teachers).
- The report does not say which checkpoint the "pre-MOPD" rows of Table 27 are; by §3.6 the MOPD input is the step-120 RL checkpoint, but Table 27 does not name it.
- Stage-1 vs stage-2 checkpoint scores are not reported; do not invent per-stage gains.
- The stage-2 rollouts-per-prompt count is not restated; only stage 1 says "one rollout each".
- Unlike Ultra, no MOPD warm-up SFT and no iterated teacher re-branching are described for Super 3.5.
- Lightning has no MOPD.
