---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "rl-rlvr"
paper_sections: ["3.2", "3.3.2", "3.6", "3.9", "Figure 4"]
title: "Nemotron 3.5 Super RL Trunk: Joint RLVR + RLHF, New Environments, and Hyperparameters"
summary: |
  Covers §3.2, the general RL stage that turns the SFT model into the step-120 trunk checkpoint used
  as both MOPD student and the root of the specialist teachers. RLVR and RLHF are applied jointly,
  with each environment's samples routed to its verifier or reward model. Records what changed vs
  Nemotron 3 Ultra (refreshed data, reward profiling, router replay, FP32 LM head), the two
  highlighted new environments (Olympiad competitive programming; SEC-filing finance agent), the
  reported hyperparameters (GBS 8,192, 16 rollouts, 80K -> 128K generation), values only implied
  elsewhere (LR 4e-6, trajectory age 1), and the medium-effort RL share from §3.9.
key_facts:
  - "§3.2: RL 'improves the general capabilities of the SFT model and provides a strong foundation for developing specialized teacher models'. RLVR and RLHF are applied jointly, routing samples from each environment to the corresponding verifier or reward model."
  - "§3.2: vs Nemotron 3 Ultra, RL data is refreshed with more recent collections and reward profiling is done before training; same training algorithm as Ultra (NVIDIA, 2026c)."
  - "§3.2: two added train-inference mismatch mitigations: router replay (Ma et al., 2025) and FP32 LM head (Chen et al., 2025); together they reduce the KL error between training and inference engines by more than an order of magnitude."
  - "§3.2: new competitive-programming RL environment (Ficek et al., 2026): 2,847 training and 372 validation Olympiad-level problems, each with a validated executable evaluator; C++ solutions compiled and run; binary reward 1 for full credit, 0 otherwise."
  - "§3.2: finance-agent environment on NeMo Gym targeting Vals.ai Finance Agent; tools = SEC filing search, document parsing, evidence retrieval, final source-supported submission, identical across trajectory generation, SFT, RL and evaluation."
  - "§3.2 finance SDG: document-grounded track (windowed contexts from SEC 10-K, 10-Q, 8-K filings of S&P 500 companies; teacher-generated analyst questions verified via multi-seed voting and GenSelect) and agentic track (coverage audit vs Vals.ai category taxonomy; reporting period, form type and company as explicit sampling variables). Validated questions are re-solved by teacher models; routed to SFT or RL pool by measured difficulty."
  - "§3.2 hyperparameters: global batch size 8,192, 16 rollouts per sample; maximum generation length initially 80K tokens, later increased to 128K."
  - "Implied by §3.3.2, not stated in §3.2: trunk RL LR is 4e-6 ('The learning rate remains 4e-6') and max trajectory age is 1 ('increase the maximum trajectory age from one to two ... compared to the initial RLVR setup')."
  - "§3.6: the trunk checkpoint after 120 steps is the MOPD student and the checkpoint all teachers branched from; teacher sections call it the 'step-120 text-RLVR checkpoint'."
  - "§3.9: approximately 4% of RL training prompts are in medium-effort mode (math, STEM, coding) with length-based reward adjustments; medium-effort uses around 2.2x fewer tokens than regular mode at approximately 9% accuracy drop, averaged over Artificial Analysis Intelligence Index benchmarks."
  - "Not reported: the full environment list or count, the GenRM used for the RLHF part of the trunk, KL/entropy settings, total steps beyond the step-120 checkpoint, and per-environment reward recipes."
related_steps:
  - "stage2_rl/stage1_rlvr"
currency: "frozen"
---

# Scope

- What does the general RL stage of Nemotron 3.5 Super do and what checkpoint does it produce?
- What changed relative to Nemotron 3 Ultra's RLVR?
- Which new environments does the report highlight?
- What batch size, rollout count, generation length, LR are used?
- How is medium-effort reasoning trained in RL?

# Role in the pipeline (§3.2, §3.6, Figure 4)

- Input: the two-stage SFT checkpoint (§3.1).
- RLVR and RLHF run **jointly** in one stage; each environment's samples are routed to the matching
  verifier or reward model. Figure 4 draws a "GenRM (txt)" box feeding this "Text Only Trunk RL student (VLM)".
- Output: the checkpoint "after 120 steps" is the MOPD student and the branch point for the teachers
  (§3.6); teacher sections call it the "step-120 text-RLVR checkpoint".

# What changed vs Nemotron 3 Ultra (§3.2)

| Change | Detail |
| --- | --- |
| Data | Refreshed with more recent collections |
| Reward profiling | Performed before training |
| Algorithm | Same as Ultra (asynchronous GRPO per the Ultra report) |
| Router replay | Replays MoE routing decisions (Ma et al., 2025) |
| FP32 LM head | Chen et al., 2025 |
| Effect | Training-vs-inference-engine KL error reduced by more than an order of magnitude |

# Hyperparameters

| Setting | Value | Source |
| --- | --- | --- |
| Global batch size | 8,192 | §3.2 |
| Rollouts per sample | 16 | §3.2 |
| Max generation length | 80K, later 128K | §3.2 |
| Learning rate | 4e-6 (implied) | §3.3.2 "remains 4e-6" |
| Max trajectory age | 1 (implied) | §3.3.2 "from one to two ... compared to the initial RLVR setup" |
| Checkpoint used downstream | step 120 | §3.3, §3.6 |
| KL penalty, entropy, prompts/rollout split | not reported | — |
| Number of environments | not reported | — |

Lightning also runs RLVR (§3); its RL hyperparameters are not to be inferred from this table.

# Highlighted new environments (§3.2)

## Olympiad competitive programming (Ficek et al., 2026)

| Item | Value |
| --- | --- |
| Train / validation problems | 2,847 / 372 |
| Source | past Olympiad-level competitions |
| Evaluator | validated executable evaluator per problem |
| Language | C++ (compiled, run on problem-specific tests) |
| Reward | binary: 1 full credit, 0 otherwise |

## Finance agent (NeMo Gym `finance_sec_search` resources server)

| Item | Value |
| --- | --- |
| Target benchmark | Vals.ai Finance Agent |
| Tools | SEC filing search, document parsing, evidence retrieval, final source-supported submission |
| Consistency | same tools in trajectory generation, SFT, RL and evaluation |
| Track 1: document-grounded | windowed contexts from SEC 10-K / 10-Q / 8-K of S&P 500 companies; teacher-generated analyst questions; answerability/quality via multi-seed voting + GenSelect (Toshniwal et al., 2025) |
| Track 2: agentic | targets coverage-audit gaps vs Vals.ai category taxonomy; relationship among documents (period, form type, company) is an explicit sampling variable |
| Trajectories | validated questions re-solved by teacher models in the same environment |
| Routing | to SFT pool or RL pool by measured difficulty |
| Sizes, teacher model identity | not reported |

# Reasoning effort in RL (§3.9)

- ~4% of RL training prompts are medium-effort (math, STEM, coding), with length-based reward adjustments.
- Same recipe as Nemotron 3 Ultra and Super; effects generalize beyond math/STEM/coding.
- Result: medium-effort uses around 2.2x fewer tokens than regular at ~9% accuracy drop (averaged over AA Intelligence Index benchmarks).
- The report does not say which RL stage(s) (trunk, teachers, MOPD) carry the 4%.

# GenRM / reward models

- §3.2 only says samples are routed "to the corresponding verifier or reward model".
- Figure 4 shows a "GenRM (txt)" feeding trunk RL.
- §3.5 describes a *new* variant of the Nemotron 3 Ultra GenRM trained with a reasoning-length penalty for
  the alignment teacher; whether the trunk uses this GenRM or another is **not reported**.

# Repo cross-reference

`docs/nemotron/super35/rl/rlvr.md` compares these values with `stage2_rl/stage1_rlvr/config/default.yaml`.
The repo docs list a reference-run blend of NeMo Gym environments (including safety and chat/GenRM); that
list comes from the recipe, not from the report text. The RL data and the report's length-penalized GenRM variant are not yet published.

# Caveats

- LR 4e-6 and trajectory age 1 are inferences from wording in §3.3.2, not explicit §3.2 statements.
- "More than an order of magnitude" KL reduction has no absolute numbers in the report.
- §3.2 is described as joint RLVR + RLHF; do not describe Super 3.5 as having a separate RLHF stage after RLVR (that was Nemotron 3 Super). Alignment-focused RLHF is a teacher (§3.5).
- The 2.2x / 9% medium-effort figures are Super 3.5's; Ultra's report gave ~2.5x / ~7%. Do not mix.
