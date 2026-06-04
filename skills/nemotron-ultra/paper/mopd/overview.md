---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "mopd-overview"
paper_sections: ["3.2", "3.3", "3.3.1"]
title: "Nemotron 3 Ultra: RLVR Framing and the MOPD Algorithm (Multi-teacher On-Policy Distillation)"
summary: |
  Covers the unified RLVR stage that produces the student model and the centerpiece MOPD algorithm.
  MOPD distills more than ten domain-specialized teachers into one student via asynchronous on-policy
  distillation, providing dense token-level reverse-KL guidance on student-generated trajectories.
  Includes the RLVR hyperparameters, the MOPD objective and asynchronous clipped surrogate, the
  decoupled behavior/proximal policy stabilization, and MOPD training settings.
key_facts:
  - "RLVR is a unified Reinforcement Learning with Verifiable Reward stage spanning all environments (terminal, office/productivity, software engineering, search, general tool-calling, math, code, STEM, safety, chat, instruction following, long-context QA, inductive/transductive reasoning, structured outputs, usability)."
  - "RLVR uses the Gaussian-based data mixture/curriculum approach from NVIDIA (2025b) and largely follows the asynchronous GRPO algorithm with stability optimizations from NVIDIA (2026)."
  - "RLVR global batch size is 8192, with each sample generating 16 rollouts."
  - "RLVR begins with a maximum generation length of 48K tokens, later increased to 64K tokens."
  - "Motivation for MOPD: as the number of RLVR environments grows, each domain contributes only a small number of samples per batch, diluting the per-domain learning signal; MOPD adds targeted specialization."
  - "MOPD trains more than ten specialized teacher models, each optimized through its own domain-specific pipeline."
  - "During MOPD, the student (from RLVR) generates rollouts across all domains and receives dense reward signals from the corresponding teacher models."
  - "MOPD is conducted asynchronously, with student rollout generation, teacher scoring, and student optimization fully pipelined."
  - "MOPD is iterative: after an MOPD checkpoint, new rounds of teacher training are branched from the updated student and merged into the next MOPD stage; Nemotron 3 Ultra used two iterations of MOPD (Figure 17)."
  - "MOPD formulation (Eq. 1): student policy pi_theta and N domain-specialized teachers {pi_T_i}; fully on-policy objective maximizes negative reverse-KL = sum over teachers of lambda_i * E[ sum_t (log pi_T_i(y_t|s_t) - log pi_theta(y_t|s_t)) ], over student-induced states; equivalently student minimizes D_KL(pi_theta(.|s_t) || pi_T_i(.|s_t)) at each prefix."
  - "Unlike sparse environment-dependent RLVR reward, MOPD gives a dense token-level learning signal from the relevant teacher distribution; lambda_i controls sampling/loss weight of domain i."
  - "Async MOPD decouples the behavior policy pi_behav from the proximal policy pi_prox used as trust-region center (Fu et al., 2026)."
  - "Dense distillation advantage (Eq. 2): A_hat_t = sg[ l_T_i_t - l_prox_t ] where l_x_t = log pi_x(y_t|s_t) and sg is stop-gradient."
  - "Importance ratios: c_t = sg[ pi_prox(y_t|s_t)/pi_behav(y_t|s_t) ] corrects stale-rollout mismatch; r_t(theta) = pi_theta(y_t|s_t)/pi_prox(y_t|s_t) is the optimized policy ratio with PPO-style clipping around pi_prox."
  - "Clipped async-MOPD surrogate (Eq. 3): maximize E[ sum_t m_t c_t min( r_t A_hat_t, clip(r_t,1-eps,1+eps) A_hat_t ) ], where m_t is token-level masking using the IcePop strategy (Team et al., 2025)."
  - "MOPD training uses a maximum generation length of 192K tokens, matching the longest generation length across all teacher training runs."
  - "Each MOPD training batch contains 1,024 prompts, with one rollout per prompt; ablations found multiple rollouts gave no additional benefit."
related_steps: []
currency: "frozen"
---

# Scope

- What does the unified RLVR stage cover and with what hyperparameters?
- Why was MOPD introduced on top of RLVR?
- What is the MOPD algorithm: objective, async on-policy distillation, dense token-level guidance, merging teachers?
- What are the MOPD training settings (gen length, batch, rollouts)?

# RLVR stage (3.2)

Claim: RLVR ("Reinforcement Learning with Verifiable Reward") is a single unified stage across all environments, run on top of the SFT model to produce the student.

| Setting | Value |
| --- | --- |
| Algorithm | Asynchronous GRPO with stability optimizations (NVIDIA, 2026) |
| Data mixture / curriculum | Gaussian-based approach (NVIDIA, 2025b) |
| Global batch size | 8192 |
| Rollouts per sample | 16 |
| Initial max generation length | 48K tokens |
| Later max generation length | 64K tokens |

Domains: terminal usage, office/productivity workflows, software engineering, search, general tool-calling, math, code, STEM, safety, chat, instruction following, long-context QA, inductive and transductive reasoning, structured outputs, general usability. Harness-based environments use diverse harness implementations/interaction formats to reduce overfitting to any one harness.

# Why MOPD (3.3)

- As RLVR environments grow, each domain contributes few samples per batch, diluting per-domain signal and making cross-domain balancing hard.
- Solution: train >10 specialized teachers (each via its own domain pipeline), then have the RLVR student generate rollouts across all domains and receive dense reward signals from the corresponding teachers.
- MOPD is asynchronous (rollout gen, teacher scoring, student optimization fully pipelined) and iterative (new teacher rounds branch from the updated student, then merge into the next MOPD stage). Two MOPD iterations were used for Ultra (Figure 17).

# MOPD algorithm (3.3.1)

Setup: student policy pi_theta; N domain-specialized teacher policies {pi_T_i}_{i=1..N}, each tied to a domain dataset D_i. For prompt q ~ D_i and student completion y=(y_1,...,y_H), prefix state s_t=(q, y_<t). MOPD trains the student to match the teacher on states the student itself induces.

Fully on-policy objective (Eq. 1):

J_MOPD(theta) = sum_{i=1..N} lambda_i * E_{q~D_i, y~pi_theta(.|q)} [ sum_{t=1..H} ( log pi_T_i(y_t|s_t) - log pi_theta(y_t|s_t) ) ]

This is the negative reverse-KL; equivalently at each prefix s_t the student minimizes D_KL(pi_theta(.|s_t) || pi_T_i(.|s_t)). lambda_i weights domain i. Key property vs RLVR: dense token-level signal from the teacher distribution instead of sparse environment reward.

## Asynchronous stabilization

A trajectory may be generated by a stale behavior policy pi_behav while the learner optimizes a newer student snapshot. The behavior policy is decoupled from a proximal policy pi_prox (trust-region center). Per sampled token, with l_x_t = log pi_x(y_t|s_t):

| Quantity | Definition |
| --- | --- |
| Dense advantage (Eq. 2) | A_hat_t = sg[ l_T_i_t - l_prox_t ] |
| Behavior->proximal ratio | c_t = sg[ pi_prox(y_t|s_t) / pi_behav(y_t|s_t) ] |
| Proximal->current ratio | r_t(theta) = pi_theta(y_t|s_t) / pi_prox(y_t|s_t) |

sg[.] = stop-gradient. PPO-style clipping is applied to r_t(theta) around pi_prox.

Clipped async-MOPD surrogate (Eq. 3):

J_async-MOPD(theta) = E_{q~D_i, y~pi_behav} [ sum_{t=1..H} m_t c_t min( r_t(theta) A_hat_t, clip(r_t(theta), 1-eps, 1+eps) A_hat_t ) ]

where m_t is token-level masking using the IcePop strategy (Team et al., 2025).

## MOPD training settings

| Setting | Value |
| --- | --- |
| Max generation length | 192K tokens (matches longest teacher-training gen length) |
| Prompts per batch | 1,024 |
| Rollouts per prompt | 1 (multiple rollouts gave no additional benefit in ablations) |

# Caveats

- "More than ten" teachers is the only count given; do not state an exact teacher number.
- MOPD merges teachers via dense token-level guidance during the distillation/co-evolution loop; the exact merge mechanism beyond the objective and iterative branch/merge described is not further specified here.
- Equations are transcribed in plain text from LaTeX; subscripts/superscripts (T_i, y_t, s_t) reflect the source notation.
- RLVR infrastructure improvements are deferred to Section 3.6 (out of this slice); not detailed here.
- This chunk covers the algorithm; specific teacher recipes are in mopd/teachers.md and results/warmup in mopd/warmup-results.md.
