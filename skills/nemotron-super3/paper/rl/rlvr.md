---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "rlvr"
paper_sections: ["3.2.1"]
title: "Stage 1: Reinforcement Learning from Verifiable Rewards"
summary: |
  The first RL stage trains Super3 with asynchronous GRPO on 21 environments and
  37 datasets spanning math, code, STEM, instruction following, safety, long
  context, and agentic tool use. The report emphasizes unified multi-environment
  training, low-effort reasoning control, and large-scale Ray/NeMo-Gym systems.
key_facts:
  - "RLVR uses 21 environments and 37 datasets simultaneously."
  - "Training on all environments together is reported to be more stable than single-environment RL."
  - "Low-effort prompts are explicitly included and token-count-aware in reward design."
  - "The released config uses async GRPO with in-flight weight updates and max sequence length 65,536."
related_steps:
  - "stage2_rl/rlvr"
  - "stage2_rl/data_prep"
currency: "frozen"
---

# Scope

Use this file for questions about:

- what RLVR means in Super3
- which environments are included
- how the reward pipeline is structured
- why the run is multi-environment rather than specialized
- which released config implements the stage

---

# What RLVR is doing

RLVR is the broad capability-focused RL stage. It takes the SFT model and improves it using environments where correctness can be checked or judged with relatively strong automated signals.

The report makes one strategic claim very clearly: **training on all environments simultaneously yields stable gains**, whereas single-environment training causes severe regressions on other skills.

That is why RLVR is framed as a unified curriculum rather than a collection of isolated fine-tunes.

---

# Environment families

The paper describes 21 environments and 37 datasets. At the category level, they include:

| Environment family | Examples of what it teaches |
|---|---|
| Math | competition math, formal proof verification, with and without tools |
| Code | competition coding and code generation |
| STEM | science and technical reasoning |
| Instruction following | rubric-scored multi-constraint instruction tasks |
| Safety | over-refusal reduction and jailbreak robustness |
| Long context | retrieval- and long-prompt reasoning |
| Agentic tool use | conversational tools and terminal-style environments |
| Reasoning Gym | a diverse suite of structured reasoning tasks |

The important point is breadth: RLVR is trying to raise the floor across many rewardable tasks at once.

---

# Data selection and curriculum

The report says RLVR does not simply sample uniformly from all prompts.

## Filtering

Prompts that the SFT model already answers consistently well are filtered out.

## Difficulty ordering

The remaining samples are then arranged with a difficulty-aware curriculum.

## Why this matters

This is one reason RLVR can stay broad without wasting too much compute on already-mastered examples.

---

# Low-effort reasoning inside RLVR

Low-effort reasoning is not only an SFT feature. RLVR also includes it explicitly.

| Detail | Reported value |
|---|---|
| Initial low-effort mix | 2% |
| Later low-effort mix | 1% |
| Reward signal | combines correctness with token-count efficiency |

The paper describes this as a way to encourage efficient reasoning rather than merely correct reasoning.

---

# Algorithmic framing

The report uses **asynchronous GRPO**.

## Main mechanics

| Mechanic | Role |
|---|---|
| Separate generation workers | continuously produce trajectories |
| Rollout buffer | accumulates experiences before training |
| Separate training engine | updates the policy independently of generation |
| In-flight weight updates | generation workers can receive new weights without waiting for all rollouts to finish |
| No KV-cache recomputation | reduces the overhead of mid-rollout updates |
| Importance-sampling correction | stabilizes training under policy lag |

This is important because Super3’s RLVR contribution is partly algorithmic and partly systems-oriented.

---

# Released operating point

The open recipe in `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/config/default.yaml` gives a concrete view of the intended production setup.

| Setting | Released value |
|---|---|
| Nodes | 109 |
| GPUs per node | 8 |
| Prompts per step | 256 |
| Generations per prompt | 16 |
| Train global batch size | 4096 |
| Max total sequence length | 65,536 |
| Tensor parallelism | 4 |
| Context parallelism | 8 |
| Expert parallelism | 8 |
| Learning rate | 3e-6 |
| KL penalty | 0 |
| Async GRPO | enabled |
| In-flight weight updates | enabled |

The released config is therefore an implementation-level confirmation of the paper’s large-scale RL story.

---

# Reward-model and judge infrastructure

The paper and released config describe a fairly rich support stack around the main policy.

| Auxiliary component | Role |
|---|---|
| Qwen3-235B-A22B-class judge models | equivalence and instruction-following judgments |
| Nemotron Content Safety model | safety judgments |
| GenRM server | comparison-based behavioral judgments |
| Tool and verifier servers | math, code, terminal, structured outputs, search, etc. |

This is one reason RLVR is better thought of as an environment platform than as a single reinforcement-learning loop.

---

# Safety inside RLVR

RLVR includes safety as first-class training content.

## Over-refusal reduction

One environment tries to reduce unnecessary refusals on benign prompts.

## Jailbreak robustness

Another environment targets jailbreak robustness. The report says seed prompts come from SFT data and are strengthened using an iterative PAIR-style attack pipeline to surface harder adversarial prompts.

This is a good example of how Super3 tries to avoid a false choice between capability and safety: both are included in the same RLVR stage.

---

# Long-context and tool use in RLVR

The report also keeps long-context and tool-use behavior inside RLVR rather than isolating them entirely to later stages.

That means RLVR is already teaching the model to:

- reason over large inputs,
- operate through tools,
- manage terminal-like or conversational tool interactions,
- and preserve these behaviors under reward optimization.

---

# Systems scaling story

A large part of the report’s RLVR section is really about operating at scale.

## Reported challenges

- intermittent hardware failures,
- long startup times,
- port conflicts between many cooperating services,
- dependency and environment initialization overhead.

## Reported mitigations

- parallelized initialization,
- prefetching venvs and binaries,
- caching in upstream projects such as vLLM and flashinfer,
- more careful port management.

These details matter because they explain why the RL stack is not trivial to reproduce even if the configs are available.

---

# How the repo maps the paper

| Function | Open file |
|---|---|
| RL data resolution | `src/nemotron/recipes/super3/stage2_rl/data_prep.py` |
| RLVR training script | `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/train.py` |
| RLVR production config | `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/config/default.yaml` |
| Reduced-scale config | `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/config/small.yaml` |

The data-prep script is especially important because it resolves placeholder records from the released HF blends into concrete JSONL train/validation splits.

---

# Short answer templates

## “What is RLVR in Super3?”

It is the first RL stage: a large multi-environment asynchronous GRPO run over 21 environments and 37 datasets with mostly verifiable reward signals.

## “Why not run one environment at a time?”

Because the paper reports that single-environment training causes regressions elsewhere; the joint curriculum is more stable.

## “Is low-effort reasoning only supervised?”

No. RLVR also includes low-effort prompts and rewards them partly for token efficiency.

---

# Caveats

1. **The paper’s environment count is not the same thing as the number of config files.**
2. **RLVR is broad but not identical to SWE-RL.** Software engineering gets its own dedicated stages later.
3. **A zero KL penalty here does not mean the whole RL pipeline is KL-free.** RLHF later uses a nonzero penalty.

---

# Related files

- `overview.md`
- `swe.md`
- `rlhf.md`
- `../../recipes/stage2_rl_rlvr.md`
