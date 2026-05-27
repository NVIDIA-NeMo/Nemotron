---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "rl-overview"
paper_sections: ["3.2", "3.2.1", "3.2.2", "3.2.3"]
title: "RL Pipeline Overview"
summary: |
  Super3 uses a multi-phase RL pipeline rather than a single alignment run.
  The sequence is RLVR across many verifiable environments, a dedicated SWE-RL
  track for long-horizon code tasks, a principle-following RLHF stage using a
  GenRM, and a final MTP-healing step to restore speculative-decoding quality.
key_facts:
  - "RL is staged as RLVR → SWE-RL → RLHF → MTP healing."
  - "RLVR covers 21 environments and 37 datasets."
  - "SWE-RL is separated because of longer contexts, slower rollouts, and sandbox requirements."
  - "RLHF adds a principle-following GenRM and a nonzero KL penalty."
related_steps:
  - "stage2_rl/rlvr"
  - "stage2_rl/swe1"
  - "stage2_rl/swe2"
  - "stage2_rl/rlhf"
currency: "frozen"
---

# Scope

Use this file for top-level questions about:

- what happens during Super3 RL
- why RL is split into multiple stages
- how RLVR, SWE-RL, and RLHF differ
- where MTP healing fits in
- how the released RL recipe mirrors the paper

---

# The stage order

The report’s RL story is sequential rather than monolithic.

| Order | Stage | Primary goal |
|---|---|---|
| 1 | RLVR | improve capability on many verifiable environments |
| 2 | SWE-RL stage 1 | bootstrap software-engineering RL with a pivot task |
| 3 | SWE-RL stage 2 | full repository-agent RL on SWE-bench-style tasks |
| 4 | RLHF | improve behavioral quality, identity, and safety via GenRM |
| 5 | MTP healing | recover MTP head quality after RL |

This is the main point to preserve in answers. Saying “Super3 does RL after SFT” is true but too coarse.

---

# Why RL is not a single stage

The report gives at least three reasons for splitting RL into multiple phases.

## 1. Different reward structures

- RLVR uses **verifiable rewards**.
- SWE-RL uses **real test execution** in isolated repositories.
- RLHF uses **principle-following preference comparison** via GenRM.

## 2. Different rollout costs

RLVR can mix many relatively shorter environments, while SWE rollouts are much longer and more expensive.

## 3. Different behavioral goals

RLVR mostly sharpens capability, SWE-RL specializes software engineering, and RLHF shapes preference-sensitive behavior such as identity and safety.

---

# RLVR first

The report starts with multi-environment RL from verifiable rewards because it offers the broadest capability gains with relatively clean reward signals.

| RLVR property | Reported value |
|---|---|
| Environments | 21 |
| Datasets | 37 |
| Domains | math, code, STEM, safety, instruction following, long context, puzzles, agentic tasks |
| Main algorithmic framing | asynchronous GRPO |

RLVR is therefore the broad “capability sharpening” stage.

---

# SWE-RL next

Software engineering is split out into its own track because its rollouts are operationally different.

| SWE-specific issue | Why it justifies a separate stage |
|---|---|
| Long rollouts | hundreds of interaction turns are possible |
| Long contexts | sequences are much larger than RLVR |
| Sandbox execution | code and tests must run in isolated environments |
| Lower throughput | mixing with RLVR would bottleneck the whole run |

The report specifically describes two SWE stages:

1. a **pivot** stage, and
2. a **full SWE-bench/OpenHands** stage.

---

# RLHF after capability-focused RL

The final RL stage is not another verifiable environment sweep. It is a preference-alignment stage using a principle-following **Generative Reward Model (GenRM)**.

That ordering is meaningful:

- RLVR and SWE-RL first improve task competence.
- RLHF then refines how the model behaves while preserving capability via a nonzero KL penalty.

This is closer to a “capability first, preference second” interpretation of alignment.

---

# MTP healing is part of the story

The report explicitly adds an **MTP healing** phase after RLHF.

Why?

Because the RL stages optimize the backbone policy for reward but do not naturally preserve the quality of the MTP heads used for speculative decoding. The report therefore freezes the backbone and retrains the MTP heads on RL-generated prompts and responses.

This is one of the most distinctive details in Super3’s RL story and is easy to forget if one only reads the released repo summaries.

---

# Common algorithmic thread: asynchronous GRPO

Although the reward sources differ, the report frames the RL system around **asynchronous GRPO**.

Key shared ideas include:

- separate generation and training workers,
- rollout buffers,
- updates pushed to generation workers with at most small lag,
- in-flight weight updates,
- no KV-cache recomputation after weight updates,
- importance-sampling correction for training/inference mismatch.

This matters because the paper’s RL contribution is not only about data and rewards; it is also about scaling the system to large multi-node runs.

---

# Infrastructure themes across RL

| Theme | Why it appears repeatedly |
|---|---|
| Ray orchestration | coordinates many workers, model servers, and environments |
| NeMo-RL | provides the GRPO training loop |
| NeMo Gym | hosts reward environments and external tools |
| vLLM/OpenAI-style serving | generation and judge-model infrastructure |
| Large-scale resiliency work | needed at hundreds to ~1K GPU scale |

The paper spends a surprising amount of attention on systems failures and startup overhead because those issues materially affected large-scale RL training.

---

# How the repo maps the RL stages

The released repo mirrors the stage structure closely:

| Paper stage | Open recipe summary |
|---|---|
| RLVR | `../../recipes/stage2_rl_rlvr.md` |
| SWE stage 1 | `../../recipes/stage2_rl_swe1.md` |
| SWE stage 2 | `../../recipes/stage2_rl_swe2.md` |
| RLHF | `../../recipes/stage2_rl_rlhf.md` |
| Stage hub | `../../recipes/stage2_rl.md` |

The repo hub is especially important because it makes the stage order operational rather than just conceptual.

---

# Answering guide

## If the user asks “what happens in RL?”

Give the short ordered list first:

1. RLVR across many verifiable environments
2. SWE-RL for software engineering
3. RLHF with GenRM
4. MTP healing

Then ask whether they want capability RL, SWE details, or RLHF behavior shaping.

## If the user asks “why is SWE separate?”

Say: much longer rollouts, much larger contexts, and containerized test execution make SWE a throughput bottleneck if mixed into RLVR.

## If the user asks “where does safety enter RL?”

Say: partly in RLVR through safety environments and partly in RLHF through principle-following GenRM judgments.

---

# Caveats

1. **Do not reduce the RL pipeline to one config.**
2. **Do not claim RLHF is the only safety stage.** Safety also appears in RLVR and SFT.
3. **Do not forget MTP healing** if the question is about serving or speculative decoding after RL.

---

# Related files

- `rlvr.md`
- `swe.md`
- `rlhf.md`
- `../../recipes/stage2_rl.md`
