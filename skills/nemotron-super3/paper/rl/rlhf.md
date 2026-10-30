---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "rlhf"
paper_sections: ["3.2.3"]
title: "Stage 3: RLHF and MTP Healing"
summary: |
  The final alignment stage uses a principle-following Generative Reward Model
  initialized from Qwen3-235B-A22B-Thinking-2507. Unlike RLVR and SWE-RL, this
  phase applies a nonzero KL penalty to preserve capability while improving
  preference-sensitive behavior; afterward, the MTP heads are healed on RL data.
key_facts:
  - "RLHF uses a principle-following GenRM initialized from Qwen3-235B-A22B-Thinking-2507."
  - "The released RLHF config applies a KL penalty of 1e-4."
  - "RLHF data includes HelpSteer 3, arena-style preference subsets, and newer human preference data."
  - "After RLHF, the backbone is frozen and the MTP heads are retrained in an MTP-healing pass."
related_steps:
  - "stage2_rl/rlhf"
currency: "frozen"
---

# Scope

Use this file for questions about:

- what the final RLHF stage is optimizing
- what the GenRM does
- how RLHF differs from RLVR and SWE-RL
- where the KL penalty appears
- why MTP healing happens after RLHF

---

# What this stage is for

After SFT, RLVR, and SWE-RL, Super3 already has strong task competence. The final RL stage therefore shifts focus toward **behavioral preference shaping** rather than raw capability acquisition.

The report highlights three themes for this stage:

1. **principle-following comparison judgments**,
2. **identity and safety behavior refinement**,
3. **capability preservation through KL regularization**.

---

# The GenRM

## What it is

The reward model used here is a **Generative Reward Model (GenRM)**.

## Initialization

The paper and recipe docs say it is initialized from **Qwen3-235B-A22B-Thinking-2507**.

## Training data sources

The report describes the GenRM as trained from sources including:

- HelpSteer 3,
- commercially friendly subsets of arena-style preference data,
- newer human preference annotations,
- principle-following prompts and comparison templates.

## Why use a generative reward model?

A GenRM can generate a judge-style answer and compare policy outputs in a more structured way than a tiny scalar preference head. In the Super3 story, it is especially important for domains like:

- identity-sensitive responses,
- safety-sensitive responses,
- nuanced assistant helpfulness and style judgments.

---

# How RLHF differs from earlier RL stages

| Stage | Main reward style | KL penalty | Primary objective |
|---|---|---|---|
| RLVR | verifiable environment rewards | 0 | broad capability gains |
| SWE-RL | repository test execution rewards | 0 | software-engineering specialization |
| RLHF | GenRM comparison + tool-use preference signals | 1e-4 | preference, identity, and safety refinement |

This is the cleanest way to explain why the final stage exists instead of simply making RLVR bigger.

---

# Nonzero KL penalty

One of the easiest-to-miss but most important details is that RLHF applies a **reference-policy KL penalty of 1e-4**.

## Why it matters

The paper and released config imply that by this point the model already has valuable capabilities learned from SFT, RLVR, and SWE-RL. RLHF should refine behavior without allowing the policy to drift too far from that competent reference.

## Contrast with earlier stages

RLVR and SWE-RL both run with **KL = 0** in the released defaults, so the RLHF KL term marks a real shift in optimization philosophy.

---

# Released RLHF operating point

From `stage2_rl/stage3_rlhf/config/default.yaml`:

| Setting | Value |
|---|---|
| Nodes | 72 |
| Prompts per step | 128 |
| Generations per prompt | 16 |
| Train global batch size | 2048 |
| Max sequence length | 49,152 |
| Tensor parallelism | 4 |
| Context parallelism | 4 |
| Expert parallelism | 8 |
| Learning rate | 1e-6 |
| KL penalty | 1e-4 |
| Async GRPO | enabled |

This places RLHF between RLVR and SWE-RL in sequence length and total scale, but with a distinct reward model and regularization regime.

---

# Environments used in the released RLHF config

The released config exposes two key environment families:

| Environment | Role |
|---|---|
| `genrm_compare` | pairwise response comparison using the GenRM |
| `single_step_tool_use_with_argument_comparison` | preserves tool-use correctness |

The practical takeaway is that the final stage is not purely style-only RL. It still keeps one foot in actionable tool behavior while using the GenRM to refine broader preferences.

---

# Principle-following behavior

The released config includes a long default judging principle that asks the GenRM to:

- generate its own answer first,
- compare assistant answers against that answer,
- judge helpfulness, relevance, and concision,
- notice ambiguity and value clarifying questions when appropriate,
- identify missing important information.

That principle-following framing is the reason this stage is repeatedly described as affecting identity and safety behavior rather than merely numerical preference optimization.

---

# Where safety enters RLHF

RLHF is one of the strongest safety-shaping stages in Super3.

## Why?

Because comparison-based judgments can explicitly encode norms about:

- harmful content handling,
- appropriate refusal vs over-refusal,
- identity and persona behavior,
- clarity under ambiguous instructions,
- helpfulness without unsafe compliance.

This is also why the report pairs RLHF closely with the principle-following GenRM rather than a narrow benchmark-specific reward.

---

# MTP healing after RLHF

The report does not end the alignment story at RLHF. It adds an **MTP healing** stage.

## Why is healing needed?

The RL stages optimize the main policy for reward, but speculative-decoding heads are not guaranteed to stay well calibrated under that objective.

## What happens in healing?

- the **backbone is frozen**,
- the **MTP heads** are retrained,
- training uses RL prompts and responses,
- the paper reports that this substantially improves MTP accuracy.

That is a very Super3-specific detail and is essential when answering deployment or speculative-decoding questions.

---

# How the repo maps the paper

| Paper concept | Open file |
|---|---|
| RLHF summary | `../../recipes/stage2_rl_rlhf.md` |
| RLHF README | `src/nemotron/recipes/super3/stage2_rl/stage3_rlhf/README.md` |
| RLHF config | `src/nemotron/recipes/super3/stage2_rl/stage3_rlhf/config/default.yaml` |

The released repo clearly exposes the RLHF config and GenRM plumbing, but it does not spell out a separate MTP-healing recipe in the same detail as the paper. When asked about healing, prefer the paper-level explanation.

---

# Common answer patterns

## “What is the GenRM doing?”

It is the principle-following comparison judge used in the final RLHF stage to refine helpfulness, identity, safety, and preference-sensitive behavior.

## “Why add KL here but not earlier?”

Because at this stage the goal is to improve behavior without sacrificing the capabilities built up during SFT, RLVR, and SWE-RL.

## “What is MTP healing?”

It is a final pass that freezes the backbone and retrains the MTP heads so speculative decoding remains accurate after RL.

---

# Caveats

1. **Do not say RLHF is the only alignment stage.** SFT, RLVR, and safety environments also align behavior.
2. **Do not forget the tool-use comparison environment.** RLHF is not purely free-form chat preference optimization.
3. **Do not skip MTP healing when answering inference-performance questions.**

---

# Related files

- `overview.md`
- `rlvr.md`
- `swe.md`
- `../../recipes/stage2_rl_rlhf.md`
