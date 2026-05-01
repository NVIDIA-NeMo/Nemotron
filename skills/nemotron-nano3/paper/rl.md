---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "rl"
paper_sections: ["3.2", "3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.2.5", "3.3", "3.3.1", "3.3.2"]
title: "RLVR, GRPO, GenRM, and RLHF"
summary: |
  Nano3 uses a unified RLVR stage trained across six environment families at once, then layers RLHF on top with a generative reward model. The paper reports synchronous GRPO with 128 prompts and 16 generations per prompt, an effective batch size of 2048, frozen MoE router weights, curriculum sampling by pass-rate difficulty, and a GenRM-driven RLHF stage with group-relative length control that cuts verbosity by about 30% without hurting accuracy.
key_facts:
  - "RLVR is trained on all environments simultaneously rather than one environment at a time."
  - "The paper’s GRPO configuration uses 128 prompts per step, 16 generations per prompt, and batch size 2048."
  - "The six environment families are math, coding, STEM QA, instruction following, agentic tool use, and structured outputs."
  - "The paper says RLVR can match or surpass a heavily fine-tuned SFT checkpoint."
  - "The RLHF stage uses a GenRM plus group-relative length control and reports roughly 30% lower verbosity without sacrificing accuracy."
related_steps:
  - "rl/nemo_rl_grpo"
  - "eval/model_eval"
  - "convert/megatron_to_hf"
  - "sft/megatron_bridge"
currency: "frozen"
---

# RLVR, GRPO, GenRM, and RLHF

## Overview

The paper presents Nano3 post-training as more than just SFT + a bit of RL.
It argues that **large-scale RL across many environments** is a defining part of Nano3’s capability profile.

There are three distinct pieces to keep straight:

1. **RLVR** — reinforcement learning from verifiable rewards across many domains
2. **GenRM training** — train a large generative reward model using RL
3. **RLHF** — use that GenRM to optimize Nano3 with group-relative length control

## Unified multi-environment RLVR

The paper makes a strong design claim up front:

- train on **all environments simultaneously**
- do **not** isolate each domain into a separate RL run

The reported reason is stability.
The paper says single-environment RL often causes unrecoverable degradation on other benchmarks, while unified multi-environment RLVR gives smooth gains across domains.

## 3.2.1 Environment families

The paper’s six high-level RL environment groups are:

| Environment family | What is rewarded |
|---|---|
| Competition Math | mathematical correctness |
| Competition Coding | unit-test success |
| Question Answering (STEM MCQA/OpenQA style) | answer correctness |
| Structured Outputs | exact schema adherence |
| Instruction Following | instruction constraint satisfaction |
| Agentic Tool Use | tool execution / state correctness |

## Concrete task counts named in the paper

| Domain | Count / detail |
|---|---|
| DAPO math | 17K tasks |
| Skywork math | 104K tasks |
| competition coding | 22K tasks after unit-test filtering |
| STEM QA | 135K tasks |
| structured outputs | 9K tasks |
| instruction following env 1 | 46K tasks |
| instruction following env 2 | 3K tasks |
| long-context QA | 12K tasks |
| workplace assistant | 690 tasks |
| multi-turn conversational agent | about 1K tasks |

## Environment details the paper highlights

### Math

The paper names:

- DAPO
- Skyworks math

These are verifiable math environments.

### Competition coding

The paper uses competitive coding problems and limits verification to **50 unit tests** to control execution time.

### Structured outputs

The structured-output environment is schema-driven.
A positive reward is given when the output matches the exact schema constraints.
No semantic reward is added beyond schema correctness.

### Instruction following

Two environments are used:

1. an IFEval/IFBench-style environment with refreshed constraints
2. a Multi-Challenge-inspired multi-turn judged environment

### Long context

Long-context RL tasks are generated from a subset of the pretraining mixture for multi-document synthesis.
The paper says:

- each question must reference at least **five documents**
- total input is limited to **32k tokens**
- Qwen3-235B-A22B-Instruct-2507 is used as judge

### Agentic tool use

Two agentic environments are named:

1. **Workplace Assistant**
2. a **multi-turn conversational agent** environment for banking-style tool workflows

## 3.2.2 Curriculum and data mixture

The RL curriculum is not random.
The paper says the pipeline first profiles all RL tasks with the SFT checkpoint and drops samples the SFT model already solves with **100% pass rate**.

Then it applies a curriculum method where:

- domain ratios per batch remain fixed
- each domain samples from a Gaussian over pass-rate difficulty
- the mean pass rate shifts from easier samples early to harder samples later
- batches are shuffled across domains

The goal is to preserve domain diversity while gradually increasing difficulty.

## 3.2.3 RLVR surpassing SFT

The paper directly asks whether RLVR can beat a very strong SFT baseline.
It compares RLVR progress against two SFT checkpoints:

| Checkpoint | Description |
|---|---|
| SFT1 | initial RLVR starting point, about 3 epochs |
| SFT2 | heavily fine-tuned checkpoint, about 5 epochs / full convergence |

The paper’s claim:

- even relatively short RLVR training can match or exceed the heavily tuned SFT model across evaluated domains

## 3.2.4 Infrastructure

The paper describes Nano3 RL as built on an integrated stack:

- **NeMo RL** for the RL training loop
- **Megatron-Core** for large-scale training
- **NeMo Gym** for environment execution
- **vLLM** for rollouts

That infrastructure description lines up closely with the public stage2 recipe.

## 3.2.5 Algorithm: synchronous GRPO

The paper says Nano3 uses **synchronous GRPO** with masked importance sampling.

### Key GRPO settings

| Field | Value |
|---|---:|
| prompts per step | 128 |
| generations per prompt | 16 |
| effective batch size | 2048 |
| update style | on-policy |

### Additional stability choices named explicitly

The paper says Nano3 RL also:

- freezes MoE router weights
- keeps aux-loss-free expert-bias updates active
- uses masked importance sampling to mitigate training/inference mismatch

These are not incidental details; they are part of the claimed RL recipe.

## GenRM training

The RLHF pipeline starts by training a **generative reward model**.
The paper argues GenRMs generalize better than Bradley-Terry reward models and reduce reward-hacking risk.

### GenRM training settings

| Field | Value |
|---|---:|
| prompts per batch | 128 |
| generations per prompt | 8 |
| optimization steps per batch | 1 |

### GenRM training data sources

The paper names:

- HelpSteer3
- a commercially friendly subset of arena-human-preference-140k
- a synthetic safety blend

## RLHF with Group Relative Length Control

After GenRM training, the paper performs RLHF on the same prompt set.
The core challenge it identifies is runaway verbosity: the model can generate more reasoning tokens to improve reward even when the task does not need much reasoning.

### Circular comparison strategy

Instead of comparing every response pair, the paper uses a **circular comparison** strategy.
That reduces pairwise comparison cost from quadratic to linear in the number of responses.

### Why length control is needed

The paper says verbosity growth is not exactly reward hacking.
Rather, it is a form of over-thinking where long reasoning traces improve reward even when user-facing tasks do not benefit.

### Group-relative length control mechanism

The mechanism described in the paper:

- decomposes each response into reasoning and answer components
- computes relative within-group length bonuses/penalties
- centers them to preserve reward scale
- adds quality-gated conciseness bonuses only for top-quality short responses

### Reported effect

The paper says the method reduces verbosity by about **30%** during training **without sacrificing accuracy**.

## What this section means operationally

When users ask “what kind of RL did Nano3 use?” the shortest accurate answer is:

- unified multi-environment RLVR
- synchronous GRPO
- 128 prompts × 16 generations
- frozen MoE router + expert-bias updates
- GenRM-based RLHF with group-relative length control

## What this section does *not* contain

It does not itself include the final benchmark table.
For performance questions, use `evaluation.md`.
For DPO-for-hallucinated-tools details, use `safety.md`.

## Common questions this file should answer

### “How many RL environments were used?”

Six environment families, with concrete public examples covering math, coding, STEM QA, instruction following, structured outputs, and agentic tool use.

### “What GRPO settings matter most?”

- 128 prompts per step
- 16 generations per prompt
- batch size 2048
- synchronous on-policy updates

### “Did RL really beat SFT?”

The paper claims yes: RLVR matched or exceeded a heavily tuned SFT checkpoint across evaluated domains.

### “How did Nano3 prevent RLHF from getting too verbose?”

By using group-relative length control plus quality-gated conciseness bonuses during GenRM-based RLHF.

## Cross-links

- `sft.md` for the starting checkpoint and reasoning-control setup
- `evaluation.md` for benchmark impact
- `safety.md` for DPO and safety-preference alignment details
- `recipes/stage2_rl.md` for the closest public runnable analogue
