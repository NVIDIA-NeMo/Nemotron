---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "pretraining"
paper_sections: ["2.2.1", "2.2.2", "2.2.3", "2.2.4"]
title: "Pre-training Data and Methodology"
summary: |
  Nemotron 3 Super is pretrained for 25T tokens in two phases: a 20T diversity-
  focused stage and a 5T quality-focused stage, using Warmup-Stable-Decay
  scheduling and selective NVFP4 training. The base run is followed by two long-
  context continuation stages that extend the model to 1M context.
key_facts:
  - "25T-token pretraining split into 20T phase 1 and 5T phase 2."
  - "Sequence length 8192, global batch 3072 sequences, AdamW, peak LR 4.5e-4."
  - "Most linear layers train in NVFP4, with key projections and late layers kept in higher precision."
  - "Long-context continuation adds 34B 1M-only tokens and 17B mixed 1M/4K tokens."
related_steps:
  - "stage0_pretrain/phase1"
  - "stage0_pretrain/phase2"
  - "stage0_pretrain/long_context_1m"
  - "stage0_pretrain/long_context_mixed"
currency: "frozen"
---

# Scope

Use this file for questions about:

- the 25T-token pretraining plan
- phase 1 vs phase 2 differences
- optimizer and schedule details
- NVFP4 pretraining methodology
- checkpoint merging
- the 1M-context continuation stages

---

# The overall program

Super3’s base model is not trained in one homogeneous pass. The report describes a staged pretraining recipe:

| Stage | Token budget | Purpose |
|---|---|---|
| Phase 1 | 20T | Broad, diversity-heavy coverage |
| Phase 2 | 5T | Higher-quality refinement and benchmark shaping |
| Long-context stage 1 | 34B | Pure 1M-context continuation |
| Long-context stage 2 | 17B | Mixed 1M / 4K continuation to recover short-context regressions |

This structure is important because the paper presents phase 2 as a deliberate curriculum change, not just “the last 20% of the same run.”

---

# Core hyperparameters

| Hyperparameter | Reported value |
|---|---|
| Sequence length | 8192 |
| Global batch size | 3072 sequences |
| Tokens per batch | ~25.17M |
| Optimizer | AdamW |
| Betas | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Peak learning rate | 4.5e-4 |
| Minimum learning rate | 4.5e-6 |
| Scheduler | Warmup-Stable-Decay (WSD) |
| Warmup span | 200B tokens |
| Final decay window | 5T tokens |
| MTP loss scaling | 0.3 |

The open recipe mirrors this structure in `stage0_pretrain/config/phase1.yaml` and `phase2.yaml` by keeping the **full schedule length** in both configs and relying on checkpoint resume semantics for phase 2.

---

# Phase 1 vs phase 2

## Phase 1: broad coverage

Phase 1 consumes **20T tokens**, or 80% of the total budget. The paper positions it as a diversity-heavy stage whose goal is to establish broad capabilities before the model enters quality-focused refinement.

Representative phase-1 blend highlights called out in the released docs include:

| Category | Approx. weight |
|---|---|
| syn-crawl-high | 22.4% |
| code | 14.0% |
| syn-crawl-medium | 11.3% |
| stem-sft | 11.1% |
| math | 6.4% |
| finepdfs | 6.1% |
| multilingual | 5.0% |

## Phase 2: quality emphasis

Phase 2 uses the final **5T tokens** with a refined, higher-quality blend and coincides with the minus-sqrt decay portion of the WSD schedule. The qualitative idea is simple: after broad capability formation, the model spends its final large-scale budget on cleaner and more benchmark-relevant sources.

Representative phase-2 blend highlights from the released docs include:

| Category | Approx. weight |
|---|---|
| syn-crawl-high | 22.4% |
| finepdfs-high | 14.3% |
| code | 14.0% |
| stem-sft | 11.8% |
| crawl-high | 6.5% |
| math | 6.4% |
| multilingual | 5.0% |

The main difference is not a total rewrite of the mixture, but a stronger preference for the highest-quality slices and a sharper end-of-training focus.

---

# The pretraining corpus

The report describes a very broad mixture built from web, code, math, STEM, PDF, multilingual, and synthetic sources. It explicitly names several newly released synthetic datasets:

- Synthetic Code Concepts
- Synthetic Algorithmic
- Synthetic Economics
- Synthetic Formal Logic
- Synthetic Multiple Choice

The paper also notes that the crawl data is stratified by quality levels, including synthetic variants of high-quality crawl categories. This is why phase 1 and phase 2 are best understood as **curriculum phases over a shared family of sources**, not as two unrelated corpora.

---

# NVFP4 pretraining

## Why this matters

The report presents Super3 as the first Nemotron 3 model to train primarily in **NVFP4**. That is a training-time efficiency claim, not merely a deployment claim.

## The selective precision recipe

The model is not trained with every operator blindly forced into FP4. Instead, NVIDIA keeps sensitive components in higher precision while pushing most linear layers into NVFP4.

| Component | Reported training precision choice |
|---|---|
| Most linear layers | NVFP4 |
| Final ~15% of the network | BF16 |
| MoE latent projections | BF16 |
| MTP layers | BF16 |
| Attention QKV / projection paths | BF16 |
| Mamba output projection | MXFP8 |
| Embeddings | BF16 |

This selective recipe is one reason the paper can make a strong efficiency claim without presenting NVFP4 as a free lunch.

## Why not use NVFP4 everywhere?

The report explains that low-precision training introduces underflow and sensitivity issues, especially in routed-expert gradients. The workaround is therefore architectural and numerical:

- move most of the training load into low precision,
- preserve sensitive or late-stage components in higher precision,
- and rely on curriculum plus checkpoint averaging to stabilize the run.

---

# Checkpoint merging

A notable methodological detail in the report is **checkpoint merging** during pretraining. Rather than reading out a single checkpoint from a noisy training trajectory, the team evaluates merged checkpoints over windows such as:

- 125B tokens,
- 250B tokens,
- 500B tokens.

## Why merge checkpoints?

The report says these merged checkpoints improved average benchmark quality by roughly **2–4 points** during the stable-LR regime. Operationally, checkpoint merging acts like a cheap stabilizer for late-stage readouts.

## Chosen merge

The final base checkpoint selected for downstream alignment is the **500B merge**.

## Why this matters for users

If a user asks why a released checkpoint differs from a single-step training snapshot, the answer is that Super3’s paper pipeline intentionally uses merged readouts to improve base-model quality before post-training.

---

# Long-context continuation

After the 25T-token base run, the paper adds two dedicated long-context stages.

## Long-context stage 1

| Item | Value |
|---|---|
| Context length | 1,048,576 |
| Duration | 34B tokens |
| Learning rate | 4.5e-6 constant |
| Global batch size | 16 |
| Context parallelism | 64 |
| Tensor parallelism | 2 |
| Expert parallelism | 64 |
| Resume checkpoint | Phase 2 final checkpoint |

The report and recipe docs both frame this as a specialized continuation run whose sole purpose is to endow the model with 1M-token behavior.

## Long-context stage 2

| Item | Value |
|---|---|
| Sequence mix | alternating 1M and 4K |
| Duration | 17B tokens |
| Learning rate | 4.5e-6 constant |
| Resume checkpoint | long-context stage 1 |
| Goal | recover short-context math/regression losses |

The open recipe explicitly warns that alternating sequence lengths may require custom dataloader handling, so this is one of the places where the paper description is cleaner than the currently released implementation surface.

## Long-context data blend

The released docs summarize the LC data mixture as:

- **20%** long-context document QA,
- **80%** downscaled phase-2-style blend.

That is a useful shorthand when explaining how Super3 gets 1M context without abandoning the original base-model distribution entirely.

---

# How the repo maps the paper

The released recipe structure mirrors the pretraining story closely:

| Paper concept | Open recipe file |
|---|---|
| Phase 1 | `src/nemotron/recipes/super3/stage0_pretrain/config/phase1.yaml` |
| Phase 2 | `src/nemotron/recipes/super3/stage0_pretrain/config/phase2.yaml` |
| LC stage 1 | `src/nemotron/recipes/super3/stage0_pretrain/config/long_context_1m.yaml` |
| LC stage 2 | `src/nemotron/recipes/super3/stage0_pretrain/config/long_context_mixed.yaml` |
| Data tokenization | `src/nemotron/recipes/super3/stage0_pretrain/data_prep.py` |
| MB training entrypoint | `src/nemotron/recipes/super3/stage0_pretrain/train.py` |

The tokenization pipeline is also worth remembering: the open code uses a Ray pipeline of **PlanStage → DownloadStage → BinIdxTokenizationStage** and emits a `blend.json` manifest for Megatron-Bridge.

---

# Reproduction caveats

1. **Open data is partial.**
   The docs estimate that the open-source corpus covers only about 8–10T tokens of the internal 25T blend.

2. **The paper’s exact internal data mix is not fully public.**
   Categories such as internal-only code and academic slices are not completely mirrored by the public recipe.

3. **Long-context stage 2 is conceptually clear but implementation-sensitive.**
   The released config notes that alternating 1M/4K support may need custom dataloader work.

4. **NVFP4 pretraining is not the same as PTQ deployment.**
   This file is about training-time precision; `quantization.md` is about post-training inference checkpoints.

---

# Short answer templates

## “What changed between phase 1 and phase 2?”

Phase 1 is the broad-capability 20T stage; phase 2 is the final 5T quality-refinement stage that runs on a stricter data mix while the learning rate decays.

## “Why does the paper mention NVFP4 before quantization?”

Because Super3 uses NVFP4 during **pretraining**, not only after training. The post-training FP8/NVFP4 release checkpoints are a separate deployment step.

## “Where does 1M context come from?”

Not from configuration alone. It comes from two dedicated long-context continuation stages after the base 25T run.

---

# Related files

- `data.md`
- `architecture.md`
- `quantization.md`
- `../recipes/stage0_pretrain.md`
