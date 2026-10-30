---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "quantization"
paper_sections: ["4", "4.1", "4.2", "4.3"]
title: "Quantization for Inference"
summary: |
  Super3 ships deployment-focused FP8 and NVFP4 checkpoints. The paper uses
  PTQ through Model Optimizer, then improves Blackwell-targeted NVFP4 quality
  with better scaling recipes, AutoQuantize mixed-precision search, QAD, and a
  careful Mamba-state quantization recipe based on FP16 with stochastic rounding.
key_facts:
  - "FP8 checkpoint targets Hopper; NVFP4 checkpoint targets Blackwell."
  - "FP8 calibration uses 256 SFT samples at 65,536 context length."
  - "NVFP4 AutoQuantize searches over {NVFP4, FP8, BF16} under a 4.75-bit budget."
  - "The final NVFP4 model reports 99.8% median accuracy relative to BF16."
related_steps:
  - "quantization/fp8"
  - "quantization/nvfp4"
  - "stage3_eval/default"
currency: "frozen"
---

# Scope

Use this file for questions about:

- which quantized Super3 checkpoints exist
- how FP8 and NVFP4 are produced
- what AutoQuantize does
- what QAD means here
- why Mamba-state quantization is a special problem

---

# Two release targets

The paper and release docs describe two main deployment checkpoints.

| Checkpoint | Target hardware | Format | Main goal |
|---|---|---|---|
| FP8 | Hopper | W8A8-style deployment path | good quality/throughput balance |
| NVFP4 | Blackwell | W4A4-style deployment path | stronger efficiency and lower memory footprint |

This is why Super3’s quantization story is hardware-specific rather than one-size-fits-all.

---

# FP8 checkpoint

## Calibration

The paper says the FP8 PTQ calibration uses:

- **256 samples**,
- from the **post-training SFT dataset**,
- at **65,536 context length**.

## What is quantized

| Operator family | FP8 checkpoint choice |
|---|---|
| Embeddings | BF16 |
| Attention QKV projection | BF16 |
| Attention output projection | BF16 |
| KV cache + attention BMM1 | FP8 |
| Attention BMM2 | BF16 |
| Sparse-expert and shared-expert GEMMs | FP8 |
| MoE latent projection | BF16 |
| Router | FP32 |
| Mamba linear / projection GEMMs | FP8 |
| Mamba state cache | FP16 |
| Output layers | BF16 |

The design is selective rather than maximalist: enough FP8 to improve inference, but not a blanket precision downgrade for every operator.

---

# NVFP4 checkpoint

## Why NVFP4 is attractive

The paper says NVFP4 is especially appealing on Blackwell because it offers higher GEMM FLOPS than FP8 and roughly halves weight footprint again, which is very attractive for prefill-heavy or MoE-heavy serving workloads.

## Why naïve FP4 PTQ is not enough

A plain FP4 PTQ recipe left more than a 1% median-accuracy gap relative to BF16. The rest of the section explains how NVIDIA closes that gap.

---

# Improved NVFP4 PTQ recipe

The report highlights a hybrid scaling strategy.

| Component | Scaling method | Why |
|---|---|---|
| Weight per-block scales | weight-MSE minimization | best offline calibration behavior |
| Activation per-block scales | max-based scaling | cheap runtime computation |

This already improves over naïve PTQ, but the report still considers it insufficient by itself.

---

# AutoQuantize

## What it does

AutoQuantize treats per-operator precision assignment as a search problem. Instead of forcing every operator into NVFP4, it decides which operators should remain in:

- NVFP4,
- FP8, or
- BF16.

## Search space and budget

| Item | Value |
|---|---|
| Candidate precisions | `{NVFP4, FP8, BF16}` |
| Effective precision budget | 4.75 bits |
| Search hardware | 1 B200 node / 8 GPUs |
| Reported search time | < 2 hours |

## Resulting intuition

Sparse-expert GEMMs remain aggressively low precision, while more sensitive attention, Mamba, or shared-expert components may stay at FP8 or BF16 where needed.

---

# Reported NVFP4 precision assignments

The paper’s Table 7 summarizes the searched mixed-precision checkpoint.

| Operator family | NVFP4 checkpoint assignment |
|---|---|
| Embedding | BF16 |
| Attention QKV projection | BF16 |
| Attention output projection | FP8 / BF16 |
| KV cache + attention BMM1 | FP8 |
| Attention BMM2 | BF16 |
| Sparse expert GEMMs | NVFP4 |
| Shared expert GEMMs | NVFP4 / FP8 / BF16 |
| MoE latent projection | FP8 / BF16 |
| Router | FP32 |
| Mamba projection GEMMs | FP8 / BF16 |
| Mamba 1D conv | BF16 |
| Mamba SSM cache | FP16 |
| Output layers | BF16 |

This table is the clearest justification for describing the NVFP4 release as a **mixed-precision** system rather than a uniform FP4 model.

---

# Reported quality

The paper says the final NVFP4 process uses **512 samples from the Nemotron 3 Super SFT dataset** for evaluation calibration and achieves **99.8% median accuracy relative to the BF16 baseline**.

Representative evaluation rows from the paper include:

| Benchmark | BF16 | FP8 | NVFP4 |
|---|---:|---:|---:|
| MMLU-Pro | 83.73 | 83.63 | 83.33 |
| HMMT Feb25 (with tools) | 94.73 | 94.38 | 95.36 |
| GPQA (no tools) | 79.23 | 79.36 | 79.42 |
| TerminalBench (hard subset) | 25.78 | 26.04 | 24.48 |
| RULER 1M | 91.64 | 91.43 | 91.60 |
| MMLU-ProX | 79.35 | 79.21 | 79.37 |

The correct interpretation is “close enough for deployment,” not “bitwise identical to BF16.”

---

# QAD: Quantization-Aware Distillation

The release docs describe **QAD** as a teacher-student refinement step.

| Role | Model |
|---|---|
| Teacher | BF16 checkpoint |
| Student | NVFP4 checkpoint |

Reported recipe details from the docs include:

| Setting | Value |
|---|---|
| Calibration / student context | 2K samples at long context from post-training reasoning SFT |
| Learning rate | 1e-5 |
| Data blend | SFT + RL on-policy rollouts (60:40) |
| Training budget | 5B tokens |

So the final NVFP4 quality story is not PTQ alone. It is PTQ plus search, with QAD as an optional further quality recovery mechanism.

---

# Mamba-state quantization

## Why this is special

Mamba cache quantization is harder than ordinary activation quantization because recurrent-state error compounds across decoding steps.

The paper explains that an error introduced at one step is propagated through later recurrent updates, so even small quantization noise can become visible as verbosity drift or quality loss.

## Tested recipes

The report compares multiple SSM-cache choices, including:

- FP32 baseline,
- direct FP16,
- INT16 block recipes,
- FP16 with stochastic rounding under several Philox settings.

## Chosen recipe

The selected deployment recipe is:

- **FP16 cache storage**,
- **stochastic rounding**,
- **Philox<5>** configuration.

The paper’s motivation is that this combination avoids the large verbosity increases seen with naïve FP16 cache casting.

---

# How the repo and docs expose quantization

The Nemotron docs summarize four named export configurations for the Mamba-MoE architecture:

| Config name | Purpose |
|---|---|
| `mamba_moe_fp8_aggressive` | more aggressive FP8 |
| `mamba_moe_fp8_conservative` | safer FP8 |
| `mamba_moe_nvfp4_aggressive` | more aggressive NVFP4 |
| `mamba_moe_nvfp4_conservative` | safer NVFP4 |

The docs then point users to Megatron-Bridge scripts for:

1. quantization,
2. Megatron-format generation/testing,
3. export back to Hugging Face format.

---

# Common answer patterns

## “Which quantized Super3 checkpoint should I use?”

Use FP8 on Hopper and NVFP4 on Blackwell.

## “Is NVFP4 just FP4 everywhere?”

No. The final release is mixed precision, with operator-level assignments searched over `{NVFP4, FP8, BF16}`.

## “Why talk about Mamba state cache separately?”

Because recurrent cache quantization error accumulates over time, so the Mamba cache needs a dedicated recipe.

---

# Caveats

1. **Do not confuse pretraining in NVFP4 with post-training NVFP4 export.**
2. **Do not describe the final NVFP4 release as uniformly FP4.**
3. **Do not omit the hardware target.** The release strategy is Hopper-vs-Blackwell specific.

---

# Related files

- `architecture.md`
- `evaluation.md`
- `pretraining.md`
- `../model-card.md`
