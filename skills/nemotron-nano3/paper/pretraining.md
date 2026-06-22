---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "pretraining"
paper_sections: ["2.3", "2.4", "2.5"]
title: "Pre-training Data Mixture, Schedule, and Long-Context Extension"
summary: |
  Nano3 pretraining is a two-phase 25T-token run with a diversity-heavy first phase and a quality-heavy second phase, followed by a dedicated long-context continuous-pretraining phase. The paper explicitly states the 23.5T / 1.5T phase split, the 94% curriculum switch, sequence length 8192, global batch size 3072, roughly 25M tokens per batch, and a later 121B-token long-context phase with 8-way CP/TP/EP and 4-way PP.
key_facts:
  - "The base model is pretrained on 25T total tokens."
  - "Phase 1 uses 23.5T tokens and Phase 2 uses 1.5T tokens."
  - "The switch to the second phase happens at 94% of training."
  - "The main pretraining run uses sequence length 8192 and batch size 3072, or roughly 25M tokens per batch."
  - "The long-context phase adds 121B tokens and mixes document QA, retrieval-focused synthetic data, and downscaled Phase 2 data."
related_steps:
  - "curate/nemo_curator"
  - "sft/megatron_bridge"
  - "eval/model_eval"
currency: "frozen"
---

# Pre-training Data Mixture, Schedule, and Long-Context Extension

## Executive summary

Nano3 pretraining is presented as a curriculum rather than a flat data stream.
The base run is split into two phases:

- **Phase 1:** 23.5T tokens, diversity-first
- **Phase 2:** 1.5T tokens, higher-quality emphasis

Then a separate **long-context continuous-pretraining phase** is applied to push the model toward up to 1M-token capability.

## Two-phase curriculum

The paper says Nano3 uses a curriculum-based pretraining strategy.

### Phase breakdown

| Phase | Tokens | Goal |
|---|---:|---|
| Phase 1 | 23.5T | maximize diversity and broad coverage |
| Phase 2 | 1.5T | emphasize higher-quality sources |

The paper also explicitly says the switch to Phase 2 happens at the **94% point of training**.

### What changes between the phases?

The paper does not provide every mixture coefficient in text form, but it does state the intended difference clearly:

- Phase 1 promotes diversity
- Phase 2 primarily uses higher-quality datasets, with Wikipedia named as an example

## Pretraining categories and quality logic

The paper says the corpus spans **15 categories**.
The guiding principle is:

- maintain coverage across domains and data types
- assign similar weights to datasets of similar estimated quality
- prioritize higher-quality sources with higher blend weight

This means the mixture is not just “more web + more code.”
It is explicitly quality-ranked and phase-aware.

## Main pretraining run: key numbers

The paper gives the following directly visible main-run settings.

| Field | Value |
|---|---:|
| total tokens | 25T |
| sequence length | 8192 |
| batch size | 3072 |
| approximate tokens per batch | ~25M |

## Optimizer and schedule notes

The paper says the run uses:

- **Warmup-Stable-Decay** learning rate schedule
- **AdamW** optimizer
- MoE aux-loss-free load balancing with expert-bias updates
- standard load-balancing loss in conjunction with the aux-loss-free method

The ar5iv HTML rendering drops some of the inline numeric values for the optimizer hyperparameters and LR magnitudes, but the schedule structure is still explicit:

1. warm up
2. hold max LR for 80% of training
3. decay over the final 20%

### What is visible in the text

| Schedule component | Visible text fact |
|---|---|
| max-LR plateau duration | 80% of training = 20T tokens |
| final decay duration | 20% of training = 5T tokens |
| optimizer family | AdamW |
| load-balancing style | DeepSeek aux-loss-free + standard load-balancing loss |

## MoE balancing during pretraining

The pretraining section says Nano3 uses DeepSeek-style aux-loss-free load balancing for MoE.
That matters because it connects the architecture choice to a training stability choice.

The paper’s point is:

- the router is not only sparse at inference time
- it is also trained with explicit load-balancing machinery designed for stable expert utilization at scale

## Why the curriculum matters

The report’s overall argument is that Nano3’s gains are not from architecture alone.
The data-ordering story matters too.

The implicit design is:

- first build broad coverage
- then sharpen with higher-quality data
- then extend context length in a later focused phase

## Long-context extension: purpose

The paper treats long context as a dedicated later phase, not something the model simply gets “for free” from the main pretraining run.

The purpose is to equip the base model with long-context ability while preserving short-context competence.

## Long-context phase settings

| Field | Value |
|---|---:|
| phase type | continuous pretraining |
| global batch size | 48 |
| context parallel | 8 |
| tensor parallel | 8 |
| expert parallel | 8 |
| pipeline parallel | 4 |
| total LC tokens | 121B |
| hardware note | H100 GPUs |

## Long-context data blend

The LC phase is built from three components:

| Component | Weight |
|---|---:|
| document QA data | 20% |
| retrieval-focused synthetic data | 1% |
| downscaled Phase 2 data | 79% |

Additional facts the paper states:

- the long-context document QA dataset is reused from Nemotron Nano 2
- that dataset is scaled up **3x** for Nano3
- synthetic retrieval-focused data uses a maximum sequence length of **256k**

## Important training observation: 512k-only batches hurt short-context quality

The long-context section contains a useful practical finding.

The authors say they first tried CPT with only **512k** sequences.
That caused a small but noticeable drop on short-context benchmarks.

They then switched to a mixture of:

- 512k sequences
- 4k sequences

The paper says this improved short-context performance again, especially on:

- MMLU-Pro
- code benchmarks

while still improving long-context scores.

## What this implies for interpretation

The paper is not claiming that “more context always helps.”
It is instead claiming that:

- long-context adaptation needs careful scheduling
- extremely long examples alone can damage other capabilities
- mixed-length training is a better compromise

## How this section relates to the public repo

The public repo’s stage0 pretraining scaffold is the operational analogue of this section.
But the paper’s exact pretraining mixture and long-context continuation are still broader than what the public configs surface directly.

So the right answer when users ask about reproduction is:

- **the public stage reflects the structure**
- **the paper contains the full training story**

## Questions this file should answer

### “How many tokens was Nano3 pretrained on?”

- 25T total
- 23.5T in Phase 1
- 1.5T in Phase 2
- plus 121B tokens in long-context CPT

### “What is the phase switch point?”

- 94% of training

### “What batch and sequence length did pretraining use?”

- 8192 sequence length
- 3072 batch size
- roughly 25M tokens per batch

### “How was 1M context achieved?”

- via a dedicated long-context continuous-pretraining phase
- with mixed long and short sequence lengths
- using high parallelism across CP/TP/EP/PP

## Cross-links

- `data.md` for the actual corpora feeding these phases
- `architecture.md` for the model design being trained
- `evaluation.md` for base-model and long-context benchmark outcomes
