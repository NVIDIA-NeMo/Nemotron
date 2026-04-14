---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "architecture"
paper_sections: ["2.1", "2.5"]
title: "Architecture, Active Parameters, and Long Context"
summary: |
  Nemotron 3 Nano is a 52-layer hybrid Mamba-Transformer model that replaces standard FFNs with sparse MoE layers. NVIDIA reports 31.6B total parameters, 3.2B active parameters per forward pass, 128 routable experts with 6 activated experts and 2 shared experts, no positional embeddings, RMSNorm, squared ReLU in MoE blocks, and a later continuous-pretraining long-context phase that extends capability up to 1M tokens.
key_facts:
  - "The model has 52 layers and model dimension 2688."
  - "It uses 128 routable experts, activates 6 experts per token, and includes 2 shared experts."
  - "The paper reports 31.6B total parameters and 3.2B active parameters per forward pass (3.6B including embeddings)."
  - "The architecture uses no positional embeddings, no dropout, and no bias on linear layers."
  - "Long-context extension is done with a separate continuous-pretraining phase using 8-way context, tensor, and expert parallelism plus 4-way pipeline parallelism."
related_steps:
  - "sft/megatron_bridge"
  - "rl/nemo_rl_grpo"
  - "eval/model_eval"
  - "convert/hf_to_megatron"
currency: "frozen"
---

# Architecture, Active Parameters, and Long Context

## At a glance

Nemotron 3 Nano is not a dense 30B-class Transformer.
It is a **hybrid Mamba-Transformer MoE** model that keeps the Nemotron-H / Nemotron 2 Nano hybrid backbone idea, then replaces standard FFN layers with sparse MoE layers.

That architectural move is the core of the paper’s efficiency story.

## Table 1 reconstructed as markdown

| Field | Nemotron 3 Nano 30B-A3B Base |
|---|---:|
| Num Layers | 52 |
| Model Dimension | 2688 |
| Q-heads | 32 |
| KV-heads | 2 |
| Head Dimension | 128 |
| Mamba State Dimension | 128 |
| Mamba Groups | 8 |
| Mamba Heads | 64 |
| Mamba Head Dimension | 64 |
| Expert Dimension | 1856 |
| Total Routable Experts | 128 |
| Number of Activated Experts | 6 |
| Number of Shared Experts | 2 |

## Parameter accounting

The paper gives three numbers that are easy to confuse.

| Quantity | Meaning |
|---|---|
| 31.6B total params | total model capacity |
| 3.2B active params | active on a forward pass, excluding embeddings |
| 3.6B active incl. embeddings | active footprint when embeddings are included |

This distinction is why the model can sit in the “30B-class” family while behaving more like a much smaller active model at inference time.

## What is hybrid about it?

The paper says Nano3 combines:

- **Mamba-2** components
- **Grouped-Query Attention (GQA)**
- **Mixture-of-Experts** sparsity

The point of the hybrid design is:

- Mamba layers help with sequence efficiency
- attention layers provide global information mixing
- MoE layers increase representational capacity without activating all weights on every token

## What is sparse about it?

Nano3 uses a **granular MoE architecture** with:

- 128 routable experts
- 6 activated experts
- 2 shared experts

This is the paper’s main answer to “how does Nano3 stay fast?”

## Core architectural choices called out explicitly by the paper

The report says Nano3:

- uses **squared ReLU** in MoE layers
- uses a **learned MLP router with sigmoid gating**
- uses **RMSNorm**
- **unties** embedding and projection weights
- uses **no positional embeddings**
- uses **no dropout**
- uses **no bias on linear layers**

Those are not generic defaults; they are explicit model-design choices.

## Why the paper thinks this architecture is better than prior Nano models

The report describes Nano3 as building on Nemotron-H and Nemotron 2 Nano, but changing the FFN blocks into sparse MoE blocks.
The claimed benefit is:

- better accuracy than the prior generation
- less active computation per token
- better throughput/accuracy trade-off relative to similarly sized open models

## Practical interpretation of the head counts

| Field | What it implies |
|---|---|
| 32 Q heads / 2 KV heads | grouped-query attention with reduced KV footprint |
| head dimension 128 | standard medium-width attention head scale |
| Mamba heads 64 | large state-space pathway presence |
| expert dim 1856 | relatively compact per-expert FFN shape compared to dense 30B-class blocks |

## What the architecture section does *not* spell out

It does **not** spell out:

- a tokenizer spec
- layer-by-layer counts of exact Mamba vs attention placement in table form
- every deployment-oriented runtime default

Those are better answered through the public model card or public recipe docs.

## Tokenizer note

**Paper note:** the architecture section itself does not provide a tokenizer identifier.

**Public repo note:** the Nano3 recipe docs and configs frequently point to Nemotron tokenizer handles rooted in the Nano/Nano2 family, for example `nvidia/NVIDIA-Nemotron-Nano-9B-v2` in public data-prep configs and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` in RL/model-card usage paths.

So for tokenizer questions, answer with this distinction:

- the **paper** emphasizes architecture and context behavior
- the **public repo/model cards** provide practical tokenizer handles and serving defaults

## Long-context extension

The paper treats long context as a distinct later pretraining phase.

### What the long-context phase is

It is a **continuous pretraining (CPT)** phase added at the end of base pretraining.
Its purpose is to equip the model with long-context capability.

### Long-context phase settings stated in the paper

| Field | Value |
|---|---|
| learning rate style | constant |
| global batch size | 48 |
| context parallel | 8-way |
| tensor parallel | 8-way |
| expert parallel | 8-way |
| pipeline parallel | 4-way |
| hardware mention | H100 GPUs |
| total LC-phase tokens | 121B |

### Long-context data blend

The LC phase reuses and expands prior Nemotron long-context ingredients.
The paper says it uses:

- the long-context document QA dataset from Nemotron Nano 2, scaled up **3x**
- a small amount of synthetic retrieval-focused data
- a maximum sequence length of **256k** tokens in the synthetic retrieval-focused addition

The phase-LC mixture is described as:

| Component | Weight |
|---|---:|
| document QA | 20% |
| synthetic retrieval-focused data | 1% |
| downscaled Phase 2 data | 79% |

### Important long-context training observation

The authors say they first tried CPT batches with only **512k** sequences, but found short-context benchmark scores degraded slightly.
They then switched to a mixture of **512k** and **4k** sequences.

Claimed effect:

- improved short-context benchmark behavior
- especially better MMLU-Pro and code scores
- still improved long-context scores

## Context-length claim vs runtime defaults

The paper claim is **up to 1M tokens**.
The public runtime examples often default to **256k** due to memory/runtime considerations.

This is the safest way to explain the difference:

- **capability claim:** 1M
- **common public serving default:** 256k

## Architecture questions this file should answer directly

### “How many experts does Nano3 use?”

Answer:

- 128 routable experts
- 6 activated experts
- 2 shared experts

### “How many parameters are active?”

Answer:

- 3.2B active per forward pass
- 3.6B including embeddings

### “Why is Nano3 faster than similarly sized open models?”

Answer:

- sparse activation via MoE
- only 6 experts activated
- hybrid design that keeps long-sequence efficiency in view
- lower active parameter footprint than its total parameter count suggests

### “Does Nano3 really support 1M context?”

Answer:

- the paper says yes
- the capability comes from a dedicated long-context CPT phase
- public deployment examples may use 256k by default, but that is a runtime default, not the paper’s upper-bound claim

## Cross-links

For questions that architecture alone cannot answer, route to:

- `pretraining.md` for optimization schedule and LC-phase placement
- `evaluation.md` for RULER and AA-LCR results
- `model-card.md` for public serving defaults and reasoning/runtime controls
