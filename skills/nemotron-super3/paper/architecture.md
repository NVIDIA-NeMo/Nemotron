---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "architecture"
paper_sections: ["2.1", "2.1.1", "2.1.2", "2.1.3"]
title: "Architecture: LatentMoE, Hybrid Interleaving, and MTP"
summary: |
  Nemotron 3 Super combines sparse LatentMoE capacity, Mamba-heavy sequence
  modeling, attention anchor layers, and shared-weight Multi-Token Prediction.
  The design goal is to improve quality and agentic capability while reducing
  the bandwidth, cache, and communication costs that usually accompany very
  large reasoning models.
key_facts:
  - "120.6B total parameters and 12.7B active parameters per forward pass."
  - "88-layer hybrid stack with Mamba-2 blocks, sparse MoE, and attention anchors."
  - "512 experts per MoE layer with top-k 22 routing and latent size 1024."
  - "Two shared-weight MTP layers support speculative decoding and better training signal."
related_steps:
  - "stage0_pretrain/phase1"
  - "stage0_pretrain/phase2"
  - "stage1_sft/default"
  - "stage2_rl/rlhf"
currency: "frozen"
---

# Scope

This file answers:

- What is the actual Super3 model architecture?
- What does LatentMoE mean in this report?
- Why does the stack mix Mamba and attention?
- What are the MTP layers doing?
- Which details matter for throughput, long context, and deployment?

---

# Architecture snapshot

| Item | Value |
|---|---|
| Total parameters | 120.6B |
| Active parameters | 12.7B |
| Active parameters excluding embeddings | 12.1B |
| Layers | 88 |
| Hidden size | 4096 |
| Attention heads | 32 query heads |
| KV heads | 2 |
| Experts per MoE layer | 512 |
| Routed experts per token | top-k 22 |
| MoE latent dimension | 1024 |
| MTP layers | 2 |
| Context length | 1,048,576 tokens |

The report organizes the architecture around three pillars:

1. **LatentMoE** for sparse scaling with lower bandwidth cost.
2. **Multi-Token Prediction (MTP)** for both training benefit and faster decoding.
3. **Periodic hybrid interleaving** of Mamba-2 blocks with attention anchors.

---

# The overall design goal

Super3 is not a conventional dense transformer scaled up to 120B parameters. The report is explicit that the dominant bottlenecks for large reasoning models are memory movement and sequence-length-dependent inference cost, not just arithmetic throughput. The architecture therefore tries to spend parameters where they help quality while keeping serving costs under control.

That leads to four recurring design choices:

| Design choice | Intended systems effect |
|---|---|
| Sparse experts | More representational capacity without activating all parameters |
| Latent routing/projections | Lower expert communication and memory bandwidth |
| Mamba-heavy stack | Lower decoding memory pressure than attention-heavy stacks |
| MTP | Fewer expensive verifier steps during speculative decoding |

---

# LatentMoE

## The problem with standard MoE

Conventional MoE increases capacity by routing each token through a small subset of experts, but it still pays significant cost in two places:

1. **Expert bandwidth**: routed tokens must read large expert weights.
2. **All-to-all communication**: tokens or activations move across devices to reach their selected experts.

For very large models, those costs start to compete directly with the quality gains from adding experts.

## What LatentMoE changes

Super3 moves routed computation into a **lower-dimensional latent space** before the token reaches the large expert projections. The paper’s key claim is that this substantially reduces the bandwidth and communication burden of expert routing, so the model can afford both:

- **more experts** overall, and
- **more active experts per token**.

That is the central reason Super3 can use **512 experts** and **top-k 22** routing while staying focused on deployment efficiency.

## Why the latent bottleneck helps

| Standard MoE intuition | LatentMoE intuition |
|---|---|
| Route full-width activations into expert weights | Route through a smaller latent representation first |
| Expert bandwidth scales with model width | Expert bandwidth scales with a much smaller latent dimension for routed work |
| Communication cost grows quickly when expert count and active top-k increase | Savings can be reinvested into larger expert count and higher top-k |
| Quality gains can be offset by serving overhead | Better chance of improving accuracy per byte and accuracy per FLOP |

The report frames this as a hardware-aware sparse architecture: not just “more capacity,” but “more capacity in a form that is easier to serve efficiently.”

## Numbers that matter

| LatentMoE setting | Value |
|---|---|
| Total experts | 512 |
| Active experts per token | 22 |
| Latent size | 1024 |
| Hidden size | 4096 |

The large gap between hidden size and latent size is the important clue: the architecture is intentionally shrinking routed computation before it hits the expert path.

---

# Periodic hybrid interleaving

## Why not pure attention?

The report argues that the main systems bottleneck for long-sequence generation is the quadratic growth of the KV cache in self-attention. If the whole 88-layer stack were attention-heavy, both 1M-context operation and high-throughput serving would be much more expensive.

## Why not pure state-space modeling?

Mamba-2 helps with linear-time sequence modeling and constant-sized recurrent state during generation, but the paper still keeps attention because some global full-token interaction is useful for information routing and agentic reasoning.

## The Super3 compromise

The 88-layer stack therefore follows a **periodic hybrid interleaving pattern**:

- many layers use **Mamba-2** blocks,
- sparse MoE capacity is retained throughout the stack,
- a limited number of attention layers act as global **anchors**.

This is the paper’s main architectural compromise between scalability and expressivity.

## Reported implementation details

| Detail | Reported choice |
|---|---|
| Attention variant | Grouped-Query Attention |
| Query heads | 32 |
| KV heads | 2 |
| Head dimension | 128 |
| Positional embeddings | omitted |
| Dropout | omitted |
| Bias terms in linear layers | omitted |
| Normalization | RMSNorm |
| Embedding/output weights | untied |

These details matter because they show Super3 is not just a standard open-weights transformer with MoE added on top; it is a tightly tuned hybrid intended to serve long-context agentic use cases.

---

# Multi-Token Prediction (MTP)

## What MTP is doing here

Super3 includes **two MTP layers** with a **shared-weight design**. The paper presents MTP as serving two goals at once:

1. **extra training signal**, because the model is trained to predict more than one future token at a time, and
2. **speculative decoding support**, because the drafted tokens can be accepted in chunks during inference.

## Why shared weights matter

The report emphasizes a shared-weight MTP design rather than duplicating full extra decoder stacks. That keeps the MTP feature from turning into an unbounded parameter or serving-cost penalty.

## Inference implication

The practical serving claim is that MTP helps Super3 achieve longer accepted draft spans during speculative decoding. The report highlights an overall **SPEED-Bench average acceptance length of 3.45**, which is the main number to remember when users ask why MTP matters for deployment.

## MTP is not the same as LatentMoE

The two ideas address different bottlenecks:

| Component | Main purpose |
|---|---|
| LatentMoE | Sparse scaling with lower communication/bandwidth cost |
| MTP | Better training signal and faster decoding |

When answering questions, keep them separate.

## MTP healing after RL

Another important paper detail is that MTP needs special care after the RL pipeline. The report describes a final **MTP healing** step that freezes the backbone and retrains the MTP heads on RL-generated prompts/responses. That is a clue that MTP quality can drift during late-stage RL and needs explicit repair.

---

# Why the architecture supports 1M context

The context length claim is not only a training artifact. Several architecture decisions make it plausible:

- Mamba-heavy sequence modeling keeps recurrent state manageable during generation.
- The hybrid design uses attention sparingly rather than everywhere.
- GQA reduces KV-head count relative to Q-head count.
- Sparse expert activation keeps active parameters low relative to total capacity.

So when the paper says Super3 supports **up to 1M tokens**, that should be read as a joint result of architecture plus dedicated long-context continuation training, not as a context-window bump from configuration alone.

---

# Throughput framing

The model card and report tie the architecture to deployment claims such as:

- up to **2.2×** higher throughput than GPT-OSS-120B,
- up to **7.5×** higher throughput than Qwen3.5-122B-A10B,

under long-generation workloads.

Those numbers should not be attributed to any single trick. The intended story is cumulative:

1. lower-cost sparse routing through LatentMoE,
2. lower decoding pressure from the hybrid Mamba stack,
3. MTP-enabled speculative decoding,
4. quantized deployment variants on top of the BF16 release.

---

# What to say when users ask “why is Super3 efficient?”

A concise answer is:

> Super3 is efficient because it does not spend compute like a dense transformer of the same total size. It keeps active parameters low, reduces expert traffic with LatentMoE, leans on Mamba to reduce decoding overhead, and adds MTP to accelerate generation.

That answer is more faithful than saying only “it’s an MoE” or only “it uses Mamba.”

---

# Architecture-related caveats

1. **Do not invent the exact per-layer interleaving order** unless you are reading the original figure/table directly.
2. **Do not collapse LatentMoE and MTP into one concept**; they solve different problems.
3. **Do not describe the model as purely transformer-based**; the hybrid Mamba stack is central.
4. **Do not quote total parameters without active parameters**; Super3’s efficiency story depends on that contrast.

---

# Related files

- `pretraining.md`
- `quantization.md`
- `rl/overview.md`
- `../model-card.md`
