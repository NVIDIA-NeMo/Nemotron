---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "overview"
paper_sections: ["Abstract", "1", "2"]
title: "Overview and Contributions"
summary: |
  Nemotron 3 Super is NVIDIA's flagship open Nemotron 3 model: a 120.6B-parameter,
  12.7B-active hybrid Mamba-attention LatentMoE model with shared-weight MTP,
  a 25T-token pretraining program, multi-stage SFT and RL, 1M-context extension,
  and deployment-oriented FP8 and NVFP4 releases.
key_facts:
  - "120.6B total parameters and 12.7B active parameters per forward pass."
  - "First Nemotron 3 model with LatentMoE, MTP, and NVFP4 pretraining."
  - "Pretraining runs for 25T tokens, then extends to 1M context."
  - "Post-training is explicitly staged: SFT, RLVR, SWE-RL, RLHF, then MTP healing."
related_steps:
  - "stage0_pretrain/phase1"
  - "stage0_pretrain/phase2"
  - "stage1_sft/default"
  - "stage2_rl/rlvr"
  - "stage2_rl/swe"
  - "stage2_rl/rlhf"
  - "stage3_eval/default"
currency: "frozen"
---

# What this file is for

Use this file when the question is broad:

- What is Nemotron 3 Super?
- What is new relative to earlier Nemotron models?
- What are the main research claims?
- How do the paper, model card, and repo recipes fit together?

For detailed mechanics, jump to the specialized chunks:

- `architecture.md` for LatentMoE, MTP, and the hybrid stack
- `pretraining.md` for the 25T-token program and 1M-context extension
- `sft.md` for the two-stage SFT loss and reasoning controls
- `rl/overview.md` for the full post-training pipeline
- `quantization.md` for FP8/NVFP4 deployment checkpoints

---

# One-paragraph synthesis

Nemotron 3 Super is the largest open model in the Nemotron 3 family and is designed for high-throughput agentic reasoning. The detailed Super3 technical report describes a hybrid architecture that combines Mamba-2 sequence modeling, sparse MoE capacity, a small number of attention anchor layers, and shared-weight Multi-Token Prediction heads. The training story is equally multi-part: the model is pretrained for 25 trillion tokens with a two-phase curriculum and NVFP4 arithmetic, extended to 1M context, then post-trained with large-scale SFT, multi-environment RL from verifiable rewards, a dedicated software-engineering RL track, principle-following RLHF, and a final MTP-healing pass. NVIDIA also releases deployment-focused FP8 and NVFP4 variants and recipe code that mirrors the stage boundaries even when open data does not exactly match the internal paper pipeline.

---

# Identity at a glance

| Item | Reported value |
|---|---|
| Model name | Nemotron 3 Super 120B-A12B |
| Total parameters | 120.6B |
| Active parameters | 12.7B |
| Architecture class | Hybrid Mamba-Transformer LatentMoE |
| Context length | Up to 1M tokens |
| MTP | 2 shared-weight MTP layers |
| Pretraining budget | 25T tokens |
| Main post-training stages | SFT → RLVR → SWE-RL → RLHF → MTP healing |
| Quantized deployment variants | FP8 for Hopper, NVFP4 for Blackwell |

---

# Why Super3 exists

The report frames Super3 around a specific systems problem: very large reasoning models often gain quality by increasing parameter count and context length, but they also become expensive to serve because memory traffic, KV-cache growth, and expert communication dominate inference cost. Super3 responds to that with a package of architectural and training choices meant to improve both quality and serving efficiency.

The three recurring design themes are:

1. **Sparse capacity with lower communication cost** through LatentMoE.
2. **Lower decoding overhead** through Mamba-heavy hybridization and MTP-assisted drafting.
3. **Task-specific post-training specialization** through staged SFT and RL rather than one monolithic alignment run.

---

# Main contributions in the report

| Contribution | What it changes | Why it matters |
|---|---|---|
| **LatentMoE** | Moves routed expert computation into a lower-dimensional latent space | Cuts expert bandwidth/all-to-all cost and lets the model scale to more experts and higher active top-k |
| **Periodic hybrid interleaving** | Uses mostly Mamba-2 blocks plus strategically placed attention anchors | Preserves long-range routing while keeping sequence modeling efficient |
| **Shared-weight MTP** | Adds two MTP layers for training and speculative decoding | Improves inference efficiency and increases accepted draft length |
| **NVFP4 pretraining** | Trains most linear layers in low precision with selective higher-precision exceptions | Lowers training cost while keeping a very large open model trainable |
| **Two-phase pretraining** | Splits 25T tokens into diversity-heavy and quality-heavy phases | Gives broad coverage first, then quality-focused refinement |
| **1M-context extension** | Adds dedicated long-context CPT after the base run | Expands the usable context window to 1,048,576 tokens |
| **Two-stage SFT loss** | Rebalances token-level and conversation-level weighting | Prevents long outputs from dominating loss on long-input/short-output tasks |
| **Multi-stage RL** | Separates RLVR, SWE-RL, and RLHF | Matches different rollout lengths, reward types, and infrastructure constraints |
| **Quantized release family** | Produces FP8 and NVFP4 checkpoints | Makes Super3 practical across Hopper and Blackwell deployments |

---

# What is distinctive versus the rest of Nemotron 3

The family paper and the detailed Super3 report together make Super3 the “capacity + agentic” end of the lineup.

| Dimension | Super3 emphasis |
|---|---|
| Model scale | Largest open Nemotron 3 model |
| Serving goal | High-volume, high-throughput agentic deployment |
| Architecture novelty | LatentMoE + MTP + hybrid Mamba-attention stack |
| Post-training depth | Most elaborate staged RL pipeline in the family |
| Quantization story | Explicit release strategy for FP8 and NVFP4 |
| Long context | Native 1M context extension and long-context evaluation |

In practical terms, Super3 is not just “Nano but larger.” The paper spends much more effort on software-engineering RL, long-horizon tool use, quantization, and throughput-sensitive systems choices.

---

# The training story in one picture

```text
25T pretraining
  ├─ Phase 1: diversity-heavy blend (20T)
  ├─ Phase 2: quality-heavy blend (5T)
  ├─ LC stage 1: 1M-only extension (34B)
  └─ LC stage 2: mixed 1M / 4K continuation (17B)

Post-training
  ├─ SFT on >7M samples
  ├─ RLVR across 21 environments / 37 datasets
  ├─ SWE-RL stage 1 (pivot)
  ├─ SWE-RL stage 2 (full SWE-bench agent loops)
  ├─ RLHF with principle-following GenRM
  └─ MTP healing

Deployment
  ├─ BF16 release
  ├─ FP8 release
  └─ NVFP4 release
```

This staged structure matters because many questions that sound similar are actually about different pieces of the pipeline. For example, “How was Super3 aligned?” could mean the SFT blend, the RLVR reward environments, the SWE-specific RL harness, or the final RLHF behavior shaping.

---

# Paper claims vs open-source reproducibility

The skill should keep these layers separate.

| Source layer | What it is good for | What it is not |
|---|---|---|
| **Technical report** | Architecture rationale, training methodology, benchmark numbers, quantization method | Not a runnable recipe by itself |
| **Model card** | Release metadata, intended use, languages, cutoffs, release-facing benchmark snapshot | Not the deepest methodological source |
| **Nemotron repo recipes** | Concrete scripts, config names, containers, artifact handoffs | Not a guaranteed exact reproduction of internal paper numbers |

The released repo follows the paper’s stage boundaries closely, but the docs explicitly warn that open data only covers a subset of the internal corpus. So the safest phrasing is:

- the **paper** describes the research system,
- the **repo** exposes a faithful implementation surface,
- the **open release** is a methodology reference rather than a promise of exact score parity.

---

# Headline numbers the skill should remember

| Topic | Headline number |
|---|---|
| Total / active params | 120.6B / 12.7B |
| Layers | 88 |
| Experts / active experts | 512 experts, top-k 22 |
| Hidden size | 4096 |
| Context length | 1,048,576 |
| Pretraining budget | 25T tokens |
| SFT scale | >7M samples |
| RLVR scale | 21 environments, 37 datasets |
| Quantization quality | NVFP4 reaches 99.8% median BF16-relative accuracy |

---

# Fast routing for common questions

| If the question is about… | Open this next |
|---|---|
| “Why is LatentMoE useful?” | `architecture.md` |
| “How do phase 1 and phase 2 differ?” | `pretraining.md` |
| “What data was used?” | `data.md` |
| “Why does the SFT stage use a two-stage loss?” | `sft.md` |
| “What exactly happens in RL?” | `rl/overview.md` |
| “How does SWE-RL differ from RLVR?” | `rl/swe.md` |
| “What makes NVFP4 work?” | `quantization.md` |
| “How do I run the released recipe?” | `../recipes/overview.md` |

---

# Key caveats

1. **Super3 is a family member and a specific release.**
   The arXiv HTML paper is family-level; the detailed Super3 PDF is model-specific.

2. **Not all reported data is open.**
   The repo recipes are real and usable, but the full internal 25T corpus is not fully reproduced in public form.

3. **“RL” is not one stage.**
   Super3’s post-training story only makes sense when separated into RLVR, SWE-RL, RLHF, and MTP healing.

4. **Quantization appears twice in the story.**
   The paper discusses both NVFP4 during pretraining and FP8/NVFP4 post-training deployment checkpoints.

---

# Related files

- `architecture.md`
- `pretraining.md`
- `data.md`
- `sft.md`
- `rl/overview.md`
- `quantization.md`
- `../model-card.md`
- `../recipes/overview.md`
