---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "overview"
paper_sections: ["Abstract", "1", "5"]
title: "Overview, Scope, and Headline Claims"
summary: |
  Nemotron 3 Nano is NVIDIA’s 30B-A3B open hybrid Mamba-Transformer MoE model for reasoning, chat, and agentic use. The paper positions it as a more efficient successor to Nemotron 2 Nano: 31.6B total parameters, 3.2B active per forward pass, 25T-token pretraining, post-training with SFT + RLVR + RLHF, support for up to 1M context, and public release of base, post-trained, and FP8 checkpoints plus most of the new data and recipes.
key_facts:
  - "Nemotron 3 Nano 30B-A3B has 31.6B total parameters and 3.2B active parameters per forward pass (3.6B including embeddings)."
  - "The model was pretrained on 25T text tokens, including more than 3T new unique tokens over Nemotron 2."
  - "NVIDIA reports up to 3.3x higher inference throughput than Qwen3-30B-A3B-Thinking-2507 and about 2.2x higher throughput than GPT-OSS-20B in the 8K input / 16K output setting."
  - "The report says the model supports context lengths up to 1M tokens."
  - "The release bundle includes Base BF16, post-trained BF16, FP8 model weights, a GenRM, recipe code, and most of the newly added data collections."
related_steps:
  - "sft/megatron_bridge"
  - "rl/nemo_rl_grpo"
  - "eval/model_eval"
  - "convert/hf_to_megatron"
currency: "frozen"
---

# Overview, Scope, and Headline Claims

## One-paragraph takeaway

Nemotron 3 Nano is NVIDIA’s open 30B-A3B “Nano” model for reasoning-heavy and agentic use. The paper argues that sparse MoE plus hybrid Mamba/Transformer layers let Nano3 improve the throughput/accuracy frontier relative to similarly sized open models. The full training story is three-part: 25T-token pretraining, a heavily expanded SFT stage, and large-scale RL using both RLVR and RLHF. The release includes base, instruct, and FP8 checkpoints; most of the newly added data; and public recipe code.

## The paper’s core claim

The report is trying to establish four things at once:

1. **Efficiency** — Nano3 activates far fewer parameters than its total parameter count suggests.
2. **Capability** — it is competitive on reasoning, tool use, agentic tasks, long-context tasks, and chat/instruction following.
3. **Scale-up in post-training** — Nano3 is NVIDIA’s first Nano line effort that aggressively scales RL during post-training.
4. **Openness** — the release includes weights, recipes, and most of the new data collections.

## Model identity at a glance

| Item | Paper claim |
|---|---|
| Model name | Nemotron 3 Nano 30B-A3B |
| Architecture family | Mixture-of-Experts hybrid Mamba-Transformer |
| Total params | 31.6B |
| Active params | 3.2B, or 3.6B including embeddings |
| Training phases | pretraining → SFT → RLVR/RLHF |
| Max context claim | 1M tokens |
| Public checkpoints | Base BF16, post-trained BF16, FP8 |

## Why NVIDIA says this model matters

The introduction frames Nano3 as a successor to Nemotron 2 Nano with two main improvements:

- **sparse scaling via MoE**, rather than dense FFNs
- **scaled-up post-training**, especially reinforcement learning

The efficiency argument is not just parameter-count marketing. The paper specifically ties the architecture to measured throughput on a single H200 GPU in a generation-heavy setting:

- 8K input
- 16K output
- best configuration chosen between vLLM and TRT-LLM per model

## Headline performance claims from the introduction

| Claim | Value |
|---|---:|
| Throughput vs Qwen3-30B-A3B-Thinking-2507 | up to 3.3x |
| Throughput vs GPT-OSS-20B | about 2.2x |
| Context support | up to 1M |
| Pretraining tokens | 25T |
| New unique tokens over Nemotron 2 | 3T+ |

Important framing note:

- the intro mixes **architectural claims**, **training claims**, and **evaluation claims**
- detailed benchmark tables live later in the paper
- public recipes in this repo do **not** claim exact paper-score reproduction

## What was released with the paper

The introduction explicitly says the paper is released alongside recipe code and the following artifacts.

### Checkpoints

| Artifact | What it is |
|---|---|
| Nemotron 3 Nano 30B-A3B FP8 | final post-trained FP8 checkpoint |
| Nemotron 3 Nano 30B-A3B BF16 | final post-trained BF16 checkpoint |
| Nemotron 3 Nano 30B-A3B Base BF16 | pretrained base checkpoint |
| Qwen-3-Nemotron-235B-A22B-GenRM | reward model used for RLHF |

### Data

| Artifact | What it is |
|---|---|
| Nemotron-CC-v2.1 | new Common Crawl-derived English corpus |
| Nemotron-CC-Code-v1 | Common Crawl code corpus |
| Nemotron-Pretraining-Code-v2 | refreshed GitHub/synthetic code corpus |
| Nemotron-Pretraining-Specialized-v1 | synthetic specialized pretraining data |
| Nemotron-SFT-Data | post-training SFT datasets |
| Nemotron-RL-Data | post-training RL datasets |

## How the paper is organized

The report itself says the remainder is split into three major parts:

1. **Pre-training**
2. **Post-Training**
3. **Quantization**

In practice, the major user-facing question clusters are:

| User question | Best paper chunk |
|---|---|
| “What is the model?” | `architecture.md` |
| “What data changed since Nemotron 2?” | `data.md` |
| “How was pretraining run?” | `pretraining.md` |
| “How was SFT done?” | `sft.md` |
| “How was RL done?” | `rl.md` |
| “What benchmarks and comparisons matter?” | `evaluation.md` |
| “What did safety alignment do?” | `safety.md` |

## What is new relative to Nemotron 2 Nano

The paper repeatedly positions Nano3 as an extension of Nemotron 2 Nano, but with meaningful changes:

- sparse MoE replaces standard FFN layers
- more than 3T new unique tokens are added during pretraining
- SFT strategy is broadened and diversified
- RL is scaled much more aggressively
- long-context ability is extended up to 1M
- FP8 PTQ is treated as a first-class deployment release

## What the paper does *not* claim

It does **not** say that every released public recipe directly reproduces every benchmark in the report.
That distinction matters when answering “how do I reproduce this?” questions.

The safe interpretation is:

- **paper** = authoritative model/training/eval story
- **public repo** = reference implementation and open-data subset path

## Practical answer patterns

### If the user asks “What is Nemotron 3 Nano?”

Answer with:

- 31.6B total / 3.2B active / 3.6B incl. embeddings
- hybrid Mamba-Transformer + sparse MoE
- 25T-token pretraining
- SFT + RLVR + RLHF post-training
- up to 1M context
- released base, BF16, and FP8 checkpoints

### If the user asks “Why is it efficient?”

Answer with:

- sparse MoE activation
- only 6 of 128 experts activated
- hybrid Mamba/Transformer design
- measured throughput gains vs Qwen3 and GPT-OSS in the intro

### If the user asks “What’s public?”

Answer with:

- three checkpoint variants
- reward model
- most of the new data collections
- recipe code
- but not every internal condition needed for exact paper-score reproduction

## Paper-to-public boundary

This overview file is also the right place to answer the most common boundary question:

> “Is the public repo the paper?”

The safest answer is:

- the **paper** is the canonical statement of architecture, training, and benchmark claims
- the **public repo** is the canonical statement of what users can run today
- the two align in stage structure, but not necessarily in every data source, mixture ratio, or score

That means the skill should use this file to set expectations before diving into a stage-specific answer.

## Source anchors

Use this file for high-level identity and release-scope questions.
For specifics, route immediately to:

- `architecture.md`
- `pretraining.md`
- `data.md`
- `sft.md`
- `rl.md`
- `evaluation.md`
- `safety.md`
