---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "sft"
paper_sections: ["3.1", "3.1.1", "3.1.2", "3.1.3", "3.1.4", "3.1.5", "3.1.6"]
title: "Supervised Fine-Tuning: Chat Template, Mixture, and Reasoning Control"
summary: |
  Nano3 SFT is designed to teach chat behavior, tool use, reasoning traces, and controllable reasoning. The paper describes a broad multi-domain SFT mixture, a unified filtering pipeline, dynamic sampling over more than 18M samples, explicit reasoning on/off and token-budget control, and a 13k-step packed-sequence run with batch size 64 and sequence length 256k.
key_facts:
  - "The paper says the exact SFT blend trains over 18M total samples."
  - "Reasoning on/off control is implemented by stripping reasoning traces from a random 10% of samples."
  - "Token-budget control is implemented by truncating 3% of reasoning traces to alternate budgets before the final answer."
  - "The SFT run trains for 13,000 steps with batch size 64 and sequence packing to 256k."
  - "The SFT domains include competition math/code, tool use, long context, formal proofs, multilingual, terminal use, safety, software engineering, and science."
related_steps:
  - "prep/sft_packing"
  - "sft/megatron_bridge"
  - "sft/automodel"
  - "convert/hf_to_megatron"
currency: "frozen"
---

# Supervised Fine-Tuning: Chat Template, Mixture, and Reasoning Control

## Role of SFT in the Nano3 pipeline

The paper presents SFT as the stage that turns the pretrained base model into a controllable chat/reasoning/agentic model before RL.
Its goals are broader than ordinary instruction tuning.
The SFT stage is meant to teach:

- chat behavior
- tool-integrated reasoning
- reasoning traces
- controllable thinking behavior
- multilingual instruction following
- long-context behavior
- safety-aware refusal behavior

## 3.1.1 Chat template

The paper says Nano3’s chat template supports:

- reasoning traces
- multi-step reuse of reasoning tokens
- multi-turn conversations
- tool use with XML-style tags

The important conceptual point is that Nano3 is trained as a **reasoning-first chat model**, not a plain text-only instruction model.

### Multi-step and multi-turn behavior

The paper distinguishes:

- **multi-step** settings, where existing reasoning tokens may be preserved and reused
- **multi-turn** settings, where reasoning from prior turns is dropped when a new user turn appears

This matches the public recipe docs, which emphasize that prior-turn reasoning is not necessarily exposed to future turns the same way user-visible content is.

## 3.1.2 SFT data: domain inventory

The paper’s SFT mixture is unusually broad.
It is not just “instruction data + code data.”
It spans many distinct capability clusters.

### Major domains named directly in the paper

| Domain | Purpose |
|---|---|
| Competition Math | teach verifiable mathematical reasoning, often with tool use |
| Competition Coding | teach code reasoning and hard problem solving |
| Conversational Tool Use | teach multi-turn tool-augmented reasoning |
| Long Context | teach reasoning over very long sequences |
| Formal Proofs | teach Lean theorem solving and proof traces |
| Multilingual | transfer post-training behaviors into FR/ES/IT/DE/JA |
| Terminal Use | teach autonomous terminal-oriented action trajectories |
| General Chat | teach broad multi-turn conversational responses |
| Instruction Following | teach strict instruction compliance |
| Safety | teach refusal and safe engagement |
| Software Engineering | teach issue resolution and codebase task behavior |
| Science | teach physics/chemistry/biology reasoning and QA |
| GenSelect | teach candidate-solution comparison and selection reasoning |
| CUDA | teach PyTorch ↔ CUDA-C paired reasoning |

### Concrete details the paper gives for some domains

| Domain | Detail |
|---|---|
| Long Context | mean token length 128k, max 256k |
| Formal Proofs | 580k natural-language theorems → 550k Lean statements → 920k traces → 300k final examples |
| Multilingual | translated to French, Spanish, Italian, German, Japanese |
| Terminal Use | adapted from Terminal Bench plus synthetic data analysis/file tasks |
| SWE | uses SWE-Gym and R2E-Gym seeds plus distilled open-source agent trajectories |
| Science | uses synthetic, real, and document-retrieved seeds processed through NeMo Data Designer |
| CUDA | 21k verified PyTorch/CUDA-C pairs |

## 3.1.3 Unified data filtering

The paper says all SFT domains go through a unified filter pipeline.
The high-level intent is to keep only:

- high-quality
- license-compliant
- structurally valid
- verifiable
- non-pathological

### Filters explicitly named

| Filter | Purpose |
|---|---|
| structural checks | drop malformed examples, such as missing tool definitions |
| repetition filtering | remove pathological repeated n-grams in reasoning traces |
| narrative filtering | remove political/nationalistic teacher artifacts |
| regex/keyword rules | catch “our nation/party” style unwanted generations |

This is important because the paper presents the SFT mixture as **heavily curated**, not simply teacher-generated at scale and accepted blindly.

## 3.1.4 Data mixture

The paper says the exact SFT blend is shown in Figure 5 and that all datasets below 1% are omitted from the figure.

### Key mixture facts visible in text

| Fact | Value |
|---|---:|
| total SFT samples | 18M+ |
| sampling logic | dynamic sampling |

The dynamic-sampling explanation is explicit:

- smaller datasets may be trained for many epochs
- larger datasets may be trained for only a few epochs
- inclusion amount per dataset is tied to how much single-task data is needed to saturate performance

That means SFT weighting is **capability-targeted**, not just proportional to raw dataset size.

## 3.1.5 Reasoning control

This is one of the most user-visible post-training ideas in Nano3.
The paper says Nano3 supports two reasoning-control modes.

### Reasoning on/off control

Implementation described by the paper:

- strip reasoning traces from a random **10%** of SFT samples

Goal:

- teach the model that some completions should omit visible reasoning

### Token-budget control

Implementation described by the paper:

- randomly truncate **3%** of reasoning traces to different budgets
- then continue with the original post-reasoning answer

Goal:

- teach the model to handle constrained internal reasoning budgets

### Why this matters

This is not just data augmentation.
It is the foundation for the public deployment behavior where the model can be asked to reason more or less, or to disable visible thinking entirely.

## 3.1.6 Hyperparameters

The paper gives the main SFT settings directly.

| Field | Value |
|---|---:|
| training steps | 13,000 |
| batch size | 64 |
| sequence packing length | 256k |
| LR warmup steps | 800 |

The text also says the run uses:

- a learning rate (numeric value omitted in the ar5iv rendering)
- a **sequence-level MoE load balancing regularizer**
- a stated loss coefficient (numeric value omitted in the ar5iv rendering)

## Why the public SFT recipe is smaller than the paper SFT

The paper’s SFT run is visibly larger than the public repo defaults.
The public recipe is best interpreted as:

- a faithful implementation pattern
- a public/open-data subset route
- not an exact mirror of the 18M-sample, 256k-packed paper run

That is why this skill should separate:

- **paper SFT facts** from this file
- **public runnable stage** from `recipes/stage1_sft.md`

## Good answers for common SFT questions

### “What did SFT add beyond ordinary instruction tuning?”

Answer:

- tool-integrated reasoning
- reasoning traces
- reasoning on/off control
- token-budget control
- long-context and agentic behaviors
- formal proof and SWE capability

### “How big was the SFT run?”

Answer:

- over 18M total samples
- 13k steps
- batch size 64
- sequence packing to 256k

### “How is reasoning control taught?”

Answer:

- remove reasoning from 10% of samples
- truncate 3% of traces to alternate budgets

### “What data domains matter most?”

Answer with the top-line domains:

- math
- code
- tool use
- long context
- proofs
- multilingual
- safety
- software engineering
- science

## Cross-links

- `data.md` for domain-by-domain source details
- `rl.md` for how RL builds on this SFT checkpoint
- `recipes/stage1_sft.md` for the public recipe surface
