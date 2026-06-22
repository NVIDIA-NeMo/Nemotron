---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "overview"
paper_sections: ["abstract", "1"]
title: "Nemotron 3 Ultra Overview: Abstract, Introduction, Contributions, Released Checkpoints"
summary: |
  Nemotron 3 Ultra is a 550B-total / 55B-active-parameter Mixture-of-Experts hybrid Mamba-Attention
  language model, pretrained on 20T text tokens, extended to 1M context, and post-trained with SFT, RL,
  and Multi-teacher On-Policy Distillation (MOPD). It claims up to ~6x higher inference throughput vs
  state-of-the-art public LLMs at on-par accuracy. NVIDIA open-sources the base, post-trained, and NVFP4
  checkpoints plus training data, recipes, and RL environments.
key_facts:
  - "550 billion total parameters, 55 billion active parameters per token (Mixture-of-Experts)."
  - "Hybrid Mamba-Attention MoE architecture; largest and most capable model in the Nemotron 3 family."
  - "Pretrained on 20 trillion text tokens, then context extended to 1M tokens."
  - "Post-training: Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and Multi-teacher On-Policy Distillation (MOPD)."
  - "Key technologies: LatentMoE, Multi Token Prediction (MTP), NVFP4 pre-training, multi-environment RLVR, reasoning budget/effort control."
  - "Claims up to ~6x higher inference throughput vs state-of-the-art publicly available LLMs at on-par accuracy."
  - "Specific throughput claims on 8K input / 64K output setting: 5.9x vs GLM-5.1-754B-A40B, 3.9x vs Kimi-K2.6-1T-A32B, 1.3x vs Qwen-3.5-397B-17B."
  - "Pretraining used a Warmup-Stable-Decay learning rate schedule, divided into two phases: 15T tokens (diversity/broad coverage) then 5T tokens (high-quality refinement)."
  - "Pretrained base model claims significantly higher accuracy than DeepSeek v3.2, Mistral Large 3, Kimi-K2, and GLM-4.5 base models."
  - "Four released checkpoints: NVFP4 (post-trained + quantized), BF16 (post-trained), Base BF16 (base model), GenRM (used for RLHF)."
  - "Released datasets named in intro: Nemotron-Pretraining-Code-v3 (173B tokens, GitHub through Sept 30 2025), Nemotron-Pretraining-Legal-v1, Nemotron-Pretraining-Specialized-v1.2."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What is Nemotron 3 Ultra (size, architecture family, modality)?
- What are the headline throughput and accuracy claims?
- How was it trained at a high level (pretrain tokens, phases, post-training stages)?
- Which checkpoints and datasets are open-sourced?
- How does the report structure the rest of the paper?

# Headline Specs (abstract + §1)

| Item | Value |
|---|---|
| Total parameters | 550B |
| Active parameters / token | 55B |
| Architecture | Mixture-of-Experts hybrid Mamba-Attention |
| Pretraining tokens | 20 trillion text tokens |
| Context length (extended) | 1M tokens |
| Pretrain LR schedule | Warmup-Stable-Decay (WSD) |
| Phase 1 | 15T tokens (diversity / broad domain coverage) |
| Phase 2 | 5T tokens (high-quality data, accuracy refinement) |
| Post-training | SFT + RL (unified RLVR) + MOPD |

# Throughput Claims (§1, Figure 1)
Reported on 8K input / 64K output token setting; throughput at max-throughput, NVFP4 on GB200 (TRT-LLM for Nemotron 3 Ultra, vLLM for others; best of with/without speculative decoding).

| Comparison model | Throughput speedup of Nemotron 3 Ultra |
|---|---|
| GLM-5.1-754B-A40B | 5.9x |
| Kimi-K2.6-1T-A32B | 3.9x |
| Qwen-3.5-397B-17B | 1.3x |

Abstract phrasing: "up to ~6x higher inference throughput" at on-par accuracy across agentic and reasoning benchmarks.

# Released Checkpoints (§1)

| Checkpoint | Description |
|---|---|
| Nemotron 3 Ultra 550B-A55B NVFP4 | post-trained and NVFP4 quantized model |
| Nemotron 3 Ultra 550B-A55B BF16 | post-trained model |
| Nemotron 3 Ultra 550B-A55B Base BF16 | base model |
| Nemotron 3 Ultra 550B-A55B GenRM | GenRM used for RLHF |

Also open-sourced: training recipes, data, and RL environments (HuggingFace + github.com/NVIDIA-NeMo/Nemotron).

# Released Datasets named in intro
- Nemotron-Pretraining-Code-v3: 173B tokens of fresh GitHub code, cut-off Sept 30, 2025.
- Nemotron-Pretraining-Legal-v1: synthetic legal datasets; one ablation boosted a proxy LegalBench average from 64.6 to 74.7 (on Nemotron 3 Nano pretraining).
- Nemotron-Pretraining-Specialized-v1.2: synthetic datasets for factual recall, moral scenarios, and diverse generative/multiple-choice questions.

(Full data details are in the data.md chunk; this chunk only carries the intro-level pointers.)

# Post-training Pipeline (§1, high level)
- Initial SFT on a curated data mixture for foundational capabilities.
- Unified RLVR across reasoning, agentic, code, safety, usability, and chat environments.
- More than ten domain-specialized teacher models trained with targeted recipes (including agentic teachers on a dedicated agentic SFT path).
- MOPD consolidates teachers into Ultra via dense token-level guidance on student-generated rollouts.
- Reasoning effort control supports inference-time accuracy-compute trade-off.

(Post-training is owned by another chunk; included here only as intro framing.)

# Report Structure
Remainder organized into: Pretraining (§2), Post-training (§3), Quantization (§4), Inference (§5).

# Caveats
- Throughput numbers are measurement-condition-specific (8K in / 64K out, GB200, NVFP4, max-throughput, TRT-LLM for Ultra vs vLLM for others). Do not generalize "6x" to other settings.
- "On-par accuracy" is a paper claim; specific benchmark figures live in evaluation chunks.
- The 64.6 -> 74.7 LegalBench gain is from a Nemotron 3 Nano ablation, NOT a measurement on Ultra itself.
- Comparison-model parameter labels (e.g. GLM-5.1-754B-A40B) are as written in the report.
