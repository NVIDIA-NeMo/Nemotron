# Nemotron 3 Super Model Card Notes

This file condenses the official **Hugging Face model card** and the Super3 paper into a release-facing reference.

---

## Identity

| Field | Value |
|---|---|
| Model | NVIDIA-Nemotron-3-Super-120B-A12B |
| Family | Nemotron 3 |
| Release date | March 11, 2026 |
| Developer | NVIDIA |
| Total parameters | 120.6B |
| Active parameters | 12.7B |
| Architecture | LatentMoE hybrid Mamba-2 + MoE + attention with MTP |
| Context length | Up to 1M tokens |
| Supported languages | English, French, German, Italian, Japanese, Spanish, Chinese |
| Reasoning control | on/off plus low-effort support via template/runtime controls |
| Minimum GPU requirement | 8× H100-80GB |
| Best for | agentic workflows, long-context reasoning, tool use, RAG, high-volume workloads |

---

## Released variants

The Super3 release family is structured around one base checkpoint, one aligned BF16 checkpoint, and deployment-oriented quantized variants.

| Variant | Purpose | Notes |
|---|---|---|
| **Base BF16** | pretrained model before SFT/RL | use for research, adaptation, or base-model evaluation |
| **BF16** | aligned/post-trained model | main instruction-following and agentic release |
| **FP8** | deployment on Hopper | balanced throughput and quality |
| **NVFP4** | deployment on Blackwell | strongest efficiency target |

Associated open releases mentioned across the report/model card:

- Nemotron 3 Super 120B-A12B Base BF16
- Nemotron 3 Super 120B-A12B BF16
- Nemotron 3 Super 120B-A12B FP8
- Nemotron 3 Super 120B-A12B NVFP4
- Qwen3-Nemotron-235B-A22B-GenRM-2603 (used as the GenRM in RLHF)

---

## Headline positioning

Super3 is positioned as the **agentic, large-capacity** member of the Nemotron 3 family:

- larger and more agentic than Nano
- optimized for collaborative agents and high-volume production workloads
- built to preserve long-context and tool-use capabilities while improving serving efficiency

The official release materials emphasize:

- **LatentMoE** for better quality at similar inference cost
- **MTP** for native speculative decoding
- **NVFP4 pretraining** as a first for the family
- **multi-stage RL** for agentic behavior

---

## Reported throughput framing

The technical report highlights these throughput comparisons under long-generation workloads:

| Comparison | Reported relative throughput |
|---|---|
| vs GPT-OSS-120B | up to **2.2×** higher |
| vs Qwen3.5-122B-A10B | up to **7.5×** higher |

These numbers are tied to the model’s architecture and serving stack, not just to quantization alone.

---

## Training windows and cutoffs

| Item | Value |
|---|---|
| Pretraining period | December 2025 – March 2026 |
| Pretraining data cutoff | June 2025 |
| Post-training data cutoff | February 2026 |

This matters when answering “what world knowledge should I expect?” questions.

---

## Intended use

The release-facing description frames Super3 as a general-purpose reasoning and chat model for:

- AI agent systems
- chatbots and assistants
- tool-using workflows
- long-context reasoning
- RAG systems
- high-volume enterprise workflows such as IT-ticket automation

It is described as ready for commercial use under the **NVIDIA Nemotron Open Model License**.

---

## Reasoning modes

The released model card and paper together imply three user-visible operating styles:

| Mode | What it means |
|---|---|
| Reasoning-off | reasoning trace removed or disabled |
| Regular reasoning | standard thought-rich behavior |
| Low-effort reasoning | shorter, efficiency-oriented reasoning traces |

The model card also notes that reasoning can be controlled via the chat template, commonly using a flag like `enable_thinking=True/False`.

---

## Benchmark snapshot

These are representative **post-trained BF16** results from the official model card.

| Benchmark | Nemotron 3 Super |
|---|---|
| MMLU-Pro | 83.73 |
| AIME25 (no tools) | 90.21 |
| HMMT Feb25 (with tools) | 94.73 |
| GPQA (with tools) | 82.70 |
| LiveCodeBench v5 | 81.19 |
| SWE-Bench (OpenHands) | 60.47 |
| SWE-Bench Multilingual | 45.78 |
| BIRD Bench | 41.80 |
| RULER @ 1M | 91.75 |
| MMLU-ProX (avg) | 79.36 |
| WMT24++ (en→xx) | 86.67 |

Use `paper/evaluation.md` when the user wants the larger comparison table or wants to separate base, BF16, FP8, and NVFP4 results.

---

## Data disclosures

The model card states:

- major portions of the pretraining corpus are released in the **Nemotron pre-training datasets** collection
- major portions of the post-training corpus are released in the **Nemotron post-training v3** collection
- RL environments and datasets are released via **NeMo Gym**

It also discloses language distribution for post-training as heavily English-dominant, with smaller targeted multilingual slices and paired translation data for the supported languages.

---

## Deployment notes

| Topic | Note |
|---|---|
| BF16 serving | higher resource use, strongest raw release format |
| FP8 | intended Hopper deployment target |
| NVFP4 | intended Blackwell deployment target |
| Single-node minimum | 8 GPUs is the stated floor in release docs |
| Best-effort decoding defaults | the model card recommends `temperature=1.0`, `top_p=0.95` |

For quantization-specific behavior, use `paper/quantization.md`.

---

## Safety and ethics

The official release guidance says:

- trustworthy AI is a shared responsibility
- downstream developers should validate fit for their domain and risk posture
- deployed systems should not remove safety guardrails without equivalent replacements
- safety, bias, privacy, and explainability subcards are part of the broader release package

This should be paired with `paper/safety.md`, which explains where safety entered the training recipe itself:

- safety SFT data
- over-refusal and jailbreak RL environments
- principle-following GenRM-based RLHF

---

## Best file to use next

| If the user asks… | Then read… |
|---|---|
| What does the shipped model look like? | `paper/architecture.md` |
| How was it trained? | `paper/pretraining.md`, `paper/sft.md`, `paper/rl/overview.md` |
| How close are FP8/NVFP4 to BF16? | `paper/quantization.md` |
| Can I reproduce the release from the repo? | `recipes/overview.md` |

