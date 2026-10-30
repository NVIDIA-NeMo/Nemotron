# Nemotron 3 Nano Knowledge Map

Start here for `/nemotron-nano3`.

## Fast Start

| Need | Read |
|---|---|
| Model identity, released variants, deployment | [`model-card.md`](./model-card.md) |
| Big-picture summary of the paper | [`paper/_overview.md`](./paper/_overview.md) |
| Architecture and long-context design | [`paper/architecture.md`](./paper/architecture.md) |
| Pretraining schedule and optimization | [`paper/pretraining.md`](./paper/pretraining.md) |
| Pretraining and post-training data sources | [`paper/data.md`](./paper/data.md) |
| SFT recipe and post-training dataset design | [`paper/sft.md`](./paper/sft.md) |
| RLVR, RLHF, GRPO, DPO | [`paper/rl.md`](./paper/rl.md) |
| Benchmark numbers and comparisons | [`paper/evaluation.md`](./paper/evaluation.md) |
| Safety preference data and alignment caveats | [`paper/safety.md`](./paper/safety.md) |
| Public recipe map | [`recipes/overview.md`](./recipes/overview.md) |
| Compact “how this maps to nemotron-customize” guide | [`context/quick-reference.md`](./context/quick-reference.md) |

---

## Paper Chunks

| File | What it covers |
|---|---|
| [`paper/_overview.md`](./paper/_overview.md) | Abstract, introduction, release scope, headline claims, and what the report actually promises |
| [`paper/architecture.md`](./paper/architecture.md) | Hybrid Mamba-Transformer-MoE design, active vs total parameters, Table 1, long-context extension, context-window notes |
| [`paper/pretraining.md`](./paper/pretraining.md) | Two-phase pretraining curriculum, optimization schedule, long-context CPT, and base-model evaluation framing |
| [`paper/data.md`](./paper/data.md) | New datasets added over Nemotron 2: CC-Code, Pretraining-Code-v2, CC-v2.1, Specialized-v1, SFT/RL data releases |
| [`paper/sft.md`](./paper/sft.md) | Chat template, SFT data domains, filtering, reasoning control, 18M-sample mixture, SFT hyperparameters |
| [`paper/rl.md`](./paper/rl.md) | RLVR environments, curriculum, synchronous GRPO details, NeMo RL + NeMo Gym infra, GenRM-based RLHF |
| [`paper/evaluation.md`](./paper/evaluation.md) | Base and post-trained evaluation tables, throughput comparisons, quantization accuracy table, prompt sensitivity |
| [`paper/safety.md`](./paper/safety.md) | Safety preference construction, DPO for tool hallucination, over-refusal handling, model-card safety notes |

---

## Recipe Summaries

These bridge the paper to the public repo.

| File | What it covers |
|---|---|
| [`recipes/overview.md`](./recipes/overview.md) | How the public 4-stage Nano3 stack maps to the paper, plus what is and is not reproduced publicly |
| [`recipes/stage0_pretrain.md`](./recipes/stage0_pretrain.md) | `src/nemotron/recipes/nano3/stage0_pretrain/`: data prep, Megatron-Bridge training, open-data pretraining caveat |
| [`recipes/stage1_sft.md`](./recipes/stage1_sft.md) | `src/nemotron/recipes/nano3/stage1_sft/`: chat-template prep, packed Parquet pipeline, Megatron-Bridge SFT |
| [`recipes/stage2_rl.md`](./recipes/stage2_rl.md) | `src/nemotron/recipes/nano3/stage2_rl/`: JSONL prep, NeMo-RL/GRPO, placeholder resolution, cluster assumptions |
| [`recipes/stage3_eval.md`](./recipes/stage3_eval.md) | `src/nemotron/recipes/nano3/stage3_eval/`: NeMo Evaluator config compilation, deployment, task selection |

---

## Model Card

| File | What it covers |
|---|---|
| [`model-card.md`](./model-card.md) | Checkpoint variants (Base BF16 / BF16 / FP8), license, data freshness, intended use, reasoning control, deployment snippets, safety notes |

---

## Common Questions → Best Files

| Question | Best file(s) |
|---|---|
| “What is Nemotron 3 Nano?” | `model-card.md`, `paper/_overview.md` |
| “How many parameters are active?” | `paper/architecture.md` |
| “Why is Nano3 efficient?” | `paper/architecture.md`, `paper/evaluation.md` |
| “What data was added over Nemotron 2 Nano?” | `paper/data.md` |
| “How was SFT done?” | `paper/sft.md` |
| “How was RL done?” | `paper/rl.md` |
| “What are the benchmark numbers?” | `paper/evaluation.md` |
| “What safety data did RLHF use?” | `paper/safety.md` |
| “Can I run the public recipe?” | `recipes/overview.md` + stage file |
| “Which `/nemotron-customize` step maps to this?” | `context/quick-reference.md` |

---

## Public Reproduction Warning

Read this before answering “how do I reproduce it?” questions:

- The **paper** reports NVIDIA’s full training/evaluation story.
- The **public Nano3 recipes** in this repo explicitly say they use **open-source subsets** and are **reference implementations**.
- So:
  - use `paper/*.md` for model facts and benchmark claims
  - use `recipes/*.md` for public execution paths
  - do **not** imply that the public recipes exactly reproduce the published scores unless a source says so

---

## Handoff Rule

If the user asks to **generate code**, **build a pipeline**, or **customize Nano3 to their own data/hardware**:

1. answer the factual part briefly from this skill
2. then hand off to `/nemotron-customize`
3. use [`context/quick-reference.md`](./context/quick-reference.md) to map the request to steps or Explorer-mode
