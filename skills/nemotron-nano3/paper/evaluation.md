---
paper: "arxiv:2512.20848"
model: "nemotron-nano3"
section: "evaluation"
paper_sections: ["2.6", "3.4", "3.4.1", "4.1", "4.2", "4.3", "Appendix E"]
title: "Benchmark Results, Quantization, and Prompt Sensitivity"
summary: |
  The paper reports strong base-model performance against Qwen3-30B-A3B-Base, strong post-trained performance against Qwen3-30B-A3B-Thinking-2507 and GPT-OSS-20B, up to 3.3x throughput gains on a single H200 in an 8K input / 16K output scenario, and selective FP8 post-training quantization with about 99% median accuracy recovery. Prompt sensitivity is reported to be below 1 across the tested datasets.
key_facts:
  - "Base Nano3 beats Qwen3-30B-A3B-Base on many code, math, and long-context benchmarks."
  - "Post-trained Nano3 posts 99.17 on AIME25 with tools, 38.76 on SWE-Bench (OpenHands), 49.04 TauBench V2 average, and 86.34 on RULER-100 @ 1M."
  - "FP8 PTQ uses a 1K-sample calibration subset from post-training reasoning SFT data."
  - "Selective PTQ keeps 6 self-attention layers and the 6 preceding Mamba layers in BF16."
  - "Prompt sensitivity stays below 1 on the datasets reported in Appendix E."
related_steps:
  - "eval/model_eval"
  - "convert/megatron_to_hf"
  - "convert/hf_to_megatron"
  - "rl/nemo_rl_grpo"
currency: "frozen"
---

# Benchmark Results, Quantization, and Prompt Sensitivity

## How to read this file

This chunk combines:

- base-model evaluation
- post-trained evaluation
- quantization evaluation
- prompt sensitivity analysis

Use it when the user asks for **numbers**.

## Throughput framing from the introduction

Before the benchmark tables, the introduction gives the paper’s top deployment-facing claim.

### Throughput setting used in the intro

| Field | Value |
|---|---|
| hardware | single H200 GPU |
| input length | 8K |
| output length | 16K |
| runtimes tried | vLLM and TRT-LLM |
| selection rule | best of the two per model |
| Nano3 / Qwen quantization during throughput test | FP8 weights and activations |
| GPT-OSS quantization during throughput test | MXFP4 weights, BF16 activations |

### Headline intro comparison

| Comparison | Value |
|---|---:|
| Nano3 vs Qwen3-30B-A3B-Thinking-2507 throughput | 3.3x |
| Nano3 vs GPT-OSS-20B throughput | 2.2x |

## 2.6 Base-model evaluation

### Table 2 reconstructed as markdown

| Task | Qwen3-30B-A3B-Base | N-3-Nano 30B-A3B Base |
|---|---:|---:|
| MMLU (5-shot, acc) | 81.07 | 78.56 |
| MMLU-Pro (5-shot, CoT EM) | 61.71 | 65.05 |
| AGIEval-En (3/5-shot, CoT acc) | 63.12 | 68.32 |
| HumanEval (0-shot) | 70.73 | 78.05 |
| MBPP-Sanitized (3-shot) | 73.15 | 75.49 |
| GSM8K (8-shot, acc) | 89.01 | 92.34 |
| MATH (4-shot, acc) | 61.14 | 82.88 |
| MATH-500 (4-shot, avg@32) | 55.08 | 78.63 |
| ARC-Challenge (25-shot, acc_norm) | 94.45 | 91.89 |
| HellaSwag (10-shot, acc_norm) | 83.14 | 85.56 |
| OpenBookQA (0-shot, acc_norm) | 44.80 | 46.20 |
| PIQA (0-shot, acc_norm) | 81.01 | 84.33 |
| WinoGrande (5-shot, acc) | 78.22 | 79.64 |
| RACE (0-shot, acc) | 90.05 | 88.04 |
| MMLU Global Lite (5-shot, avg acc) | 76.84 | 74.47 |
| MGSM (8-shot, avg acc) | 82.53 | 83.00 |
| RULER (64K, 0-shot, acc) | 63.55 | 87.50 |
| RULER (128K, 0-shot, acc) | 60.69 | 82.92 |
| RULER (256K, 0-shot, acc) | – | 75.44 |

### What the base table says qualitatively

Nano3 Base is especially strong relative to Qwen3 Base on:

- code
- math
- long context
- several commonsense tasks

Qwen3 Base remains ahead on:

- MMLU
- ARC-Challenge
- RACE
- MMLU Global Lite

So the paper’s base-model claim is not “wins everywhere.”
It is “wins on many of the technically important axes, especially math/code/long-context.”

## 3.4 Post-trained model evaluations

### Table 3 reconstructed as markdown

| Benchmark | N-3-Nano | Qwen3 | GPT-OSS |
|---|---:|---:|---:|
| MMLU-Pro | 78.30 | 80.90 | 75.00 |
| AIME25 (no tools) | 89.06 | 85.00 | 91.70 |
| AIME25 (with tools) | 99.17 | – | 98.70 |
| GPQA (no tools) | 73.04 | 73.40 | 71.50 |
| GPQA (with tools) | 75.00 | – | 74.20 |
| LiveCodeBench | 68.25 | 66.00 | 61.00 |
| SciCode (subtask) | 33.28 | 33.00 | 34.00 |
| HLE (no tools) | 10.57 | 9.80 | 10.90 |
| HLE (with tools) | 15.48 | – | 17.30 |
| MiniF2F pass@1 | 50.03 | 5.72* | 12.05* |
| MiniF2F pass@32 | 79.92 | 16.80* | 43.03* |
| Terminal Bench (hard subset) | 8.51 | 5.00 | 10.00 |
| SWE-Bench (OpenHands) | 38.76 | 22.00* | 34.00* |
| TauBench V2 Airline | 48.00 | 58.00 | 38.00 |
| TauBench V2 Retail | 56.91 | 58.80 | 54.80 |
| TauBench V2 Telecom | 42.21 | 26.30 | 49.70 |
| TauBench V2 Average | 49.04 | 47.70 | 47.50 |
| BFCL v4 | 53.76 | 46.40* | – |
| IFBench (prompt) | 71.51 | 51.00 | 65.00 |
| Scale AI Multi Challenge | 38.45 | 44.75 | 33.75 |
| Arena-Hard-V2 (Hard Prompt) | 72.10 | 49.60* | 71.20* |
| Arena-Hard-V2 (Creative Writing) | 63.20 | 66.00* | 25.90* |
| Arena-Hard-V2 (Average) | 67.65 | 57.80 | 48.55 |
| AA-LCR | 35.85 | 59.00 | 34.00 |
| RULER-100 @ 256k | 92.92 | 89.40 | – |
| RULER-100 @ 512K | 91.25 | 84.00 | – |
| RULER-100 @ 1M | 86.34 | 77.50 | – |
| MMLU-ProX (avg over langs) | 59.50 | 77.60* | 69.10* |
| WMT24++ (enxx) | 86.20 | 85.60 | 83.20 |

### Paper’s qualitative interpretation of Table 3

The paper explicitly claims Nano3:

- is competitive with GPT-OSS on reasoning
- surpasses Qwen3 on reasoning in many places
- significantly outperforms both models on agentic, chat, and long-context categories

### Important nuance

Table 3 is broad enough that the cleanest honest summary is:

- Nano3 is not the top model on every single benchmark
- but it is very strong across many categories at once, especially the multi-capability mix NVIDIA cares about

## Evaluation tooling and reproducibility notes

The paper says results were collected with:

- Nemo Evaluator SDK
- Nemo Skills Harness for most benchmarks
- dedicated packaged containers for some suites such as TauBench, ArenaHard v2, and AA-LCR
- official open-source implementations for Terminal Bench, SWE-Bench, and Scale AI Multi Challenge when not yet onboarded in NVIDIA’s tooling

This is important because it tells you where to look when a user asks “how were these numbers measured?”

## 4.1 Quantization calibration dataset

The paper says FP8 PTQ calibration used:

- a small **1K-sample** subset from the post-training reasoning SFT dataset

It also says this recovered accuracy slightly better than using CNN/DailyMail and that using on-policy BF16 generations did not help further.

## 4.2 Selective PTQ strategy

The selective FP8 strategy is specific and important.

### Layers kept in BF16

The paper says:

- the **6 self-attention layers** are most sensitive and are kept in BF16
- the **6 Mamba layers feeding into those attention layers** are also kept in BF16
- Conv1D inside all Mamba layers stays in BF16

### What is quantized to FP8

The paper says the following are quantized to FP8:

- model weights
- activations
- KV cache

### Claimed sweet spot

The authors say this mixed-precision choice gives the best trade-off between recovery and efficiency.

## 4.3 Quantization accuracy table

### Table 4 reconstructed as markdown

| Benchmark | N-3-Nano BF16 | N-3-Nano FP8 |
|---|---:|---:|
| MMLU-Pro | 78.30 | 78.10 |
| AIME25 (no tools) | 89.06 | 87.71 |
| AIME25 (with tools) | 99.17 | 98.80 |
| GPQA (no tools) | 73.04 | 72.47 |
| GPQA (with tools) | 75.00 | 73.40 |
| LiveCodeBench | 68.25 | 67.62 |
| SciCode (subtask) | 33.28 | 31.88 |
| HLE (no tools) | 10.57 | 10.33 |
| HLE (with tools) | 15.48 | 14.27 |
| TauBench V2 Airline | 48.00 | 44.79 |
| TauBench V2 Retail | 56.91 | 55.59 |
| TauBench V2 Telecom | 42.21 | 40.75 |
| TauBench V2 Average | 49.04 | 47.04 |
| BFCL v4 | 53.76 | 53.15 |
| IFBench (prompt) | 71.51 | 72.19 |
| AA-LCR | 35.85 | 36.06 |
| MMLU-ProX | 59.50 | 59.63 |

### Paper’s quantization takeaways

The paper says:

- FP8 achieves about **99% median accuracy recovery** relative to BF16
- FP8 KV cache significantly helps throughput by allowing larger batch sizes
- more aggressive quantization hurts accuracy more
- selective quantization preserves most of the BF16 accuracy while improving efficiency

## Appendix E Prompt Sensitivity

The paper evaluates prompt sensitivity as the standard deviation of prompt-averaged accuracy across multiple prompt variants and seeds.

### Table 8 reconstructed as markdown

| Benchmark | N-3-Nano | Qwen3 | GPT-OSS |
|---|---:|---:|---:|
| GPQA (no tools) | 0.42 | 0.59 | 1.91 |
| MMLU-Pro | 0.41 | 0.31 | 1.46 |
| Comp-Math-24-25 (no tools) | 0.77 | 0.51 | 1.14 |
| LiveCodeBench | 0.83 | 1.05 | 1.02 |

### Prompt sensitivity conclusion

The paper says Nano3 stays below **1** on all reported prompt-sensitivity datasets, which it interprets as strong stability under ordinary prompt variation.

## Best compact answer patterns

### “How good is base Nano3?”

Use Table 2 and emphasize:

- strong code
- strong math
- strong long context

### “How good is post-trained Nano3?”

Use Table 3 and emphasize:

- AIME25 with tools: 99.17
- SWE-Bench: 38.76
- TauBench average: 49.04
- IFBench: 71.51
- RULER-100 @ 1M: 86.34

### “How much does FP8 hurt?”

Use Table 4 and say:

- generally modest impact
- about 99% median accuracy recovery overall
- strong throughput upside from selective PTQ and FP8 KV cache

### “Is Nano3 prompt-stable?”

Use Table 8 and say:

- yes, according to the paper’s metric, scores stay below 1 on the datasets reported

## Cross-links

- `architecture.md` for why the model is efficient
- `pretraining.md` for LC-phase training setup
- `rl.md` for how the post-trained model gets its capability mix
- `model-card.md` for public deployment-facing checkpoint guidance
