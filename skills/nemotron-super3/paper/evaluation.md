---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "evaluation"
paper_sections: ["evaluation", "benchmark tables"]
title: "Evaluation and Reported Results"
summary: |
  Super3 is evaluated across general knowledge, reasoning, agentic behavior,
  chat/instruction following, long context, and multilingual tasks. The paper
  and release materials distinguish base-model validation, post-trained BF16
  results, and deployment-oriented FP8/NVFP4 comparisons.
key_facts:
  - "The evaluation stack is built around NeMo Evaluator SDK and, for most tasks, NeMo Skills Harness."
  - "Post-trained BF16 results emphasize agentic reasoning, SWE-bench, long context, and multilingual breadth."
  - "The released repo evaluation stage covers a subset useful for development, not the full paper suite."
  - "Quantized FP8/NVFP4 models remain close to BF16 on the paper's evaluation suite."
related_steps:
  - "stage3_eval/default"
  - "stage2_rl/rlhf"
  - "quantization/fp8"
  - "quantization/nvfp4"
currency: "frozen"
---

# Scope

Use this file for questions about:

- what benchmarks Super3 was evaluated on
- how the post-trained model compares to peers
- what base-model validation looked like
- how FP8 and NVFP4 compare with BF16
- how the repo evaluation stage relates to the full paper suite

---

# Evaluation categories

The report and model card organize results into six broad groups.

| Category | Example benchmarks |
|---|---|
| General knowledge | MMLU-Pro |
| Reasoning | AIME25, HMMT, GPQA, LiveCodeBench, SciCode, HLE |
| Agentic | TerminalBench, SWE-Bench, TauBench, BrowseComp, BIRD |
| Chat & instruction following | IFBench, Multi-Challenge, Arena-Hard-V2 |
| Long context | AA-LCR, RULER |
| Multilingual | MMLU-ProX, WMT24++ |

This grouping is important because Super3 is not framed as a pure coding model or a pure math model; it is evaluated as a broad agentic reasoning model.

---

# Evaluation stack

The report/model card say results were collected through:

- **NeMo Evaluator SDK**, and
- for most tasks, the **NeMo Skills Harness**.

The released repo evaluation stage also reflects this design: it compiles an evaluator config and delegates to `nemo-evaluator-launcher` rather than using a standard recipe training script.

---

# Base-model validation

Before looking at post-trained numbers, the released evaluation docs show a base-model validation table to confirm that the Megatron-Bridge deployment path matches internal research-team measurements.

| Benchmark | MBridge deployment | Research team | Delta |
|---|---|---|---|
| MMLU (5-shot) | 85.86 | 85.89 | -0.03 |
| ARC-Challenge (25-shot) | 95.82 | 95.65 | +0.17 |
| Winogrande (5-shot) | 78.37 | 78.69 | -0.32 |
| HellaSwag (10-shot) | 88.96 | 88.99 | -0.03 |
| OpenBookQA (0-shot) | 48.80 | 50.20 | -1.40 |

This is a useful answer when users ask whether the released evaluation stack is trustworthy.

---

# Post-trained BF16 benchmark snapshot

The model card and release docs report the following representative BF16 scores.

| Benchmark | Nemotron 3 Super | Comparator notes |
|---|---|---|
| MMLU-Pro | 83.73 | below Qwen3.5-122B-A10B, above GPT-OSS-120B |
| AIME25 (no tools) | 90.21 | close to frontier open reasoning models |
| HMMT Feb25 (with tools) | 94.73 | strong tool-assisted math result |
| GPQA (with tools) | 82.70 | tool use materially helps |
| LiveCodeBench v5 | 81.19 | strong coding result |
| SWE-Bench (OpenHands) | 60.47 | strong software-engineering agent result |
| BIRD Bench | 41.80 | text-to-SQL capability |
| RULER @ 1M | 91.75 | strong long-context retention |
| MMLU-ProX (avg over langs) | 79.36 | multilingual breadth |
| WMT24++ (en→xx) | 86.67 | multilingual translation quality |

These are the easiest “headline” numbers to surface in a quick answer.

---

# Full comparison table used in release docs

The release docs compare Super3 with Qwen3.5-122B-A10B and GPT-OSS-120B.

| Benchmark | N-3-Super | Qwen3.5-122B-A10B | GPT-OSS-120B |
|---|---:|---:|---:|
| MMLU-Pro | 83.73 | 86.70 | 81.00 |
| AIME25 (no tools) | 90.21 | 90.36 | 92.50 |
| HMMT (no tools) | 93.67 | 91.67 | 92.33 |
| GPQA (no tools) | 79.23 | 86.60 | 80.10 |
| GPQA (with tools) | 82.70 | — | 80.09 |
| LiveCodeBench v5 | 78.73-81.19 | 78.93 | 88.00 |
| SciCode (subtask) | 42.05 | 42.00 | 39.00 |
| HLE (no tools) | 18.26 | 25.30 | 14.90 |
| HLE (with tools) | 22.82 | — | 19.00 |
| TerminalBench (hard subset) | 25.78 / 22.30 in different release tables | 26.80 | 24.00 |
| SWE-Bench (OpenHands) | 60.47 | 66.40 | 41.90 |
| SWE-Bench Multilingual | 45.78 | — | 30.80 |
| TauBench V2 average | ~61.15–64.64 depending on table | 74.53 | 61.00 |
| BrowseComp with Search | 31.28 | — | 33.89 |
| BIRD Bench | 41.80 | — | 38.25 |
| IFBench (prompt) | ~72.56–75.03 depending on table | 73.77–76.10 | 65.00–68.32 |
| Arena-Hard-V2 | 73.88 | 75.15 | 90.26 |
| AA-LCR | 58.31–59.67 | 66.90 | 51.00 |
| RULER @ 256k | 96.30 | varies by released table | 52.30 |
| RULER @ 512k | 95.67 | varies by released table | 46.70 |
| RULER @ 1M | 91.75 | varies by released table | 22.30 |
| MMLU-ProX | 79.36–80.00 | 82.20–85.06 | 75.90–76.59 |
| WMT24++ | 86.67–87.30 | 78.30–87.84 | 87.80–88.89 |

Some released tables differ slightly because they come from different report/model-card snapshots or harness versions. When precision matters, cite the exact source layer you used.

---

# What the results say qualitatively

## Strongest storylines

| Theme | Evidence |
|---|---|
| Tool-augmented reasoning | HMMT with tools and GPQA with tools improve further over no-tool settings |
| Software engineering | SWE-Bench and multilingual SWE results are strong relative to other open models |
| Long-context competence | RULER remains high even at 1M context |
| Agentic breadth | TerminalBench, TauBench, BrowseComp, and BIRD cover different action styles |
| Quantization robustness | FP8/NVFP4 remain close to BF16 |

## Where Super3 is not simply dominant

The comparison tables also make clear that Super3 is not “best on everything.” Qwen3.5 or GPT-OSS can still lead on some general-reasoning or style-heavy benchmarks. That is worth saying explicitly in any balanced summary.

---

# Long-context evaluation

Long-context evaluation matters more for Super3 than for many models because the architecture and training pipeline explicitly target 1M context.

| Long-context benchmark | Reported Super3 score |
|---|---|
| AA-LCR | 58.31–59.67 |
| RULER @ 256k | 96.30 |
| RULER @ 512k | 95.67 |
| RULER @ 1M | 91.64–91.75 |

The small spread in the RULER 1M number is mostly source-level rounding or harness-version variation, not a conceptual disagreement.

---

# Quantized evaluation

The paper’s quantization section reports that the FP8 and NVFP4 checkpoints remain close to BF16.

Representative table entries include:

| Benchmark | BF16 | FP8 | NVFP4 |
|---|---:|---:|---:|
| MMLU-Pro | 83.73 | 83.63 | 83.33 |
| HMMT Feb25 (with tools) | 94.73 | 94.38 | 95.36 |
| GPQA (no tools) | 79.23 | 79.36 | 79.42 |
| LiveCodeBench v5/v6 | close to BF16 | close to BF16 | close to BF16 |
| TerminalBench (hard subset) | 25.78 | 26.04 | 24.48 |
| RULER 1M | 91.64 | 91.43 | 91.60 |
| MMLU-ProX | 79.35 | 79.21 | 79.37 |

The main deployment takeaway is not that quantization never changes ranking, but that the quality drop is modest enough to make FP8/NVFP4 practical serving targets.

---

# How the repo evaluation stage differs from the paper

The open recipe in `src/nemotron/recipes/super3/stage3_eval/` is intentionally narrower.

## What the released stage is for

- validate that training artifacts are healthy,
- reproduce a development-friendly subset of standard tasks,
- provide artifact-aware evaluation through the Nemotron CLI.

## Default released tasks

| Task name | Benchmark |
|---|---|
| `adlr_mmlu` | MMLU |
| `adlr_arc_challenge_llama_25_shot` | ARC-Challenge |
| `hellaswag` | HellaSwag |
| `openbookqa` | OpenBookQA |
| `adlr_winogrande_5_shot` | Winogrande |

So if a user asks “does the repo run the full paper benchmark suite?”, the answer is **no** — it covers a subset, and the broader reproducibility story lives in NeMo Evaluator configs/docs.

---

# Common answer patterns

## “Are the repo evals the paper evals?”

Not completely. The repo stage is a practical subset for development; the paper and model card report a much broader suite.

## “Is Super3 mainly a long-context model?”

No, but long context is one of its strongest differentiators and is explicitly evaluated up to 1M tokens.

## “Do FP8 and NVFP4 destroy quality?”

No. The paper reports they remain close to BF16 across a broad evaluation suite.

---

# Caveats

1. **Do not mix base-model and post-trained numbers without labeling them.**
2. **Do not ignore source drift across released tables.** Small differences come from different snapshots/harnesses.
3. **Do not overclaim exact reproducibility from the repo stage alone.**

---

# Related files

- `quantization.md`
- `pretraining.md`
- `rl/overview.md`
- `../recipes/stage3_eval.md`
