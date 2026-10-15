# Stage 3: Evaluation

This page has two parts: the tech report's results for the released checkpoints, and the `nemotron super35 eval` stage, which runs a subset of those benchmarks with [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) to check your own training runs.

The report collects every post-trained result, for Super 3.5 and for the baselines, with [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), and points to a Super 3.5 reproducibility guide in the NeMo Gym repository (`nemotron_recipes/super-3.5/reproducibility.md`). `super35 eval` uses NeMo Evaluator instead, so its scores are not expected to match the tables below exactly.

:::{admonition} Pending open-source release
:class: warning

The NeMo Gym Super 3.5 benchmark recipes and reproducibility guide are pending publication. The report lists BabyVision, CharXiv-RQ, MathVision, MMLongBench-Doc, OCR-Reasoning, OCRBench-v2, V\*, and Video-MME-v2 as not yet onboarded in its open-source tools (it used their official implementations or internal scaffolding). NV Chat Benchmark is internal to NVIDIA.
:::

---

## Evaluation Setup (report §3.8)

All models were evaluated under the same settings: agentic resources (CPU, timeouts, tools, search APIs), input data, prompt templates, number of repeats, and extraction and metric implementations. Sampling parameters (temperature, top-p, maximum tokens) follow each model's card.

| Category | Benchmarks and protocol |
|----------|-------------------------|
| **Agentic** | SWE-bench Verified / Multilingual / Pro in the OpenCode harness (sandboxed, web search and fetch disabled, average of 3 runs); Terminal-Bench 2.1 with Terminus 2 and a 3-hour task timeout (8 runs); τ²-Bench Retail / Airline / Telecom with a Qwen3-235B-A22B-Instruct-2507 user simulator and τ³-Bench Banking (8 trials); GDPVal-AA v2 (open-source implementation loosely following the Artificial Analysis v2.1 methodology); BrowseComp on an open-source NeMo Gym harness backed by Exa search with a discard-all context policy, plus an OpenCode-based harness |
| **Code** | LiveCodeBench v6 (454 problems, Aug 2024–May 2025, no tools, 3 samples); SciCode with scientist background (8 repeats; subtask and problem accuracy) |
| **Math and science** | GPQA-Diamond (8 repeats); HLE text-only, 2,158 questions, without tools and with tools (Python and web search, separate internal harness); Apex-Shortlist (16 repeats, symbolic equivalence, no tools); CritPt (official Artificial Analysis grading, 5 samples per problem); AA-Omniscience (600 questions, 10 repeats) |
| **Multimodal** | BabyVision, CharXiv-RQ, MathVision, OCR-Reasoning, OCRBench-v2 (EN / ZH), V\* (static multiple choice, no visual-search tool), Video-MME-v2 (no subtitles), MMLongBench-Doc (up to 100 pages), HLE full set (2,500 questions, 1 sample), OSWorld-Verified (361 tasks, screenshots only, up to 100 steps); 3 repeats unless noted |
| **Long context** | AA-LCR v1.1 (16 repeats); GraphWalks (the 950 of 1,150 prompts that fit in 1M tokens, 1 sample, F1) |
| **Instruction following and chat** | IFBench (8 samples, strict); NV Chat Benchmark (3,713 prompts, pairwise win rate, judged in both orders with length-based style control) |
| **Multilingual** | MMLU-ProX (10 languages); WMT24++ English→9 languages with XCOMET-XXL |

## Results

### Post-trained model (Table 10)

Nemotron 3.5 Super (BF16) against the six open models the report compares with. A dash means the report has no result.

| Benchmark | N3.5-Super 120B-A12B | N3-Ultra 550B-A55B | Qwen3.5 122B-A10B | Qwen3.8 Flash-Next | DS-V4 Flash-0731 | Inkling Small | GLM-5.3 Flash |
|-----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| **Agentic** | | | | | | | |
| SWE-bench Verified | 71.10 | 73.93 | 71.87 | 72.60 | 77.80 | 75.27 | 79.10 |
| SWE-bench Multilingual | 67.72 | 68.78 | 62.00 | 79.00 | 72.00 | 71.44 | 78.30 |
| SWE-bench Pro | 51.78 | 49.66 | 45.60 | 60.37 | 56.91 | 56.54 | 59.70 |
| τ²-Bench Retail | 69.24 | 73.68 | 55.37 | 71.60 | 73.79 | 71.82 | 78.18 |
| τ²-Bench Airline | 76.50 | 79.75 | 63.50 | 80.75 | 83.50 | 62.75 | 81.75 |
| τ²-Bench Telecom | 73.19 | 83.00 | 87.17 | 90.35 | 98.25 | 90.12 | 98.46 |
| τ³-Bench Banking | 18.88 | 18.35 | 14.23 | 44.54 | 38.56 | — | 34.85 |
| Terminal-Bench 2.1 | 60.67 | 54.78 | 48.70 | 85.80 | 76.80 | 53.40 | 82.70 |
| GDPVal-AA-v2 ELO | 1,211.20 | 1,148.30 | 860.73 | 1,561.69 | 1,382.30 | 1,313.27 | 1,563.39 |
| GDPVal-AA-v2 norm ELO | 0.356 | 0.324 | 0.180 | 0.531 | 0.441 | 0.407 | 0.532 |
| BrowseComp | 78.91 | 67.85 | 58.10 | 89.02 | 84.20 | 74.10 | 86.57 |
| BrowseComp (OpenCode) | 72.35 | 61.06 | 45.42 | 76.07 | 64.77 | 55.37 | 81.91 |
| **Code** | | | | | | | |
| SciCode (subtask) | 47.13 | 41.96 | 39.83 | 46.62 | 49.30 | 48.28 | 50.05 |
| SciCode (problem) | 15.10 | 13.46 | 10.77 | 14.42 | 18.27 | 18.65 | 17.53 |
| LiveCodeBench v6 | 88.22 | 89.21 | 76.65 | 94.13 | 94.64 | 88.25 | 80.62 |
| **Math and science** | | | | | | | |
| GPQA-Diamond | 89.52 | 86.99 | 86.74 | 92.68 | 89.65 | 88.57 | 91.22 |
| CritPt | 10.86 | 3.14 | 1.14 | — | 12.57 | 9.14 | — |
| HLE (no tools, text) | 31.79 | 29.24 | 24.24 | 39.57 | 35.82 | 33.27 | 36.47 |
| HLE (with tools, text) | 46.01 | 37.40 | 47.50 | — | 45.10 | 47.80 | 55.30 |
| Apex-Shortlist | 88.36 | 75.00 | 54.65 | 91.76 | 88.70 | 71.54 | 77.13 |
| AA-Omniscience accuracy | 26.20 | 25.30 | 26.25 | 27.87 | 44.95 | 35.38 | 34.22 |
| AA-Omniscience non-hallucination | 69.96 | 77.91 | 15.5 | 58.69 | 4.09 | 40.39 | 69.75 |
| **Multimodal** | | | | | | | |
| BabyVision | 29.60 | — | 35.65 | 46.82 | — | 24.74 | 39.09 |
| CharXiv-RQ | 80.82 | — | 83.47 | 85.77 | — | 77.77 | 81.93 |
| MathVision | 78.16 | — | 83.29 | 87.54 | — | 78.77 | 82.52 |
| OCR-Reasoning | 65.08 | — | 64.27 | 67.10 | — | 54.69 | 64.61 |
| OCRBench-v2 (EN) | 68.12 | — | 68.27 | 67.91 | — | 54.21 | 61.81 |
| OCRBench-v2 (ZH) | 59.15 | — | 67.26 | 68.28 | — | 52.38 | 56.81 |
| Video-MME-v2 | 32.85 | — | 31.20 | 45.05 | — | — | 14.79 |
| V\* | 85.60 | — | 91.10 | 93.02 | — | 70.51 | 88.13 |
| HLE (full) | 29.20 | — | 24.44 | 38.76 | — | 31.60 | 35.08 |
| MMLongBench-Doc | 62.82 | — | 63.74 | 67.93 | — | 49.06 | 65.83 |
| OSWorld | 52.14 | — | 51.47 | 55.76 | — | — | 68.86 |
| **Long context** | | | | | | | |
| AA-LCR | 68.63 | 75.19 | 75.13 | 78.38 | 79.69 | 77.31 | 79.50 |
| GraphWalks (≤1M) | 89.71 | 63.28 | 86.12 | 76.15 | 65.38 | 55.74 | 88.41 |
| **Instruction following and chat** | | | | | | | |
| IFBench | 70.86 | 82.94 | 67.56 | 75.63 | 76.19 | 79.31 | 57.67 |
| NV Chat Benchmark | 52.67 | 34.27 | 46.98 | 35.16 | 49.78 | 53.60 | 61.00 |
| **Multilingual** | | | | | | | |
| MMLU-ProX | 76.91 | 83.10 | 83.19 | 82.98 | 85.53 | 83.97 | 82.99 |
| WMT24++ COMET | 84.80 | 82.06 | 86.34 | 86.20 | 85.50 | 84.60 | 86.13 |

With about 4.6× fewer active parameters than Nemotron 3 Ultra, Super 3.5 scores higher than Ultra on SWE-bench Pro, τ³-Bench Banking, Terminal-Bench 2.1, GDPVal-AA, both BrowseComp settings, SciCode, GPQA-Diamond, CritPt, both HLE settings, Apex-Shortlist, AA-Omniscience accuracy, GraphWalks, NV Chat Benchmark, and WMT24++. It scores lower on SWE-bench Verified and Multilingual, the three τ²-Bench domains, LiveCodeBench v6, AA-Omniscience non-hallucination, AA-LCR, IFBench, and MMLU-ProX.

> **Note**: These numbers come from the full post-training pipeline, including data that is not yet published, and from the MTP-boosted release checkpoint. Checkpoints trained with these recipes on other data will score differently.

### Text Arena (Table 11)

Elo ratings from human pairwise preference votes. The report lists both Nemotron models as NVFP4 checkpoints.

| Text Arena | N3.5-Super 120B-A12B | N3-Ultra 550B-A55B | Qwen3.5 122B-A10B | DS-V4 Flash | Inkling Small | GLM-5.3 Flash |
|------------|-----:|-----:|-----:|-----:|-----:|-----:|
| Overall, with style control | 1418 | 1426 | 1416 | 1436 | 1404 | 1474 |
| Overall, no style control | 1411 | 1444 | 1417 | 1432 | 1412 | 1471 |
| English, with style control | 1442 | 1439 | 1430 | 1445 | 1417 | 1478 |
| English, no style control | 1432 | 1454 | 1429 | 1441 | 1421 | 1473 |

### Agentic coding across harnesses (§3.10)

The report evaluates the model under six agent harnesses with unmodified system prompts and standard configurations (Figure 8). For SWE-bench Pro, Git history is removed and a user instruction prohibits cheating or seeking online resources, so these scores are not comparable with Table 10.

| Harness | SWE-bench Pro | Terminal-Bench 2.1 |
|---------|--------------:|-------------------:|
| Claude Code | **49.45** | 59.55 |
| Mini-SWE-agent | 47.40 | 54.68 |
| Codex | 47.26 | **61.80** |
| Hermes | 47.20 | 61.42 |
| Pi | 46.92 | 48.31 |
| OpenCode | 41.04 | 55.06 |

The harness ordering differs between the two benchmarks, so report the harness and its configuration with any agentic result.

### Test-time scaling (§3.11)

N independent copies of the same agent attempt each task in parallel, and the same checkpoint then picks one trajectory with LLM-as-a-Verifier: pairwise comparisons on three criteria (specification adherence, output match, error-signal detection), both presentation orders, aggregated with a Probabilistic Pivot Tournament. No extra training or separate verifier is needed (Figure 9):

| Benchmark | Candidates | Tasks | Pass@1 | With selection | Gain (pp) |
|-----------|-----------:|------:|-------:|---------------:|----------:|
| Terminal-Bench 2.1 | 5 | 89 | 62.7 | 71.9 | +9.2 |
| SWE-bench Verified | 3 | 500 | 71.3 | 72.4 | +1.1 |
| SWE-bench Pro | 3 | 731 | 52.2 | 54.6 | +2.4 |
| SWE-bench Multilingual | 3 | 300 | 69.8 | 71.7 | +1.9 |
| DeepSWE | 5 | 113 | 24.6 | 34.5 | +9.9 |
| BrowseComp (400-problem subset) | 3 | 400 | 71.5 | 79.8 | +8.3 |

On the BrowseComp subset, selection recovers about 70% of the gap between pass@1 and oracle best-of-three.

### Efficiency on SWE-bench Pro (§3.9)

Figure 7 measures the aggregate GPU time to complete SWE-bench Pro three times (2,193 tasks) on four B200 GPUs, replaying each model's recorded trajectories (NVFP4 for all models except DeepSeek V4 Flash 0731, which uses MXFP4):

| Model | Pass@1 | GPU-hours |
|-------|-------:|----------:|
| Nemotron 3.5 Super 120B-A12B (NVFP4) | 51.3% | 8.3 |
| Qwen-3.8-125B-A6B-L51B (NVFP4) | 60.4% | 37.1 |
| DeepSeek V4 Flash 0731 284B-A13B (MXFP4) | 56.9% | 40.4 |
| Inkling Small 276B-A12B (NVFP4) | 56.5% | 98.9 |

### Reasoning modes (§3.9)

The model supports **reasoning-off**, **regular**, and **medium-effort** modes; regular and medium-effort can be combined with inference-time budget control. Averaged over the Artificial Analysis Intelligence Index benchmarks, medium-effort uses about 2.2× fewer tokens than regular mode for about a 9% accuracy drop.

### Base model results

Base-model results (Table 6) were collected with the NeMo Evaluator SDK and its LM Evaluation Harness container. Nemotron 3.5 Lightning is shown for reference.

| Task | Metric | Nemotron 3.5 Super Base | Nemotron 3.5 Lightning Base |
|------|--------|------------------------:|----------------------------:|
| **General knowledge** | | | |
| MMLU | exact match | 84.14 | 78.63 |
| MMLU-Pro | exact match, TigerLab extract | 76.25 | 67.50 |
| AGIEval-En | CoT exact match | 75.65 | 68.62 |
| SuperGPQA | CoT | 47.65 | 33.21 |
| GPQA Diamond | 5-shot, CoT exact match | 66.16 | 33.33 |
| **Math** | | | |
| GSM8K | 8-shot, CoT exact match | 91.74 | 91.74 |
| Minerva MATH | exact match | 87.66 | 83.10 |
| MATH-500 | 4-shot, sampled pass@1, n = 32 | 81.56 | 79.53 |
| **Code** | | | |
| HumanEval | sampled pass@1, n = 32, EvalPlus sanitized | 81.31 | 76.91 |
| MBPP-Sanitized | 3-shot, sampled pass@1, n = 32, EvalPlus sanitized | 79.06 | 76.98 |
| LiveCodeBench CodeGen | pass@1 | 47.61 | 29.77 |
| BigCodeBench Complete | greedy | 48.42 | 50.09 |
| BigCodeBench Instruct | greedy | 49.39 | 51.75 |
| **Commonsense understanding** | | | |
| CommonsenseQA | 7-shot, acc | 85.26 | 81.08 |
| ARC-Challenge | 25-shot, acc_norm | 95.99 | 93.17 |
| HellaSwag | 10-shot, acc_norm | 88.96 | 85.25 |
| OpenBookQA | 0-shot, acc_norm | 48.80 | 47.80 |
| PIQA | 0-shot, acc_norm | 85.09 | 85.09 |
| TruthfulQA MC2 | 0-shot, acc | 71.18 | 55.53 |
| WinoGrande | 5-shot, acc | 79.08 | 78.06 |
| SocialIQA | 0-shot, acc | 49.03 | 47.80 |
| **Reading comprehension** | | | |
| RACE | 0-shot, acc | 91.39 | 87.85 |
| **Multilingual** | | | |
| Global MMLU | avg. across 8 languages | 86.56 | 75.09 |
| MGSM | 8-shot, native CoT, avg. across 6 languages | 87.53 | 82.27 |

---

## Recipe: `nemotron super35 eval`

### Quick Start

```bash
# Evaluate the final MOPD checkpoint (default: run.model=super35-rl-mopd-phase2-model:latest)
uv run nemotron super35 eval --run YOUR-CLUSTER

# Evaluate a specific artifact version (for example the phase-2 step-30 checkpoint)
uv run nemotron super35 eval --run YOUR-CLUSTER run.model=super35-rl-mopd-phase2-model:vN

# Filter tasks
uv run nemotron super35 eval --run YOUR-CLUSTER -t nemo_skills.ns_gpqa -t ns_ifbench

# Preview the compiled config
uv run nemotron super35 eval --dry-run
```

Like the [Nemotron 3 Super evaluation](../super3/evaluate.md), `super35 eval` has no recipe script: the CLI compiles `src/nemotron/recipes/super35/stage3_eval/config/default.yaml`, resolves the model artifact, and passes the job to [nemo-evaluator-launcher](https://github.com/NVIDIA-NeMo/Evaluator). The checkpoint must be in Hugging Face format (the RL stages produce HF checkpoints; Megatron checkpoints from pretrain and SFT need a Megatron-Bridge export first).

### Default tasks

The config runs the five Table 10 benchmarks that NeMo Evaluator supports without a judge model or extra services. Task names follow NeMo Evaluator's Nemotron 3 Super example.

| Table 10 benchmark | Task | Repeats (config) | Repeats (report) |
|--------------------|------|-----------------:|-----------------:|
| GPQA-Diamond | `nemo_skills.ns_gpqa` | 8 | 8 |
| SciCode | `ns_scicode` | 8 | 8 |
| LiveCodeBench v6 | `ns_livecodebench` (`test_v6_2408_2505`) | 3 | 3 |
| IFBench | `ns_ifbench` | 8 | 8 |
| MMLU-ProX | `ns_mmlu_prox` | 1 | not stated |

HLE, AA-LCR, AA-Omniscience, CritPt, τ²-Bench, and WMT24++ are listed but commented out in the config, because they need judge or user-simulator models, API keys, or a local XCOMET-XXL checkpoint. The agentic, multimodal, Apex-Shortlist, GraphWalks, OSWorld, and NV Chat Benchmark results are not covered by this stage.

### Deployment and sampling

| Setting | Value |
|---------|-------|
| Serving | vLLM, tensor parallel 8 on one node, `--trust-remote-code`, FP32 Mamba SSM cache, `qwen3_coder` tool-call parser, `nemotron_v3` reasoning parser |
| Sampling | Reasoning on (`enable_thinking: true`), temperature 1.0, top-p 0.95, up to 131,072 new tokens |
| Container | `run.env.container` (a vLLM image that serves Super 3.5 VL; pin a released version) |
| Export | Results are exported to the W&B project from `env.toml` |

The eval stage has not been run against Table 10 from this repository.

## Reference

- Nemotron 3.5 tech report, §2.6 (base model), §3.8–3.11 (post-trained evaluation, reasoning control, harnesses, test-time scaling), Tables 6, 10, 11
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym): evaluation environments and harnesses
- [Quantization](./quantization.md): BF16 vs NVFP4 across 35 benchmarks
- **Recipe Source**: `src/nemotron/recipes/super35/stage3_eval/`
- [Back to Overview](./README.md)
