---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "evaluation"
paper_sections: ["1", "2.6", "3.8", "3.8.1", "3.9", "3.10", "3.11", "4"]
title: "Base-model and Post-trained Evaluations: Table 6, Table 10, Text Arena, Reasoning Control, Harnesses, Test-time Scaling"
summary: |
  Combines §2.6 base-model evaluations (Table 6: Nemotron 3.5 Super 120B-A12B Base, with Lightning 30B-A3B
  Base as the only other column) and §3.8 post-trained evaluations (setup + Table 10 vs six open models,
  Table 11 Text Arena Elo), plus §3.9 reasoning modes and the Figure 7 SWE-bench Pro GPU-hour comparison,
  §3.10 six-harness agentic coding (Figure 8), and §3.11 parallel-agent test-time scaling with
  LLM-as-a-Verifier (Figure 9). Every number is tagged BASE, POST-TRAINED BF16, or NVFP4. Table 10 has
  no precision label in its caption; its Super column equals the BF16 column of Table 22 (§4), so it is
  BF16. Table 11 (Text Arena) and Figure 7 are explicitly NVFP4. Figure 9's precision is not reported.
key_facts:
  - "BASE (Table 6, p.11): Super 3.5 Base MMLU 84.14, MMLU-Pro 76.25, GPQA Diamond 66.16 (5-shot CoT), GSM8K 91.74, Minerva MATH 87.66, MATH-500 81.56, HumanEval 81.31, MBPP-Sanitized 79.06, LiveCodeBench CodeGen 47.61, Global MMLU 86.56, MGSM 87.53. No external base models are compared; the only other column is Lightning 3.5 30B-A3B Base."
  - "Base evals: NeMo Evaluator SDK + NVIDIA's open-source LM Evaluation Harness container, unless otherwise stated (§2.6)."
  - "Post-trained evals: 'All evaluation results for Nemotron 3.5 Super and baselines were collected via Nemo Gym' (§3.8.1), with a Super-3.5 reproducibility tutorial in the NeMo Gym recipes folder."
  - "Not yet onboarded in open-source tools (official implementation or internal scaffolding NVIDIA 'plan[s] to open source in the future'): BabyVision, CharXiv-RQ, MathVision, MMLongBench-Doc, OCR-Reasoning, OCRBench-v2, V*, Video-MME-v2. NV Chat Benchmark is an internal NVIDIA chat benchmark. HLE with tools comes from 'a separate internal harness' (§3.8.1)."
  - "POST-TRAINED BF16 (Table 10, p.23) comparison models: Nemotron 3 Ultra 550B-A55B, Qwen3.5 122B-A10B, Qwen3.8 Flash-Next, DeepSeek-V4 Flash-0731, Inkling Small, GLM-5.3 Flash."
  - "POST-TRAINED BF16 agentic: SWE-bench Verified 71.10, Multilingual 67.72, Pro 51.78 (OpenCode, avg 3 runs); Terminal-Bench 2.1 60.67 (Terminus 2, 3h timeout, avg 8 runs); tau2 Retail 69.24 / Airline 76.50 / Telecom 73.19; tau3 Banking 18.88; GDPVal-AA-v2 ELO 1,211.20 (norm 0.356); BrowseComp 78.91 (NeMo Gym harness) / 72.35 (OpenCode harness)."
  - "POST-TRAINED BF16 reasoning: GPQA-Diamond 89.52; HLE text 31.79 (no tools) / 46.01 (with tools); Apex-Shortlist 88.36; CritPt 10.86; AA-Omniscience accuracy 26.20, non-hallucination 69.96; SciCode 47.13 subtask / 15.10 problem; LiveCodeBench v6 88.22."
  - "POST-TRAINED BF16 multimodal: BabyVision 29.60, CharXiv-RQ 80.82, MathVision 78.16, OCR-Reasoning 65.08, OCRBench-v2 68.12 EN / 59.15 ZH, Video-MME-v2 32.85, V* 85.60, HLE (full, 2,500 q) 29.20, MMLongBench-Doc 62.82, OSWorld 52.14."
  - "POST-TRAINED BF16 long context / chat / multilingual: AA-LCR 68.63 (Table 10) — but 68.62 in Table 22 BF16 column; GraphWalks (<=1M) 89.71; IFBench 70.86; NV Chat Benchmark 52.67; MMLU-ProX 76.91; WMT24++ COMET 84.80."
  - "Headline (Abstract, §1): SWE-bench Pro 51.78% 'surpassing Nemotron 3 Ultra' (49.66) with ~4.6x fewer active parameters (12B vs 55B)."
  - "NVFP4 Text Arena Elo (Table 11, p.25), Super / Ultra both NVFP4: Overall with style control 1418 (Ultra 1426, GLM-5.3 Flash 1474 best); English with style control 1442 (Ultra 1439)."
  - "§3.9: three reasoning modes (reasoning-off, regular, medium-effort); regular and medium-effort can be combined with inference-time budget control. Medium-effort uses ~2.2x fewer tokens than regular, averaged over the Artificial Analysis Intelligence Index benchmarks, for ~9% accuracy drop. ~4% of RL prompts are medium-effort."
  - "NVFP4 Figure 7 (SWE-bench Pro, 2,193 tasks = 3 x 731, 4x B200): Super 51.3% in 8.3 GPU-h; Qwen-3.8-125B-A6B-L51B 60.4% / 37.1 GPU-h; DeepSeek V4 Flash 0731 284B-A13B (MXFP4) 56.9% / 40.4 GPU-h; Inkling Small 276B-A12B 56.5% / 98.9 GPU-h."
  - "§3.10 Figure 8 (six harnesses, Git history removed, anti-cheat user instruction): SWE-bench Pro best Claude Code 49.45%, worst OpenCode 41.04%; Terminal-Bench 2.1 best Codex 61.80%, worst Pi 48.31%. Not comparable with Table 10."
  - "§3.11 Figure 9 (LAV self-verification, same checkpoint as verifier, no extra training): Terminal-Bench 2.1 62.7 -> 71.9 (+9.2, Bo5); SWE-bench Verified 71.3 -> 72.4; Pro 52.2 -> 54.6; Multilingual 69.8 -> 71.7 (Bo3); DeepSWE 24.6 -> 34.5 (+9.9, Bo5); BrowseComp 400-problem internal subset 71.5 -> 79.8 (+8.3, Bo3)."
related_steps:
  - "stage3_eval"
currency: "frozen"
---

# Scope
Answers:
- How does Nemotron 3.5 Super 120B-A12B Base score (Table 6), and what is it compared against?
- Which harnesses, protocols, judges, and repeats define the post-trained evaluation (§3.8.1)?
- Which benchmarks are internal, or "to be open-sourced"?
- What does the post-trained BF16 model score against the six open comparison models (Table 10)?
- How does Super rank in Text Arena (NVFP4, Table 11)?
- What do the reasoning modes cost and buy (§3.9), and how efficient is Super on SWE-bench Pro (Figure 7)?
- How sensitive are agentic coding scores to the harness (§3.10, Figure 8)?
- How much does parallel-agent self-verification add (§3.11, Figure 9)?

Precision legend used throughout: **BASE** = pre-trained base checkpoint; **POST-TRAINED BF16** = released post-trained BF16 model; **NVFP4** = released NVFP4-quantized post-trained checkpoint. Lightning numbers appear only where the source table puts them side by side and are labelled; never cite them for Super.

# §2.6 Base Model Evaluations (Table 6, report p.11) — BASE

Collected via NeMo Evaluator SDK and NVIDIA's open-source container of LM Evaluation Harness, unless otherwise stated; configs in the NeMo Evaluator SDK examples. **Table 6 has no external base models**: its only columns are Super 3.5 Base and Lightning 3.5 Base. The caption mentions "A dash indicates that a corresponding result is not available" but no dashes appear.

| Task | Metric | Super 3.5 120B-A12B Base | (Lightning 3.5 30B-A3B Base, reference only) |
|---|---|---|---|
| **General knowledge** | | | |
| MMLU | exact match | 84.14 | 78.63 |
| MMLU-Pro | exact match, TigerLab extract | 76.25 | 67.50 |
| AGIEval-En | CoT exact match, custom extract | 75.65 | 68.62 |
| SuperGPQA | CoT | 47.65 | 33.21 |
| GPQA Diamond | 5-shot, CoT exact match | 66.16 | 33.33 |
| **Math** | | | |
| GSM8K | 8-shot, CoT exact match | 91.74 | 91.74 |
| Minerva MATH | exact match | 87.66 | 83.10 |
| MATH-500 | 4-shot, sampled pass@1, n=32 | 81.56 | 79.53 |
| **Code** | | | |
| HumanEval | sampled pass@1, n=32, EvalPlus sanitized (0-shot) | 81.31 | 76.91 |
| MBPP-Sanitized | 3-shot, sampled pass@1, n=32, EvalPlus sanitized | 79.06 | 76.98 |
| LiveCodeBench CodeGen | pass@1 | 47.61 | 29.77 |
| BigCodeBench Complete | greedy | 48.42 | 50.09 |
| BigCodeBench Instruct | greedy | 49.39 | 51.75 |
| **Commonsense** | | | |
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
| Global MMLU | avg. of 8 langs (de, es, fr, it, ja, ko, pt, zh) | 86.56 | 75.09 |
| MGSM | 8-shot native CoT, avg. of 6 langs (es, de, fr, zh, ja, ru) | 87.53 | 82.27 |

Notes:
- GSM8K and PIQA are identical for Super and Lightning (91.74, 85.09) as printed; verified against the PDF layout.
- Lightning beats Super on both BigCodeBench variants as printed.
- No base long-context (RULER) row appears in Table 6. A RULER-1M recovery from 33.5 to 81.0 after a 9B-token recovery stage is mentioned in §1 (p.2), outside this table.

# §3.8 Post-trained Model Evaluations

## §3.8.1 Evaluation Setup

- "All evaluation results for Nemotron 3.5 Super and baselines were collected via Nemo Gym"; the report points to a Super-3.5 reproducibility tutorial in the NeMo Gym recipes folder.
- **Not yet onboarded in open-source tools** (official open-source implementation, or "an internal scaffolding that we plan to open source in the future"): BabyVision, CharXiv-RQ, MathVision, MMLongBench-Doc, OCR-Reasoning, OCRBench-v2, V*, Video-MME-v2.
- **Internal:** NV Chat Benchmark ("an internal chat benchmark developed at NVIDIA"); HLE with tools ("comes from a separate internal harness"); the BrowseComp 400-problem subset used in §3.11 ("used for our internal evaluation tracking").
- **Open-source harnesses named:** BrowseComp harness "released in NeMo Gym"; "our open-source implementation of GDPVal".
- Identical settings across all models: agentic resources (CPU, timeouts, tools, search APIs), input data, prompt templates, repeats, extraction/metric implementation. Temperature, top_p, and max tokens taken from each model's card.

| Category | Benchmark | Protocol (§3.8.1) |
|---|---|---|
| Agentic | SWE-bench Verified / Multilingual / Pro | OpenCode harness, sandbox, web search + web fetch disabled, resolve rate avg over 3 runs |
| | Terminal-Bench 2.1 | Terminus 2 harness, 3-hour task timeout (Table 10 caption), avg over 8 runs |
| | tau2-Bench Retail / Airline / Telecom | avg over 8 trials; user simulator Qwen3-235B-A22B-Instruct-2507 |
| | tau3-Bench Banking | avg over 8 trials; agent retrieves from bank KB with BM25 and grep tools; user simulator GPT-5.4-mini |
| | GDPVal-AA v2 | open-source implementation loosely following AA v2.1 methodology; same sandbox (python/system packages); Stirrup agent. Reference committee: DeepSeek V4 Flash 0731, DeepSeek V4 Pro, Inkling Small, GLM-5.1, Kimi K2.6, Nemotron 3 Ultra, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, GPT-OSS-120b, Gemma 4 26B-A4B, Qwen3-30B-A3B-Thinking-2507. Judge committee as AA: GPT-5.6 Sol (medium), Gemini 3.8 Flash (high), Claude Opus 5 (high effort). ELO normalization method not reported |
| | BrowseComp | single open-source NeMo Gym harness; search/browse tools backed by Exa (deep mode); per-sample workspace via read-only shell tool; discard-all context policy with a memory tool. Second row: OpenCode-based harness with minor tool-config changes |
| Code | LiveCodeBench v6 | 454 problems (Aug 2024 - May 2025), no tools, pass@1 avg over 3 samples |
| | SciCode | 65 problems / 288 subproblems, scientist background included, avg over 8 repeats; subtask and problem accuracy |
| Math & science | GPQA-Diamond | avg over 8 repeats |
| | HLE (text) | 2,158 text-only questions; no tools graded by GPT-5.6 Luna with official HLE judge prompt; with tools = Python + web search, separate internal harness |
| | Apex-Shortlist (MathArena) | avg over 16, symbolic-equivalence checking, no tools |
| | CritPt | official Artificial Analysis grading API, 5 samples for each of 70 problems |
| | AA-Omniscience | 600-question public set, 10 repeats, graded by GPT-5.6 Luna; accuracy and non-hallucination rate |
| Multimodal | BabyVision, CharXiv-RQ, MathVision, OCR-Reasoning, OCRBench v2 (EN/ZH), V*, Video-MME-v2, MMLongBench-Doc, HLE (full), OSWorld-Verified | avg over 3 repeats unless noted. Judges: GPT-4o-mini (BabyVision, MathVision, OCR-Reasoning), Claude Opus 4.7 (CharXiv), GPT-4o (MMLongBench-Doc), GPT-3.5-Turbo (V* answer extraction). OCRBench v2 and Video-MME-v2 rule-based. V* static multiple choice, no visual-search tool. Video-MME-v2 without subtitles. MMLongBench-Doc up to 100 rendered pages, one image per page. HLE (full) 2,500 questions, 1 sample, GPT-5.6 Luna. OSWorld-Verified: 361 tasks not needing Google Drive, cloud desktop VMs, screenshots only, pyautogui actions, up to 100 steps, avg task score (partial credit) over 3 runs |
| Long context | AA-LCR v1.1 | 100 questions over ~100K-token document sets, 16 repeats, GPT-5.6 Luna per official protocol |
| | GraphWalks | BFS and parent tasks, all context-length buckets; 950 of 1,150 prompts that fit within 1M tokens (same 950 for all models); F1 of node sets, single sample |
| IF & chat | IFBench | 58 out-of-domain verifiable constraints; strict; avg over prompts and 8 samples |
| | NV Chat Benchmark (internal) | 3,713 prompts; opponent response from human-preference battles; Gemini 3.1 Pro judge, both presentation orders, length-based style control; win rate |
| | Text Arena (Table 11) | Elo from human pairwise preference votes; all prompts and English only; with and without style control |
| Multilingual | MMLU-ProX | accuracy avg over 10 langs: en, de, es, fr, it, ja, zh, hi, pt, ko |
| | WMT24++ | en -> the 9 non-English languages of that set; reference-based XCOMET-XXL, macro-averaged over pairs |

## §3.8 Results — Table 10 (report p.23) — POST-TRAINED BF16

Caption: "Evaluation suite for Nemotron 3.5 Super. We compare against six open models. All SWE evaluations use OpenCode as a default harness. Terminal Bench 2.1 uses Terminus 2 and a 3 hours task timeout." The caption does not state precision; every Super value that also appears in Table 22 (§4, p.42) matches its **BF16** column, so Table 10 is treated as BF16. Comparison-model precision is not reported. "—" = no result in the source. Re-checked against `pdftotext -layout` of p.23.

| Benchmark | N3.5-Super 120B-A12B | N3-Ultra 550B-A55B | Qwen3.5 122B-A10B | Qwen3.8 Flash-Next | DS-V4 Flash-0731 | Inkling Small | GLM-5.3 Flash |
|---|---|---|---|---|---|---|---|
| **Agentic** | | | | | | | |
| SWE-bench Verified | 71.10 | 73.93 | 71.87 | 72.60 | 77.80 | 75.27 | 79.10 |
| SWE-bench Multilingual | 67.72 | 68.78 | 62.00 | 79.00 | 72.00 | 71.44 | 78.30 |
| SWE-bench Pro | 51.78 | 49.66 | 45.60 | 60.37 | 56.91 | 56.54 | 59.70 |
| Tau2 Retail | 69.24 | 73.68 | 55.37 | 71.60 | 73.79 | 71.82 | 78.18 |
| Tau2 Airline | 76.50 | 79.75 | 63.50 | 80.75 | 83.50 | 62.75 | 81.75 |
| Tau2 Telecom | 73.19 | 83.00 | 87.17 | 90.35 | 98.25 | 90.12 | 98.46 |
| Tau3 Banking | 18.88 | 18.35 | 14.23 | 44.54 | 38.56 | — | 34.85 |
| Terminal-Bench 2.1 | 60.67 | 54.78 | 48.70 | 85.80 | 76.80 | 53.40 | 82.70 |
| GDPVal-AA-v2 ELO | 1,211.20 | 1,148.30 | 860.73 | 1,561.69 | 1,382.30 | 1,313.27 | 1,563.39 |
| GDPVal-AA-v2 norm ELO | 0.356 | 0.324 | 0.180 | 0.531 | 0.441 | 0.407 | 0.532 |
| BrowseComp | 78.91 | 67.85 | 58.10 | 89.02 | 84.20 | 74.10 | 86.57 |
| BrowseComp (OpenCode) | 72.35 | 61.06 | 45.42 | 76.07 | 64.77 | 55.37 | 81.91 |
| **Code generation** | | | | | | | |
| SciCode (subtask) | 47.13 | 41.96 | 39.83 | 46.62 | 49.30 | 48.28 | 50.05 |
| SciCode (problem) | 15.10 | 13.46 | 10.77 | 14.42 | 18.27 | 18.65 | 17.53 |
| LiveCodeBench v6 | 88.22 | 89.21 | 76.65 | 94.13 | 94.64 | 88.25 | 80.62 |
| **Math and science** | | | | | | | |
| GPQA-Diamond | 89.52 | 86.99 | 86.74 | 92.68 | 89.65 | 88.57 | 91.22 |
| CritPt | 10.86 | 3.14 | 1.14 | — | 12.57 | 9.14 | — |
| HLE (no tools, text) | 31.79 | 29.24 | 24.24 | 39.57 | 35.82 | 33.27 | 36.47 |
| HLE (with tools, text) | 46.01 | 37.40 | 47.50 | — | 45.10 | 47.80 | 55.30 |
| Apex-Shortlist | 88.36 | 75.00 | 54.65 | 91.76 | 88.70 | 71.54 | 77.13 |
| AA-Omniscience Accuracy | 26.20 | 25.30 | 26.25 | 27.87 | 44.95 | 35.38 | 34.22 |
| AA-Omniscience Non-Hallucination | 69.96 | 77.91 | 15.5 | 58.69 | 4.09 | 40.39 | 69.75 |
| **Multimodal** | | | | | | | |
| BabyVision | 29.60 | — | 35.65 | 46.82 | — | 24.74 | 39.09 |
| CharXiv-RQ | 80.82 | — | 83.47 | 85.77 | — | 77.77 | 81.93 |
| MathVision | 78.16 | — | 83.29 | 87.54 | — | 78.77 | 82.52 |
| OCR-Reasoning | 65.08 | — | 64.27 | 67.10 | — | 54.69 | 64.61 |
| OCRBench-v2 (EN) | 68.12 | — | 68.27 | 67.91 | — | 54.21 | 61.81 |
| OCRBench-v2 (ZH) | 59.15 | — | 67.26 | 68.28 | — | 52.38 | 56.81 |
| Video-MME-v2 | 32.85 | — | 31.20 | 45.05 | — | — | 14.79 |
| V* | 85.60 | — | 91.10 | 93.02 | — | 70.51 | 88.13 |
| HLE (full) | 29.20 | — | 24.44 | 38.76 | — | 31.60 | 35.08 |
| MMLongBench-Doc | 62.82 | — | 63.74 | 67.93 | — | 49.06 | 65.83 |
| OSWorld | 52.14 | — | 51.47 | 55.76 | — | — | 68.86 |
| **Long context** | | | | | | | |
| AA-LCR | 68.63 | 75.19 | 75.13 | 78.38 | 79.69 | 77.31 | 79.50 |
| GraphWalks (<=1M) | 89.71 | 63.28 | 86.12 | 76.15 | 65.38 | 55.74 | 88.41 |
| **Instruction following and chat preference** | | | | | | | |
| IFBench | 70.86 | 82.94 | 67.56 | 75.63 | 76.19 | 79.31 | 57.67 |
| NV Chat Benchmark | 52.67 | 34.27 | 46.98 | 35.16 | 49.78 | 53.60 | 61.00 |
| **Multilingual** | | | | | | | |
| MMLU-ProX | 76.91 | 83.10 | 83.19 | 82.98 | 85.53 | 83.97 | 82.99 |
| WMT24pp COMET | 84.80 | 82.06 | 86.34 | 86.20 | 85.50 | 84.60 | 86.13 |

Readings (derived from the table, not claims of the report unless quoted):
- Report claim (Abstract, §1): SWE-bench Pro 51.78% "surpassing Nemotron 3 Ultra despite using approximately 4.6x fewer active parameters."
- Super beats Ultra on SWE-bench Pro, tau3 Banking, Terminal-Bench 2.1, both GDPVal rows, both BrowseComp rows, both SciCode rows, GPQA-Diamond, CritPt, both HLE text rows, Apex-Shortlist, AA-Omniscience accuracy, GraphWalks, NV Chat Benchmark, WMT24++. Ultra is ahead on SWE-bench Verified and Multilingual, all three tau2 domains, LiveCodeBench v6, AA-Omniscience non-hallucination, AA-LCR, IFBench, MMLU-ProX. Ultra has no multimodal results.
- Super is the column maximum on GraphWalks (89.71); it is lowest of the seven on AA-LCR (68.63) and MMLU-ProX (76.91).
- Figure 1 (p.2; repo asset `docs/assets/super35/benchmarks.png`) plots 12 of these rows rounded to one decimal; spot-checked values match Table 10. Its right-hand throughput panel is covered in `inference.md`.

## Table 11 — Text Arena Elo (report p.25) — NVFP4 for both Nemotron columns

The header row marks N3.5-Super and N3-Ultra as **NVFP4**; precision of the other four models is not reported. Qwen3.8 Flash-Next is not in this table.

| Text Arena | N3.5-Super 120B-A12B (NVFP4) | N3-Ultra 550B-A55B (NVFP4) | Qwen3.5 122B-A10B | DS-V4 Flash | Inkling Small | GLM-5.3 Flash |
|---|---|---|---|---|---|---|
| Overall, with style control | 1418 | 1426 | 1416 | 1436 | 1404 | 1474 |
| Overall, no style control | 1411 | 1444 | 1417 | 1432 | 1412 | 1471 |
| English, with style control | 1442 | 1439 | 1430 | 1445 | 1417 | 1478 |
| English, no style control | 1432 | 1454 | 1429 | 1441 | 1421 | 1473 |

Confidence intervals, vote counts, and collection dates are not reported.

## Cross-check: Table 22 (§4, p.42) — POST-TRAINED BF16 vs NVFP4

Table 22 ("Super 3.5 BF16 versus released NVFP4 across the full benchmark suite") repeats Table 10's BF16 values and adds the released NVFP4 checkpoint. Full treatment belongs to the quantization chunk; selected rows for orientation:

| Benchmark | BF16 | NVFP4 | Recovery % |
|---|---|---|---|
| SWE-Bench Verified | 71.10 | 71.07 | 99.95 |
| SWE-Bench Pro | 51.78 | 51.30 | 99.08 |
| Terminal-Bench v2.1 | 60.67 | 62.78 | 103.47 |
| GDPVal (metric not stated; not ELO) | 49.98 | 55.74 | 111.53 |
| AA-LCR | **68.62** | 67.69 | 98.63 |
| MMLU-ProX | 76.91 | 73.76 | 95.90 |
| Median recovery | | | 99.52% |

Table 22 also has rows absent from Table 10 (BIRD text2sql 59.58 BF16, ScreenSpot Pro 43.53 BF16, WMT24++ (en->xx) 52.32 BF16 with an unnamed metric) and omits Table 10's BrowseComp (both), OSWorld, CritPt, and HLE (with tools) rows.

# §3.9 Reasoning Efficiency and Control

- Three trained modes: **reasoning-off**, **regular**, **medium-effort**. Regular and medium-effort can be combined with inference-time budget control.
- Medium-effort is introduced in SFT with data generated by Kimi K3's low-effort mode, then optimized in RL. Budget-control prep follows the Nemotron 3 Ultra recipe: reasoning traces of a small fraction of SFT data are truncated to random budgets, responses unchanged, `</think>` masked from the loss.
- ~4% of RL training prompts are medium-effort (math, STEM, coding), with length-based reward adjustments; the effect generalizes beyond those domains (report claim).
- **Medium-effort uses ~2.2x fewer tokens than regular, averaged over the Artificial Analysis Intelligence Index benchmarks, at ~9% accuracy drop.** Per-benchmark numbers, the budget-control curve, and reasoning-off accuracy are not reported. Whether "9%" is relative or absolute is not stated.

## Figure 7 — SWE-bench Pro accuracy vs GPU time (p.27) — NVFP4

Each model is evaluated, trajectories are recorded, and the captured traces are replayed to benchmark throughput. Y = Pass@1 averaged over 3 runs; X = aggregate GPU-hours to complete 2,193 tasks (3 x 731 public set); 4x NVIDIA B200. All NVFP4 except DeepSeek V4 Flash 0731 (MXFP4). Values read from the figure labels.

| Model (precision) | Pass@1 | GPU-hours |
|---|---|---|
| Nemotron Super 3.5 120B-A12B (NVFP4) | 51.3% | 8.3 |
| Qwen-3.8-125B-A6B-L51B (NVFP4) | 60.4% | 37.1 |
| DeepSeek V4 Flash 0731 284B-A13B (MXFP4) | 56.9% | 40.4 |
| Inkling Small 276B-A12B (NVFP4) | 56.5% | 98.9 |

51.3% matches Table 22's NVFP4 SWE-Bench Pro (51.30). Serving engine, concurrency, and MTP settings for Figure 7 are not reported. The figure labels the Qwen model "Qwen-3.8-125B-A6B-L51B"; the report does not state whether this is the Table 10 "Qwen3.8 Flash-Next" (its 60.4% matches Table 10's 60.37 within rounding).

# §3.10 Agentic Coding Across Harnesses (Figure 8, p.27) — precision not reported

Six harnesses, unmodified harness system prompts, standard configurations. For SWE-bench Pro, Git history is removed and a user instruction prohibits cheating or seeking online resources. Subtitles: SWE-Bench Pro "731 tasks · 2 repeats"; Terminal-Bench 2.1 "89 tasks · 3 repeats". Values and harness versions read from the figure.

| Harness (version) | SWE-bench Pro | Terminal-Bench 2.1 |
|---|---|---|
| Claude Code (v2.1.126) | **49.45** | 59.55 |
| Mini-SWE-agent (v2.4.5) | 47.40 | 54.68 |
| Codex (v0.144.6) | 47.26 | **61.80** |
| Hermes (v0.18.2) | 47.20 | 61.42 |
| Pi (v0.80.10) | 46.92 | 48.31 |
| OpenCode (v1.17.8) | 41.04 | 55.06 |

Report claim: harness ordering differs between benchmarks, so report the harness and configuration with results. Note: OpenCode SWE-bench Pro here (41.04, 2 repeats, anti-cheat setup) differs from Table 10's OpenCode SWE-bench Pro (51.78, 3 runs); the Figure 8 setup is stricter and not comparable. Terminus 2 (Table 10's Terminal-Bench harness) is not among the six. Training-side harness diversity (mid-training ablation, Table 5) is in §2.5, not this chunk.

# §3.11 Test-Time Scaling via Parallel Agents and Self-Verification (Figure 9, p.28)

- N independent instances of the same post-trained agent attempt each task in parallel (no inter-agent communication). The **same checkpoint** then selects one trajectory with LLM-as-a-Verifier (LAV, Kwok et al. 2026). No parameter updates, no separately trained verifier.
- Pairwise comparison on three criteria: specification adherence, output match, error-signal detection. 20-level grade scale (single-token labels A-T, best to worst); trajectory score = probability-weighted average over grades normalized to [0,1].
- Both presentation orders (K=2) x 3 criteria = 6 evaluations per pair; averaged pair scores R_i, R_j converted with Bradley-Terry p_ij = 1/(1+exp(-(R_i - R_j))).
- Probabilistic Pivot Tournament: random ring of neighbor comparisons, top-mean candidates become pivots, remaining candidates compared against pivots, preferences aggregated, highest mean preference wins (normalized by number of comparisons). Terminal-Bench and the three SWE-bench variants use 2 pivots; DeepSWE and BrowseComp use 1.
- Precision of the "final post-trained checkpoint" is not reported.

| Benchmark (pool, tasks) | Pass@1 (mean over candidates) | With LAV selection | Gain (pp) |
|---|---|---|---|
| Terminal-Bench 2.1 (Bo5, 89) | 62.7 | 71.9 | +9.2 |
| SWE-bench Verified (Bo3, 500) | 71.3 | 72.4 | +1.1 |
| SWE-bench Pro (Bo3, 731) | 52.2 | 54.6 | +2.4 |
| SWE-bench Multilingual (Bo3, 300) | 69.8 | 71.7 | +1.9 |
| DeepSWE (Bo5, 113) | 24.6 | 34.5 | +9.9 |
| BrowseComp (Bo3, 400-problem internal subset) | 71.5 | 79.8 | +8.3 |

Terminal-Bench, DeepSWE, BrowseComp values and all gains are in the §3.11 text; SWE-bench absolute values are from the Figure 9 bar labels. On the BrowseComp subset, selection recovers ~70% of the gap between pass@1 and oracle best-of-three (oracle value not reported). DeepSWE is not in Table 10. The Figure 9 pass@1 baselines differ from Table 10 (e.g., Terminal-Bench 62.7 vs 60.67; SWE-bench Pro 52.2 vs 51.78) — different rollout sets; the report does not reconcile them.

# Table 27 (p.62, Appendix C.1) — output length pre-MOPD vs MOPD, both BF16

Referenced from Appendix C.1 (runaway-loop early-warning indicators; see `quantization-ablations.md`); physically placed at the top of p.62, just before the Appendix C heading. OSL = total over a rollout's policy-model calls, excluding the user simulator. Caption claim: "Post-training RL removed the non-terminating tail, not baseline verbosity."

| Benchmark | Modality | Rollouts | ISL (k) | OSL (k)/rollout pre-MOPD -> MOPD | Trunc. % |
|---|---|---|---|---|---|
| Baby Vision | vision | 388 x 3 | 0.5 | 42.3 -> 11.2 (0.27x) | 39.0 -> 1.9 |
| CharXiv (RQ) | vision | 1,000 x 3 | 3.1 | 11.8 -> 3.3 (0.28x) | 9.2 -> 0.6 |
| MMLongBench-Doc | vision | 1,091 x 3 | 75.0 | 10.8 -> 3.0 (0.28x) | 9.9 -> 0.4 |
| OCR Reasoning | vision | 1,069 x 3 | 1.5-1.6 (a) | 16.8 -> 5.2 (0.31x) | 13.7 -> 0.6 |
| OSWorld | vision/agentic | 361 x 3 | — | 13.6 -> 5.1 (0.38x) | 1.2 -> 0.0 |
| Tau3 Banking | text/agentic | 97 x 8 | — | 35.2 -> 13.7 (0.39x) | 20.9 -> 0.3 |
| AA-Omniscience | text | 600 x 10 | 0.1 | 10.0 -> 5.6 (0.56x) | 1.9 -> 0.3 |
| SciCode | text | 65 x 8 | 3.0-3.1 (a) | 7.2 -> 6.5 (0.90x) | 0.0 -> 0.0 |
| AA-LCR | text | 100 x 16 | 108.0 | 4.7 -> 4.7 (1.00x) | 0.0 -> 0.2 |
| Tau2-AA | text/agentic | 278 x 8 | — | 5.0 -> 5.1 (1.03x) | 0.0 -> 0.0 |

(a) evaluation harness update resulted in ISL difference.

# Contradictions and inconsistencies in the source
- **AA-LCR BF16:** 68.63 in Table 10 (p.23) vs 68.62 in Table 22's BF16 column (p.42). Unreconciled; quote both with their table.
- **"All via Nemo Gym" vs exceptions:** §3.8.1 says all results were collected via Nemo Gym, then lists eight benchmarks run with official implementations or internal scaffolding, and says HLE with tools uses a separate internal harness.
- **GDPVal metric:** Table 10 reports GDPVal-AA-v2 as ELO (1,211.20) and normalized ELO (0.356); Table 22 reports "GDPVal" BF16 49.98 / NVFP4 55.74 with no metric definition.
- **Repeats differ across sections:** Table 10 SWE-bench avg over 3 runs and Terminal-Bench avg over 8 runs; Figure 8 uses 2 repeats (SWE-bench Pro) and 3 repeats (Terminal-Bench 2.1).
- **Pass@1 baselines differ:** Terminal-Bench 2.1 is 60.67 (Table 10 BF16), 62.78 (Table 22 NVFP4), 62.7 (Figure 9, precision unstated), 48.31-61.80 across harnesses (Figure 8). SWE-bench Pro is 51.78 (Table 10), 51.30 (Table 22 NVFP4), 51.3% (Figure 7 NVFP4), 52.2 (Figure 9), 41.04-49.45 (Figure 8).
- **Table 6 caption** mentions dashes that the table does not contain.
- **Figure 1 throughput panel** (GB200, 4,331.88 TPS/GPU, 1.23x for Qwen3.8 Flash-Next -> 1.90x Super/Flash-Next derived) vs Abstract/§1 "4,632 output tokens/s/GPU" and "2.88x the throughput of Qwen3.8 Flash-Next"; see `inference.md`.

# Caveats
- Never mix Table 6 (BASE) with Table 10 (POST-TRAINED BF16); never attribute the Lightning column of Table 6 to Super.
- Table 10's BF16 status is inferred from Table 22's identical values; the Table 10 caption does not state precision.
- Figure 7, 8, 9 values are read from figure labels (raster images), not from tables.
- Comparison-model sizes are only given where the source prints them (Table 10 header; Figure 7 labels).
- `related_steps: stage3_eval` refers to this repository's `super35 eval` stage, which runs a subset (GPQA-Diamond, SciCode, LiveCodeBench v6, IFBench, MMLU-ProX) with NeMo Evaluator rather than NeMo Gym; scores are not expected to match Table 10 exactly.
