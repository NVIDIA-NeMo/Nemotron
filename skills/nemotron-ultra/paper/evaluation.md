---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "evaluation"
paper_sections: ["2.6", "3.7", "3.7.1", "3.7.2", "3.7.3", "3.7.4"]
title: "Base-model and Post-trained Model Evaluations"
summary: |
  Combines §2.6 base-model evaluations (Nemotron 3 Ultra 550B-A55B-Base vs
  DeepSeek-V3.2-Exp-Base, Mistral-Large-3-675B-Base, Kimi-K2-Base, GLM-4.5-Base)
  and §3.7 post-trained evaluations (setup, results vs six open models,
  test-time scaling on Olympiad math, and harness robustness). Base and
  post-trained numbers are kept in clearly separated tables. The paper reports
  BF16 post-trained results; no NVFP4 post-trained numbers appear in this slice.
key_facts:
  - "Base-model evals collected via Nemo Evaluator SDK and NVIDIA's open-source LM Evaluation Harness container unless otherwise stated."
  - "Base model: Nemotron-3-Ultra-550B-A55B-Base. MMLU 89.08 (5-shot acc), MMLU-Pro 79.07 (5-shot CoT EM), AGIEval-En 78.73, GPQA 50.00 (5-shot CoT EM)."
  - "Base math: GSM8K 88.10 (8-shot CoT EM), MATH 82.00 (4-shot EM, Minerva Math)."
  - "Base code: HumanEval 83.84 (sampled pass@1 n=32, EvalPlus sanitized), MBPP-Sanitized 85.97 (3-shot pass@1 n=32, EvalPlus sanitized)."
  - "Base commonsense: ARC-Challenge 97.35, HellaSwag 90.51, OpenBookQA 48.60, PIQA 83.79, WinoGrande 79.32."
  - "Base reading comprehension: RACE 92.15 (0-shot acc). Base multilingual: MMLU Global Lite 90.13 (5-shot avg), MGSM 87.73 (8-shot native CoT avg)."
  - "Base long context (RULER 0-shot): 64K 95.30, 128K 92.49, 256K 86.22, 512K 84.54, 1M 76.83."
  - "Post-trained evals collected via Nemo Evaluator SDK using three harnesses: Nemo Gym, Nemo Skills, and Harbor (extended sandboxing via AWS ECS), plus dedicated containers for Multi-Challenge multi-turn instruction following."
  - "Not yet onboarded in open-source tools (used official/internal scaffolding): BrowseComp, Tau Bench 3, ProfBench, PinchBench, Vals.ai Financial Agent, LongBench v2."
  - "PinchBench and ProfBench used as held-out generalization gates: not used for training-time monitoring, checkpoint selection, or development decisions; evaluated only once after the final model."
  - "Post-trained Nemotron 3 Ultra (550B-A55B): PinchBench 90.0 (within 1.3 pts of best); ProfBench 56.0 (ties Kimi-K2.6, a 1T-parameter model)."
  - "Post-trained Terminal Bench 2.1 56.4; GDPVal 46.7; SWE-Bench Verified 71.9; SWE-Bench Multilingual 67.7; TauBench V3 average 70.9; BrowseComp 44.4."
  - "Post-trained IOI 2025 score 570.0 (top-3-human-level, between 2nd and 3rd ranked official human contestants); LiveCodeBench v6 89.0."
  - "Post-trained IMOAnswerBench 88.6 (no tools) / 92.3 (with tools); Apex-Shortlist 74.9 (no tools) / 84.8 (with tools)."
  - "Post-trained GPQA 87.0 (no tools); MMLU-Pro 86.8; HLE 26.7 (no tools) / 37.4 (with tools); SciCode 44.6; CritPt 3.1 (no tools)."
  - "Post-trained AA-Omniscience: Accuracy 24.1, Non-Hallucination 78.7 (highest non-hallucination score in comparison)."
  - "Post-trained chat/IF: IFBench (prompt loose) 81.7; Multi-Challenge 63.8; Arena-Hard-V2 88.1 (Claude 4.6 Opus judge, style-controlled)."
  - "Post-trained long context: AA-LCR 65.4; RULER (1M) 94.7; LongBench v2 (<=1M) 61.9. Multilingual: MMLU-ProX 83.0; WMT24++ (en->xx) 83.7."
  - "Test-time scaling (Table 11, generate-verify-refine per Shao et al. 2025; 128 proof attempts/problem, 512k context): IMO-ProofBench Advanced 83.3% (175/210); IMO 2025 83.3% (35/42); Putnam 2025 97.5% (117/120); USAMO 2026 97.6% (41/42)."
  - "Harness robustness: five task verticals, each model trained under at least two of: Stirrup, OpenHands, OpenCode, Terminus, Droid, Custom Internal Harnesses (prevents single-harness overfitting)."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- How does the base model (550B-A55B-Base) compare to other open base models?
- What harnesses/benchmarks define the post-trained evaluation setup?
- What are the post-trained results across agentic, reasoning, chat, long-context, multilingual?
- What were the held-out generalization gate results (PinchBench, ProfBench)?
- What test-time scaling results were obtained on Olympiad math?
- How is harness robustness ensured?

# §2.6 Base Model Evaluations (Nemotron-3-Ultra-550B-A55B-Base)

Collected via Nemo Evaluator SDK + NVIDIA open-source LM Evaluation Harness container. Best available results bold in source Table 2. **These are BASE-model numbers — do not mix with post-trained Table 10.**

| Task | Metric | N3-Ultra-550B-A55B-Base | DeepSeek-V3.2-Exp-Base | Mistral-Large-3-675B-Base-2512 | Kimi-K2-Base | GLM-4.5-Base |
|---|---|---|---|---|---|---|
| MMLU | 5-shot, acc | 89.08 | 87.82 | 87.35 | 87.60 | 86.50 |
| MMLU-Pro | 5-shot, CoT EM | 79.07 | 63.26 | 67.42 | 69.15 | 65.78 |
| AGIEval-En | 3/5-shot, CoT EM | 78.73 | 70.13 | 69.30 | 72.55 | 70.06 |
| GPQA | 5-shot, CoT EM | 50.00 | 31.82 | 34.85 | 43.43 | 34.85 |
| GSM8K | 8-shot, CoT EM | 88.10 | 84.38 | 91.21 | 91.05 | 85.37 |
| MATH | 4-shot, EM | 82.00 | 60.12 | 62.88 | 68.40 | 57.58 |
| HumanEval | sampled pass@1 n=32, EvalPlus sanitized | 83.84 | 61.85 | 66.71 | 78.20 | 78.16 |
| MBPP-Sanitized | 3-shot pass@1 n=32, EvalPlus sanitized | 85.97 | 58.66 | 84.08 | 72.14 | 76.69 |
| ARC-Challenge | 25-shot, acc_norm | 97.35 | 95.22 | 97.27 | 95.82 | 96.59 |
| HellaSwag | 10-shot, acc_norm | 90.51 | 89.44 | 88.88 | 90.92 | 90.17 |
| OpenBookQA | 0-shot, acc_norm | 48.60 | 48.20 | 51.40 | 50.80 | 49.60 |
| PIQA | 0-shot, acc_norm | 83.79 | 85.09 | 84.82 | 85.47 | 85.09 |
| WinoGrande | 5-shot, acc | 79.32 | 83.43 | 82.08 | 84.21 | 85.24 |
| RACE | 0-shot, acc | 92.15 | 93.21 | 93.30 | 91.96 | 92.15 |
| MMLU Global Lite | 5-shot, avg | 90.13 | 85.59 | 87.34 | 85.63 | 85.81 |
| MGSM | 8-shot, native CoT avg | 87.73 | 82.33 | 82.93 | 85.20 | 81.27 |
| RULER 64K | 0-shot | 95.30 | 93.30 | 90.11 | 93.79 | 16.12 |
| RULER 128K | 0-shot | 92.49 | 91.88 | 55.77 | 88.61 | 0.00 |
| RULER 256K | 0-shot | 86.22 | – | 35.50 | – | – |
| RULER 512K | 0-shot | 84.54 | – | – | – | – |
| RULER 1M | 0-shot | 76.83 | – | – | – | – |

Notes: MATH evaluated as Minerva Math (4-shot EM). HumanEval/MBPP use EvalPlus-sanitized; pass@1 from 32 generations where available. Missing entries = result not available in comparison set.

# §3.7 Post-trained Model Evaluations

## §3.7.1 Evaluation Setup
- All post-trained results (Ultra + baselines) collected via Nemo Evaluator SDK.
- Three main harnesses: Nemo Gym, Nemo Skills, Harbor (extended sandboxing via AWS ECS on Nemo Evaluator); plus dedicated open-source containers for Multi-Challenge multi-turn instruction following.
- Not yet onboarded in open-source tools (official/internal scaffolding used): BrowseComp, Tau Bench 3, ProfBench, PinchBench, Vals.ai Financial Agent, LongBench v2.
- All models evaluated under identical settings (agentic resources, input data, prompt templates, repeats, extraction/metric). Temperature, top_p, max tokens taken from each model card.
- Benchmark domains: agentic, reasoning & knowledge, conversation & instruction following, long-context, multilingual (full list in §3.7.1 of source; see Appendix B).

## §3.7.2 Evaluation Results — Table 10 (post-trained, BF16)
**These are POST-TRAINED numbers (six open-model comparison). No NVFP4 post-trained numbers appear in this slice.**

Columns: N-3-Ultra 550B-A55B | MiniMax-2.7 230B-A10B | GLM-5.1 744B-A40B | Kimi-K2.6 1T-A32B | Qwen-3.5 397B-17B | DS-v4-Pro 1.6T-A49B | DS-v4-Flash 284B-A13B

| Benchmark | N-3-Ultra | MiniMax-2.7 | GLM-5.1 | Kimi-K2.6 | Qwen-3.5 | DS-v4-Pro | DS-v4-Flash |
|---|---|---|---|---|---|---|---|
| Terminal Bench 2.1 | 56.4 | 55.5 | 59.3 | 67.2 | 49.9 | 49.2 | 54.2 |
| GDPVal | 46.7 | 47.6 | 54.7 | 50.4 | 34.6 | 54.6 | 50.2 |
| SWE-Bench Verified | 71.9 | 72.2 | 73.8 | 69.5 | 69.9 | 74.0 | 72.4 |
| SWE-Bench Multilingual | 67.7 | 69.2 | 73.8 | 65.9 | 67.7 | 71.9 | 72.1 |
| ProfBench (Search) | 56.0 | 52.0 | 46.0 | 56.0 | 53.0 | 59.9 | 57.0 |
| PinchBench | 90.0 | 77.6 | 81.2 | 90.2 | 86.6 | 88.6 | 91.3 |
| TauBench V3 Airline | 81.5 | 75.3 | 85.0 | 85.8 | 76.5 | 80.8 | 80.8 |
| TauBench V3 Retail | 86.4 | 84.9 | 84.1 | 82.9 | 88.5 | 88.9 | 89.1 |
| TauBench V3 Telecom | 92.9 | 89.6 | 96.9 | 97.8 | 98.0 | 96.3 | 98.3 |
| TauBench V3 Banking | 22.6 | 14.6 | 12.8 | 23.1 | 20.9 | 25.9 | 26.7 |
| TauBench V3 Average | 70.9 | 66.1 | 69.7 | 72.4 | 71.0 | 73.2 | 73.7 |
| BrowseComp | 44.4 | 54.1 | 59.4 | 61.3 | 40.5 | 59.4 | 46.9 |
| Vals.ai Financial Agent 1.1 (no web search) | 60.1 | 51.3 | 60.2 | 54.0 | 61.3 | 58.9 | 58.4 |
| Vals.ai Financial Agent 1.1 (with web search) | 53.7 | 50.5 | 60.7 | 58.8 | 59.0 | 62.3 | 60.1 |
| IOI 2025 | 570.0 | – | 456.5 | 585.0 | 441.3 | 580.1 | – |
| LiveCodeBench (v6) | 89.0 | 77.2 | 85.7 | 90.2 | 79.3 | 92.5 | 90.9 |
| IMOAnswerBench (no tools) | 88.6 | 68.3 | 86.8 | 91.1 | 83.1 | 93.0 | 91.1 |
| IMOAnswerBench (with tools) | 92.3 | 75.1 | 91.1 | 93.71 | 84.51 | 85.4 | 89.6 |
| Apex-Shortlist (no tools) | 74.9 | 28.9 | 71.1 | 77.4 | 61.4 | 85.8 | 82.4 |
| Apex-Shortlist (with tools) | 84.8 | 51.9 | 79.0 | 73.2 | 60.4 | 86.5 | 82.0 |
| GPQA (no tools) | 87.0 | 86.6 | 86.1 | 91.0 | 87.1 | 87.8 | 88.5 |
| SciCode (subtask) | 44.6 | 38.3 | 47.7 | 52.0 | 48.0 | 50.5 | 48.2 |
| HLE (no tools) | 26.7 | 23.1 | 27.2 | 34.8 | 28.5 | 37.7 | 32.2 |
| HLE (with tools) | 37.4 | – | 50.4 | 54.0 | 48.3 | 48.2 | 45.1 |
| CritPt (no tools) | 3.1 | 0.6 | 3.7 | 9.1 | 2.4 | 14.0 | 10.6 |
| MMLU-Pro | 86.8 | 81.9 | 85.9 | 88.1 | 88.3 | 87.5 | 86.4 |
| OmniScience Accuracy | 24.1 | 20.5 | 31.3 | 35.5 | 35.9 | 46.8 | 39.9 |
| OmniScience Non-Hallucination | 78.7 | 74.4 | 66.8 | 67.1 | 7.4 | 5.7 | 2.8 |
| IFBench (prompt loose) | 81.7 | 74.6 | 76.6 | 73.7 | 78.2 | 79.1 | 82.0 |
| Multi-Challenge | 63.8 | 42.5 | 63.0 | 63.1 | 63.9 | 64.1 | 63.5 |
| Arena-Hard-V2 | 88.1 | 66.0 | 78.7 | 85.6 | 81.5 | 78.1 | 77.0 |
| AA-LCR | 65.4 | 69.8 | 66.9 | 70.2 | 68.3 | 67.3 | 62.7 |
| RULER (1M) | 94.7 | – | – | – | 90.1 | 94.2 | 87.7 |
| Longbench v2 (<=1M) | 61.9 | – | – | – | 68.9 | 62.1 | 57.0 |
| MMLU-ProX (avg 10 langs) | 83.0 | 78.4 | 85.8 | 85.0 | 86.4 | 85.6 | 84.3 |
| WMT24++ (en->xx) | 83.7 | 82.8 | 84.4 | 84.5 | 86.8 | 85.9 | 85.9 |

MMLU-ProX languages: en/de/fr/es/it/ja/zh/hi/pt/ko. Arena-Hard-V2 judged by Claude 4.6 Opus, style-controlled scores.

Key claims: Ultra is agentic-first and well-rounded, competitive with larger open models. Held-out gates: PinchBench 90.0 (within 1.3 pts of best), ProfBench 56.0 (ties Kimi-K2.6, a 1T-param model). IOI 2025 570.0 = top-3-human-level (between 2nd and 3rd ranked human contestants). IMOAnswerBench with tools 92.3. AA-Omniscience non-hallucination 78.7 = highest in comparison.

## §3.7.3 Test-time Scaling in Math Olympiad Problems (Table 11)
Generate-verify-refine methodology (Shao et al., 2025), evaluated on IMO-ProofBench Advanced subset, IMO 2025, Putnam 2025, USAMO 2026. Started with 128 proof attempts per problem, allowed 512k context length; all other pipeline hyperparameters identical to the original paper. Accuracy reported with graded score in parentheses (human expert graders, except USAMO 2026 which follows Dekoninck et al. 2026).

| Competition | Accuracy |
|---|---|
| IMO-ProofBench Advanced | 83.3% (175/210) |
| IMO 2025 | 83.3% (35/42) |
| Putnam 2025 | 97.5% (117/120) |
| USAMO 2026 | 97.6% (41/42) |

Figure 12 plots accuracy vs rounds of refinement (proxy for compute), comparing against SOTA Aletheia (Gemini Deep Think) math research agent (Feng et al., 2026).

## §3.7.4 Harness Robustness
- Five task verticals: Zero-to-One Terminal Use & Software Engineering; Existing Repo Bug Fixing; Office/General Productivity; General/Multi-domain Knowledge; Search.
- Each input task distribution: model trained under at least two of: Stirrup, OpenHands, OpenCode, Terminus, Droid, Custom Internal Harnesses.
- Rationale: avoiding single-harness training per task distribution improves generalization/robustness under dynamic real-world execution contexts. Figure 13 shows agent x model matrices for SWE-bench Verified and Terminal-Bench 2.1.

# Caveats
- §2.6 base numbers and §3.7 post-trained numbers are distinct; never merge the two tables.
- This slice reports BF16 post-trained results; no NVFP4 post-trained benchmark numbers appear here (do not claim NVFP4 eval results from this chunk).
- "Best" markings in base Table 2 are per the source; this chunk lists raw values without re-deriving bolding.
- Source notes IMOAnswerBench-with-tools values 93.71 (Kimi-K2.6) and 84.51 (Qwen-3.5) verbatim with trailing digits; copied as-is.
- §3.7.3 carried a "[TODO: ... results ... ETA 9 AM PST Jun 3]" editorial note in the source; Table 11 values are as printed and may be provisional.
- Comparison-model sizes (e.g., Kimi-K2.6 1T-A32B, DS-v4-Pro 1.6T-A49B) are from the Table 10 header; Ultra is 550B-A55B.
