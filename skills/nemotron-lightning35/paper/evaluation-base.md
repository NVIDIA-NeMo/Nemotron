---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "evaluation-base"
paper_sections: ["2.3", "2.6"]
title: "Nemotron 3.5 Lightning Base-Model Evaluation: Table 6 (Lightning 3.5 30B-A3B Base)"
summary: |
  Base-model results for Nemotron 3.5 Lightning 30B-A3B Base from §2.6 Table 6 (report p.11), re-checked
  against the PDF layout. The table's only other column is Super 3.5 120B-A12B Base; there are no external
  base models. Results were collected with NeMo Evaluator SDK and NVIDIA's LM Evaluation Harness
  container. Highlights: MMLU 78.63, MMLU-Pro 67.50, GPQA Diamond 33.33, GSM8K 91.74, MATH-500 79.53,
  HumanEval 76.91, LiveCodeBench 29.77, Global MMLU 75.09, MGSM 82.27. No RULER or long-context base
  score is reported. The repo docs (evaluate.md) reproduce Table 6. The Base model card has its own table
  (slightly different numbers, e.g. MMLU 78.59) plus RULER 256K 76.88 and 1M 69.62.
key_facts:
  - "BASE (Table 6): General knowledge: MMLU 78.63, MMLU-Pro 67.50, AGIEval-En 68.62, SuperGPQA 33.21, GPQA Diamond 33.33 (5-shot CoT)."
  - "BASE (Table 6): Math: GSM8K 91.74 (8-shot CoT), Minerva MATH 83.10, MATH-500 79.53 (4-shot, pass@1, n=32)."
  - "BASE (Table 6): Code: HumanEval 76.91, MBPP-Sanitized 76.98 (both sampled pass@1, n=32, EvalPlus), LiveCodeBench CodeGen 29.77, BigCodeBench Complete 50.09, Instruct 51.75 (greedy)."
  - "BASE (Table 6): Commonsense: CommonsenseQA 81.08, ARC-Challenge 93.17, HellaSwag 85.25, OpenBookQA 47.80, PIQA 85.09, TruthfulQA MC2 55.53, WinoGrande 78.06, SocialIQA 47.80; RACE 87.85."
  - "BASE (Table 6): Multilingual: Global MMLU 75.09 (8 languages), MGSM 82.27 (6 languages, 8-shot native CoT)."
  - "Lightning Base beats Super 3.5 Base only on BigCodeBench Complete (50.09 vs 48.42) and Instruct (51.75 vs 49.39); it ties on GSM8K (91.74) and PIQA (85.09) as printed."
  - "Harness: NeMo Evaluator SDK + NVIDIA's open-source LM Evaluation Harness container, unless otherwise stated (§2.6)."
  - "No base long-context (RULER) score is reported for Lightning in the report. The Base model card gives RULER 256K 76.88 and RULER 1M 69.62 (NeMo Skills harness)."
  - "Indic research ablations (Table 3) report a 'base reference' four-benchmark Indic mean of 45.5; this is not a Table 6 metric (see data.md)."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- How does Lightning 3.5 30B-A3B Base score on standard base-model benchmarks?
- Which shots, metrics, and harness were used?
- How does it compare with Super 3.5 Base, the only other column?
- Are there model-card or repo-doc base numbers, long-context base scores, or comparisons with other base models?

Precision / stage label: every number here is **BASE** (pre-trained Lightning 3.5 Base BF16 checkpoint). Post-trained BF16 and NVFP4 results (Table 21, §4.8) belong to a separate post-training evaluation chunk.

# Setup (§2.6)
- "All evaluation results reported for Nemotron 3.5 Super 120B-A12B Base and Lightning 3.5 30B-A3B Base were collected via NeMo Evaluator SDK and NVIDIA's open-source container of LM Evaluation Harness, unless otherwise stated." Configs are in the NeMo Evaluator SDK examples, and the container is on NGC.
- Sampled code and math benchmarks estimate pass@1 from 32 generations per prompt where indicated.
- Per-category settings (report list):
  - GPQA Diamond: 5-shot CoT.
  - GSM8K: 8-shot CoT exact match. Minerva MATH: exact match. MATH-500: 4-shot, 32 samples.
  - HumanEval: 0-shot. MBPP-Sanitized: 3-shot. Both EvalPlus-sanitized.
  - Global MMLU: de, es, fr, it, ja, ko, pt, zh. MGSM: es, de, fr, zh, ja, ru, 8-shot native CoT, average exact match.
- No external base models (e.g. Qwen3-30B-A3B-Base, Nano 3 Base) appear in Table 6. The caption says "A dash indicates that a corresponding result is not available", but no dashes are printed.

# Table 6 — Lightning 3.5 30B-A3B Base (report p.11; re-checked with `pdftotext -layout`)

| Task | Metric | Lightning 3.5 Base | (Super 3.5 Base, reference only) |
|---|---|---|---|
| **General knowledge** | | | |
| MMLU | exact match | 78.63 | 84.14 |
| MMLU-Pro | exact match, TigerLab extract | 67.50 | 76.25 |
| AGIEval-En | CoT exact match, custom extract | 68.62 | 75.65 |
| SuperGPQA | CoT | 33.21 | 47.65 |
| GPQA Diamond | 5-shot, CoT exact match | 33.33 | 66.16 |
| **Math** | | | |
| GSM8K | 8-shot, CoT exact match | 91.74 | 91.74 |
| Minerva MATH | exact match | 83.10 | 87.66 |
| MATH-500 | 4-shot, sampled pass@1, n=32 | 79.53 | 81.56 |
| **Code** | | | |
| HumanEval | sampled pass@1, n=32, EvalPlus sanitized (0-shot) | 76.91 | 81.31 |
| MBPP-Sanitized | 3-shot, sampled pass@1, n=32, EvalPlus sanitized | 76.98 | 79.06 |
| LiveCodeBench CodeGen | pass@1 | 29.77 | 47.61 |
| BigCodeBench Complete | greedy | 50.09 | 48.42 |
| BigCodeBench Instruct | greedy | 51.75 | 49.39 |
| **Commonsense understanding** | | | |
| CommonsenseQA | 7-shot, acc | 81.08 | 85.26 |
| ARC-Challenge | 25-shot, acc_norm | 93.17 | 95.99 |
| HellaSwag | 10-shot, acc_norm | 85.25 | 88.96 |
| OpenBookQA | 0-shot, acc_norm | 47.80 | 48.80 |
| PIQA | 0-shot, acc_norm | 85.09 | 85.09 |
| TruthfulQA MC2 | 0-shot, acc | 55.53 | 71.18 |
| WinoGrande | 5-shot, acc | 78.06 | 79.08 |
| SocialIQA | 0-shot, acc | 47.80 | 49.03 |
| **Reading comprehension** | | | |
| RACE | 0-shot, acc | 87.85 | 91.39 |
| **Multilingual** | | | |
| Global MMLU | avg. of 8 languages | 75.09 | 86.56 |
| MGSM | 8-shot native CoT, avg. of 6 languages | 82.27 | 87.53 |

Observations (from the printed values):
- Lightning trails Super 3.5 Base on 20 of 24 rows. It ties on GSM8K and PIQA and leads only on both BigCodeBench variants.
- The largest gaps are GPQA Diamond (-32.83), LiveCodeBench (-17.84), TruthfulQA MC2 (-15.65), SuperGPQA (-14.44), and Global MMLU (-11.47).
- Lightning's OpenBookQA and SocialIQA are both printed as 47.80.
- GPQA Diamond 33.33 equals 66/198 (GPQA Diamond has 198 questions). This is arithmetic, not a report statement.

# Not Reported
- **Long-context base scores** (RULER etc.) for Lightning: none in the report. The RULER-1M figures in §1/§2.4 (79.9 -> 33.5 -> 81.0) are Super's. The Base model card gives RULER 256K 76.88 and 1M 69.62.
- **Base vs other open base models** (Nano 3 Base, Qwen3-30B-A3B-Base): not in the report. Lightning's own initialization (Nano 3 Base) is not shown, so the report gives no direct measure of what MTP warm-up + 1T distillation + LC added.
- **Model-card / repo-doc base numbers:** docs evaluate.md reproduces Table 6 and says the repo's base evaluation uses a short-context lm-evaluation-harness suite plus RULER via nemo-evaluator-launcher (configs in the NeMo Gym repo under `scripts/more/base/`). The Base model card has its own benchmark table with comparison models; its numbers differ slightly from Table 6 (e.g. MMLU 78.59 vs 78.63).
- **Indic base numbers:** Table 3's "Base reference" four-benchmark Indic mean is 45.5 (MILU, MMLU-Pro, ARC-Challenge, GSM8K). The report does not identify that checkpoint exactly; see data.md.

# Caveats
- Do not mix these BASE numbers with Lightning's post-trained Table 21 numbers (§4.8, NeMo Gym, BF16 vs W4A16 NVFP4) or with Super's Table 10.
- Never cite the Super column for Lightning or vice versa. In the Super skill this table is reproduced with Lightning as the reference column.
- Earlier versions of the recipe README claimed "better or on-par accuracy than competitive models", "up to 3.3x higher inference throughput", and "outperforming both GPT-OSS-20B and Qwen3-30B-A3B on RULER". That text was Nano 3's and has been removed; it is not a Lightning result.
- §4.8 footnote: evaluation results are subject to run-to-run variation, and "Scores may therefore differ from those reported on the model cards". It is written for the post-trained tables, but it is the report's only statement on model-card score agreement.
