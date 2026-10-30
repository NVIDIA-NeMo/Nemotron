---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "evaluation"
paper_sections: ["2.3", "2.6", "4.8", "Table 3", "Table 6", "Table 21"]
title: "Nemotron 3.5 Lightning Evaluation: Base (Table 6), BF16 and NVFP4 Post-Trained (Table 21)"
summary: |
  Every Lightning benchmark number in the report, labelled by checkpoint. Base model: Table 6 (24
  lm-eval-harness tasks via NeMo Evaluator, e.g. MMLU 78.63, MMLU-Pro 67.50, GSM8K 91.74). Post-trained
  BF16 and NVFP4: Table 21 (18 NeMo Gym benchmarks, e.g. SWE-Bench Verified 52.20 / 52.20, GPQA Diamond
  76.45 / 75.38, RULER-v1 1M 91.52 / 90.83). These are the report's only post-trained Lightning scores:
  Table 10 and Figure 1 cover Super only, and there are no comparisons with other models. Also lists Table
  3's research Indic ablations (not the released model) and the repo's eval setup. The repo docs now
  reproduce Table 6 and Table 21 (README Key Results, evaluate.md).
key_facts:
  - "Checkpoints (§1): Nemotron 3.5 Lightning 30B-A3B Base BF16 (base), 30B-A3B BF16 (post-trained), 30B-A3B NVFP4 (post-trained, NVFP4-quantized)."
  - "Base (Table 6, Lightning 3.5 30B-A3B Base): MMLU 78.63; MMLU-Pro 67.50; AGIEval-En 68.62; SuperGPQA 33.21; GPQA Diamond (5-shot CoT) 33.33; GSM8K 91.74; Minerva MATH 83.10; MATH-500 79.53; HumanEval 76.91; MBPP-Sanitized 76.98; LiveCodeBench 29.77; BigCodeBench Complete 50.09 / Instruct 51.75; CommonsenseQA 81.08; ARC-C 93.17; HellaSwag 85.25; OpenBookQA 47.80; PIQA 85.09; TruthfulQA MC2 55.53; WinoGrande 78.06; SocialIQA 47.80; RACE 87.85; Global MMLU (8 langs) 75.09; MGSM (6 langs) 82.27."
  - "Table 6 oddities (as printed in the PDF): Lightning GSM8K 91.74 and PIQA 85.09 equal Super's values exactly; Lightning beats Super Base on BigCodeBench Complete (50.09 vs 48.42) and Instruct (51.75 vs 49.39)."
  - "Base eval harness (§2.6): NeMo Evaluator SDK + NVIDIA's LM Evaluation Harness container; pass@1 from 32 samples where marked."
  - "Post-trained (Table 21, BF16 / NVFP4 W4A16): SWE-Bench Verified 52.20 / 52.20; SWE-Bench Multilingual 37.13 / 37.00; Tau2 Telecom 59.65 / 60.96; Tau3 Banking 9.90 / 9.48; Terminal-Bench v2.1 24.44 / 22.05; GDPval 17.19 / 19.09; PinchBench 83.86 / 84.78; BrowseComp 39.50 / 39.50; SciCode (subtask) 32.74 / 31.53; GPQA Diamond (no tools) 76.45 / 75.38; HLE (no tools) 10.80 / 10.70; AA-Omniscience Accuracy 17.47 / 16.35; AA-Omni Non-Hallucination 68.88 / 69.20; CritPt 0.00 / 0.00; AA-LCR 52.38 / 50.00; RULER-v1 (1M) 91.52 / 90.83; IFBench (loose) 72.00 / 74.04; LM Arena Proxy 12.06 / 11.10."
  - "§4.8: Lightning is evaluated with the NeMo Gym framework used for Super; both BF16 and NVFP4 are served with FP8 KV cache and FP32 Mamba SSM cache. Median accuracy recovery 99.24%, verbosity (ratio of totals) -3.63%."
  - "Footnote 19 (§4.8): scores vary run to run (stochastic sampling, judge variability) 'and may therefore differ from those reported on the model cards'."
  - "Not reported for Lightning: any comparison with other models; Table 10-style suite (SWE-bench Pro, MMLU-Pro post-trained, multilingual, NV Chat Benchmark, etc.); reasoning-off or budget-controlled results; agentic harness comparisons (§3.10) and test-time scaling (§3.11), which are Super only."
  - "Table 3 (research ablations, not the released model, §2.3): 200B-token runs with 10/20/30% Indic data; four-benchmark mean (MILU, MMLU-Pro, ARC-Challenge, GSM8K) 63.1 / 64.6 / 65.6 vs base reference 45.5. 'The released Lightning 3.5 model ... did not include this Indic CPT stage.'"
  - "Appendix C.3: MRCR-128k disambiguation loss under quantization 'is also observed on Nemotron 3.5 Lightning'; on Lightning, moving the KV cache FP8 -> BF16 recovers +1.43 pp (2.4 sigma) and moving projections FP8 -> BF16 recovers +2.97 pp (2.8 sigma). Lightning's MRCR-128k scores themselves are not reported."
  - "Repo eval recipe (evaluate.md, stage3_eval): entries for 10 NeMo Gym benchmarks (gpqa, scicode, hle, aalcr, omniscience, browsecomp, tau2, critpt, gdpval, pinchbench); only gpqa is enabled by default. The docs say this is a subset of Table 21, which also covers SWE-Bench Verified/Multilingual, Tau3 Banking, Terminal-Bench v2.1, RULER-v1 1M, IFBench and LM Arena Proxy. (The stage README lists Tau3 where evaluate.md and the config list tau2.)"
  - "Repo docs (evaluate.md): base model evaluated with a short-context lm-evaluation-harness suite plus RULER via nemo-evaluator-launcher (configs in the Gym repo). Table 6: 24 short-context tasks, no RULER. The Base model card reports RULER 256K 76.88 and 1M 69.62 (NeMo Skills harness)."
  - "Repo docs reproduce Table 21 (evaluate.md, README Key Results) and Table 6 (evaluate.md), citing the report."
related_steps: []
currency: "frozen"
---

# Scope

- What base-model scores does the report give for Lightning?
- What are the post-trained BF16 and NVFP4 scores?
- What is missing (competitor comparisons, Table 10-style suite)?
- How does the repo's eval stage relate to the published numbers?

# Base model: Lightning 3.5 30B-A3B Base (Table 6, §2.6)

Full table, metrics and settings are in `evaluation-base.md`. Headline base numbers (NeMo Evaluator SDK +
LM Evaluation Harness container): MMLU 78.63, MMLU-Pro 67.50, GPQA Diamond (5-shot CoT) 33.33, GSM8K 91.74,
MATH-500 79.53, HumanEval 76.91, LiveCodeBench 29.77, BigCodeBench Complete / Instruct 50.09 / 51.75,
Global MMLU 75.09, MGSM 82.27. GSM8K and PIQA are identical to Super Base's values in the PDF (page 11);
the report does not comment. Do not compare base GPQA (5-shot CoT, 33.33) with post-trained GPQA
(no tools, 76.45): different checkpoints and protocols.

# Post-trained: BF16 and NVFP4 (Table 21, §4.8)

Evaluated with NeMo Gym. Both checkpoints served with FP8 KV cache and FP32 Mamba SSM cache. NVFP4 is the
W4A16 configuration (see quantization.md). Accuracy below; token counts and recoveries are in
quantization.md.

| Category | Benchmark | BF16 | NVFP4 |
| --- | --- | --- | --- |
| Agentic | SWE-Bench Verified | 52.20 | 52.20 |
| Agentic | SWE-Bench Multilingual | 37.13 | 37.00 |
| Agentic | Tau2 Telecom | 59.65 | 60.96 |
| Agentic | Tau3 Banking | 9.90 | 9.48 |
| Agentic | Terminal-Bench v2.1 | 24.44 | 22.05 |
| Agentic | GDPval | 17.19 | 19.09 |
| Agentic | PinchBench | 83.86 | 84.78 |
| Agentic | BrowseComp | 39.50 | 39.50 |
| Coding | SciCode (subtask) | 32.74 | 31.53 |
| Reasoning & knowledge | GPQA Diamond (no tools) | 76.45 | 75.38 |
| Reasoning & knowledge | HLE (no tools) | 10.80 | 10.70 |
| Reasoning & knowledge | AA-Omniscience Accuracy | 17.47 | 16.35 |
| Reasoning & knowledge | AA-Omni Non-Hallucination | 68.88 | 69.20 |
| Reasoning & knowledge | CritPt | 0.00 | 0.00 |
| Long context | AA-LCR | 52.38 | 50.00 |
| Long context | RULER-v1 (1M) | 91.52 | 90.83 |
| IF & chat | IFBench (loose) | 72.00 | 74.04 |
| IF & chat | LM Arena Proxy | 12.06 | 11.10 |

Summary rows: median recovery 99.24%; verbosity change (ratio of totals) -3.63%. Values re-checked against
the PDF (page 41).

Harness details for these benchmarks (SWE harness, Terminal-Bench timeout, tau user simulators, GDPval
committee, BrowseComp harness) are given in §3.8.1 for Super. §4.8 only says Lightning uses "the NeMo Gym
evaluation framework described for Super 3.5" with Lightning-specific settings in the Gym recipes folder,
so assume the same harnesses but expect differences in detail.

# Not reported for Lightning

- Comparison with any other model (Table 10 and Figure 1 are Super only).
- SWE-bench Pro, MMLU-Pro (post-trained), multilingual, NV Chat Benchmark, MRCR scores, vision (Lightning is text-only).
- Reasoning-off / medium-effort / budget-control results (§3.9 is Super only).
- Agentic-harness robustness (§3.10) and test-time scaling (§3.11).
- Any safety benchmark (see safety.md).
- Whether §3.12 benchmark decontamination was applied to Lightning's post-training data.

# Research ablations, not the released model (Table 3, §2.3)

Indic mixture ablations (10/20/30% Indic, 200B tokens each; four-benchmark mean 63.1 / 64.6 / 65.6 vs base
reference 45.5) are research runs; the released Lightning model "did not include this Indic CPT stage".
Details in `data.md`.

# Repo evaluation setup (model card / repo docs)

- Stage `stage3_eval`: serves the HF checkpoint with vLLM (TP=4, expert parallel, `nemotron_v3` reasoning
  parser, `qwen3_coder` tool parser, FP32 Mamba SSM cache, `kv_cache_dtype: auto` with a note that
  "published runs use fp8 on GB200", MTP speculation with 5 tokens), then runs `gym eval` per benchmark.
- Default config enables only `gpqa`; the others need judge models, API keys or extra data.
- Base model: evaluate.md says a short-context lm-eval-harness suite plus RULER via nemo-evaluator-launcher,
  and reproduces Table 6 (24 tasks, no RULER). Base RULER scores appear only on the Base model card.
- evaluate.md reproduces Table 21 (BF16 and W4A16 NVFP4, 99.24% median recovery, -3.63% verbosity).

# Contradictions

| Topic | Report | Repo |
| --- | --- | --- |
| Instruct benchmark list | Table 21: 18 benchmarks incl. SWE-Bench, Terminal-Bench, RULER, IFBench, LM Arena Proxy, Tau2 Telecom + Tau3 Banking | eval recipe: 10 benchmarks (no SWE-Bench, Tau3 Banking, Terminal-Bench, RULER, IFBench, LM Arena Proxy); docs say so |
| Base long context | no RULER in Table 6 | Base model card: RULER 256K 76.88, 1M 69.62 (NeMo Skills) |
| Model-card scores | Footnote 19: Table 21 may differ from model cards | docs quote the report's numbers and repeat the footnote |

Earlier docs versions claimed "up to 3.3x higher inference throughput" and RULER wins over GPT-OSS-20B and Qwen3-30B-A3B (copied from the Nano 3 README) and said "21 short-context tasks". Those were removed.
