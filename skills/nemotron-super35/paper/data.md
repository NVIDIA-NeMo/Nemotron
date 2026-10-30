---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "data"
paper_sections: ["2.3", "2.5"]
title: "Nemotron 3.5 Super Pretraining Data: Nemotron-Pretraining-Indic-v1, Indic CPT Mixture, Agentic Mid-training Blend"
summary: |
  Covers the two data contributions of Super 3.5's base-model training. First, Nemotron-Pretraining-Indic-v1:
  ~2.83T tokens of synthetic Indic text translated from (mostly Nemotron-CC) English documents across 22
  Indian languages and 34 language-script configurations, built with a four-stage NeMo Data Designer
  pipeline (Gemma-4-31B-it translate, score, rewrite below 75, re-check). Super's 100B-token Indic CPT uses
  10% Indic data and improves five Indic benchmarks by +4.98 to +9.82 points (Table 2). Second, the Table 4
  agentic software-engineering mid-training blend (issue-PR pairs, repositories, agent traces, commits, CI
  logs, replay).
key_facts:
  - "Nemotron-Pretraining-Indic-v1: ~2.83T (2,825.6B) generated output tokens, 22 Indian languages, 34 language-script configurations, counted with a Gemma-4 tokenizer (Table 1)."
  - "12 priority languages (native + romanized): ~2.50T tokens incl. 224.6B romanized; 10 additional lower-resource languages (native only): ~327B tokens."
  - "English sources mostly from Nemotron-CC (general web, education, math, STEM, QA); documents < 4,096 Gemma-4 tokens, translated whole without splitting."
  - "Four-stage NeMo Data Designer pipeline (priority languages), all with Gemma-4-31B-it: translate -> quality score 0-100 -> rewrite if < 75 -> final check; keep if >= 75."
  - "Additional 10 languages: Bodhan-MT translation, Gemma-4-31B-it quality assessment, language-specific thresholds."
  - "Translation cost cut from 0.835 to 0.335 GPU-hours per million English input tokens (~60%, 2.49x tokens per GPU-hour) via speculative decoding, FP8 weights, FlashAttention."
  - "Super Indic CPT: nominal 100B tokens = 10% (10B) synthetic + natural Indic, 90% (90B) preceding mixture proportions."
  - "Table 2 gains (before -> after Indic CPT): MILU 72.48 -> 77.90, Indic MMLU-Pro 51.83 -> 56.81, Indic ARC-C 84.17 -> 92.83, Indic ARC-E 90.73 -> 96.77, Indic GSM8K strict 72.67 -> 82.49."
  - "IGB/FLORES chrF++ (12 languages): En->Indic 35.43 -> 47.47; Indic->En 58.75 -> 64.30."
  - "Lightning 3.5 ablations (200B tokens, separate from released Lightning): 10/20/30% Indic share -> four-benchmark mean 63.1 / 64.6 / 65.6 (base reference 45.5)."
  - "Mid-training blend (Table 4): Issue-PR 41%, Repositories 20%, Agentic traces 10%, Commit diffs 3%, Build/CI logs 1%, Pretraining replay 20%, Long-context data 5%."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What is Nemotron-Pretraining-Indic-v1 (languages, scripts, token counts, sources)?
- How is the synthetic Indic data generated and filtered (Figure 2 pipeline)?
- How much Indic data did Super's CPT use, and what did it improve (Table 2, Figure 3)?
- What do the Lightning mixture ablations say, and why are they not Super results?
- What is in the agentic mid-training data blend (Table 4)?
- What is planned for release?

# §2.3 Indic Data: Motivation, Coverage, Scale
- Natural Indic text is scarce in many languages, and coverage varies across languages, scripts, and subjects. The approach is to translate curated English documents into Indic languages.
- English sources: mostly **Nemotron-CC**, covering general web content, education, mathematics, STEM, and question answering. Source documents are kept so the output can be checked for preserved meaning and structure.
- **22 Indian languages**: native-script **and** romanized text for 12 priority languages, plus native-script text for 10 additional lower-resource languages, giving **34 language-script configurations**.
- **Nemotron-Pretraining-Indic-v1** total: ~**2.83T** generated tokens (2,825.6B, Gemma-4 tokenizer). Priority languages ~2.50T (incl. 224.6B romanized). Additional languages ~327B native-script.
- The corpus is much larger than Super's Indic CPT used. The report says the broader corpus will be released publicly.

## Table 1: Token counts (billions, Gemma-4 tokenizer)
Input counts may overlap across configurations; they are not unique source volume or total pipeline processing. Subtotals sum the displayed rounded values. The output total 2,825.6B is the corpus size, **not** the number of tokens used to train Super.

| Priority language | Native in | Native out | Romanized in | Romanized out |
|---|---|---|---|---|
| Bengali | 114.0 | 138.9 | 12.2 | 19.0 |
| Gujarati | 115.5 | 196.4 | 12.0 | 17.2 |
| Hindi | 118.1 | 149.3 | 12.8 | 18.4 |
| Kannada | 103.5 | 180.5 | 11.3 | 19.1 |
| Malayalam | 108.9 | 183.5 | 11.4 | 19.3 |
| Marathi | 114.4 | 152.8 | 11.6 | 17.6 |
| Nepali | 118.0 | 171.3 | 12.4 | 19.7 |
| Odia | 109.3 | 320.1 | 12.2 | 19.0 |
| Punjabi | 112.3 | 262.6 | 12.0 | 17.6 |
| Tamil | 110.5 | 157.5 | 10.9 | 20.0 |
| Telugu | 115.3 | 192.4 | 12.2 | 18.8 |
| Urdu | 117.8 | 168.5 | 12.8 | 18.9 |
| **Subtotal** | **1,357.6** | **2,273.8** | **143.8** | **224.6** |

| Additional language (native only) | Input | Output |
|---|---|---|
| Assamese | 16.3 | 31.1 |
| Bodo | 8.6 | 18.7 |
| Dogri | 23.4 | 39.9 |
| Kashmiri | 11.4 | 33.1 |
| Konkani | 16.7 | 28.9 |
| Maithili | 20.0 | 31.3 |
| Manipuri | 6.6 | 50.1 |
| Sanskrit | 12.6 | 22.8 |
| Santali | 8.3 | 40.8 |
| Sindhi | 14.6 | 30.5 |
| **Subtotal** | **138.5** | **327.2** |

Check: 2,273.8 + 224.6 + 327.2 = 2,825.6B.

# Synthetic-Data Pipeline (§2.3, Figure 2)

![Four-stage Indic synthetic-data pipeline built with NeMo Data Designer for the twelve priority languages (tech report Figure 2).](../../../docs/assets/super35/indic-sdg-pipeline.png)

Built with **NeMo Data Designer** for the 12 priority languages. **Gemma-4-31B-it** (chosen from translation-model comparisons) does translation, quality assessment, and rewriting.

| Stage | Model | What happens |
|---|---|---|
| Source selection | — | English docs (primarily Nemotron-CC) with **< 4,096 Gemma-4 tokens**, translated **in full, no splitting**; prompts preserve facts, math expressions, code, formatting; source metadata retained |
| 1. Document translation | Gemma-4-31B-it | Native-script candidate |
| 2. Translation quality check | Gemma-4-31B-it | Score q in 0-100 and list errors; **q >= 75: keep** |
| 3. Targeted rewrite (if q < 75) | Gemma-4-31B-it | Rewrite using source + translation + identified errors |
| 4. Final quality check | Gemma-4-31B-it | Re-score revised text; q >= 75 keep, otherwise discard |
| Romanization | Gemma-4-31B-it (script-specific prompts) | Starts from accepted native-script text; repeats stages 2-4 for Latin script, checking meaning and script |
| Human review | — | Sampled outputs across languages and domains, used to improve prompts and checks; does not cover every document |

- Accepted outputs keep source ID, language, script, text, and quality metadata for traceability.
- The report notes a caveat: using the same model to translate and judge may favor its own translations.
- **10 additional languages:** **Bodhan-MT** translates and Gemma-4-31B-it assesses quality, with language-specific acceptance thresholds. Overlapping windows are used only for longer Santali documents. Manipuri translations are converted from Bengali script to Meitei Mayek. Details are deferred to the companion report *Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs*.
- **Efficiency (12-language pipeline):** scheduling lets translation and quality checks run concurrently. Speculative decoding, FP8 weights, and FlashAttention cut median translation cost from **0.835 to 0.335 GPU-hours per million English input tokens** (~60% reduction, 2.49x tokens per GPU-hour). Measured on the same 12-language benchmark with ~46.8M input tokens per language. The saving covers translation only; quality checks and rewriting add cost.

# Super Indic CPT Mixture (§2.3)
- Nominal **100B-token** budget: **10% (10B tokens)** from selected synthetic text from the corpus **plus natural Indic text**, and **90% (90B tokens)** keeping the preceding training mixture's relative source proportions.
- These are configured sampling allocations, not measured per-source totals or unique-token counts. The synthetic vs natural split inside the 10B is not given.
- 10% was "a practical allocation within this budget", informed by the Lightning ablations (Table 3). Higher shares helped further, but with smaller incremental gains.

# Super Results (Table 2, Figure 3)
Evaluation: MILU plus human-verified Indic translations of MMLU-Pro, ARC-Challenge, ARC-Easy, and GSM8K. The number of languages is in parentheses. MILU covers ten supported languages. GSM8K excludes Odia because its evaluation exceeded the serving context limit.

| Checkpoint | MILU (10) | Indic MMLU-Pro (12) | Indic ARC-C (12) | Indic ARC-E (12) | Indic GSM8K strict (11) |
|---|---|---|---|---|---|
| Before Indic CPT (distilled) | 72.48 | 51.83 | 84.17 | 90.73 | 72.67 |
| After Indic CPT | 77.90 | 56.81 | 92.83 | 96.77 | 82.49 |
| Change (pp) | +5.42 | +4.98 | +8.66 | +6.04 | +9.82 |

- Changes are computed from unrounded scores. The results describe the combined CPT stage and **do not separate synthetic vs natural Indic contributions**.
- Translation (12-language IGB/FLORES, chrF++): English->Indic **35.43 -> 47.47**; Indic->English **58.75 -> 64.30**.
- The later mid-training checkpoint "largely retains" the Indic gains. No regressions were observed in the tested English benchmark categories.

**Figure 3 (released Nemotron 3.5 Super vs Nemotron 3 Super, whole pipeline, not CPT alone):** 3.5 Super is higher on 5 of 6 task metrics and on all 4 IndiVibe dimensions. Labels as rendered: MILU 1.5 +7.63, GPQA +11.66, MATH500 +4.65, MBPP +3.28, WMT24++ +13.69 (chrF points), IFEval 1.5 **-2.04**. IndiVibe conditional mean preference: Usefulness +23.7, Fluency +25.6, Language and script fidelity +22.3, Appropriate verbosity +21.0. IndiVibe scores are preliminary and are not head-to-head win rates. Benchmark and checkpoint details are deferred to the companion report.

# Lightning 3.5 Mixture-Ratio Ablations (Table 3), Context Only
These are separate Lightning research runs. The **released Lightning 3.5 did not include Indic CPT**. Each run uses a fixed 200B-token budget. The mean covers MILU, MMLU-Pro, ARC-Challenge, GSM8K.

| Checkpoint | Indic share | Four-benchmark mean |
|---|---|---|
| Base reference | — | 45.5 |
| Mixture ablation | 10% | 63.1 |
| Mixture ablation | 20% | 64.6 |
| Mixture ablation | 30% | 65.6 |

The report: these runs "inform mixture design but do not establish an optimal share for Super".

# Planned Releases (§2.3)
Alongside the report the authors will release the broader synthetic corpus and open-source the generation pipelines. They will also publish the companion technical report (translation-model comparisons, data-source ablations on Nemotron 3 Nano, Lightning mixture experiments). With AI4Bharat they will release a human-verified Indic LLM evaluation suite.

# §2.5 Agentic Mid-training Data (Table 4)
Shares of training tokens:

| Data source | Description | Share |
|---|---|---|
| Issue-PR | GitHub issues with repository context and the merged patch | 41% |
| Repositories | Whole repositories, with files that import each other placed together | 20% |
| Agentic traces | Trajectories from many agent scaffolds, tool responses loss-masked | 10% |
| Commit diffs | Multi-file commits and per-file edit histories | 3% |
| Build and CI logs | Raw build and continuous-integration logs | 1% |
| Pretraining replay | Phase-2 pretraining blend | 20% |
| Long-context data | Long-context phase data | 5% |

- Mostly natural data in its original format, without chat templates. Agent traces are the only SFT-style data, kept to a small share, and train only on the model's own turns.
- Every source is decontaminated against SWE-bench and Terminal-Bench and has secrets removed.
- 65.4B tokens at 256K. Training settings and the Table 5 effect are in pretraining.md.

# Caveats
- 2.83T is the generated corpus size; Super trained on a 10B-token Indic allocation (synthetic + natural) inside a 100B-token stage.
- Table 3 numbers are Lightning research ablations; never report them as Super results.
- Figure 3 compares the full released models (Nemotron 3 vs 3.5 Super), not the Indic CPT stage. Table 2 is the stage-level comparison.
- Input token counts in Table 1 may overlap across configurations.
- Figure 2 shows a 0-100 score and a threshold of 75 for the priority-language pipeline. The 10 additional languages use language-specific thresholds that the report does not list.
- The report does not name the natural Indic corpora used in CPT, the Table 4 sources, or a mid-training dataset release name.
