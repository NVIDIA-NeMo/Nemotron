---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "data"
paper_sections: ["2.2", "2.3"]
title: "Nemotron 3.5 Lightning Pretraining Data: Ultra 3 Phase 2 Distillation Data, Indic-Share Ablations (Table 3), Open Recipe Blend"
summary: |
  The report says little about Lightning's data. Distillation used 1T tokens of the Ultra 3 Phase 2 data,
  with top-128 teacher logits from Super 3 Base. Long-context and MTP warm-up data are not reported for
  Lightning. The released Lightning model did not get Indic CPT. Lightning was, however, the test bed for
  Indic-share research ablations at a 200B-token budget (Table 3): a 10/20/30% Indic share gives a
  four-benchmark mean of 63.1 / 64.6 / 65.6, vs 45.5 for the base reference. These runs informed Super's
  10% choice. The repo's open-data recipe blend has 23 equal-weight subsets of four public Nemotron
  pretraining datasets and a single mix (no phase curriculum); it does not reproduce the released model.
key_facts:
  - "Distillation data: 1T tokens of the Ultra 3 Phase 2 data (NVIDIA, 2026c), with top-128 logits from Super 3 Base as teacher signal (§2.2)."
  - "MTP warm-up data (100B tokens) and long-context data for Lightning: not reported."
  - "Released Lightning 3.5 did NOT include the Indic CPT stage (§2.3)."
  - "Table 3 Lightning research ablations, 200B-token budget each, mean over MILU, MMLU-Pro, ARC-Challenge, GSM8K: base reference 45.5; 10% Indic 63.1; 20% 64.6; 30% 65.6."
  - "The mean rises with Indic share and 30% scores highest of the tested settings. The runs 'inform mixture design but do not establish an optimal share for Super' (§2.3)."
  - "Super chose a 10% Indic share 'informed by' these ablations: 10% already gave substantial gains, and higher shares helped further with smaller increments (§2.3)."
  - "Language-level results, mixture configurations, and English-retention analysis for these runs are deferred to the companion 'Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs' report (§2.3)."
  - "Repo open-data blend (data_blend_raw.json): 23 subsets, all weight 1.0, from Nemotron-CC-Math-v1, Nemotron-CC-v2, Nemotron-CC-v2.1, Nemotron-Pretraining-Specialized-v1; tokenized with the Lightning Base tokenizer (add_eos true, add_bos false)."
  - "Repo docs: recipes use only the open-sourced subset; 'the published model-card benchmarks ... used additional proprietary data'."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What data did Lightning's distillation use?
- What does the report say about Lightning's MTP warm-up and long-context data?
- What are the Lightning Indic-share ablations (Table 3), and what do they not show?
- What open datasets does the repo's `lightning35` pretraining recipe use, and how does that relate to the released model?

# Distillation Data (report §2.2)
- "For Lightning 3.5 Base, we used top-128 logits from Super 3 Base on 1 trillion tokens of the Ultra 3 Phase 2 data (NVIDIA, 2026c)."
- Ultra 3 Phase 2 is the higher-quality second phase of Nemotron 3 Ultra's pretraining mixture. For its composition see `skills/nemotron-ultra/paper/data.md`. The 3.5 report does not restate it.
- The report says nothing about whether the 1T tokens are a subset, a resample, or one full pass of that blend.

# Other Lightning Pretraining Data: Not Reported
- MTP warm-up (100B tokens, §2.1): data not stated.
- Long-context training (§1): data not stated for Lightning. §2.4's "same with that in Nemotron 3 Ultra" blend is attributed to the LC-Phase described for Super.
- No Indic CPT and no agentic mid-training data for the released model.
- Nano 3 Base (the initialization) was trained on Nemotron 3 Nano's 25T-token two-phase mixture; see `skills/nemotron-nano3/paper/data.md`.

# Indic-Share Ablations on Lightning (report §2.3, Table 3)
Context: the report built a synthetic Indic corpus (Nemotron-Pretraining-Indic-v1: about 2.83T generated tokens across 22 Indian languages and 34 language-script configurations, translated mostly from Nemotron-CC with Gemma-4-31B-it). It used 10% of a 100B-token CPT stage **for Super**. Corpus and pipeline details are in `skills/nemotron-super35/paper/data.md`.

"To study how the Indic share affects performance, we ran separate Lightning 3.5 research experiments with 10%, 20%, and 30% Indic data under a fixed 200B-token training budget. These experiments are separate from the released Lightning 3.5 model, which did not include this Indic CPT stage."

Table 3 | Lightning 3.5 research ablations (200B-token budget each; verified against the PDF, p.9):

| Checkpoint | Indic share | Four-benchmark mean |
|---|---|---|
| Base reference | — | 45.5 |
| Mixture ablation | 10% | 63.1 |
| Mixture ablation | 20% | 64.6 |
| Mixture ablation | 30% | 65.6 |

- Mean over **MILU, MMLU-Pro, ARC-Challenge, and GSM8K**. The caption does not say "Indic" for MMLU-Pro / ARC-C / GSM8K; by context these are presumably the Indic-translated versions used in Table 2, but Table 3 does not say so.
- "The base checkpoint is shown for reference; the fixed-budget comparison covers the three ablation runs." Which Lightning checkpoint is the base reference (e.g. distilled, post-LC) is not stated.
- Increments: +17.6 (base -> 10%), +1.5 (10 -> 20%), +1.0 (20 -> 30%). This is my arithmetic from the printed values and matches the report's "higher shares improved results further, but with smaller incremental gains".
- Report conclusions: "The four-benchmark mean increased with the Indic share, with the 30% mixture scoring highest among the tested settings." "These experiments inform mixture design but do not establish an optimal share for Super." Super used 10% as "a practical allocation within this budget".
- Not reported: the non-Indic 80-90% of the ablation mixtures, per-language and per-benchmark scores, English retention, natural vs synthetic split, and seeds or variance. All of these are deferred to the companion Indic report, which will also cover "Lightning 3.5 mixture experiments".

# Repo Open-Data Pretraining Blend (repo docs / recipe config, not the report)
`src/nemotron/recipes/lightning35/stage0_pretrain/config/data_prep/data_blend_raw.json` (identical content in `data_blend_raw_large.json`) has 23 entries, all `weight: 1.0`:

| HF dataset | Subsets |
|---|---|
| nvidia/Nemotron-CC-Math-v1 | 3, 4plus, 4plus_MIND |
| nvidia/Nemotron-CC-v2 | Diverse-QA, High-Quality, High-Quality-Synthetic, Medium-High-Quality, Medium-Quality, Translated-Diverse-QA |
| nvidia/Nemotron-CC-v2.1 | High-Quality, High-Quality-Synthetic, High-Quality-Translated-To-English, High-Quality-Translated-To-English-Synthetic, Medium-High-Quality, Medium-High-Quality-Synthetic, Medium-High-Quality-Translated-To-English, Medium-Quality |
| nvidia/Nemotron-Pretraining-Specialized-v1 | InfiniByte-Reasoning, Math-Textbooks, RQA, STEM-SFT, Scientific-Coding, Wiki-Rewrite |

- `data_blend_raw_small.json`: 5 entries: Math-v1 `3` and `4plus`, plus CC-v2 High-Quality-Synthetic, Medium-High-Quality, and Medium-Quality.
- Tokenizer: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`, `add_bos: false`, `add_eos: true`. Output: Megatron bin/idx, 128 train shards, 1 valid, 1 test.
- Docs: "These recipes train exclusively on the open-sourced subset of training data. Results will differ from the published model-card benchmarks, which used additional proprietary data."

# Caveats
- **Open blend in the docs:** docs pretrain.md lists the same four families and 23 subsets as the config, and says the recipe trains on this single mix with no multi-phase curriculum. (Earlier docs versions had a Nano 3 table with code datasets and a 23.5T / 1.5T two-phase "Data Mixture" section; both were removed. The blend contains no code dataset.)
  - The stage README calls the small blend "math-only", but it also contains three Nemotron-CC-v2 web subsets.
  - Uniform 1.0 weights in the open blend say nothing about the released model's mixtures (Nano 3's pretraining mixture for the initialization, then Ultra 3 Phase 2 data for distillation).
- The repo recipe pretrains on raw open text. It does not reproduce the report's distillation (which needs Super 3 Base logits on Ultra 3 Phase 2 data).
- Table 3 numbers are Lightning **research** runs; never report them as properties of the released Lightning model or as Super results. Super's own Indic results are Table 2 (Super chunk).
- Languages: the BF16 model card lists English, Spanish, French, German, Italian, and Japanese for the post-trained model (the recipe README repeats this); the Base model card says the pretraining corpus covers English plus 19 other spoken languages and 43 programming languages. The report gives no language count for Lightning.
