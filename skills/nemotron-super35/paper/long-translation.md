---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "long-translation"
paper_sections: ["3.1", "B"]
title: "Long-Document Translation: PG19 / FinePDFs-Edu SFT Data and PG19 Book-level Evaluation (Figure 12)"
summary: |
  Appendix B (pp.60-61) supports the §3.1 "Long-Context Data" paragraph: Nemotron 3.5 Super's SFT
  includes synthetic long-document translation data built from PG19 (Table 25) and FinePDFs-Edu
  (Table 26), each document translated in chunks by DeepSeek V4 Flash. Figure 12 evaluates book-level
  translation on a 100-book PG19 test set into six languages (600 translations) scored with the SEGALE
  pipeline: Nemotron 3.5 Super 0.684 vs Nemotron 3 Super 0.450, DeepSeek V4 Flash 0.672, Qwen 3.5 122B
  0.715. Precision of the evaluated Super checkpoint is not reported. Lightning is not in Figure 12.
key_facts:
  - "Concerns Super: Figure 12 evaluates Nemotron 3.5 Super (120B); the data is described in Super's SFT section (§3.1). Whether Lightning's text-only SFT used it is not reported."
  - "Data (§3.1): synthetic long-context translation data from PG19 and FinePDFs-Edu; each instance is one long document translated in chunks from the source to a target language using DeepSeek V4 Flash."
  - "§3.1: long-translation performance improved by also including synthetic transduction tasks (Willette et al., 2026) that reinforce position tracking, retrieval, mutation, and output. No ablation numbers are reported."
  - "PG19 data (Table 25): 4,335 English source records, 258,667,792 source tokens -> 4,335 target records, 299,276,122 target tokens across Japanese, Chinese (Simplified), Italian, Spanish (Mexico), German, French."
  - "FinePDFs-Edu data (Table 26): 1,438 source records (English, Spanish, French, Mandarin Chinese, Japanese), 620,467,507 source tokens -> 1,438 target records in 21 target languages, 701,331,885 target tokens."
  - "Evaluation (Figure 12): PG19 test set, 100 books, each translated into German, Spanish, French, Italian, Japanese, Chinese = 600 translations, scored with SEGALE (Wang et al., 2025a); y-axis 'Mean COMET-QE / SEGALE'."
  - "Figure 12 scores (read from bar labels): Nemotron 3 Super 120B 0.450; DeepSeek V4 Flash 284B 0.672; Nemotron 3.5 Super 120B 0.684; Qwen 3.5 122B 0.715."
  - "Super 3.5 (0.684) scores above DeepSeek V4 Flash (0.672), the model that generated its long-translation training data, and below Qwen 3.5 122B (0.715) (derived from Figure 12)."
  - "Book-level translation is separate from Table 10's WMT24++ (sentence-level, XCOMET-XXL, en -> 9 languages, Super 84.80 BF16)."
related_steps:
  - "stage1_sft"
currency: "frozen"
---

# Scope
Answers:
- Does the long-translation appendix concern Super? (Yes.)
- What long-translation SFT data was built, from which sources, by which model, and how big is it?
- How was book-level translation evaluated and how does Super 3.5 compare?
- What is not reported?

# Context: §3.1 Long-Context Data (report p.15)

Within Super's SFT data, long-context data reuses the 512K long-context data from Nemotron 3 Ultra, adds synthetic graph-structured data, and agent trajectories from long-horizon search and coding. Then: "we synthetically generate long-context translation data from both the PG19 (Rae et al., 2019) and FinePDFs-Edu (Kydlíček et al., 2025) datasets where each instance is one long document translated in chunks from the source language to a target language using Deepseek V4 Flash." Long-translation performance "was improved by also including synthetic transduction tasks (Willette et al., 2026) that reinforce position tracking, retrieval, mutation, and output." Evaluation uses the SEGALE pipeline; results in Figure 12.

Not reported: chunk size, how chunks are stitched, how many long-translation samples enter the final SFT blend or its token share, the transduction-task volume, and any with/without ablation numbers.

# Appendix B (report pp.60-61)

## Figure 12 — PG19 book-level translation (p.60)

Caption: book-level translation scores on the PG19 test set of 100 books; each book is translated into German, Spanish, French, Italian, Japanese, and Chinese, for 600 translations, each scored with the SEGALE pipeline. Y-axis label: "Mean COMET-QE / SEGALE".

| Model (as labelled) | Score |
|---|---|
| Nemotron 3 Super 120B | 0.450 |
| DeepSeek V4 Flash 284B | 0.672 |
| **Nemotron 3.5 Super 120B** | **0.684** |
| Qwen 3.5 122B | 0.715 |

Values read from the rendered figure. The plain-text extraction of p.60 lists the labels in an order that can mis-assign 0.672 and 0.684; the rendered bars show 0.672 = DeepSeek V4 Flash and 0.684 = Nemotron 3.5 Super.

Derived readings: +0.234 over Nemotron 3 Super; +0.012 over DeepSeek V4 Flash (the generator of the training translations); -0.031 behind Qwen 3.5 122B.

Not reported: precision (BF16 / NVFP4) of the evaluated checkpoint, reasoning mode, sampling settings, per-language scores, variance, whether the PG19 test books are disjoint from the PG19 training books (implied by "test set" but not stated), and how the "COMET-QE / SEGALE" metric combines its components.

## Table 25 — PG19 long-translation data (p.60)

| Side | Language | Records | Tokens |
|---|---|---|---|
| Source | English | 4,335 | 258,667,792 |
| Target | Japanese | 785 | 64,757,505 |
| Target | Chinese (Simplified) | 786 | 55,994,705 |
| Target | Italian | 757 | 51,381,730 |
| Target | Spanish (Mexico) | 766 | 49,770,267 |
| Target | German | 728 | 48,193,156 |
| Target | French | 513 | 29,178,759 |
| | Target total | 4,335 | 299,276,122 |

Target languages match the Figure 12 evaluation languages. French has notably fewer records (513) than the others (728-786); no reason is given. Totals re-add correctly.

## Table 26 — FinePDFs-Edu long-translation data (p.61)

| Side | Language | Records | Tokens |
|---|---|---|---|
| Source | English (Latin) | 1,031 | 451,571,611 |
| Source | Spanish (Latin) | 173 | 74,851,307 |
| Source | French (Latin) | 112 | 46,515,356 |
| Source | Mandarin Chinese (Han) | 85 | 31,270,147 |
| Source | Japanese | 37 | 16,259,086 |
| | Source total | 1,438 | 620,467,507 |
| Target | Italian | 97 | 49,297,516 |
| Target | Vietnamese | 83 | 38,620,351 |
| Target | Danish | 71 | 38,607,034 |
| Target | Norwegian Bokmål | 72 | 38,171,425 |
| Target | Portuguese (Brazil) | 76 | 37,615,920 |
| Target | Indonesian | 72 | 37,144,561 |
| Target | Japanese | 78 | 36,626,127 |
| Target | Finnish | 69 | 35,973,396 |
| Target | Portuguese | 70 | 35,528,679 |
| Target | Korean | 81 | 35,360,466 |
| Target | Dutch | 66 | 35,240,892 |
| Target | French | 69 | 35,112,951 |
| Target | German | 66 | 34,144,385 |
| Target | Thai | 72 | 34,013,548 |
| Target | Spanish | 65 | 32,888,083 |
| Target | Swedish | 60 | 31,457,768 |
| Target | Turkish | 62 | 31,304,928 |
| Target | Hindi | 75 | 28,435,582 |
| Target | Chinese (Simplified) | 55 | 26,234,775 |
| Target | Arabic | 62 | 22,652,904 |
| Target | English | 17 | 6,900,594 |
| | Target total | 1,438 | 701,331,885 |

21 target languages; English is a target for 17 records (presumably the non-English sources; not stated). Totals re-add correctly. Average document length is large: ~431K source tokens per FinePDFs-Edu record and ~60K per PG19 record (derived), consistent with long-context training. The tokenizer used for these counts is not reported.

# Other content on the Appendix B pages
- Appendix A is a placeholder: "A. TBD — [TODO: TBD: add supplementary material to the appendix]" (p.60).
- Table 27 (p.62, output length and truncation pre-MOPD vs MOPD, BF16) sits after Table 26 under the Appendix B heading but is unrelated to translation; it is referenced from Appendix C.1 (runaway-loop termination failure) and is recorded in `evaluation.md`.

# Contradictions and inconsistencies in the source
- No numeric contradictions found in Tables 25-26 (all totals re-add).
- Text extraction of Figure 12 mis-orders bar values relative to labels (see above); use the rendered figure.
- The y-axis says "Mean COMET-QE / SEGALE" while the caption says scores come from "the SEGALE pipeline"; the exact metric definition is not given.

# Caveats
- Figure 12 is the only long-translation result; no Lightning, no per-language, no ablation.
- Do not compare Figure 12 scores with WMT24++ COMET in Table 10 (different task, metric, and scale).
- `related_steps: stage1_sft` points to this repository's SFT stage; whether the open SFT data release includes the long-translation subsets is not stated in the report.
