---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "safety"
paper_sections: ["1", "3.1", "3.2", "3.5", "3.8", "Table 10", "Appendix C.1"]
title: "Nemotron 3.5 Super Safety and Alignment: What the Report Says and What It Does Not"
summary: |
  Collects every safety- and alignment-relevant statement in the Nemotron 3.5 report. Safety appears
  in the SFT blend (Nemotron 3 Ultra's safety blend largely reused, plus additional dual-risk CBRNE
  and demographic-bias data) and in the alignment teacher (RLHF against a length-penalized Ultra
  GenRM variant, model-identity data, language-mixing emphasis, Ultra's abstention training). The
  trunk RL is joint RLVR + RLHF but the report does not list a safety environment. There is no
  safety benchmark, no red-teaming/prompt-injection work, and no responsible-use section in the
  report; the closest evaluation signals are AA-Omniscience Non-Hallucination, IFBench and the internal
  NV Chat Benchmark.
key_facts:
  - "§1: the two-stage SFT pipeline develops 'reasoning, chat, safety, long-context, and agentic capabilities'; §3.1: SFT stage 1 includes safety data in its general mix."
  - "§3.1: 'For safety data, we largely use the same blend as Nemotron 3 Ultra, plus some additional data focused on dual-risk CBRNE content and demographic biases.' Sizes, languages and generation method of the additional data are not reported."
  - "Context from the Ultra report (not restated in this report): Ultra's safety SFT blend was ~135K samples (~45K English from Nemotron 3 Super + ~15K in each of six translated languages)."
  - "§3.2: trunk RL jointly applies RLVR and RLHF, routing each environment's samples to its verifier or reward model; the report does not name a safety environment or describe safety rewards for Super 3.5."
  - "§3.5 alignment teacher: trained primarily with RLHF to have superior chat capabilities and be well calibrated with human preferences; reward from a new Nemotron 3 Ultra GenRM variant trained with a reasoning length penalty; responses compared to a fixed Gemma 4 31B-it response; mean of per-rubric scores."
  - "§3.5 alignment blend: similar to Ultra, plus newly collected preference data, updated model identity and capabilities data, more emphasis on addressing language mixing; mixed with Ultra's abstention training strategy and a small reasoning length penalty."
  - "§3.6: alignment teacher gets 20% of MOPD stage-1 prompts and 30% of stage-2 prompts."
  - "Table 10 (final Super 3.5): AA-Omniscience Accuracy 26.20 and Non-Hallucination 69.96; IFBench 70.86; NV Chat Benchmark 52.67 (internal NVIDIA chat benchmark)."
  - "Not reported for Super 3.5: any content-safety, jailbreak, over-refusal, bias or toxicity benchmark; agentic-safety / prompt-injection teacher or environment; red-teaming; responsible-use or license guidance; safety results for the NVFP4 checkpoint."
related_steps:
  - "stage1_sft"
  - "stage2_rl/stage2_teachers/rlhf"
currency: "frozen"
---

# Scope

Answers:
- What safety data is in Nemotron 3.5 Super's SFT?
- Is there a safety environment or safety teacher in RL / MOPD?
- How is alignment (RLHF, GenRM, abstention, identity) handled?
- What safety-relevant evaluation numbers exist, and what is explicitly absent?

# SFT safety data (§3.1)

| Aspect | Report statement |
| --- | --- |
| Where | SFT stage 1 general mix includes safety data |
| Base blend | "largely ... the same blend as Nemotron 3 Ultra" |
| Additions | "some additional data focused on dual-risk CBRNE content and demographic biases" |
| Chat (related) | Ultra recipe; refreshed with more recent Qwen, Kimi and Gemma responses |
| Size, languages, generation pipeline, filtering | not reported |

Context only (from the Ultra report, not repeated here): Ultra's blend was ~135K samples, ~45K English
(inherited from Nemotron 3 Super) plus ~15K each in German, Spanish, French, Japanese, Italian and Chinese.
Whether Super 3.5 kept all of it ("largely") is not specified.

# RL (§3.2)

- RLVR and RLHF run jointly; each environment is routed to its verifier or reward model.
- The report highlights only competitive-programming and finance-agent environments. It **does not name a
  safety environment**, over-refusal environment or jailbreak environment for Super 3.5.
- Repo note: `docs/nemotron/super35/rl/rlvr.md` lists a reference-run NeMo Gym blend that includes safety;
  that is a recipe statement, not a report statement.

# Alignment teacher (§3.5) and MOPD (§3.6)

| Component | Detail |
| --- | --- |
| Method | primarily RLHF |
| Reward model | new variant of the Nemotron 3 Ultra GenRM, trained with a reasoning length penalty (more token-efficient; with MTP inference, substantially faster training) |
| Comparison baseline | fixed response from Gemma 4 31B-it (instead of cyclical self-comparison) |
| Score | mean of per-rubric scores (instead of overall score) |
| Data | Ultra-like blend + new preference data + updated model identity and capabilities data + more emphasis on language mixing |
| Abstention | mixed with Ultra's abstention training strategy |
| Length | small reasoning length penalty |
| Eval | internal chat benchmark |
| MOPD share | 20% (stage 1), 30% (stage 2) |

# Safety-adjacent evaluation signals (Table 10, final Super 3.5)

| Benchmark | Super 3.5 | What it measures |
| --- | --- | --- |
| AA-Omniscience Non-Hallucination | 69.96 | factual abstention / hallucination resistance |
| AA-Omniscience Accuracy | 26.20 | factual knowledge |
| IFBench | 70.86 | instruction following |
| NV Chat Benchmark | 52.67 | chat preference win rate, 3,713 prompts vs human-preference-battle opponents, Gemini 3.1 Pro judge, both orders, length style control (internal NVIDIA benchmark, §3.8) |

These are not content-safety metrics. Appendix C.1 discusses "runaway loop" termination failures (e.g.
tau3-banking 19.7% pre-MOPD -> one failure after MOPD), a reliability rather than safety property.

# Not reported

- Any content-safety, harmful-request, jailbreak, over-refusal, bias, or toxicity benchmark for Super 3.5 or Lightning 3.5.
- A dedicated safety RL environment or safety teacher; no agentic-safety / indirect-prompt-injection work (Ultra had an Agentic Safety teacher; Super 3.5's report does not mention one).
- Red-teaming procedures or results.
- Size or composition of the CBRNE / demographic-bias additions.
- Responsible-use, acceptable-use, or license guidance (the abstract says model cards accompany the release; the report itself does not reproduce them).
- Whether NVFP4 quantization affects safety behavior.

# Caveats

- "Largely the same blend as Ultra" is the report's wording; do not assert the full Ultra blend (135K, six languages) was reused verbatim.
- Do not import Ultra's agentic-safety teacher, Ultra's safety-translation pipeline, or Nemotron 3 Super's safety RL environments into Super 3.5 answers.
- AA-Omniscience Non-Hallucination is a factuality/abstention metric; NV Chat Benchmark is internal and not reproducible from public tools.
- Lightning's SFT is text-only and single-stage; the report does not separately describe its safety data.
