---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "safety"
paper_sections: ["3", "3.1", "3.5", "3.6", "3.12", "4.8", "Table 21", "Appendix D.2"]
title: "Nemotron 3.5 Lightning Safety: What Is and Is Not Reported"
summary: |
  The report says almost nothing about Lightning safety. Every safety statement in it (SFT safety blend
  with dual-risk CBRNE and demographic-bias data; RLHF alignment teacher; MOPD) is made for Super, and
  Lightning skips the teacher and MOPD stages entirely. For Lightning there is no safety data
  description, no safety benchmark, no red-teaming and no responsible-use section. The nearest signals
  are Table 21's AA-Omniscience Non-Hallucination (68.88 BF16 / 69.20 NVFP4) and IFBench. The BF16 model
  card describes a post-training filter that removes trajectories aligned with political entities or
  nationalistic narratives.
key_facts:
  - "Report, Lightning-specific: post-training = single-stage text-only SFT + RLVR (§1, §3); no MOPD (§3.6). Nothing in the report describes safety data, safety rewards or safety evaluation for Lightning."
  - "§3.1 safety data ('largely the same blend as Nemotron 3 Ultra, plus ... dual-risk CBRNE content and demographic biases') is stated for Super's SFT. The report does not say whether Lightning's 'text only SFT data' includes it."
  - "§3.5 alignment teacher (RLHF with an Ultra GenRM variant, abstention training, model-identity data) and §3.6 MOPD are Super only. Lightning gets neither."
  - "§1/§3 describe Lightning's RL as RLVR; no RLHF, preference or safety environment is mentioned for Lightning."
  - "Closest evaluation signals (Table 21, BF16 / NVFP4): AA-Omni Non-Hallucination 68.88 / 69.20; AA-Omniscience Accuracy 17.47 / 16.35; IFBench (loose) 72.00 / 74.04; LM Arena Proxy 12.06 / 11.10. None of these is a safety benchmark."
  - "§3.12 / Appendix D.2: post-training data was decontaminated against 30 benchmarks, including CBRN and BBQ. The section does not say which model's data, and does not mention Lightning; CBRN and BBQ results are not reported for either model."
  - "Not reported for Lightning: content-safety, jailbreak, over-refusal, bias, toxicity or CBRN results; red-teaming; prompt-injection / agentic-safety work; responsible-use or license guidance; safety behaviour of the NVFP4 checkpoint."
  - "BF16 model card (quoted in docs sft.md): post-training data filters include keyword- and regex-based removal of trajectories that align with specific political entities or promote nationalistic narratives, plus structural and repetition filters."
  - "Earlier docs versions had an SFT 'Safety: refusal behaviors' domain and a 'DPO for tool hallucination' table (8.33% -> 0.7%), both copied from Nano 3; they were removed. The report does not mention DPO for Lightning, and the repo RL recipe has no DPO stage."
  - "Repo: weights, data and recipes are released under OpenMDW-1.1 (docs/nemotron/lightning35/README.md, root README). The report does not discuss licensing."
related_steps: []
currency: "frozen"
---

# Scope

- What does the report say about Lightning's safety training and evaluation?
- Which safety statements belong to Super only?
- What do the repo docs add, and how reliable is it?

# Report: Lightning-specific statements

| Topic | Statement | Section |
| --- | --- | --- |
| Pipeline | single-stage text-only SFT + RLVR | §1, §3 |
| MOPD / teachers | none ("did not have an MOPD stage") | §3.6 |
| Safety data | not described | - |
| Safety evaluation | none | - |

# Super-only safety material (do not transfer)

| Item | Section |
| --- | --- |
| SFT safety blend: Ultra's blend plus dual-risk CBRNE and demographic-bias data | §3.1 |
| Alignment teacher: RLHF with a length-penalized Ultra GenRM variant, abstention training, model identity, language mixing | §3.5 |
| Joint RLVR + RLHF trunk RL | §3.2 |
| Alignment teacher's share of MOPD prompts (20% / 30%) | §3.6 |

Lightning may share some of this SFT data through its "text only SFT data", but the report does not say so.

# Nearest evaluation signals (Table 21)

| Benchmark | BF16 | NVFP4 | What it measures |
| --- | --- | --- | --- |
| AA-Omni Non-Hallucination | 68.88 | 69.20 | avoiding confident wrong answers (not safety) |
| AA-Omniscience Accuracy | 17.47 | 16.35 | knowledge accuracy |
| IFBench (loose) | 72.00 | 74.04 | instruction following |
| LM Arena Proxy | 12.06 | 11.10 | chat preference proxy |

# Decontamination (§3.12, Appendix D.2)

The decontamination benchmark list includes CBRN and BBQ, which suggests those evaluations exist
internally. No CBRN or BBQ score is reported for either model, and §3.12 does not say whether
Lightning's post-training data went through the same pipeline.

# Repo docs (model card / repo docs)

| Claim | Source | Reliability |
| --- | --- | --- |
| Filter removing political/nationalistic narratives | BF16 model card, quoted in lightning35/sft.md | Lightning model card statement |
| Open-data recipes are a subset of the release data | all Lightning docs | Lightning-specific statement |
| License OpenMDW-1.1 | lightning35/README.md, root README | Lightning-specific statement |

# Gaps

- No safety benchmark, red-teaming, refusal or bias result for Lightning in the report or the repo.
- No statement on whether NVFP4 quantization changes safety behaviour.
- The model card may carry safety / responsible-AI sections that the repo does not reproduce. They are not in this knowledge base.
