---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "safety"
paper_sections: ["3.1.1", "3.2", "3.3.2"]
title: "Nemotron 3 Ultra Safety: SFT Safety Blend, RLVR Safety Environment, Agentic Safety Teacher, GenRM/RLHF Alignment"
summary: |
  Synthesizes the safety-relevant content scattered across the Nemotron 3 Ultra report: the multilingual
  SFT safety data blend (§3.1.1), the unified RLVR "safety" environment (§3.2), the Agentic Safety teacher
  for indirect prompt-injection robustness (§3.3.2), and alignment via the Ultra-based principle-following
  GenRM used in RLHF plus the Instruction-following & Factuality teacher's abstention training (§3.3.2).
  The report does not contain a dedicated responsible-use / license / safety-eval section.
key_facts:
  - "SFT retains the 45K safety data blend curated in Nemotron 3 Super; prompts from diverse sources, responses synthetically generated conditioned on a per-prompt response policy."
  - "A two-stage response-and-reasoning generation framework enables deliberate reflection on safety guidelines in the reasoning traces and aims for consistent, policy-compliant, contextually appropriate final responses."
  - "Safety blend translated into six languages — German, Spanish, French, Japanese, Italian, Chinese — via NeMo Skills chunked translation (sentence-by-sentence parity) with NVIDIA Riva Translate 4B v1.1 as the translation backbone."
  - "Translation quality control: each translated example back-translated to English and compared to the original; examples with semantic similarity below 0.8 filtered out, removing approximately 10-15% per translated language; highest/lowest-scoring translations manually spot-checked."
  - "Final safety blend after stratified sampling: approximately 135K samples = approximately 45K English + approximately 15K from each of the six translated languages."
  - "RLVR (§3.2) is a single unified stage whose environment domains explicitly include safety (alongside terminal, office/productivity, software engineering, search, tool-calling, math, code, STEM, chat, instruction following, long-context QA, reasoning, structured outputs, usability)."
  - "Agentic Safety teacher (§3.3.2) targets indirect prompt-injection attacks where malicious instructions are embedded in tool-response data rather than issued by the user."
  - "Agentic Safety dataset: benign user request requires a read tool whose returned content hides an adversarial instruction (chart notes, case summaries, product descriptions, resumes, support tickets); the injected instruction targets a sensitive write tool distinct from the task tool, making attack compliance observable from the tool-call trace."
  - "Agentic Safety covers four attack categories: unauthorized actions, data modification, denial of service, data exfiltration."
  - "Agentic Safety uses an automated red-teaming loop: attacker model iteratively rewrites the injected instruction against a defender until the defender complies; only successful attacks retained; attacker = Nemotron 3 Super, defender = Nemotron 3 Nano."
  - "Agentic Safety deterministic verifier marks an injection as resisted only if the agent does not invoke the attacker's target tool with the target arguments."
  - "Chat teacher (§3.3.2) is an Ultra-based principle-following Generative Reward Model (GenRM) trained on top of the Ultra SFT model to mitigate reward hacking during RLHF; evaluates a pair of candidate responses given conversational context, conditioned on user-defined principles when provided; only overall scores used as RLHF reward; multiple RLHF iterations with targeted data per cycle."
  - "Instruction-following & Factuality teacher (§3.3.2) adds abstention training (encourage abstaining when uncertain rather than hallucinate); abstention reward dynamically calibrated; RLHF data added to avoid behavioral collapse/overfitting and preserve helpfulness/alignment."
  - "AA-Omniscience Non-Hallucination 78.7 is the highest non-hallucination score in the post-trained comparison set (see evaluation.md)."
  - "The report does not include a dedicated responsible-use, license, content-safety-benchmark, or over-refusal-environment section for Ultra; such guidance is not reported in the report text."
related_steps: []
currency: "frozen"
---

# Scope

Answers:
- How is safety addressed in Nemotron 3 Ultra SFT data?
- What safety-relevant environments/teachers exist in RLVR and MOPD?
- How is alignment / RLHF / reward modeling handled (GenRM, abstention)?
- What does the report say (and not say) about responsible use, over-refusal, and safety evaluation?

# SFT safety data (§3.1.1)

| Aspect | Value |
|---|---|
| Base blend | 45K safety samples retained from Nemotron 3 Super |
| Prompt sources | diverse; responses synthetic, conditioned on a per-prompt response policy |
| Generation | two-stage response-and-reasoning framework (safety-guideline reflection in reasoning traces) |
| Translation | six languages: German, Spanish, French, Japanese, Italian, Chinese |
| Translation tool | NeMo Skills chunked translation (sentence-by-sentence parity); backbone NVIDIA Riva Translate 4B v1.1 |
| Quality filter | back-translate to English, compare to original; drop semantic similarity < 0.8 (~10-15% removed/language); manual spot-check of extremes |
| Final blend | ~135K samples = ~45K English + ~15K each of the six translated languages (stratified sampling) |

# RLVR safety environment (§3.2)

- RLVR is one unified verifiable-reward stage; its enumerated domains **explicitly include `safety`** (and
  `usability`), alongside agentic, reasoning, code, chat, instruction-following, long-context, etc.
- Per-environment safety reward design, over-refusal/over-safety environments, and jailbreak-specific
  environments for Ultra: **not reported** in this report (the domain is named but its internal recipe is
  not detailed here).

# Agentic Safety teacher (§3.3.2)

Targets **indirect prompt injection** — malicious instructions embedded in tool-response data, not issued
by the user.

| Aspect | Value |
|---|---|
| Task setup | benign user request → read tool whose returned content hides an adversarial instruction |
| Injection carriers | chart notes, case summaries, product descriptions, resumes, support tickets |
| Attack target | a sensitive write tool distinct from the task tool (compliance observable in tool-call trace) |
| Attack categories | unauthorized actions, data modification, denial of service, data exfiltration |
| Attack generation | automated red-teaming loop; attacker rewrites injection until defender complies; keep only successful attacks |
| Attacker / defender | attacker = Nemotron 3 Super; defender = Nemotron 3 Nano |
| Verifier | deterministic; injection "resisted" only if the agent does NOT invoke the attacker's target tool with the target arguments |

Provides verifiable supervision: complete the user's task while ignoring untrusted environment instructions.

# Alignment: GenRM, RLHF, abstention (§3.3.2)

| Component | Role |
|---|---|
| Chat teacher (Ultra-based GenRM) | Principle-following Generative Reward Model trained on the Ultra SFT model; mitigates reward hacking. Scores a pair of candidate responses given conversational context; conditions on user-defined principles when provided (else general helpfulness/quality). Predicts individual scores + ranking (triplet per principle, then overall); only overall scores used as RLHF reward. Multiple RLHF iterations with targeted data per cycle. |
| Instruction-following & Factuality teacher | Domain-focused RLVR combining instruction-following, abstention-focused, and RLHF environments. Abstention training encourages abstaining when uncertain rather than hallucinating; abstention reward dynamically calibrated. RLHF data added to avoid behavioral collapse/overfitting and preserve quality/helpfulness/alignment. |

A released checkpoint, **Nemotron 3 Ultra 550B-A55B GenRM**, is the GenRM used for RLHF (§1).

# Safety-relevant evaluation signal

- The report has no dedicated safety/content-safety benchmark table for Ultra.
- The closest reported safety-relevant metric is **AA-Omniscience Non-Hallucination 78.7** (post-trained),
  the highest non-hallucination score in the comparison set (see `evaluation.md`); this reflects
  abstention/factuality, not content-safety or jailbreak robustness.

# Not reported

- A responsible-use, acceptable-use, or license/guardrail-guidance section (unlike a typical release card).
- An explicit Ultra over-refusal / over-safety RLVR environment or jailbreak-robustness benchmark.
- Bias, toxicity, or content-safety benchmark numbers for Ultra.
- Quantitative results for the Agentic Safety teacher (attack-resistance rates are not given here).

# Caveats

- Safety figures (135K, 45K, 15K, 10-15%, 0.8 threshold) carry "approximately"/"≈" qualifiers in the source.
- The RLVR `safety` and `usability` domains are named but their internal reward recipes are not detailed
  in the report; do not invent over-refusal or jailbreak environment specifics for Ultra.
- The Agentic Safety attacker/defender are Nemotron 3 Super / Nano (not Ultra itself).
- "Not reported" items above are genuinely absent from the report text; do not infer them from Super3.
- AA-Omniscience Non-Hallucination is a factuality/abstention metric, not a content-safety score.
