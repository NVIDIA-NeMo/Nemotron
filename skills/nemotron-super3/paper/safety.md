---
paper: "arxiv:2512.20856"
model: "nemotron-super3"
section: "safety"
paper_sections: ["3.1", "3.2.1", "3.2.3", "model card"]
title: "Safety, Alignment, and Limitations"
summary: |
  Safety in Super3 is distributed across the whole post-training stack rather
  than confined to one final guardrail layer. The model receives safety-focused
  SFT data, RLVR environments for over-refusal and jailbreak robustness, and a
  final principle-following RLHF stage; the model card then adds deployment-side
  guidance about misuse, validation, and downstream responsibility.
key_facts:
  - "Safety enters in SFT, RLVR, and RLHF rather than a single final stage."
  - "RLVR includes explicit over-refusal and jailbreak-focused environments."
  - "RLHF uses a principle-following GenRM to shape identity and safety behavior."
  - "The model card stresses downstream validation and preserving equivalent guardrails if safety layers are changed."
related_steps:
  - "stage1_sft/default"
  - "stage2_rl/rlvr"
  - "stage2_rl/rlhf"
  - "stage3_eval/default"
currency: "frozen"
---

# Scope

Use this file for questions about:

- where safety enters the Super3 training pipeline
- how the model balances refusal and over-refusal
- what role RLHF plays in safe behavior
- what limitations and deployment cautions the release materials emphasize

---

# Safety is distributed across stages

Super3 does not treat safety as an afterthought layered on top of a finished capability model. Instead, safety signals appear in multiple stages.

| Stage | Safety role |
|---|---|
| SFT | teaches baseline safe behavior, refusal style, prompt-injection handling, and domain boundaries |
| RLVR | sharpens jailbreak robustness and reduces over-refusal with environment rewards |
| RLHF | applies principle-following preference optimization for identity/safety behavior |
| Model card / deployment docs | tells downstream users how to preserve or validate guardrails |

That is the single most important framing for safety questions.

---

# Safety in SFT

The paper and release docs say the SFT blend includes a dedicated safety domain.

## Topics mentioned in the release docs

- harmful-content handling,
- jailbreak behavior,
- over-safety / over-refusal balance,
- bias-related behavior,
- prompt injection,
- copyright-sensitive responses.

## Why this matters

It means the model enters RL with some existing safety priors rather than learning all safe behavior from later preference optimization.

---

# Safety in RLVR

The RLVR stage includes explicit safety-oriented environments.

## Over-refusal reduction

One environment trains the model to avoid refusing benign or appropriate requests just because they look safety-adjacent.

## Jailbreak robustness

Another environment improves robustness to jailbreak attacks. The report says the team seeds attacks from SFT prompts and then strengthens them with a PAIR-style iterative adversarial pipeline.

## Why both are needed

| If you optimize only for refusal | If you optimize only for compliance |
|---|---|
| you risk over-refusal on harmless requests | you risk unsafe behavior on adversarial requests |
| safe-but-unhelpful assistant behavior | capable-but-unsafe assistant behavior |

The report’s design tries to push against both failure modes at once.

---

# Safety in RLHF

The final RLHF stage uses a principle-following GenRM to shape behavior on more nuanced assistant dimensions.

## Why RLHF matters for safety

Comparison-based judgments are better suited than simple verifiable rewards for subtle questions such as:

- whether the model asked clarifying questions when a prompt was ambiguous,
- whether it responded helpfully without unnecessary verbosity,
- whether refusal was appropriate rather than excessive,
- whether identity/persona behavior remained consistent with policy goals.

That is why the report says RLHF is used on important areas like **identity and safety-related topics**.

---

# Principle-following behavior

The released RLHF config includes a long judging principle emphasizing:

- impartial judging,
- generating an answer before comparison,
- correctness and error correction,
- helpfulness, relevance, and concision,
- asking for clarification under ambiguity,
- surfacing missing important information.

Even though this is broader than “safety” alone, it contributes directly to safer assistant behavior because many risky failures are really failures of ambiguity handling, omission, or inappropriate compliance.

---

# Safety and tool use

Super3’s safety story matters especially because the model is trained for tool use and agentic action.

Potential risk surfaces include:

- shell or terminal actions,
- web/search behavior,
- code execution,
- tool arguments that can affect external systems,
- long-context prompt injection.

That is why the paper’s safety interventions are embedded in the same pipeline that teaches tool use rather than bolted on afterward.

---

# Model-card deployment guidance

The official release guidance adds a second layer of safety messaging aimed at downstream deployers.

## Main themes

| Guidance | Meaning |
|---|---|
| Trustworthy AI is a shared responsibility | the model release is not a complete end-product safety guarantee |
| Validate for your own domain | high-risk domains need task-specific testing |
| Preserve equivalent guardrails if modifying behavior | do not remove safety layers without replacement controls |
| Review bias/privacy/explainability subcards | safety is broader than content moderation alone |

This is useful when users ask “is the model safe by default?” The best answer is: it contains multiple safety-training layers, but deployment responsibility still remains with the application builder.

---

# Limitations the skill should surface

## 1. Safety is not a proof of harmlessness

A multi-stage safety pipeline reduces risk but does not guarantee safe behavior under all prompts or tools.

## 2. Open reproduction is incomplete

The paper’s exact internal data and environment mixtures are not fully open, so public reproductions may not match the internal balance of capability and safety interventions.

## 3. Agentic models have broader failure surfaces

Because Super3 is trained for tool use, browsing, terminal actions, and software-engineering tasks, its risk profile is broader than that of a pure text-only chatbot.

## 4. Refusal quality is a balancing act

The report explicitly targets both jailbreak robustness and over-refusal reduction. That means “more refusal” is not automatically “safer.”

---

# How to answer common safety questions

## “Where does safety enter training?”

Answer: in all of SFT, RLVR, and RLHF, with different roles at each stage.

## “Did RL make the model safer or just more capable?”

Both. RLVR includes safety environments, while RLHF shapes principle-following behavior on identity and safety-sensitive topics.

## “Can I remove the safety system if I fine-tune it?”

The model card warns against removing safety guardrails without equivalent replacements and asks downstream deployers to validate domain-specific risk.

---

# How the repo maps the paper

| Paper concept | Open file |
|---|---|
| SFT data and behavior | `sft.md` and `../recipes/stage1_sft.md` |
| RLVR safety environments | `rl/rlvr.md` and `../recipes/stage2_rl_rlvr.md` |
| RLHF safety shaping | `rl/rlhf.md` and `../recipes/stage2_rl_rlhf.md` |
| Release-facing cautions | `../model-card.md` |

---

# Caveats

1. **Do not answer safety questions from benchmark numbers alone.** Safety is mostly a training-method question here.
2. **Do not say RLHF is the whole safety story.** Earlier stages matter.
3. **Do not promise domain safety without deployment-specific validation.**

---

# Related files

- `sft.md`
- `rl/rlvr.md`
- `rl/rlhf.md`
- `../model-card.md`
