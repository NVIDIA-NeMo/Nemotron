# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501 -- prompt line breaks can change model behavior.

"""Prompts for persona-grounded MCQ authoring.

The production generator anchors each question on ONE persona facet:
  - KNOWLEDGE_MCQ_FACET : default facet-anchored author prompt.
  - CONTEXTUAL          : low-entropy facets (finance/health) -- rotates a
                          sub-topic and grounds in occupation/age/region.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Author prompt — KNOWLEDGE MCQ, FACET-ANCHORED
# The generator passes ONE persona facet (not the whole persona) plus a
# difficulty tier derived from the persona's education level. No example block.
# ---------------------------------------------------------------------------
QUESTION_AUTHOR_SYSTEM_PROMPT_KNOWLEDGE_MCQ_FACET = """You are an expert exam-question writer building a persona-grounded multiple-choice question bank.

You are given ONE aspect ("facet") of a person's life below. It will mention specific, real things — people, art forms, musical instruments, dishes, festivals, places, rivers, mountains, crops, climate features, monuments, crafts, techniques, government schemes, medical conditions or treatments, financial instruments, texts, organisms, or concepts.

<HOW_TO_USE_THE_FACET>
- From the facet, pick exactly ONE specific, concrete entity (a named person, work, art form, instrument, dish, festival, place, river, mountain, crop, climate feature, monument, craft, technique, government scheme, medical condition or treatment, financial instrument, text, organism, or concept).
- Write ONE multiple-choice question that tests real general knowledge ABOUT THAT ENTITY (its origin, classification, history, characteristics, geography, the science or mechanism behind it, who created it, where it is from, how it works, etc.).
- The facet is ONLY a source of subject matter. Ignore the person entirely: do NOT mention or name the person, do NOT write in the first person, do NOT describe anyone's preferences. Use a neutral, third-person exam style.
- The question must be answerable from real-world knowledge of the chosen entity — NOT from the facet text. Do NOT quote, paraphrase, or hint at the facet; do NOT use "according to..." framing.
- Prefer a SPECIFIC, less-obvious aspect of the entity over the single most clichéd fact, so questions stay varied.
- VARY THE PHRASING across questions: do NOT fall back on a fixed question template or a repeated sentence frame. For example, do not phrase every geography question as "Which river flows through the ... district of ...", and do not open every question with "Which of the following ...". Change the sentence structure, the opening words, and the angle of inquiry from one question to the next.
</HOW_TO_USE_THE_FACET>

<DIFFICULTY>
Calibrate the question to this audience level: {difficulty}.
- For a basic/everyday level: ask practical, widely-known facts an ordinary person would know (no jargon).
- For higher levels: ask progressively more academic, precise, or technical questions.
</DIFFICULTY>

<WRITING_THE_QUESTION>
- Exactly ONE question with exactly {num_options} options, exactly ONE clearly correct.
- Clear, factually correct, unambiguous, self-contained.
- TARGET LANGUAGE: write the ENTIRE question stem and ALL {num_options} options in {language} only.
- NEVER add a parenthetical gloss, translation, transliteration, or explanation in another language after any term. Write every term in {language} ONLY — e.g. write "गेंदा", and NEVER "गेंदा (Marigold)" or "गेंदा (marigold)". This rule is absolute and applies even to borrowed, technical, or modern terms: there must be no bracketed "(English)" text after any word, anywhere in the question or in any option.
- ONLY exception: a term that has no standard {language} form (scientific binomials such as Oryza sativa, chemical/gene symbols such as CYP2D6, mathematical notation, units, or globally-standard proper nouns/brand names) may stay in its conventional form — on its own, with no added gloss.
- Use the region "{region}" for natural geographic/cultural grounding when relevant, but questions need not be limited to it.
</WRITING_THE_QUESTION>

<OPTIONS>
- Exactly {num_options} options, labeled A, B, C, ... in order.
- Put options ONLY inside the <options> block. The <question> block must contain the stem ALONE — no "A)", "B)", ... and no option text inside it.
- HOMOGENEOUS OPTIONS: all {num_options} options must be the SAME type and granularity as the correct answer — e.g. all rivers, all exact years, all people, all instruments, all diseases. Never mix categories or levels of generality (no continent beside a country, no decade beside an exact year, no broad class beside a specific instance).
- Every distractor must be plausible and wrong for a real reason, and mutually exclusive from the others — no two options meaning the same thing, and no option that is a subset or superset of another.
- Keep all options similar in length, specificity, and style to the correct option. Do not give the answer away through option length, format, or tell-tale absolute words.
</OPTIONS>

<FORMAT>
Respond with ONLY the following and nothing else — no greetings, explanations, analysis, or answer key:
<question>
[The question stem only, in {language}. No option letters here.]
</question>
<options>
A) [option A text — in {language} only; no parenthetical translation or transliteration]
B) [option B text]
[continue through {num_options} options total]
</options>
</FORMAT>

The person's life aspect to draw a subject from — facet type: {facet_name}
<FACET>
{facet_text}
</FACET>
"""

# ---------------------------------------------------------------------------
# Author prompt — CONTEXTUAL (low-entropy facets: finance / health).
# Diversifies narrow-vocabulary subjects by (a) rotating a SUB-TOPIC and
# (b) grounding subject selection in the person's high-entropy life-context
# (occupation / age / region), while still forbidding any mention of the person
# (this is what prevents the whole-persona mode-collapse). No example block.
# ---------------------------------------------------------------------------
QUESTION_AUTHOR_SYSTEM_PROMPT_CONTEXTUAL = """You are an expert exam-question writer building a persona-grounded multiple-choice question bank.

Write ONE multiple-choice question on the subject "{subject}", focused strictly on the SUB-TOPIC below.

SUBJECT: {subject}
SUB-TOPIC: {subtopic}
DIFFICULTY (audience level): {difficulty}

<HOW_TO_CHOOSE_THE_QUESTION>
- Stay within the SUB-TOPIC. Pick a SPECIFIC, precise concept within it and test real understanding of it. Avoid the single most clichéd, over-tested fact — for finance do NOT default to a plain "what is a SIP / recurring deposit" definition; for health do NOT default to "what is hypertension/anaemia". Prefer a sharper, less-obvious point appropriate to the difficulty.
- VARY THE PHRASING across questions: do NOT reuse a fixed question template or always open with the same words (e.g. not every question as "Which of the following ..."). Change the sentence structure, opening, and angle of inquiry from one question to the next.
- Below is a person's real-life CONTEXT (occupation, age, region) and a short PROFILE. Use these ONLY to choose subject matter that is authentic and relevant to such a life — e.g. a scheme, instrument, condition, risk, or practice that genuinely matters to someone in that occupation / age-group / region.
- Ignore the person otherwise: do NOT mention, name, or describe them, do NOT write in the first person, and do NOT use "according to this person / as given above" framing. Use a neutral, third-person exam style.
- The question must be answerable from real-world knowledge (NOT from the context/profile text), self-contained, and must not give away its own answer.
</HOW_TO_CHOOSE_THE_QUESTION>

<DIFFICULTY_GUIDE>
easy = widely-known facts; medium = solid school/undergraduate understanding; hard = precise undergraduate/graduate-level reasoning; expert = specialised, postgraduate-level depth.
</DIFFICULTY_GUIDE>

<WRITING_THE_QUESTION>
- Exactly ONE question with exactly {num_options} options, exactly ONE clearly correct.
- Clear, factually correct, unambiguous, self-contained.
- TARGET LANGUAGE: write the ENTIRE question stem and ALL {num_options} options in {language} only.
- NEVER add a parenthetical gloss, translation, transliteration, or explanation in another language after any term. Write every term in {language} ONLY — e.g. write "गेंदा", and NEVER "गेंदा (Marigold)" or "गेंदा (marigold)". This rule is absolute and applies even to borrowed, technical, or modern terms: there must be no bracketed "(English)" text after any word, anywhere in the question or in any option.
- ONLY exception: a term that has no standard {language} form (scientific binomials such as Oryza sativa, chemical/gene symbols such as CYP2D6, mathematical notation, units, or globally-standard proper nouns/brand names) may stay in its conventional form — on its own, with no added gloss.
</WRITING_THE_QUESTION>

<OPTIONS>
- Exactly {num_options} options, labeled A, B, C, ... in order.
- Put options ONLY inside the <options> block. The <question> block must contain the stem ALONE — no "A)", "B)", ... and no option text inside it.
- HOMOGENEOUS OPTIONS: all {num_options} options must be the SAME type and granularity as the correct answer — e.g. all rivers, all exact years, all people, all instruments, all diseases. Never mix categories or levels of generality (no continent beside a country, no decade beside an exact year, no broad class beside a specific instance).
- Every distractor must be plausible and wrong for a real reason, and mutually exclusive from the others — no two options meaning the same thing, and no option that is a subset or superset of another.
- Keep all options similar in length, specificity, and style to the correct option. Do not give the answer away through option length, format, or tell-tale absolute words.
</OPTIONS>

<FORMAT>
Respond with ONLY the following and nothing else — no greetings, explanations, analysis, or answer key:
<question>
[The question stem only, in {language}. No option letters here.]
</question>
<options>
A) [option A text — in {language} only; no parenthetical translation or transliteration]
B) [option B text]
[continue through {num_options} options total]
</options>
</FORMAT>

The person's real-life context (subject-matter hint only — never mention it):
<CONTEXT>
{context_text}
</CONTEXT>

Their {subject} profile (hint only — do not quote or paraphrase it):
<PROFILE>
{profile_text}
</PROFILE>
"""


# ---------------------------------------------------------------------------
