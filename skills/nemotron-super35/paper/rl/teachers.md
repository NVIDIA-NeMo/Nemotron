---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "rl-teachers"
paper_sections: ["3.3", "3.3.1", "3.3.2", "3.4", "3.4.1", "3.4.2", "3.4.3", "3.5"]
title: "Nemotron 3.5 Super Specialist Teachers: Agentic, Reasoning, Video, Vision Tool Use, Vision Reasoning, Alignment"
summary: |
  Details the six specialist teachers distilled into Nemotron 3.5 Super via MOPD: the agentic teacher
  (end-to-end GRPO on executable Terminal/SWE/Search tasks), the reasoning teacher (90% science
  curriculum, 131,072-token generation), the video reasoning and tracking teacher (Video-QA + SA-V
  tracking with verifiable rewards), the vision tool-use teacher (XML image tools), the two-stage
  vision reasoning RLVR teacher, and the alignment teacher (RLHF against a length-penalized Ultra
  GenRM variant, comparing to fixed Gemma 4 31B-it responses). Captures each teacher's initialization,
  recipe, data, batch/LR settings and selected checkpoint where reported, plus the only per-teacher
  results the report gives (video teacher).
key_facts:
  - "The agentic, reasoning, video, vision tool-use and vision-reasoning teachers use GRPO (asynchronous where stated); the alignment teacher is 'primarily' RLHF with no algorithm named. The agentic, reasoning and video teachers initialize from the 'step-120 text-RLVR checkpoint'; vision tool use from 'the text-RLVR checkpoint'; vision reasoning from 'an existing multimodal RLVR checkpoint'; alignment initialization not reported."
  - "Agentic teacher (§3.3.1): full end-to-end GRPO on sandboxed, fully executable Terminal, SWE and Search tasks; no single-step environments; 64 prompts x 16 generations = 1,024 batch; max trajectory age 1; 2-hour max timeout per rollout; tasks difficulty-profiled against the step-120 checkpoint. Selected step not reported."
  - "Reasoning teacher (§3.3.2): same async GRPO; reasoning-focused curriculum of 25,600 examples, each 256-prompt batch 90% science / 5% math / 5% competitive coding; max generation 131,072 throughout; 256 prompts x 16 responses = 4,096; max trajectory age 2 (up from 1); LR 4e-6; math/STEM rewards via answer verification with DeepSeek-V4-Flash as LLM judge, coding via execution tests; checkpoint after 80 updates selected."
  - "Video reasoning and tracking teacher (§3.4.1): Video-QA from the Nemotron 3 Nano Omni video subset (multiple choice over 64 ordered frames as one video) + SA-V tracking (box, point, referring-object; frames as ordered multi-image). Binary reward for Video-QA; continuous SA-V reward on localization quality, object-visibility accuracy, track validity/completeness; missing/invalid/extraneous entries penalized. No LLM judge."
  - "Video teacher shaping: response without exactly one matched <think>...</think> block gets task reward 0 and advantage overridden to -5; soft length preference on correct responses only: full reward up to 1,024 reasoning tokens, linear 1.0 -> 0.95 between 1,024 and 4,096, 0.95 beyond (no truncation; at most 5% preference)."
  - "Video teacher training: async GRPO in NeMo-RL from the step-120 text-RLVR checkpoint; 128 prompts x 16 responses = 2,048 trajectories per update; MoE routing replay; MoE router and vision modules frozen; constant LR 3e-6; 128 NVIDIA GB200 GPUs; step 60 selected."
  - "Video teacher results (step 60): Video-MME 74.90 (256-frame reasoning protocol), Video-MME-v2 19.10 (256-frame reasoning protocol), GOT-10k 52.59, Molmo2-VideoTrack 60.00."
  - "Vision tool-use teacher (§3.4.2): curated mixture of internal data; tools = crop, zoom, rotation, flipping, difference detection, side-by-side comparison, overlays, object counting, color analysis, as structured XML; synthetic multiple-choice and numerical questions filtered to those nearly impossible without tools; outcome reward + auxiliary tool-use reward penalizing duplicate/malformed calls; async GRPO; checkpoint selected by V*, HRBench8K and BabyVision accuracy. Batch, LR, steps not reported."
  - "Vision reasoning RLVR teacher (§3.4.3): chart, document/table, OCR, geometry and visual reasoning tasks. Stage 1: 16,090 profiled examples. Stage 2: 13,848 examples = 8,366 chart-reasoning + 5,482 document/table/OCR; exact QA duplicates removed. Outcome-correctness reward via math, multiple-choice and string-matching verifiers; 256 prompts x 16 responses; stage 2 starts from stage-1 step 114 with a fresh optimizer; checkpoint after 20 further updates used; 128 GPUs across 32 nodes."
  - "Alignment teacher (§3.5): primarily RLHF. New variant of the Nemotron 3 Ultra GenRM trained with a reasoning length penalty (more token efficient; with MTP inference, substantially faster training). Policy responses compared to a fixed response from Gemma 4 31B-it instead of cyclical self-comparison; mean of per-rubric scores used instead of an overall score."
  - "Alignment blend (§3.5): similar to Ultra plus newly collected preference data, updated model identity and capabilities data, more emphasis on language mixing; mixed with Ultra's abstention training and a small reasoning length penalty; evaluated on an internal chat benchmark."
related_steps:
  - "stage2_rl/stage2_teachers/agentic"
  - "stage2_rl/stage2_teachers/reasoning"
  - "stage2_rl/stage2_teachers/video"
  - "stage2_rl/stage2_teachers/vision"
  - "stage2_rl/stage2_teachers/vision_reasoning"
  - "stage2_rl/stage2_teachers/rlhf"
currency: "frozen"
---

# Scope

- Which specialist teachers feed Super 3.5's MOPD, and from which checkpoint does each start?
- What data, rewards and hyperparameters does each teacher use?
- Which checkpoint of each teacher was selected?
- What results does the report give per teacher?

# Teacher roster

| Teacher | § | Init (report wording) | Recipe | Used in MOPD stage |
| --- | --- | --- | --- | --- |
| Agentic | 3.3.1 | step-120 text-RLVR | end-to-end GRPO, executable Terminal/SWE/Search | 1 (35%) |
| Reasoning | 3.3.2 | step-120 text-RLVR | async GRPO, reasoning curriculum | 1 (25%) |
| Video reasoning & tracking | 3.4.1 | step-120 text-RLVR | async GRPO, verifiable video rewards | 2 (20%) |
| Vision tool use | 3.4.2 | "the text-RLVR checkpoint" | async GRPO, XML image tools | 1 (15%), 2 (20%) |
| Vision reasoning RLVR | 3.4.3 | "an existing multimodal RLVR checkpoint" | two-stage async GRPO | 2 (20%) |
| Alignment | 3.5 | not reported | RLHF with length-penalized GenRM + abstention | 1 (20%), 2 (30%) |

MOPD shares from §3.6 (see `mopd.md`). Lightning has no teachers.

# Agentic teacher (§3.3.1)

| Setting | Value |
| --- | --- |
| Init | step-120 text-RLVR checkpoint |
| Algorithm | full end-to-end GRPO |
| Tasks | sandboxed, fully executable Terminal, SWE, Search; no single-step environments |
| Batch | 64 prompts x 16 generations = 1,024 |
| Max trajectory age | 1 |
| Rollout timeout | 2 hours max per rollout |
| Difficulty profiling | against the step-120 checkpoint |
| LR, steps, selected checkpoint, data size | not reported |

# Reasoning teacher (§3.3.2)

| Setting | Value |
| --- | --- |
| Init | step-120 text-RLVR checkpoint |
| Algorithm | same asynchronous GRPO framework |
| Curriculum | 25,600 examples; each 256-prompt batch = 90% science, 5% math, 5% competitive coding |
| Max generation length | 131,072 throughout |
| Batch | 256 prompts x 16 responses = 4,096 |
| Max trajectory age | 2 (vs 1 in initial RLVR) |
| LR | 4e-6 ("remains") |
| Rewards | math/STEM: answer verification, DeepSeek-V4-Flash as LLM judge; coding: execution tests |
| Selected checkpoint | after 80 updates |

Note: 25,600 examples / 256 prompts per batch = 100 batches; 80 updates were used. (Arithmetic only; the report does not state epochs.)

# Video reasoning and tracking teacher (§3.4.1)

| Setting | Value |
| --- | --- |
| Data | Video-QA (from Nemotron 3 Nano Omni video subset; MCQ over 64 ordered frames, processed as one video) + SA-V tracking (box, point, referring-object; ordered multi-image) |
| Rewards | Video-QA: binary MCQ correctness. SA-V: continuous, from localization quality, object-visibility accuracy, track validity/completeness; missing/invalid/extraneous entries penalized. No LLM judge. |
| Format shaping | not exactly one matched `<think>...</think>` -> reward 0, advantage overridden to -5 |
| Length shaping (correct only) | <= 1,024 tokens: x1.0; 1,024-4,096: linear 1.0 -> 0.95; > 4,096: x0.95 |
| Init | step-120 text-RLVR checkpoint |
| Algorithm | asynchronous GRPO in NeMo-RL |
| Batch | 128 prompts x 16 = 2,048 trajectories |
| Stability | replay MoE routing decisions from rollout |
| Frozen | MoE router and vision modules |
| LR | constant 3e-6 |
| Hardware | 128 NVIDIA GB200 GPUs |
| Selected | step 60 |

| Benchmark (teacher, step 60) | Score |
| --- | --- |
| Video-MME (256-frame reasoning protocol) | 74.90 |
| Video-MME-v2 (256-frame reasoning protocol) | 19.10 |
| GOT-10k | 52.59 |
| Molmo2-VideoTrack | 60.00 |

No pre-teacher baseline is given for these four numbers.

# Vision tool-use teacher (§3.4.2)

| Setting | Value |
| --- | --- |
| Data | "curated mixture of internal data"; synthetic MCQ and numerical questions needing >= 1 tool; filtered to samples nearly impossible without tools |
| Tools (XML) | crop, zoom, rotation, flipping, difference detection, side-by-side comparison, overlays, object counting, color analysis |
| Rewards | outcome correctness (primary) + auxiliary tool-use reward penalizing duplicate and malformed calls |
| Init | "the text-RLVR checkpoint" |
| Algorithm | asynchronous GRPO in NeMo-RL |
| Checkpoint selection | V* (Wu & Xie, 2024), HRBench8K, BabyVision accuracy |
| Batch, LR, steps, data size, scores | not reported |

# Vision reasoning RLVR teacher (§3.4.3)

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Data | 16,090 profiled examples (chart, document/table, OCR, geometry, visual reasoning) | 13,848 = 8,366 chart reasoning + 5,482 document/table/OCR; formats audited; exact QA duplicates removed |
| Init | "an existing multimodal RLVR checkpoint" | stage-1 checkpoint at step 114, fresh optimizer |
| Batch | 256 prompts x 16 responses | same |
| Length | step 114 (used to init stage 2) | 20 additional updates |
| Reward | outcome correctness via math, MCQ, string-matching verifiers | same |
| Hardware | 128 GPUs across 32 nodes | same |
| LR, scores | not reported | not reported |

# Alignment teacher (§3.5)

| Aspect | Value |
| --- | --- |
| Goal | superior chat capability, well calibrated with human preferences |
| Main method | RLHF |
| Reward model | new variant of the Nemotron 3 Ultra GenRM, trained with a reasoning length penalty (more token-efficient; with MTP inference enables substantially faster training) |
| Comparison | each policy response vs a fixed response from Gemma 4 31B-it (instead of Ultra's cyclical self-comparison) |
| Scoring | mean of per-rubric scores (instead of an overall score) |
| Blend | similar to Ultra + new preference data + updated model identity/capabilities data + more emphasis on language mixing |
| Mixed with | Ultra's abstention training strategy; small reasoning length penalty |
| Evaluation | internal chat benchmark (no number given in §3.5) |
| Init, batch, LR, steps | not reported |

# Repo cross-reference

`docs/nemotron/super35/rl/teachers.md` separates report values from config values per teacher. Values
found there but not in the report (agentic selected step, vision tool-use step, two alignment teachers each
with two substages, MTP head settings) come from reference-run configs. The repo docs flag the agentic
teacher as blocked (its Terminal-Bench environment is not in public NeMo Gym).

# Caveats

- Only the video teacher reports per-teacher benchmark results; do not invent numbers for the others.
- The vision tool-use teacher's checkpoint step and the vision-reasoning teacher's root checkpoint are not identified in the report.
- The report says the alignment teacher is "primarily" RLHF; its starting checkpoint is not stated (Figure 4 draws it branching from the trunk RL student).
- GPU counts are given in different units: video teacher "128 NVIDIA GB200 GPUs"; vision reasoning "128 GPUs across 32 compute nodes"; MOPD "128 GB200 nodes". Keep the units as stated.
- "Text-RLVR" vs "multimodal RLVR" checkpoint wording is the report's; it does not explain the difference.
