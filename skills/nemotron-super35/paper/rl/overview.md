---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "rl-overview"
paper_sections: ["1", "3", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "4", "Figure 4"]
title: "Nemotron 3.5 Super Post-Training Pipeline Overview: RLVR, Specialist Teachers, Two-Stage MOPD, MTP Boosting"
summary: |
  Top-level map of Nemotron 3.5 Super post-training after SFT. A single joint RLVR+RLHF trunk run
  produces a step-120 checkpoint that is both the MOPD student and the branch point for specialist
  teachers (agentic, reasoning, video reasoning/tracking, vision tool use, vision reasoning,
  alignment). Two MOPD stages merge the teachers back into one student, routing un-owned environments
  to a frozen self-anchor. MTP boosting then trains only the MTP head on the frozen MOPD output, and
  the result is NVFP4-quantized. Explains why teachers + MOPD are used and how this differs from
  Nemotron 3 Super (sequential RL) and Nemotron 3 Ultra (iterated MOPD). Lightning skips teachers/MOPD.
key_facts:
  - "§3: Super's pipeline 'is similar to that used in Nemotron 3 Ultra', consisting of SFT, RLVR, expert teacher model training, and finally MOPD (Figure 4). Lightning: text-only SFT and RLVR only."
  - "Figure 4 caption: teacher models are developed from a shared trunk through SFT and RL stages, specializing in different capabilities, and are merged into a unified student through MOPD."
  - "Figure 4 shows: Base 1M context (agentic image training included) -> Multi-modal + text SFT (VLM) -> Text Only Trunk RL student (VLM), with GenRM (txt) feeding the trunk RL; teachers branch from the trunk: Chat & Alignment (TXT RL), Vision teachers (VLM RL), Reasoning + Agentic (TXT RL); MOPD stage 1 student -> MOPD stage 2 student, each with a self-anchor -> Final Post-Trained Weights."
  - "§3.2: RL jointly applies RLVR and RLHF, routing samples from each environment to the corresponding verifier or reward model."
  - "§3.6: the MOPD student is the RL checkpoint of §3.2 after 120 steps, 'the same checkpoint the teachers were branched from'. §3.3-§3.4.1 name it the 'step-120 text-RLVR checkpoint'."
  - "Teachers (§3.3-§3.5): agentic (§3.3.1), reasoning (§3.3.2), video reasoning and tracking (§3.4.1), vision tool use (§3.4.2), vision reasoning RLVR (§3.4.3), alignment/RLHF (§3.5)."
  - "MOPD (§3.6) runs in two stages because benchmarks improve at different rates and some peak then decline: stage 1 weighted to fast-improving/regress-later domains (agentic 35%, reasoning 25%, alignment 20%, vision tool use 15%, self-anchor 5%); stage 2 to slow ones (alignment 30%, vision tool use 20%, vision reasoning 20%, video 20%, self-anchor 10%)."
  - "MTP boosting (§3.7) runs on the MOPD output with the backbone frozen; MTP head trained 25k steps with the LK loss, initialized from a 4k-step MTP-only run on the SFT checkpoint. Average AL at temp=1 goes 3.787 -> 4.612 (Table 8/9)."
  - "§4: the boosted model is quantized to NVFP4; 99.52% median accuracy recovery vs BF16 across 35 benchmarks for the released quantized Super checkpoint (§1/abstract)."
  - "Lightning (§3, §3.6, §3.7): no teachers, no MOPD; MTP boosting 48k steps with no SFT warm start. Do not attribute Super's pipeline to Lightning."
related_steps:
  - "stage1_sft"
  - "stage2_rl/stage1_rlvr"
  - "stage2_rl/stage2_teachers"
  - "stage2_rl/stage3_mopd"
currency: "frozen"
---

# Scope

Use this file for top-level questions about:

- what happens after SFT for Nemotron 3.5 Super, and in which order
- why the report trains specialist teachers and then distills them (MOPD) instead of running sequential RL stages
- which checkpoint each stage starts from
- where MTP boosting and NVFP4 quantization fit
- how the Super 3.5 pipeline differs from Lightning 3.5, Nemotron 3 Super, and Nemotron 3 Ultra

Detail lives in `rlvr.md`, `teachers.md`, `mopd.md`, `mtp-boosting.md`, and `../sft.md`.

---

# Stage order

![Super 3.5 VL post-training pipeline (report Figure 4)](../../../../docs/assets/super35/post-training-pipeline.png)

| Order | Stage | Starts from | Output used by | Section |
| --- | --- | --- | --- | --- |
| 1 | Two-stage VLM SFT (after projector warm-up) | text-only base | RL trunk; MTP warm start | §3.1 |
| 2 | RL trunk: joint RLVR + RLHF | SFT checkpoint | step 120 -> MOPD student and teacher root | §3.2 |
| 3 | Specialist teachers (6) | mostly the step-120 RL checkpoint (see below) | MOPD | §3.3-§3.5 |
| 4 | MOPD stage 1 (30 steps) | RL step 120 (student and self-anchor) | stage 2 | §3.6 |
| 5 | MOPD stage 2 (40 steps) | stage-1 model (student and self-anchor) | MTP boosting | §3.6 |
| 6 | MTP boosting (25k steps, head only) | MOPD output; head initialized from a 4k-step SFT-stage MTP run | release BF16 | §3.7 |
| 7 | NVFP4 PTQ / QAD | boosted BF16 model | release NVFP4 | §4 |

Figure 4 itself shows stages 1-5 only; MTP boosting and NVFP4 come from §3.7 and §4.

## Teacher initializations as stated

| Teacher | Initialization (report wording) |
| --- | --- |
| Agentic (§3.3.1) | "step-120 text-RLVR checkpoint" |
| Reasoning (§3.3.2) | "step-120 text-RLVR checkpoint" |
| Video reasoning and tracking (§3.4.1) | "step-120 text-RLVR checkpoint" |
| Vision tool use (§3.4.2) | "the text-RLVR checkpoint" (step not stated) |
| Vision reasoning RLVR (§3.4.3) | "an existing multimodal RLVR checkpoint" (not further identified) |
| Alignment (§3.5) | not reported; Figure 4 draws it branching from the trunk RL student |

---

# Why specialist teachers + MOPD

The report's reasons (§3, §3.6):

1. **Specialization from a shared trunk.** Figure 4's caption: teachers are "developed from a shared trunk"
   and "specializ[e] in different capabilities". Each teacher trains on a narrow mix
   (e.g. reasoning: 90% science; agentic: only end-to-end executable tasks); the report does not spell
   out why a single broad mix would be worse.
2. **Dense token-level signal.** In MOPD every student token gets an advantage from one teacher forward
   pass, "with no verifier and no sampling from the teacher"; the signal is zero where student and teacher
   agree and concentrates where the student departs from the specialist (§3.6).
3. **Uneven learning rates across domains.** "Benchmarks improve at different rates under distillation, and
   ... some peak within a few steps and then decline". Two MOPD stages handle this: stage 1 for fast
   domains (agentic, reasoning), stage 2 for slow ones (chat quality, vision) with agentic/reasoning
   folded into the self-anchor so they are held rather than re-distilled (§3.6).
4. **Self-anchor for un-owned environments.** Environments no teacher owns go to a frozen copy of the
   student's starting checkpoint, which gives no signal while the student matches it and regularizes
   departures (§3.6).

Side effect reported in Appendix C.1 / Table 27: pre-MOPD vs MOPD, output length per rollout and
truncation drop sharply on several benchmarks (e.g. Tau3 Banking 35.2k -> 13.7k tokens/rollout, truncation
20.9% -> 0.3%); see `mopd.md`.

---

# Relation to earlier Nemotron pipelines

| Model | Post-SFT structure (as described by the respective reports) |
| --- | --- |
| Nemotron 3 Super | Sequential RL: RLVR -> SWE-RL -> RLHF -> MTP healing |
| Nemotron 3 Ultra | RLVR -> MOPD warm-up -> MOPD (x N cycles, two iterations) -> MTP boosting |
| Nemotron 3.5 Super | RL trunk (RLVR + RLHF jointly) -> 6 specialist teachers -> MOPD stage 1 -> MOPD stage 2 -> MTP boosting -> NVFP4 |
| Nemotron 3.5 Lightning | text-only SFT -> RLVR -> MTP boosting (48k steps) |

Differences from Ultra stated in this report: RL data refreshed and reward-profiled; router replay and FP32
LM head added (§3.2); MTP boosting uses the LK loss instead of temperature-scaled forward KL (§3.7).
Described for Super 3.5 but not explicitly framed as differences from Ultra: MOPD in two stages with
different teacher sets and a proximal-teacher target in stage 2 (§3.6); an SFT-stage MTP warm start (§3.7).
An Ultra-style "MOPD warm-up" SFT step is not mentioned for Super 3.5.

---

# Repo cross-reference

- `docs/nemotron/super35/rl/index.md`, `rlvr.md`, `teachers.md`, `mopd.md` map these stages onto
  `src/nemotron/recipes/super35/stage2_rl/{stage1_rlvr,stage2_teachers,stage3_mopd}`.
- The repo's RL recipe stops at MOPD; MTP boosting is not part of it (per `docs/nemotron/super35/rl/mopd.md`).
- Where the repo docs give values the report does not (e.g. selected teacher steps for agentic and vision
  tool use, two alignment teachers), those come from reference-run configs, not the report.

# Caveats

- "Step-120" is the report's identifier for the RL trunk checkpoint; the total length of the RL run is not reported.
- The report names the trunk a "text-RLVR" checkpoint and Figure 4 says "Text Only Trunk RL student (VLM)": the model is a VLM but trunk RL appears to use text environments. The vision-reasoning teacher's "existing multimodal RLVR checkpoint" is not further identified.
- Figure 4 does not show MTP boosting or quantization; do not claim it does.
- Whether the GenRM drawn in Figure 4 for trunk RL is the same model as the alignment teacher's new length-penalized GenRM (§3.5) is not stated.
- The MOPD student/self-anchor in stage 1 is the RL checkpoint; in stage 2 it is the stage-1 model.
