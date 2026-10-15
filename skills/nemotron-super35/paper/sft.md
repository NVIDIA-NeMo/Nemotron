---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "sft"
paper_sections: ["3", "3.1", "3.9", "Table 7", "Tables 25-26"]
title: "Nemotron 3.5 Super Post-Training: Stage Order and Two-Stage Vision-Language SFT"
summary: |
  Covers the opening of the Nemotron 3.5 Super post-training pipeline (§3, Figure 4) and the SFT
  stage (§3.1). Super's base model is pre-trained on text only, so SFT is where it becomes a VLM:
  after a vision-projector warm-up, a two-stage SFT jointly fine-tunes language and vision components
  on vision, agentic-vision, reasoning, chat, safety, long-context and agentic data. Records the SFT
  data domains (teachers, filters, sizes where given), the training config (524,288-token packed
  sequences, GBS 32, constant LR 1e-5, two shared-weight MTP layers at aux-loss 0.3), and the
  SFT-side preparation for reasoning-effort control (§3.9). Lightning (30B-A3B) is mentioned only
  to disambiguate: it gets a single text-only SFT stage.
key_facts:
  - "§3: Super's post-training pipeline is 'similar to that used in Nemotron 3 Ultra': SFT, RLVR, expert teacher model training, and finally MOPD (Figure 4). Lightning goes through a lighter pipeline: text-only SFT and RLVR."
  - "§3.1: Super is trained to be a VLM; Lightning is text-only. Because the Super base model is pre-trained on text-only data, large quantities of vision data are introduced in SFT, along with agentic and reasoning text data."
  - "§3.1: the vision projector is warmed up first (citing NVIDIA, 2026a, the Nemotron 3 Nano Omni report); warm-up hyperparameters are not reported in this report."
  - "§3.1: SFT is two-stage 'because learning vision and text capabilities simultaneously is challenging'. Stage 1 = general mix of vision, reasoning, chat, long-context, safety, and agentic data. Stage 2 = a 'small' second stage with changed blend composition to boost capabilities stage 1 did not learn adequately. Lightning uses a single SFT stage on text-only data."
  - "Vision data (§3.1): Nemotron 3 Nano Omni training corpus (perception/VQA over charts, tables, OCR, multi-page documents, grounding, GUI, video; visual math/science/spatial/temporal reasoning; multilingual mainly in document/OCR), augmented with newly curated data generated using open models such as Qwen3.5 (chart understanding, grounded perception, multi-hop reasoning, GUI/web navigation, multimodal coding)."
  - "Vision reasoning traces are organized as an explicit sequence of concise steps, each grounded in visual evidence; automated quality checks filter ambiguous/unsupported questions, incomplete traces, hallucinated labels/values, wrong visual/numerical interpretation, excessive repetition, unproductive self-correction (Figure 5). A chart-understanding version is released as a NeMo Data Designer recipe."
  - "Agentic vision data (§3.1): image tool operations (crop, rotate, zoom, flip, ...) on high-resolution images generated with Kimi K2.6 and K3; SpatialClaw video-agent demonstrations generated with Kimi K2.6. Final SpatialClaw corpus = 14,169 trajectories and 81,366 assistant actions; free-form answers need same-teacher text-only equivalence judging with confidence >= 0.90."
  - "Table 7 (SpatialClaw harness): agentic vision training raises the average over the listed spatial/video benchmarks from 50.5 (baseline, no agentic vision training) to 54.8 (+4.3)."
  - "Reasoning data (§3.1): same problem sets as Nemotron 3 Ultra (math, STEM, coding, proof). Math/STEM CoT and TIR trajectories regenerated with Kimi K3; proofs with DeepSeek V4 Pro; competitive programming with GLM-5.2. Responses shorter than 5,000 tokens removed."
  - "§3.1 observation: with DeepSeek V4 Pro as primary reasoning teacher (Ultra), reasoning plateaued after one pass; with Kimi K3 (Super 3.5), performance kept improving over multiple passes. Authors hypothesize K3's more concise, information-dense reasoning needs more exposure for a smaller student."
  - "Chat and safety (§3.1): largely Ultra's recipe; chat refreshed with more recent Qwen, Kimi and Gemma responses; safety = largely the same blend as Ultra plus additional data on dual-risk CBRNE content and demographic biases (see safety.md)."
  - "Long-context data (§3.1): Ultra's 512K long-context data; synthetic graph-structured data (trace nodes/edges); agent trajectories from long-horizon search and coding; long-context translation from PG19 and FinePDFs-Edu translated in chunks with DeepSeek V4 Flash (Tables 25-26); synthetic transduction tasks; long-context vision (multi-page documents up to hundreds of pages) via NeMo Data Designer."
  - "Training config (§3.1): language and vision components fine-tuned jointly; packed sequences of up to 524,288 tokens; global batch size 32; LR warms up over 100 packed-sample-equivalents (approximately 9,100 original records) to 1e-5, then constant; Adam beta1 0.9, beta2 0.95, weight decay 0.1, gradient clipping 1.0; two shared-weight MTP layers retained with auxiliary-loss scaling 0.3."
  - "§3.9: medium-effort mode is introduced in SFT with data from Kimi K3's low-effort mode; following Ultra, a small fraction of SFT reasoning traces are truncated to random budgets (responses unchanged, </think> masked from the loss) to prepare for inference-time budget control."
related_steps:
  - "stage1_sft"
currency: "frozen"
---

# Scope

- What is the order of Nemotron 3.5 Super post-training stages, and where does SFT sit?
- Why is SFT two-stage, and what is in each stage?
- Which vision, agentic-vision, reasoning, chat, safety and long-context data are used, and with which teacher models?
- What are the SFT hyperparameters (sequence length, GBS, LR, optimizer, MTP)?
- How does Super's SFT differ from Lightning's and from Nemotron 3 Ultra's?

# Post-training stage order (§3, Figure 4)

The report says Super's pipeline "is similar to that used in Nemotron 3 Ultra", consisting of SFT, RLVR,
expert teacher model training, and MOPD (Figure 4). Combining §3.1-§3.7 and §4:

| Order | Stage | Section |
| --- | --- | --- |
| 0 | Vision-projector warm-up | §3.1 (details cited to NVIDIA 2026a; not given here) |
| 1 | SFT stage 1 (general mix) then SFT stage 2 (small, re-weighted) | §3.1 |
| 1b | Short MTP-only warm start (4k steps) from the SFT checkpoint, backbone frozen | §3.7 |
| 2 | RL (joint RLVR + RLHF); step-120 checkpoint = MOPD student and teacher root | §3.2, §3.6 |
| 3 | Specialist teachers (agentic, reasoning, video, vision tool use, vision reasoning, alignment) | §3.3-§3.5 |
| 4 | MOPD stage 1, then MOPD stage 2 | §3.6 |
| 5 | MTP boosting (25k steps, backbone frozen) | §3.7 |
| 6 | NVFP4 quantization (PTQ / QAD) | §4 |

See `rl/overview.md` for the full picture. Lightning: single text-only SFT stage + RLVR, no MOPD (§3, §3.6).

# Two-stage SFT (§3.1)

| Item | Report statement |
| --- | --- |
| Starting point | Text-only pre-trained Super 3.5 base |
| Before SFT | Vision projector warm-up (NVIDIA, 2026a) |
| Why two stages | "learning vision and text capabilities simultaneously is challenging" |
| Stage 1 | General mix of vision, reasoning, chat, long-context, safety, and agentic data |
| Stage 2 | "small second stage"; blend composition changed to boost capabilities stage 1 did not learn adequately |
| Trainable parts | Language and vision components fine-tuned jointly |
| Lightning | Single SFT stage, text-only data |

Not reported: stage-2 blend composition, stage-1/stage-2 sample or token counts, number of steps,
projector warm-up hyperparameters, or whether stage 2 uses different hyperparameters.

# SFT training config (§3.1)

| Setting | Value |
| --- | --- |
| Packed sequence length | up to 524,288 tokens |
| Global batch size | 32 |
| LR warm-up | 100 packed-sample-equivalents (~9,100 original records) |
| Peak LR | 1e-5 |
| LR after warm-up | constant |
| Optimizer | Adam, beta1 = 0.9, beta2 = 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| MTP | two shared-weight MTP layers retained, auxiliary-loss scaling factor 0.3 |
| Total samples / steps | not reported |

For comparison only (Ultra report, not this report): Ultra SFT used 294,912-token packs, GBS 64, cosine
LR 1.5e-5 -> 1e-6 and MTP aux scaling 0.1. Do not mix these into Super 3.5 answers.

# SFT data domains (§3.1)

| Domain | What the report says |
| --- | --- |
| Vision | Nemotron 3 Nano Omni corpus (charts, tables, OCR, multi-page documents, grounding, GUI, video; visual math/science/spatial/temporal reasoning; multilingual mostly doc/OCR) + new data generated with open models such as Qwen3.5 (chart understanding, grounded perception, multi-hop reasoning, GUI/web navigation, multimodal coding). Grounded step-wise reasoning traces; quality filters (Figure 5). Chart pipeline released as a NeMo Data Designer recipe. |
| Agentic vision | Image tools (crop, rotate, zoom, flip, ...) on high-res images: Kimi K2.6 and K3. Video: Kimi K2.6 generates executable SpatialClaw demonstrations from video QA pairs (attribute perception, temporal, object reasoning, OCR, counting, synopsis). Actions = Purpose / Reasoning / Next Goal / Python code in a persistent kernel (frame inspection, VQA calls, SAM3 segmentation, scene-geometry reconstruction, ReturnAnswer). Keep correct, tool-successful trajectories; free-form answers judged by the same teacher text-only with confidence >= 0.90. Final: 14,169 trajectories, 81,366 assistant actions. |
| Reasoning | Ultra's problem sets (math, STEM, coding, proof). Math/STEM CoT + TIR regenerated with Kimi K3; proofs with DeepSeek V4 Pro; competitive programming with GLM-5.2. Drop responses < 5,000 tokens. Sizes not reported. |
| Chat | Ultra's recipe; refreshed with more recent Qwen, Kimi and Gemma responses. |
| Safety | Ultra's blend largely reused + extra dual-risk CBRNE and demographic-bias data. Size not reported. |
| Long context | Ultra's 512K long-context data; synthetic graph-structured data; long-horizon search/coding agent trajectories (Su et al., 2026); chunked long-document translation (PG19, FinePDFs-Edu) with DeepSeek V4 Flash; synthetic transduction tasks (Willette et al., 2026); long-context vision documents (up to hundreds of pages) from NeMo Data Designer. |
| Finance agent | §3.2: the finance-agent SDG produces trajectories routed to the SFT or RL pool by measured difficulty (see rl/rlvr.md). |

## Agentic-vision effect (Table 7, SpatialClaw harness)

Baseline = no agentic vision training. Average over the 20 listed benchmarks: **50.5 -> 54.8 (+4.3)**.
Largest gains: SpatialTree +10.4, SPAR-Bench +10.3, ERQA +8.3, VSI-Bench-U +7.4. Smallest: OSI-Bench +0.3,
PercepComp +0.3, Video-MME +0.9.

## Long-context translation data (Tables 25-26)

| Source | Records | Source tokens | Target tokens |
| --- | --- | --- | --- |
| PG19 (English source -> JA, ZH-Hans, IT, ES-MX, DE, FR) | 4,335 | 258,667,792 | 299,276,122 |
| FinePDFs-Edu (EN/ES/FR/ZH/JA sources -> 21 targets) | 1,438 | 620,467,507 | 701,331,885 |

Evaluated with SEGALE on PG19 (100 books x 6 languages = 600 translations, Figure 12).

# Reasoning-effort preparation in SFT (§3.9)

- Three modes: reasoning-off, regular, medium-effort.
- Medium-effort is introduced in SFT with data from **Kimi K3's low-effort mode**, then optimized in RL.
- Following Ultra, a small fraction of SFT reasoning traces are truncated to random budgets (responses
  unchanged, `</think>` masked from loss) to prepare for inference-time budget control. Fraction not reported.

# Repo cross-reference

`docs/nemotron/super35/sft.md` maps this section onto the `stage1_sft` recipe (`-c default` for stage 1,
`-c stage2` for stage 2). The SFT data used for Super 3.5 is not yet published.

# Caveats

- One training config is reported for SFT; the report does not say whether stage 2 uses different values.
- "~9,100 original records" and the 0.90 confidence threshold are as stated; sample/step totals are not reported.
- The "512K long-context data" (reused from Ultra) and the 524,288-token pack length are numerically the same (512 x 1024), but the report states them separately and does not say they are the same setting; do not conflate.
- Table 7 compares with/without agentic vision training using the SpatialClaw harness; it is not the main eval suite (Table 10).
- Lightning's SFT is single-stage and text-only; do not attribute Super's vision data or config to Lightning.
- The report contains a placeholder "TODO UPDATE URL" for the chart-understanding dev note in §3.1; the NeMo Data Designer footnote URLs are given.
