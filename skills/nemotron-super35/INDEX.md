# Nemotron 3.5 Super Knowledge Map

This skill is organized around four source layers:

1. **Skill routing**: how `/nemotron-super35` should locate and cite sources
2. **Model card**: release identity, checkpoints, deployment facts
3. **Paper chunks**: architecture, pretraining, SFT, RL/MOPD, evaluation, quantization, inference, safety
4. **Recipe tracker**: what `nemotron super35` runs, and how close it is to the paper

---

## Start here

- `context/quick-reference.md`: compact reference
- `SKILL.md`: invocation rules, source priority, caveats, handoff boundaries
- `model-card.md`: release-facing facts
- `paper/`: research and benchmark content
- `recipes/`: runnable recipe map and gaps

---

## Question → file map

| Question type | Primary file(s) | Notes |
|---|---|---|
| What is Nemotron 3.5 Super? | `model-card.md`, `paper/_overview.md` | identity, checkpoints, headline claims |
| How is it different from Super 3? | `SKILL.md`, `paper/architecture.md`, `paper/rl/overview.md` | VLM via SFT, distillation pretraining, Ultra-style MOPD |
| Architecture / vision stack / MTP head | `paper/architecture.md` | no dims table in the report; backbone inherited from Super 3 |
| How was the base model trained? | `paper/pretraining.md` | MTP warm-up, Ultra 3 distillation, 1M context, mid-training, recovery |
| Indic data / mid-training mix | `paper/data.md` | Nemotron-Pretraining-Indic-v1, Figure 2 pipeline, Tables 1–5 |
| How does SFT add vision? | `paper/sft.md` | two stages, vision and agentic-vision data, hyperparameters |
| Long-document translation data | `paper/long-translation.md` | PG19 / FinePDFs-Edu SFT data, Figure 12 |
| Post-training pipeline overview | `paper/rl/overview.md` | Figure 4 |
| RLVR trunk | `paper/rl/rlvr.md` | joint RLVR + RLHF, new environments, step 120 |
| Specialist teachers | `paper/rl/teachers.md` | agentic, reasoning, video, vision tool use, vision reasoning, alignment |
| What is MOPD? | `paper/rl/mopd.md` | reverse-KL dense advantage, routing, self-anchor, proximal teacher |
| MTP boosting / acceptance length | `paper/rl/mtp-boosting.md` | Tables 8–9, Figure 6 |
| Benchmark results | `paper/evaluation.md` | base (Table 6), post-trained BF16 (Table 10), NVFP4 (Table 11, Fig 7) |
| Contamination | `paper/decontamination.md` | §3.12, Appendix D |
| NVFP4 recipe and results | `paper/quantization.md` | Tables 12–22 (Table 21 is Lightning) |
| Why some layers stay in higher precision | `paper/quantization-ablations.md` | Appendix C |
| Serving and throughput | `paper/inference.md` | Figures 10–11, Tables 23–24 |
| Safety | `paper/safety.md` | what is and is not reported |
| How do I run it here? | `recipes/overview.md` | stage map, CLI, artifacts, gaps |
| What does `super35 pretrain` train? | `recipes/stage0_pretrain.md` | VL image-caption, not the paper's text pretraining |
| How do the SFT configs relate to §3.1? | `recipes/stage1_sft.md` | Megatron-Bridge verified defaults vs production settings |
| RL configs, lineage, selected steps | `recipes/stage2_rl.md` | RLVR, teachers, MOPD |
| How do I evaluate? | `recipes/stage3_eval.md` | NeMo Evaluator subset of Table 10 |
| Build it with `/nemotron-customize` | `context/quick-reference.md` → step map | no catalog step lists Super 3.5 VL yet |

---

## Important distinctions

### 1. Super vs Lightning
The report covers both. Lightning (30B-A3B) is text-only, has one SFT stage, RLVR, and no MOPD. Never attribute a Lightning number to Super.

### 2. Paper vs reference run vs repo
- **Paper**: what NVIDIA reports.
- **Reference run**: the internal training run the repo's RL configs were ported from. It adds detail the paper omits (for example two alignment teachers and the exact selected steps). `recipes/` labels these as repo facts.
- **Repo**: what `nemotron super35` runs today, including gaps.

### 3. Base vs post-trained BF16 vs NVFP4 vs throughput
- **Base**: Table 6, text-only base model.
- **Post-trained BF16**: Table 10 (precision inferred from Table 22).
- **NVFP4**: Table 11 (Text Arena), Figure 7, Table 22.
- **Throughput**: condition-specific serving measurements (`paper/inference.md`).

### 4. Three MTP training phases
Pretraining MTP warm-up (§2.1), a 4K-step MTP-only run on the SFT checkpoint, and 25K-step MTP boosting after MOPD (§3.7).

---

## Suggested read sequences

- **Architecture**: `model-card.md` → `paper/architecture.md` → `paper/rl/mtp-boosting.md`
- **Training methodology**: `paper/_overview.md` → `paper/pretraining.md` → `paper/sft.md` → `paper/rl/overview.md` → `paper/rl/mopd.md`
- **Deployment**: `model-card.md` → `paper/quantization.md` → `paper/inference.md`
- **Reproduction**: `recipes/overview.md` → `recipes/stage2_rl.md` → `docs/nemotron/super35/`
