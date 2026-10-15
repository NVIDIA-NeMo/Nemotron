---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "recipes-stage2-rl"
title: "Super 3.5 RL recipes: RLVR, teachers, MOPD"
source_path: "src/nemotron/recipes/super35/stage2_rl/"
currency: "evolving"
---

# Stage 2 recipe summary — RL

Maps the paper's RL stages (§3.2–3.6) to `src/nemotron/recipes/super35/stage2_rl/`. All stages run NeMo-RL's `examples/nemo_gym/run_grpo_nemo_gym.py` (NeMo-RL `super-v3.5-posttraining` @ `f299847f1`, NeMo Gym `e24db4e6`) through the shared runner `_nemo_rl.py`. The configs are ported from the reference run that produced the released checkpoint.

Values below are from the checked-in YAMLs; re-read the YAML before quoting, since configs can change.

---

## Lineage and selected steps (reference run)

| Stage | Starts from | Step used downstream |
|---|---|---|
| RLVR | SFT | 120 |
| Reasoning teacher | RLVR | 80 |
| Vision (tool use) teacher | RLVR | 40 |
| Video teacher | RLVR | 60 |
| Vision-reasoning teacher | RLVR → stage 1 (114) → stage 2 | 20 |
| Alignment (RLHF) teacher 1 | SFT | 120, then full resume to 500 |
| Alignment (RLHF) teacher 2 | RLVR | stage 1 step 200 → stage 2 (weights only) step 120 |
| Agentic teacher | RLVR | 14 (config only; cannot be launched publicly) |
| MOPD phase 1 | RLVR + agentic, reasoning, RLHF 1, vision | 30 |
| MOPD phase 2 | MOPD phase 1 + RLHF 2, vision, vision reasoning, video | 30 (= released backbone, before MTP boosting) |

---

## Key config values

| Config | Nodes (4 GPUs each) | Prompts × gens | Train GBS | Max seq | LR | TP / CP / EP |
|---|---:|---:|---:|---:|---:|---|
| `stage1_rlvr/default` | 144 | 512 × 16 | 8192 | 81,920 | 4e-6 | 4 / 2 / 16 |
| `teachers/reasoning/default` | 64 | 256 × 16 | 4096 | 262,144 | 4e-6 | 4 / 16 / 32 |
| `teachers/vision/default` | 16 | 16 × 8 | 128 | 32,768 | 3e-6 | 8 / 2 / 16 |
| `teachers/video/default` | 32 | 128 × 16 | 2048 | 65,536 | 3e-6 | 2 / 1 / 16 |
| `teachers/vision_reasoning/stage1` | 32 | 256 × 16 | 4096 | 32,768 | 3e-6 | 2 / 1 / 16 |
| `teachers/vision_reasoning/stage2` | 33 (32 + Ray head) | 256 × 16 | 4096 | 32,768 | 3e-6 | 2 / 2 / 16 |
| `teachers/rlhf/teacher1_stage{1,2}` | 32 | 32 × 32 | 1024 | 49,152 | 2e-6 | 8 / 8 / 16 |
| `teachers/rlhf/teacher2_stage1` | 64 | 32 × 32 | 1024 | 49,152 | 2e-6 | 8 / 8 / 16 |
| `teachers/rlhf/teacher2_stage2` | 32 | 32 × 32 | 1024 | 49,152 | 2e-6 | 8 / 8 / 16 |
| `teachers/agentic/default` | 128 | 64 × 16 | 1024 | 196,608 | 3e-6 | 8 / 8 / 8 |
| `stage3_mopd/phase1` | 128 | 512 × 1 | 512 | 131,072 | 2e-6 | 4 / 4 / 16 |
| `stage3_mopd/single_stage` | 112 | 512 × 1 | 512 | 131,072 | 2e-6 | 4 / 4 / 16 |
| `stage3_mopd/phase2` | 128 | 1024 × 1 | 1024 | 131,072 | 3e-6 | 8 / 8 / 16 |

All configs: GB200-style 4-GPU nodes, KL penalty 0, async GRPO enabled. RLVR's judges and GenRM run as separate vLLM servers outside this allocation (26 more nodes in the reference run). Every RL config keeps the shared MTP head in training: `mtp_num_layers: 5` with a repeated (weight-shared) layer, detached heads, loss scale 0.3, as in the reference run.

---

## Runner behaviour (`_nemo_rl.py`)

- resolves only `${art:...}` artifact references; secrets stay as `oc.env` interpolations, so their values never land in the handed-off YAML;
- converts Megatron-Bridge and NeMo-RL `step_N` checkpoints to HF under `<ckpt>/hf`, or `<converted_checkpoint_dir>/<name>_hf` when set (policy, and each MOPD teacher);
- maps `pretrained_checkpoint.path` to `step_N/policy/weights` for weights-only resumes;
- names the output artifact from `run.output_artifact`.

---

## Gaps to state

- Data: every stage reads a data artifact (`super35-rl-*-data`) that users must register (`super35 data import rl`).
- The report's length-penalized GenRM variant is not public; NeMo-RL's public recipe uses `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM`.
- Agentic teacher: blocked (legacy Terminal-Bench agent and configs not in public NeMo Gym). `rl mopd -c single_stage` (port of NeMo-RL's `mopd.yaml`) is MOPD phase 1 without it: agentic routes go to the self-anchor.
- Vision-reasoning stage 2: structured-answer verifier unpublished. The reasoning teacher's STEM judge prompt ships with NeMo-RL (`examples/nemo_gym/nemotron-3.5-super/prompts/reasoning_semantic.txt`) and is the config default.
- NeMo-RL's own [Super 3.5 guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3.5-posttraining/docs/guides/models/nemotron/nemotron-3.5-super-vl.md) is a shorter lineage (one RLHF teacher, no video/vision-reasoning/agentic teachers, single-stage MOPD); see `docs/nemotron/super35/rl/index.md → Relation to NeMo-RL's Super 3.5 guide`.
- MTP boosting (§3.7) has no recipe.

Human guides: `docs/nemotron/super35/rl/{index,rlvr,teachers,mopd}.md`.

---

## Reproduce with nemotron-customize

`rl/nemo_rl/rlvr` (NeMo-RL GRPO) matches the RLVR stage shape, but not this recipe's NeMo Gym environment stack, pinned NeMo-RL branch, or vision-language checkpoint handling. Specialist teachers and MOPD have no catalog step. Ground Super 3.5 RL builds on `src/nemotron/recipes/super35/stage2_rl/` (configs, `_nemo_rl.py`, `Dockerfile`), and treat anything beyond RLVR as Explorer mode.
