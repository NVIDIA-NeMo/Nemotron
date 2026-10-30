---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "recipes-stage2-rl"
title: "Lightning 3.5 stage 2: GRPO RLVR on NeMo-RL"
source_path: "src/nemotron/recipes/lightning35/stage2_rl/"
currency: "evolving"
---

# Stage 2 recipe summary — Reinforcement Learning

Maps the report's RLVR stage (§1, §3; §3.2's RL details are Super-only; Lightning has no teachers and no MOPD) to `src/nemotron/recipes/lightning35/stage2_rl/`.

---

## What the repo exposes

| Path | Role |
|---|---|
| `data_prep.py` | resolves placeholder records and writes NeMo Gym JSONL, one file per split the HF dataset publishes (`validation` → `val`); no split ratio is applied. Output `output/stage2_rl_resolved/` with `manifest.json` (`lightning35 data prep rl`) → `lightning35-rl-data` |
| `train.py` | NeMo-RL GRPO with Ray and NeMo Gym environments (`lightning35 rl`) → `lightning35-rl-model` |
| `config/default.yaml` | see below |
| `config/data_prep/data_blend_raw.json` | `nvidia/Nemotron-3.5-Lightning-RL-Training-Blend` |

Key values in `config/default.yaml` (re-read the YAML before quoting):

| Setting | Value |
|---|---|
| Container | `nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_5_lightning` |
| Cluster | 32 nodes × 8 GPUs |
| Prompts × generations | 512 × 16 (train GBS 8,192) |
| Max sequence length | 73,728 |
| LR | 4e-6 |
| TP / EP | 4 / 16 |
| MTP | 5 layers trained at loss scale 0.3 |
| Environments | math_with_judge (judge off), code_gen, workplace_assistant, mcqa, instruction_following, structured_outputs_json; single-turn |
| Async GRPO / overlong filtering | both off |

---

## What it does not reproduce

- The report's full RLVR environment set and data beyond the released RL blend (`../paper/rl.md`).
- MTP boosting (§3.7: 48K MTP-only steps for Lightning, no warm start) is not a stage.
- The BF16 model card's asynchronous RL with MTP-accelerated rollouts; the recipe runs synchronously.
- There is no RLHF, reward-model, or DPO stage, matching the report's "SFT and RLVR" description. (Earlier docs versions had GenRM/DPO sections copied from Nano 3; they were removed.)

---

## Reproduce with nemotron-customize

`rl/nemo_rl/rlvr` matches the stage shape but does not list Lightning in its `[[models]]`. Ground RL builds on `src/nemotron/recipes/lightning35/stage2_rl/` (config and container), with `data_prep/rl_prep` for the JSONL conversion. DGX Station GRPO: `usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/grpo.md`.
