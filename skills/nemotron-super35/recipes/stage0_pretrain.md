---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "recipes-stage0-pretrain"
title: "Super 3.5 stage 0: Megatron-Bridge vision-language pretraining (not the paper's pretraining)"
source_path: "src/nemotron/recipes/super35/stage0_pretrain/"
currency: "evolving"
---

# Stage 0 recipe summary — Pretraining

Maps the paper's pretraining (§2) to `src/nemotron/recipes/super35/stage0_pretrain/`.

---

## What the repo exposes

`nemotron super35 pretrain` runs Megatron-Bridge's **vision-language image-caption** pretraining recipe, `megatron.bridge.recipes.nemotronh_multimodal.nemotron_35_super_vl_pretrain_config`, with forward step `nemotron_omni_step`. It warm-starts from a Super-VL checkpoint converted from Hugging Face and trains every stack.

| Path | Role |
|---|---|
| `train.py` | loads the recipe named in the YAML `recipe:` block, merges YAML/CLI overrides, calls Megatron-Bridge `pretrain()` |
| `config/default.yaml` | Megatron-Bridge verified H100 card: 8 × 8 H100, TP2 / PP2 / EP32, seq 4,096, GBS 1,280, 200 steps |
| `config/tiny.yaml` | 10-step smoke test on the same layout (still the full 120B model) |

Inputs come from environment variables, not artifacts: `SUPER35_INIT_CHECKPOINT` (converted Megatron checkpoint) and `SUPER35_PRETRAIN_DATA` (Energon image-caption dataset). Output: `super35-pretrain-model`.

Verification: Megatron-Bridge reports 200 steps on 64 × H100 with DataComp image-caption data (loss 3.37 → 1.92). That is bounded support verification, not convergence.

---

## What it does not reproduce

The paper's pretraining is **text-only** and continues from Nemotron 3 Super Base: MTP warm-up (100B), distillation from Ultra 3 Base (1T), Indic CPT (100B), 1M context (35B), agentic mid-training (65.4B at 256K), LC recovery (~9B). It needs Ultra 3 Base teacher logits and non-public data. See `../paper/pretraining.md` and `../paper/data.md`.

Most users should skip this stage and start SFT from a converted checkpoint (`super35 model import pretrain`).

---

## Reproduce with nemotron-customize

No catalog step covers this model: `pretrain/megatron_bridge` lists only text models in its `[[models]]` and has no vision-language data path. Treat a Super 3.5 pretraining build as an **Explorer-mode** or direct recipe task grounded on `src/nemotron/recipes/super35/stage0_pretrain/`. `convert/hf_to_megatron` covers the checkpoint conversion only if the user confirms it supports the Super-VL bridge.
