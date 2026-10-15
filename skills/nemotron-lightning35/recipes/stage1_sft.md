---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "recipes-stage1-sft"
title: "Lightning 3.5 stage 1: text-only SFT on Megatron-Bridge"
source_path: "src/nemotron/recipes/lightning35/stage1_sft/"
currency: "evolving"
---

# Stage 1 recipe summary — Supervised Fine-Tuning

Maps the report's single text-only SFT stage (§3.1) to `src/nemotron/recipes/lightning35/stage1_sft/`.

---

## What the repo exposes

| Path | Role |
|---|---|
| `data_prep.py` | OpenAI chat format → chat template, role-based loss mask, packed Parquet (`lightning35 data prep sft`) → `lightning35-sft-data` |
| `train.py` | Megatron-Bridge `nemotron_3_5_lightning_sft_config` (`lightning35 sft`) → `lightning35-sft-model` |
| `config/default.yaml` | from `lightning35-pretrain-model`; sequence length = the data artifact's `pack_size` (4,096 in the data-prep config); GBS 128; 100 iterations; LR 5e-6; TP 2 / EP 8 |
| `config/data_prep/data_blend_raw.json` | open SFT blend: Nemotron-Agentic-v1, Nemotron-Competitive-Programming-v1, Nemotron-Instruction-Following-Chat-v1, Nemotron-Math-Proofs-v1, Nemotron-SWE-v1 |

Runspec default: 1 node × 8 GPUs. DGX Station SFT and LoRA variants: `usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/`.

---

## What it does not reproduce

- The report's SFT data and production hyperparameters for Lightning (`../paper/sft.md`); the recipe trains on the open subset at a short pack size.
- Open data only; results will differ from the model card.

---

## Reproduce with nemotron-customize

This stage maps cleanly to catalog steps, and Lightning is listed in `sft/megatron_bridge`'s `[[models]]`:

1. `data_prep/sft_packing` (packed Parquet)
2. `sft/megatron_bridge`

`peft/megatron_bridge` and `sft/automodel` are alternatives for LoRA or smaller GPU counts; confirm Lightning support in their manifests before relying on them.
