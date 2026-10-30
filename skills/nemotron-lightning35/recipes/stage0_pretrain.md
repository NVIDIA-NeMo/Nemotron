---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "recipes-stage0-pretrain"
title: "Lightning 3.5 stage 0: Megatron-Bridge pretraining on open data"
source_path: "src/nemotron/recipes/lightning35/stage0_pretrain/"
currency: "evolving"
---

# Stage 0 recipe summary — Pretraining

Maps the report's base-model training (§2) to `src/nemotron/recipes/lightning35/stage0_pretrain/`.

---

## What the repo exposes

| Path | Role |
|---|---|
| `data_prep.py` | tokenizes the blend into Megatron bin/idx shards (`lightning35 data prep pretrain`) → `lightning35-pretrain-data` |
| `train.py` | Megatron-Bridge `nemotron_3_5_lightning_pretrain_config` (`lightning35 pretrain`) → `lightning35-pretrain-model` |
| `config/default.yaml`, `config/tiny.yaml` | production template and a 1-node test (1,700 iterations, GBS 8) |
| `config/data_prep/data_blend_raw.json` | open pretraining blend: Nemotron-CC-v2, Nemotron-CC-v2.1, Nemotron-CC-Math-v1, Nemotron-Pretraining-Specialized-v1 (plus `_small` / `_large` variants) |

Runspec default: 2 nodes × 8 GPUs. Tokenizer: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`. `lightning35 pipe` chains pretrain → SFT.

---

## What it does not reproduce

- The report builds Lightning's base from **Nemotron 3 Nano Base**: an MTP head initialized by projection fusion and pruning, a 100B-token MTP warm-up, **1T tokens of distillation from Super 3 Base**, then long-context training (`../paper/pretraining.md`). The recipe instead pretrains the Lightning architecture from random initialization with Megatron-Bridge on open data; `docs/nemotron/lightning35/pretrain.md` and the stage README say so and describe the report's stages.
- The open blend is a single mix (23 subsets, all weight 1.0); there is no phase curriculum. The MTP loss scale is 0.3 (report distillation: 0.1); the LR schedule and iteration count are inherited from Megatron-Bridge's Nano 3 pretraining config.
- The Base model card describes the base differently from the report (>20T tokens, NVFP4 recipe, MTP continued pretraining, no distillation); the docs follow the report and note the card. See `../paper/pretraining.md`.
- Open data only; results will differ from the model card.

---

## Reproduce with nemotron-customize

`pretrain/megatron_bridge` matches the stage shape but does not list Lightning in its `[[models]]`; `data_prep/pretrain_prep` matches the tokenization step. Ground a Lightning pretraining build on `src/nemotron/recipes/lightning35/stage0_pretrain/` (Explorer mode for anything the catalog does not cover).
