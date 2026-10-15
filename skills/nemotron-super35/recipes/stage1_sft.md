---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "recipes-stage1-sft"
title: "Super 3.5 stage 1: two-stage vision-language SFT on Megatron-Bridge"
source_path: "src/nemotron/recipes/super35/stage1_sft/"
currency: "evolving"
---

# Stage 1 recipe summary — Supervised Fine-Tuning

Maps the paper's SFT (§3.1) to `src/nemotron/recipes/super35/stage1_sft/`.

---

## What the repo exposes

| Config | CLI | Recipe | Input → output |
|---|---|---|---|
| `default` (SFT stage 1) | `super35 sft` | `nemotron_35_super_vl_sft_config` | `super35-pretrain-model` + `super35-sft-data` → `super35-sft-stage1-model` |
| `stage2` (SFT stage 2) | `super35 sft -c stage2` | same | `super35-sft-stage1-model` + `super35-sft-stage2-data` → `super35-sft-model` |
| `peft` | `super35 sft -c peft` | `nemotron_35_super_vl_peft_config` (LoRA) | → `super35-sft-peft-model` |
| `long_context` | `super35 sft -c long_context` | `nemotron_35_super_vl_sft_long_context_128gpu_gb200_bf16_config` | → `super35-sft-long-context-model` (unverified in Megatron-Bridge) |
| `tiny` | `super35 sft -c tiny` | same as default | 10-step smoke test |

- Data: Energon WebDataset (images and video) through the Nemotron Omni task encoder; register with `super35 data import sft`.
- Default freeze matrix: language model and projector trained, vision encoder frozen (`model.freeze_vision_model=false` unfreezes it; the paper fine-tunes language and vision jointly).
- Shared MTP block trained with two depths at loss scale 0.3, as in the paper.

---

## What it does not reproduce

| Setting | Repo default (Megatron-Bridge verified card) | Paper §3.1 |
|---|---|---|
| Sequence length | 4,096 | packed up to 524,288 |
| Global batch | 1,280 | 32 |
| LR | recipe default, 200 steps (stage 2: 100) | warm-up to 1e-5, then constant |
| Data | user-supplied | non-public SFT blends |
| Projector warm-up | not a separate stage | precedes SFT (settings not reported) |

The production values and their override keys are documented in the `default.yaml` header. The Super 3.5 SFT blends are not public.

Verification: Megatron-Bridge reports 200-step image-caption runs on 64 × H100 and 64 × GB200 (not video, not convergence).

---

## Reproduce with nemotron-customize

`sft/megatron_bridge` and `peft/megatron_bridge` are the matching catalog steps, but neither lists Super 3.5 VL in `[[models]]`, and `data_prep/sft_packing` produces packed text Parquet, not the Energon multimodal data this recipe needs. Ground a Super 3.5 SFT build on `src/nemotron/recipes/super35/stage1_sft/` directly, or treat it as Explorer mode; do not claim catalog support until the step manifests add the model.
