# Stage 0: Omni SFT

Run supervised fine-tuning for Nemotron Omni starting from the GA checkpoint.

## Overview

Omni does not ship a pre-baked training container. This stage owns:

| Component | Description |
| --- | --- |
| `Dockerfile` | Builds the Megatron-Bridge `dev/nomni` environment |
| `build.py` | Exports the image as `omni3-sft.tar` |
| `data_prep.py` | Validates or stages a prepared Valor32k Energon dataset |
| `train.py` | Thin wrapper around `scripts/training/run_recipe.py` |
| `config/` | QA-guide-derived training and data-prep configs |

## Quick start

```bash
# 1. Build the container
uv run nemotron omni3 build sft --run YOUR-CLUSTER

# 2. Validate or stage the Valor32k Energon dataset
uv run nemotron omni3 data prep sft --run YOUR-CLUSTER

# 3. Convert the GA HF checkpoint to Megatron format
uv run nemotron omni3 model import pretrain --run YOUR-CLUSTER \
  --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning \
  --megatron-path /checkpoints/nemotron_omni

# 4. Launch SFT
uv run nemotron omni3 sft --run YOUR-CLUSTER
```

## Data prep

`data_prep.py` assumes a prepared Valor32k Energon dataset already exists (matching
the QA guide's DFW flow). It:

- validates `dataset_path`
- optionally runs a site-local `builder_command` if the dataset is missing
- writes a small `manifest.json` under `metadata_dir`
- optionally refreshes a convenience symlink with `link_path`

This keeps the recipe usable before the upstream Valor32k shard-builder is
public while still giving the CLI a concrete staging hook.

## Container image

The canonical image archive path used by configs is:

```text
oci-archive:///home/$USER/.cache/nemotron/containers/omni3-sft.tar
```

`build.py` respects `NEMOTRON_CACHE_DIR`, which the `nemotron omni3 build`
dispatcher sets to `/nemotron-cache` on remote builds.

## Training configs

The training configs port the QA guide variants:

- `default.yaml` — full Valor32k SFT
- `image_text_sft.yaml` — image-text projector-only SFT
- `image_text_peft.yaml` — image-text PEFT
- `audio_text.yaml` — audio-text SFT
- `peft_valor32k.yaml` — Valor32k LoRA / PEFT
- `tiny.yaml` — small smoke-test config

Use `-c <name>` to select a variant:

```bash
uv run nemotron omni3 sft -c image_text_peft --run YOUR-CLUSTER
```
