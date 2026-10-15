# Stage 1: Supervised Fine-Tuning (SFT)

Turn the Megatron-format Super-VL checkpoint into an instruction-following
vision-language model with Megatron-Bridge's Nemotron 3.5 Super VL SFT recipes.

## Overview

The tech report (§3.1) post-trains the text-only base into a VLM by warming up the vision
projector and then running **two SFT stages**: a general mix of vision, reasoning, chat,
long-context, safety and agentic data, followed by a small second stage with a re-balanced
blend that boosts capabilities the first stage under-learned. This stage provides both SFT
steps on top of Megatron-Bridge's `nemotron_35_super_vl_sft_config`.

| Component | Description |
|-----------|-------------|
| `train.py` | Loads the MB recipe named in the YAML `recipe:` block, merges YAML/CLI overrides, exports the recipe's HybridEP/NVTE env vars and calls Megatron-Bridge `finetune()` with `nemotron_omni_step` |
| `config/default.yaml` | **SFT stage 1** — MB's verified 64x H100 SFT configuration |
| `config/stage2.yaml` | **SFT stage 2** — continues from stage 1 on a second blend; output `super35-sft-model` |
| `config/peft.yaml` | LoRA (rank 32) on 16x H100 — MB's verified LoRA configuration |
| `config/long_context.yaml` | 128K text SFT on 128x GB200 — **unverified in Megatron-Bridge** |
| `config/tiny.yaml` | 10-step smoke test on the default layout (still the full 120B model) |

### What the recipe trains

| Component | SFT (`default`, `stage2`) | LoRA (`peft`) |
|-----------|---------------------------|---------------|
| Language model (hybrid Mamba-Transformer latent MoE) + shared MTP | trained | LoRA adapters |
| Vision projector | trained | frozen |
| RADIO vision encoder + temporal video embedder | frozen | frozen |

The shared MTP block is trained with two prediction depths at an auxiliary loss scale of 0.3.
Data is read from an Energon WebDataset through the Nemotron Omni task encoder (images and
video; no audio). The report fine-tunes language and vision components jointly; set
`model.freeze_vision_model=false` to unfreeze RADIO as well (see the freeze matrix in MB's
`examples/models/nemotron/nemotron_3/super_vl/README.md`).

### Default vs. production settings

The configs default to Megatron-Bridge's **verified** card so they run as-is. The report's
production run used a non-public blend and much longer packed sequences:

| Setting | Default (MB verified, H100) | Tech report §3.1 |
|---------|-----------------------------|------------------|
| Hardware | 8 nodes x 8 H100 | not stated |
| Parallelism | TP2 (SP) / PP2 / EP32, full recompute | not stated |
| Sequence length | 4,096 | packed up to 524,288 |
| Global batch | 1,280 | 32 |
| LR schedule | recipe default, 200 steps | warm-up over 100 packed samples (~9,100 records) to 1e-5, then constant |
| Optimizer | recipe default (distributed Adam, FP32 state) | Adam, beta1 0.9, beta2 0.95, weight decay 0.1, grad clip 1.0 |
| MTP | 2 shared depths, loss scale 0.3 | 2 shared depths, loss scale 0.3 |

The production values are documented in the `default.yaml` header with the matching override
keys (`optimizer.lr`, `scheduler.lr_warmup_iters`, `scheduler.lr_decay_style=constant`, ...).

## Data

The Super 3.5 SFT blends are not public. Bring your own Energon dataset (`energon prepare`
output: tar shards plus `.nv-meta/`) and register it:

```bash
# stage 1 blend
uv run nemotron super35 data import sft /path/to/energon/stage1
# stage 2 blend
uv run nemotron super35 data import sft /path/to/energon/stage2 --name super35-sft-stage2-data
```

TODO(open-source): replace with the public dataset ids once the Super 3.5 SFT data is released.

The `long_context` config builds its own text dataset (Hugging Face CoderForge at a pinned
revision, offline-packed to 131,072 tokens) and ignores `run.data`.

## Starting checkpoint

`run.model` must be a Megatron-format checkpoint. Either use
[Stage 0](../stage0_pretrain/README.md)'s output (`super35-pretrain-model`), or convert the
released Hugging Face checkpoint with Megatron-Bridge and register it:

```bash
uv run nemotron super35 model import pretrain /path/to/megatron/super35-vl
```

## Quick Start

```bash
# Preview the resolved config
uv run nemotron super35 sft --dry-run

# Smoke test
uv run nemotron super35 sft -c tiny --run YOUR-CLUSTER

# SFT stage 1 -> super35-sft-stage1-model
uv run nemotron super35 sft --run YOUR-CLUSTER

# SFT stage 2 -> super35-sft-model
uv run nemotron super35 sft -c stage2 --run YOUR-CLUSTER

# LoRA (2 nodes)
uv run nemotron super35 sft -c peft --run YOUR-CLUSTER

# Long-context SFT (32 GB200 nodes x 4 GPUs; unverified)
uv run nemotron super35 sft -c long_context --run YOUR-GB200-CLUSTER

# Production-style overrides (example)
uv run nemotron super35 sft --run YOUR-CLUSTER \
    optimizer.lr=1e-5 scheduler.lr_warmup_iters=100 scheduler.lr_decay_style=constant \
    optimizer.adam_beta2=0.95 optimizer.weight_decay=0.1 optimizer.clip_grad=1.0
```

## Artifacts

| Config | Input model | Input data | Output |
|--------|-------------|------------|--------|
| `default` | `super35-pretrain-model` | `super35-sft-data` | `super35-sft-stage1-model` |
| `stage2` | `super35-sft-stage1-model` | `super35-sft-stage2-data` | `super35-sft-model` |
| `peft` | `super35-pretrain-model` | `super35-sft-data` | `super35-sft-peft-model` |
| `long_context` | `super35-pretrain-model` | (CoderForge, from recipe) | `super35-sft-long-context-model` |
| `tiny` | `super35-pretrain-model` | `super35-sft-data` | `super35-sft-model-tiny` |

## Handing off to RL

The stage-2 output (`super35-sft-model`, Megatron format) feeds RL directly: the RL runner
converts it to Hugging Face on first use (written to `<ckpt>/hf`), so no manual export is
needed. To start RL from an externally produced checkpoint instead, register it:

```bash
uv run nemotron super35 model import sft /path/to/super35-sft-hf
```

For LoRA, MB notes that standalone HF adapter export does not preserve the trained router
expert-bias buffers: reload natively, merge into the full model, then export.

## Software

Megatron-Bridge commit
[`f7f37b26`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/f7f37b263307812702ee57bbc413a8a26c0924b5)
and its Megatron-LM submodule revision
[`bab851bf`](https://github.com/NVIDIA/Megatron-LM/commit/bab851bfe3abc805256155962389ec0fe5de7571)
are mounted over the NeMo 26.08 container (`run.env.mounts`).

## Next Steps

- [Stage 2: RL](../stage2_rl/README.md)
- [Stage 3: Evaluation](../stage3_eval/README.md)
- [Family overview](../README.md)
