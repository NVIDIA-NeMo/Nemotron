# Stage 0: Vision-Language Pretraining (Megatron-Bridge)

> **This is Megatron-Bridge's vision-language (image-caption) pretraining recipe
> for Nemotron 3.5 Super VL. It is _not_ a reproduction of the tech report's text
> pretraining**, which is not publicly reproducible (see
> [What the tech report did](#what-the-tech-report-did)).

## Overview

`nemotron super35 pretrain` runs Megatron-Bridge's
`nemotron_35_super_vl_pretrain_config`: image-caption training on an
[Energon](https://github.com/NVIDIA/Megatron-Energon) WebDataset with **every model stack
trainable** (hybrid Mamba-Transformer latent-MoE language model with its shared MTP head,
RADIO vision encoder, vision projector and temporal video embedder), warm-started from a
Super-VL checkpoint converted from Hugging Face.

| Component | Description |
|-----------|-------------|
| `train.py` | Loads the MB recipe named in the YAML `recipe:` block, merges YAML/CLI overrides, exports the recipe's HybridEP/NVTE env vars and calls Megatron-Bridge `pretrain()` with `nemotron_omni_step` |
| `config/default.yaml` | MB's verified 64x H100 configuration (below) |
| `config/tiny.yaml` | 10-step smoke test on the same layout (still the full 120B model) |

### Default configuration

The defaults reproduce Megatron-Bridge's verified H100 pretrain card
(`docs/models/nemotron/nemotron3.5-super-vl.md` at the pinned MB commit):

| Setting | Value |
|---------|-------|
| Hardware | 8 nodes x 8 H100 (64 GPUs), BF16, FP32 optimizer state |
| Parallelism | TP2 (sequence parallel) / PP2 / EP32 / ETP1, natural HybridEP routing |
| Sequence length | 4,096 |
| Global / micro batch | 1,280 / 1 |
| Steps | 200 (LR decay over 200 steps), checkpoints every 100 |
| MTP | two shared prediction depths, loss scale 0.3 |
| Forward step | `nemotron_omni_step` |

The MB card trained on a DataComp image-caption Energon dataset and reports a
loss drop from 3.37 to 1.92 over 200 steps. It is bounded support verification, not a
convergence run.

### Software

Megatron-Bridge is mounted at commit
[`f7f37b26`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/f7f37b263307812702ee57bbc413a8a26c0924b5)
(the latest MB `main` commit touching `recipes/nemotronh_multimodal/nemotron_35_super_vl.py`),
and Megatron-LM at
[`bab851bf`](https://github.com/NVIDIA/Megatron-LM/commit/bab851bfe3abc805256155962389ec0fe5de7571),
the `3rdparty/Megatron-LM` submodule revision that MB commit records. Both are
mounted over the NeMo 26.08 container via `${auto_mount:...}` in `run.env.mounts`.

## Prerequisites

1. **A Megatron-format starting checkpoint.** Convert the Hugging Face checkpoint with
   Megatron-Bridge (MB's verified import used 1 node x 4 GPUs, TP1/PP2/EP2):

   ```bash
   # from a Megatron-Bridge checkout at the pinned commit (command from MB's card)
   ./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 4 \
       --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
       --megatron-path /path/to/megatron/super35-vl \
       --torch-dtype bfloat16 --tp 1 --pp 2 --ep 2 --etp 1 --trust-remote-code
   ```

   The released Base checkpoint
   ([`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16))
   converts with the same command and `--hf-model` swapped; MB verified the round trip on the
   post-trained BF16 checkpoint only.

2. **An Energon image-caption dataset** (`energon prepare` output: tar shards plus
   `.nv-meta/`). Samples may contain text, images and video, but no audio.

Point the config at both (paths must be visible from the compute nodes):

```bash
export SUPER35_INIT_CHECKPOINT=/path/to/megatron/super35-vl/iter_0000000
export SUPER35_PRETRAIN_DATA=/path/to/energon/image-caption-dataset
```

## Quick Start

```bash
# Preview the resolved config
uv run nemotron super35 pretrain --dry-run

# Smoke test (10 steps, same 8-node layout)
uv run nemotron super35 pretrain -c tiny --run YOUR-CLUSTER

# Verified MB configuration (200 steps)
uv run nemotron super35 pretrain --run YOUR-CLUSTER

# Overrides use Hydra-style dotlists
uv run nemotron super35 pretrain --run YOUR-CLUSTER train.train_iters=1000 scheduler.lr_decay_iters=1000
```

The checkpoint is written to `/nemo_run/super35-pretrain-model` and registered as the
`super35-pretrain-model` artifact, which [Stage 1: SFT](../stage1_sft/README.md) consumes.

To start SFT from the released checkpoint instead, register the converted checkpoint
directly:

```bash
uv run nemotron super35 model import pretrain /path/to/megatron/super35-vl
```

## What the tech report did

The Nemotron 3.5 Super base model was built with a long **text-only** pipeline starting
from Nemotron 3 Super Base. Its data and teacher logits are not public, so it cannot be
reproduced from this repository:

1. **MTP initialization and warm-up** — the shared MTP head is initialized from the first of
   Nemotron 3 Super Base's two MTP heads, then trained for 100B tokens with the backbone frozen,
   using hidden-state mixing to reduce the train/inference mismatch of recursive drafting.
2. **Distillation** — 1T tokens of distillation from Nemotron 3 Ultra Base (Skew Reverse KL,
   alpha 0.1, top-128 logits, MTP loss scale 0.1), LR ramped from 4.5e-6 to 4.5e-5 over 200B tokens
   and decayed back over the remaining 800B.
3. **Indic continued pretraining** — 100B tokens mixing 10B tokens of synthetic and natural Indic
   data (from a ~2.83T-token synthetic corpus over 22 Indian languages) with 90B replay tokens.
4. **Long-context extension** — 35B tokens of continued pretraining at 1M context
   (constant LR 4.5e-6, CP16 / TP8 / EP64).
5. **Agentic software-engineering mid-training** — 65.4B tokens at 256K context on repositories,
   issue-PR pairs, commit histories, build/CI logs and a small share of agent traces
   (LR warm-up to 1e-5, then constant).
6. **Long-context recovery** — ~9B tokens with the long-context recipe, restoring RULER-1M from 33.5
   to 81.0 after mid-training.

The resulting base model is text-only. Vision is introduced in post-training: the report warms
up the vision projector and then runs a two-stage SFT on vision, reasoning, chat, long-context,
safety and agentic data ([Stage 1](../stage1_sft/README.md)). The image-caption pretraining in
this stage is the closest public analogue of that projector/vision alignment step, not of the
text pretraining above.

## Next Steps

- [Stage 1: SFT](../stage1_sft/README.md)
- [Family overview](../README.md)
