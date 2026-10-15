# Nemotron 3.5 Super Training Recipe

Training recipes for **Nemotron 3.5 Super** (120B total / 12B active parameters): Megatron-Bridge
vision-language pretraining and two-stage SFT, the NeMo-RL post-training pipeline (RLVR,
specialist teachers, multi-teacher on-policy distillation), and NeMo Evaluator evaluation.

## Paper

*Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer
Models for Agentic AI* (NVIDIA, 2026). Section references below (§) point to that report.

## Model Overview

| Property | Value |
|----------|-------|
| Parameters | 120B total, 12B active |
| Language model | Hybrid Mamba-Transformer with latent Mixture-of-Experts, initialized from Nemotron 3 Super |
| Vision | C-RADIO encoder + vision-language projector; the Megatron-Bridge implementation adds a temporal video embedder (the Nemotron Omni image/video stack, no audio) |
| Speculative decoding | One recursively shared MTP head |
| Context | Up to 1M tokens |
| Released checkpoints | Base BF16, post-trained BF16, NVFP4 |
| Hugging Face ids | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-{Base-BF16,BF16,NVFP4}` |

Post-training trains repeated MTP heads at loss coefficient 0.3; the report then boosts the head
(MTP boosting, §3.7), which this recipe does not include.

The base model is text-only; vision is introduced during SFT. The post-trained model reaches
51.78% on SWE-bench Pro (Table 10).

## Training Pipeline

| Stage | Directory | CLI | Framework | What it does |
|-------|-----------|-----|-----------|--------------|
| 0. Pretrain | [`stage0_pretrain/`](stage0_pretrain/README.md) | `nemotron super35 pretrain` | Megatron-Bridge | Megatron-Bridge vision-language (image-caption) pretraining, all components trainable. Not the report's text pretraining. |
| 1. SFT | [`stage1_sft/`](stage1_sft/README.md) | `nemotron super35 sft` (stage 1), `sft -c stage2` | Megatron-Bridge | Two SFT stages (general mix, then a small re-balanced stage). Also `-c peft` (LoRA) and `-c long_context` (128K, GB200, unverified). |
| 2. RL — RLVR | [`stage2_rl/stage1_rlvr/`](stage2_rl/stage1_rlvr/README.md) | `nemotron super35 rl rlvr` | NeMo-RL + NeMo Gym | Text-only RLVR from the SFT checkpoint; seeds the teachers and the MOPD student. |
| 2. RL — teachers | [`stage2_rl/stage2_teachers/`](stage2_rl/stage2_teachers/README.md) | `nemotron super35 rl teacher {reasoning,vision,video,vision-reasoning,rlhf,agentic}` | NeMo-RL + NeMo Gym | Six specialist teachers; agentic is config-only. See [Teachers](stage2_rl/stage2_teachers/README.md). |
| 2. RL — MOPD | [`stage2_rl/stage3_mopd/`](stage2_rl/stage3_mopd/README.md) | `nemotron super35 rl mopd` (phase 1), `rl mopd -c phase2` | NeMo-RL | Two-phase multi-teacher on-policy distillation; phase 2 produces the final backbone. `-c single_stage` runs NeMo-RL's one-stage variant without the agentic teacher. |
| 3. Eval | [`stage3_eval/`](stage3_eval/README.md) | `nemotron super35 eval` | NeMo Evaluator | vLLM-served checkpoint on the Table 10 benchmarks NeMo Evaluator supports. |

### About pretraining

The report's base model comes from a text-only pretraining pipeline (§2) that is not publicly
reproducible. Stage 0 instead runs Megatron-Bridge's Super-VL image-caption pretraining recipe,
warm-started from the released weights; most users should start at Stage 1 from the released
checkpoint. [Stage 0](stage0_pretrain/README.md#what-the-tech-report-did) lists the report's stages.

## Prerequisites

- **Megatron-Bridge stages (0, 1)** run in the NeMo 26.08 container with Megatron-Bridge commit
  [`f7f37b26`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/f7f37b263307812702ee57bbc413a8a26c0924b5)
  and Megatron-LM [`bab851bf`](https://github.com/NVIDIA/Megatron-LM/commit/bab851bfe3abc805256155962389ec0fe5de7571)
  mounted on top (`run.env.mounts`). Default layouts need 8 nodes × 8 H100 (LoRA: 2 nodes;
  long-context: 32 GB200 nodes × 4).
- **RL stages (2)** run in a container built from [`stage2_rl/Dockerfile`](stage2_rl/Dockerfile)
  (NeMo-RL `super-v3.5-posttraining` branch with NeMo Gym and Megatron-Bridge):

  ```bash
  uv run nemotron kit slurm build YOUR-CLUSTER --recipe super35 --stage rl
  ```

- **Eval (3)** needs `nemo-evaluator-launcher` (`pip install "nemotron[evaluator]"`) and a vLLM
  image that serves Super 3.5.
- An `env.toml` with your cluster profile and `[wandb]` project/entity (see
  [docs/nemo_runspec/nemo-run.md](../../../../docs/nemo_runspec/nemo-run.md)).

## Quick Start

```bash
# 0. Register a Megatron-format starting checkpoint (converted from Hugging Face with
#    Megatron-Bridge), or run stage 0 to produce one
uv run nemotron super35 model import pretrain /path/to/megatron/super35-vl
# uv run nemotron super35 pretrain --run YOUR-CLUSTER

# 1. Register SFT data (Energon) and run both SFT stages
uv run nemotron super35 data import sft /path/to/energon/stage1
uv run nemotron super35 data import sft /path/to/energon/stage2 --name super35-sft-stage2-data
uv run nemotron super35 sft --run YOUR-CLUSTER
uv run nemotron super35 sft -c stage2 --run YOUR-CLUSTER

# super35-sft-model feeds RL directly (the RL runner converts it to HF).
# To start RL from an external SFT checkpoint instead:
# uv run nemotron super35 model import sft /path/to/super35-sft-hf

# 2. Register each RL stage's data and run the RL stages
uv run nemotron super35 data import rl /data/rlvr/train.jsonl --val /data/rlvr/val.jsonl
uv run nemotron super35 rl rlvr --run YOUR-CLUSTER
uv run nemotron super35 data import rl /data/video/train.jsonl --val /data/video/val.jsonl \
    --name super35-rl-teacher-video-data
uv run nemotron super35 rl teacher video --run YOUR-CLUSTER
#    ... remaining teachers, then
uv run nemotron super35 rl mopd --run YOUR-CLUSTER
uv run nemotron super35 rl mopd -c phase2 --run YOUR-CLUSTER

# 3. Evaluate the final checkpoint, exported to HF (see stage3_eval/README.md)
uv run nemotron super35 eval --run YOUR-CLUSTER run.model=super35-rl-mopd-phase2-hf-model:latest

# Preview any stage without submitting
uv run nemotron super35 sft --dry-run
```

## CLI Commands

### Import

The Super 3.5 data blends are not public, so data is registered rather than prepared:

```bash
# SFT data: an Energon dataset (`energon prepare` output: tar shards + .nv-meta/)
uv run nemotron super35 data import sft <energon_dir> [--name super35-sft-stage2-data]

# RL data: NeMo Gym JSONL, one artifact per RL stage (--val defaults to the training file)
uv run nemotron super35 data import rl <train.jsonl> [--val <val.jsonl>] [--name super35-rl-<stage>-data]

# Checkpoints: register an existing checkpoint under the name a stage reads
uv run nemotron super35 model import {pretrain,sft,rl} <checkpoint_dir> [--step N] [--name <artifact>]
```

### Training

```bash
# Vision-language pretraining (Megatron-Bridge)
uv run nemotron super35 pretrain [--run <profile>] [-c default|tiny] [overrides...]

# Supervised fine-tuning (Megatron-Bridge)
uv run nemotron super35 sft [--run <profile>] [-c default|stage2|peft|long_context|tiny] [overrides...]

# Reinforcement learning (NeMo-RL)
uv run nemotron super35 rl rlvr [--run <profile>] [overrides...]
uv run nemotron super35 rl teacher {reasoning,vision,video,vision-reasoning,rlhf,agentic} [-c <variant>] [--run <profile>]
uv run nemotron super35 rl mopd [-c phase1|phase2|single_stage] [--run <profile>]

# Evaluation (NeMo Evaluator)
uv run nemotron super35 eval [--run <profile>] [-t <task>] [overrides...]
```

Teacher variants: `vision-reasoning -c {stage1,stage2}` and
`rlhf -c {teacher1_stage1,teacher1_stage2,teacher2_stage1,teacher2_stage2}`; the other teachers have a single
`default` config.

### Execution Options

| Option | Description |
|--------|-------------|
| `--run <profile>` | **Attached**: submits the job and waits, streaming logs to the terminal |
| `--batch <profile>` | **Detached**: submits the job and exits immediately |
| `-c <config>` | Select a config (for example `-c tiny` for a smoke test) |
| `--dry-run` | Print the resolved config without executing |
| `--stage` | Stage files for interactive debugging |
| `key=value` | Override config values (Hydra-style) |

Use `--run` for short runs you want to watch, and `--batch` for long training runs. The RL
stages allocate 16 to 144 nodes each (`run.env.nodes`).

## Configuration Files

| Stage | Configs | Notes |
|-------|---------|-------|
| `stage0_pretrain/config/` | `default`, `tiny` | Megatron-Bridge recipe named in the `recipe:` block; the rest of the YAML overrides it |
| `stage1_sft/config/` | `default` (stage 1), `stage2`, `peft`, `long_context`, `tiny` | `default.yaml`'s header lists the report's production settings and their override keys |
| `stage2_rl/stage1_rlvr/config/` | `default` | NeMo-RL GRPO config with the NeMo Gym environments |
| `stage2_rl/stage2_teachers/<teacher>/config/` | one per teacher sub-stage | see the teacher table in [`stage2_rl/README.md`](stage2_rl/README.md) |
| `stage2_rl/stage3_mopd/config/` | `phase1`, `phase2`, `single_stage` | teacher checkpoints and per-domain routing |
| `stage3_eval/config/` | `default` | vLLM deployment and NeMo Evaluator task list |

Every config has a `run:` block with its input and output artifacts and its cluster settings
(`run.env`). Override any value on the command line:

```bash
# Shorter SFT run
uv run nemotron super35 sft -c tiny train.train_iters=5

# Start RLVR from a specific SFT artifact version
uv run nemotron super35 rl rlvr run.model=super35-sft-model:v2

# Pin MOPD phase 2 to the phase-1 artifact version saved at step 30
uv run nemotron super35 rl mopd -c phase2 run.model=super35-rl-mopd-phase1-model:<version>
```

## Artifact Flow

Stages exchange checkpoints and data as W&B (or manifest) artifacts referenced from each
config's `run:` block (`${art:model,path}`, `${art:data,path}`, ...).

```
super35-pretrain-model ──► sft (stage 1) ──► super35-sft-stage1-model ──► sft -c stage2 ──► super35-sft-model
   (stage 0 output or         ▲ super35-sft-data                            ▲ super35-sft-stage2-data      │
    `model import pretrain`)                                                                         ──────┘
                                                                                                           ▼
super35-sft-model ──► rl rlvr ──► super35-rl-rlvr-model ──► rl teacher <name> ──► super35-rl-teacher-<name>-model
                                          │                                                   │
                                          └──────────────► rl mopd (phase 1) ◄────────────────┘
                                                 super35-rl-mopd-phase1-model ──► rl mopd -c phase2
                                                                                        │
                                                         super35-rl-mopd-phase2-model ◄─┘ ──► eval
```

| Artifact | Produced by | Consumed by |
|----------|-------------|-------------|
| `super35-pretrain-model` | `pretrain`, or `model import pretrain` | `sft` (stage 1, peft, long_context) |
| `super35-sft-data`, `super35-sft-stage2-data` | `data import sft` | `sft`, `sft -c stage2` |
| `super35-sft-stage1-model` | `sft` | `sft -c stage2` |
| `super35-sft-model` | `sft -c stage2` (Megatron; RL converts it to HF automatically), or `model import sft` | `rl rlvr`, RLHF teacher 1 |
| `super35-rl-<stage>-data` | `data import rl --name ...` | the matching RL stage |
| `super35-rl-rlvr-model` | `rl rlvr` | teachers, MOPD phase 1 |
| `super35-rl-teacher-<name>-model` | `rl teacher <name>` (`rlhf1` / `rlhf2` for RLHF teachers 1 and 2; two-stage teachers also write a `-stage1-` intermediate) | MOPD phase 1 / 2 |
| `super35-rl-mopd-phase1-model` | `rl mopd` | `rl mopd -c phase2` |
| `super35-rl-mopd-phase2-model` | `rl mopd -c phase2` (final) | `eval` |
| `super35-rl-mopd-single-stage-model` | `rl mopd -c single_stage` | `eval` with `run.model=super35-rl-mopd-single-stage-model:<version>` |

Any checkpoint can be registered by hand with `nemotron super35 model import {pretrain,sft,rl}`
(`--name` selects the artifact name).

## Execution Methods

### nemotron CLI (recommended)

The CLI resolves artifacts, builds the job with [NeMo-Run](https://github.com/NVIDIA-NeMo/Run), and
submits it to the profile's cluster:

```bash
# Submit to a Slurm cluster
uv run nemotron super35 sft -c tiny --run YOUR-CLUSTER

# Check the resolved config before submitting
uv run nemotron super35 sft -c tiny --run YOUR-CLUSTER --dry-run
```

### Direct script execution

The Megatron-Bridge stages are plain `train.py --config <yaml> [overrides...]` scripts, so they
can be launched by hand inside the NeMo container (with Megatron-Bridge and Megatron-LM mounted
as in `run.env.mounts`):

```bash
cd src/nemotron/recipes/super35/stage0_pretrain
export SUPER35_INIT_CHECKPOINT=/path/to/megatron/super35-vl SUPER35_PRETRAIN_DATA=/path/to/energon
# one torchrun per node, 8 nodes × 8 GPUs
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29500 \
    train.py --config config/tiny.yaml
```

To run without this repository, Megatron-Bridge's own launcher (`scripts/training/train.sh`) submits
the same recipes to Slurm; see
[Pretraining → Direct Script Execution](../../../../docs/nemotron/super35/pretrain.md#direct-script-execution-megatron-bridge)
and the [Megatron-Bridge Nemotron 3.5 Super VL guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/nemotron/nemotron3.5-super-vl.md).

Even `tiny` loads the full 120B model with the default parallel layout (TP2 / PP2 / EP32), so
it needs all 64 GPUs.

The RL `train.py` scripts launch a multi-node Ray job inside the RL container (`NEMO_RL_DIR`
points at NeMo-RL), so run them through the CLI. To run RL without this repository, use NeMo-RL's
own `super35_launch.sh` on the `super-v3.5-posttraining` branch, which runs the shorter lineage in
its [Super 3.5 guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3.5-posttraining/docs/guides/models/nemotron/nemotron-3.5-super-vl.md); see
[RL → Using super35_launch.sh](../../../../docs/nemotron/super35/rl/index.md#using-super35_launchsh-direct).

## Open-Source Status

| Item | Status |
|------|--------|
| Text pretraining pipeline (§2) | Not reproducible (data and teacher logits not public) |
| Megatron-Bridge Super-VL recipes | Public (Megatron-Bridge `main`); pretrain/SFT/LoRA verified on H100 (SFT/LoRA also GB200); long-context SFT unverified |
| SFT data blends | Not public yet — bring your own Energon dataset |
| RL data blends | Not public yet (NeMo-RL says it will be released later) — register your own JSONL per stage |
| NeMo-RL `super-v3.5-posttraining` branch | Public; pinned to `f299847f1` (NeMo Gym `e24db4e6`) in `stage2_rl/Dockerfile` (see [RL container](stage2_rl/README.md#container)) |
| Agentic teacher NeMo Gym environment | Not public (the Terminal-Bench agent environment); config only |
| RL recipe validation | Ports of the reference configs; vision and vision-reasoning teachers not yet GPU-validated on the shared RL stack |
| RL checkpoints | Intermediate RL checkpoints and the RLHF GenRM not published; the final post-trained model is released as `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` |
| End-to-end pipeline | Not run end to end from this repository (including the SFT → RL hand-off) |
| NeMo Gym Super 3.5 reproducibility guide | Public ([`jakub/marlin` branch](https://github.com/NVIDIA-NeMo/Gym/blob/jakub/marlin/nemotron_recipes/super-3.5/reproducibility.md) until merged); one recipe per Table 10 benchmark |
| NVFP4 quantization | Public: Model Optimizer's [NemotronH Omni VL W4A4 tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD) (released precision recipe; smaller public QAD) |
| Multimodal benchmarks (BabyVision, CharXiv-RQ, MathVision, ...) | Not yet onboarded in open-source evaluation tools |

## Stage Documentation

- [Stage 0: Vision-Language Pretraining](stage0_pretrain/README.md)
- [Stage 1: SFT](stage1_sft/README.md)
- [Stage 2: RL](stage2_rl/README.md)
- [Stage 3: Evaluation](stage3_eval/README.md)

## Further Reading

- [Megatron-Bridge Nemotron 3.5 Super VL guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/nemotron/nemotron3.5-super-vl.md)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
