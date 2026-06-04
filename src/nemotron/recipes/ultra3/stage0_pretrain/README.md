# Stage 0: Pretraining

Super3/nano3-style two-step stage: a Ray **data prep** that tokenizes the open Ultra
datasets into Megatron `bin/idx`, then Megatron-Bridge **training** that consumes it.

## Overview

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Ray + xenna tokenization → Megatron `bin/idx`. CLI: `nemotron ultra3 data prep pretrain`. Blends in `config/data_prep/`. |
| `train.py` | Runs Megatron-Bridge `nemotron_3_ultra_pretrain_config` through the runspec CLI |
| `config/data_prep/*.yaml` + `data_blend_raw_*.json` | Data-prep configs + dataset blends (phase1 / phase2) |
| `config/default.yaml`, `config/tiny.yaml` | `default`: paper-aligned 20T pretrain (seq 8192, GBS 3072, WSD); `tiny`: 36-node 550B-A55B smoke test (seq 2048, TP2/PP36/EP4). |
| `Dockerfile` | Builds the Ultra3 pretrain training image on top of `nvcr.io/nvidia/nemo:26.04.01`. |

## Container image

Ultra3 does **not** ship a released pretrain container tag. Build the squashfs
image before running this stage:

```bash
uv run nemotron ultra3 build pretrain --run YOUR-CLUSTER --dry-run
uv run nemotron ultra3 build pretrain --run YOUR-CLUSTER -- \
  --build-arg MEGATRON_BRIDGE_BRANCH=<ultra-release-branch> \
  --build-arg MEGATRON_CORE_BRANCH=<ultra-mcore-branch>
```

The Dockerfile follows the Ultra Megatron-Bridge README's Day-0 code flow:
start from `nvcr.io/nvidia/nemo:26.04.01`, clone the Ultra MB branch, checkout
the matching Megatron-LM branch, run `uv lock && uv sync`, and sanity-import
`nemotron_3_ultra_pretrain_config`. The output is
`~/.cache/nemotron/containers/ultra3-pretrain.sqsh` by default, and the train
configs/runspec header point at that path.

## 1. Data prep — tokenize the open datasets

```bash
uv run nemotron ultra3 data prep pretrain -c phase1 --run YOUR-RAY-CLUSTER --dry-run   # diversity (~15T)
uv run nemotron ultra3 data prep pretrain -c phase2 --run YOUR-RAY-CLUSTER              # quality (~5T)
```

Blends (`config/data_prep/data_blend_raw_{phase1,phase2,long_context}.json`) carry the
paper's mixture weights plus a `_missing_categories` honesty block. New-for-Ultra
datasets — Multiple-Choice, Generative, Fact-Seeking, Moral-Scenarios (subsets of
`Nemotron-Pretraining-Specialized-v1.2`) and Legal/Case-Law-Summary — are tagged
`"_new_in_ultra": true`. The `code` category (14%) stays in `_missing_categories` because
`Nemotron-Pretraining-Code-v3` ships as a repo manifest (metadata), not tokenized text. Confirm the released tokenizer id
and dataset repo ids/subsets (`TODO(ultra3)` markers) before a full run. Output is a
tokenized `bin/idx` dataset registered as the `ultra3-pretrain-data-*` artifact.

## 2. Train — consume the tokenized data

```bash
uv run nemotron ultra3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
uv run nemotron ultra3 pretrain --run YOUR-CLUSTER
```

`config/default.yaml` declares the `data` input artifact and forwards
`data.per_split_data_args_path: ${art:data,path}/blend.json` to the recipe function —
the same tokenized-data wiring used by the MB Ultra pretrain path.

## Training defaults

- MB recipe: `megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_pretrain_config`
- HF path: `nvidia/nemotron-ultra-rl-052726` · Container: `~/.cache/nemotron/containers/ultra3-pretrain.sqsh`
- Launch: `torchrun` via runspec; `default` resources 96 nodes × 8 GPUs; `tiny` resources 36 nodes × 8 GPUs
- `default`: paper §2.4 values — seq length 8192, global batch 3072, 794,728 iterations, LR 2.5e-4 → 2.5e-6, WSD warmup 7,946 iters and final-5T WSD tail (`lr_wsd_decay_iters: 198682`), MTP loss scaling 0.1
- `default` parallelism: TP=2, PP=12, EP=32, ETP=1, CP=1, sequence parallel enabled. This is the runnable MB Ultra shape; the paper does not state main-phase parallelism.
- `tiny`: smoke test only, not convergence — seq length 2048, global batch 8, 10 iters, TP=2, PP=36, EP=4, CP=1, full uniform activation recompute

## Slurm wiring

No bespoke Slurm script. `nemotron ultra3 pretrain --run <profile>` uses the shared
`nemo_runspec` / NeMo-Run path; `data prep` uses the Ray code-packager path. Configure
clusters in `env.toml`.

## Long-context phase (not included)

The Ultra LC-Phase (1M context) is **not provided** because its data is not open-source: per §2.5
the LC blend is 46% long-context data (document-QA reused from Super & Nano + synthetic
long-context SFT-style data) + 54% Phase 2 data, and the long-context half is not part of the
dataset release. To replicate: run a CPT phase from the Phase 2 checkpoint with your own
long-context corpus blended ~46/54 against `data_blend_raw_phase2.json`, constant LR `2.5e-6`,
~33B tokens, 92% iters at 1M / 8% at 4K (math/code only), CP=32/TP=8/EP=128/PP=2, no RULER-style data.
