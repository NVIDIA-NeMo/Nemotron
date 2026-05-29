# Stage 0 Recipe Bridge: Pretraining

This file connects the paper’s pretraining story to `src/nemotron/recipes/nano3/stage0_pretrain/`.

## What Exists Publicly

Public files:

- `src/nemotron/recipes/nano3/stage0_pretrain/data_prep.py`
- `src/nemotron/recipes/nano3/stage0_pretrain/train.py`
- `src/nemotron/recipes/nano3/stage0_pretrain/config/default.yaml`
- `src/nemotron/recipes/nano3/stage0_pretrain/config/tiny.yaml`
- `src/nemotron/recipes/nano3/stage0_pretrain/README.md`

The stage is designed as:

1. raw text blend →
2. tokenized Megatron `bin/idx` shards + `blend.json` →
3. Megatron-Bridge pretraining run →
4. base Megatron checkpoint artifact

## What It Maps To In The Paper

| Paper section | Public recipe element |
|---|---|
| §2.2 pretraining data | `data_prep.py` + data blend configs |
| §2.3 data mixture and ordering | blend JSONs and stage-level config selection |
| §2.4 hyperparameters | Megatron recipe target + train/checkpoint settings |
| §2.5 long-context extension | conceptually stage0-adjacent, but not a one-command public paper clone |

## Public Defaults

From `config/default.yaml`:

| Setting | Value |
|---|---|
| container | `nvcr.io/nvidian/nemo:26.02.super.rc1` |
| recipe target | `megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_pretrain_config` |
| data input | `${art:data,path}/blend.json` |
| checkpoint save dir | `/nemo_run/nano3-pretrain-model` |
| checkpoint save interval | `20` |

From `config/tiny.yaml`:

| Setting | Value |
|---|---|
| recipe target | `megatron.bridge.recipes.qwen.qwen3.qwen3_8b_pretrain_config` |
| train iterations | `1700` |
| global batch size | `32` |
| LR warmup iters | `32` |
| checkpoint path | `/nemo_run/nano3-pretrain-model-tiny` |

Interpretation:

- the public stage exposes a **Nano3-shaped Megatron pretraining path**
- the tiny config is a **debug/smoke path**, not a miniature faithful Nano3 run

## Data Prep Behavior

`data_prep.py` is a real pipeline, not a placeholder.
Its structure shows that stage0 public pretraining depends on Nemotron’s data-prep stack:

- `DataBlend`
- tokenizer config
- plan/download/tokenization stages
- split distribution
- W&B artifact logging

Its output shape is the expected Megatron pretraining corpus form:

- shard `.bin/.idx` files
- per-split directories
- `blend.json`
- artifact registration for downstream training

## What The Recipe Reproduces Faithfully

The public recipe faithfully reproduces the *operational pattern* of the paper:

- Megatron-format corpus preparation
- Megatron-Bridge training entrypoint
- distributed checkpointing
- artifact lineage
- cluster execution via NeMo-Run

## What It Does Not Reproduce Exactly

The paper reports:

- 25T tokens total
- two curriculum phases
- a 94% phase switch
- long-context CPT
- a full fifteen-category training mixture
- internal and open data combined

The public recipe does **not** claim a literal paper rerun.
Its README explicitly says the recipe uses **open-source data only** and that results will differ from the paper.

## Operator Guidance

If a user asks:

> “Can I reproduce Nano3 pretraining?”

The best precise answer is:

- **yes, structurally** — the repo exposes the same kind of stage
- **no, not exactly** — the paper’s full data mix and conditions are not fully public

## Useful File Anchors

| Need | File |
|---|---|
| actual data-prep pipeline | `src/nemotron/recipes/nano3/stage0_pretrain/data_prep.py` |
| training launcher | `src/nemotron/recipes/nano3/stage0_pretrain/train.py` |
| public defaults | `src/nemotron/recipes/nano3/stage0_pretrain/config/default.yaml` |
| smoke config | `src/nemotron/recipes/nano3/stage0_pretrain/config/tiny.yaml` |
| usage docs | `src/nemotron/recipes/nano3/stage0_pretrain/README.md` |

## Reproduce with nemotron-customize

Important limitation:

- there is **no catalog pretraining step yet** in `src/nemotron/steps/STEPS.md`

So the correct `/nemotron-customize` handoff is:

1. use **Explorer mode** grounded on `src/nemotron/recipes/nano3/stage0_pretrain/`
2. optionally use `curate/nemo_curator` for corpus acquisition/filtering upstream
3. treat Megatron pretraining config as recipe-driven rather than step-manifest-driven

## Good One-Sentence Handoff

> “Nano3 stage0 exists as a public recipe, but not yet as a catalog `nemotron-customize` step, so I’d hand this off as an Explorer-mode pretraining task grounded on `src/nemotron/recipes/nano3/stage0_pretrain/`.”
