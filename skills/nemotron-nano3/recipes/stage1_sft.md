# Stage 1 Recipe Bridge: SFT

This file connects the paper’s SFT section to `src/nemotron/recipes/nano3/stage1_sft/`.

## What Exists Publicly

Public files:

- `src/nemotron/recipes/nano3/stage1_sft/data_prep.py`
- `src/nemotron/recipes/nano3/stage1_sft/train.py`
- `src/nemotron/recipes/nano3/stage1_sft/config/default.yaml`
- `src/nemotron/recipes/nano3/stage1_sft/config/tiny.yaml`
- `src/nemotron/recipes/nano3/stage1_sft/README.md`

The stage is a two-part pipeline:

1. OpenAI-format chat data → chat template → tokenization → role-based loss masking → packed Parquet
2. packed Parquet + Megatron checkpoint → Megatron-Bridge SFT

## What It Maps To In The Paper

| Paper section | Public recipe element |
|---|---|
| §3.1.1 chat template | `data_prep.py` and SFT docs |
| §3.1.2 data | SFT blend config and data-prep stage |
| §3.1.3 filtering | validation and prep logic documented in `docs/nemotron/nano3/sft.md` |
| §3.1.4 data mixture | blend JSON + public subset caveat |
| §3.1.5 reasoning control | public docs explain reasoning-on/off and budget-control shaping, though not as a separate runtime knob in stage1 |
| §3.1.6 hyperparameters | `config/default.yaml`, `config/tiny.yaml`, and `train.py` |

## Public Data Prep Behavior

The public SFT prep path does real transformation work:

- apply Nano3 chat template
- split text into role-labeled chunks
- tokenize
- build assistant-only loss masks
- pack examples into fixed-size sequences
- write packed Parquet shards for Megatron-Bridge

That matches the paper’s core SFT mechanics even though the public data mixture is smaller and more open.

## Public Training Defaults

From `config/default.yaml`:

| Setting | Value |
|---|---|
| run data artifact | `nano3-sft-data:latest` |
| run model artifact | `nano3-pretrain-model:latest` |
| container | `nvcr.io/nvidian/nemo:26.02.super.rc1` |
| recipe target | `megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_finetune_config` |
| packed sequence | `true` |
| PEFT | `null` (full SFT) |
| train iterations | `1700` |
| global batch size | `4` |
| tensor parallel | `4` |
| context parallel | `2` |
| pipeline parallel | `1` |
| checkpoint finetune mode | `true` |

The public config also mounts pinned Megatron-LM and Megatron-Bridge revisions into the container.

## Tiny Config Notes

From `config/tiny.yaml`:

| Setting | Value |
|---|---|
| recipe target | `qwen3_8b_finetune_config` |
| train iterations | `1700` |
| global batch size | `32` |
| tensor parallel | `4` |
| pipeline parallel | `1` |

Interpretation:

- `tiny.yaml` is for smoke/debug runs
- it is **not** a paper-faithful Nano3 SFT schedule

## Important Public-vs-Paper Difference

The paper reports SFT on:

- 18M total samples
- 13,000 steps
- sequence packing to **256k**

The public stage1 recipe defaults to a much smaller operational footprint.
That means the right answer is:

- **methodology aligns**
- **exact scale does not**

## Why This Stage Is Still Valuable

Even with that scale gap, stage1 is the best public explanation of how Nano3 SFT works in practice:

- the chat-template format
- assistant-only loss masking
- packed-sequence training
- Megatron checkpoint loading
- artifact handoff from stage0 to stage1

## Where To Point Users

| Need | File |
|---|---|
| public SFT data prep code | `src/nemotron/recipes/nano3/stage1_sft/data_prep.py` |
| public SFT train code | `src/nemotron/recipes/nano3/stage1_sft/train.py` |
| default config | `src/nemotron/recipes/nano3/stage1_sft/config/default.yaml` |
| tiny config | `src/nemotron/recipes/nano3/stage1_sft/config/tiny.yaml` |
| explanatory docs | `docs/nemotron/nano3/sft.md` |

## Reproduce with nemotron-customize

This stage maps cleanly to catalog steps:

1. `prep/sft_packing`
2. `sft/megatron_bridge`

Optional surrounding steps:

- `convert/hf_to_megatron` if the user starts from released HF weights instead of a Megatron checkpoint
- `sft/automodel` if the user wants a smaller-GPU or LoRA-first path rather than Megatron SFT

## Good Handoff Pattern

> “For a public Nano3-style SFT build, use `prep/sft_packing` to produce packed Parquet and `sft/megatron_bridge` to run Megatron-Bridge fine-tuning. That reproduces the stage shape, but not the paper’s full 18M-sample, 256k packed run.”
