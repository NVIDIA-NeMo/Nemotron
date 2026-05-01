# Stage 1 Recipe Summary — Supervised Fine-Tuning

This file maps the Super3 SFT section of the paper to the released recipe under:

- `src/nemotron/recipes/super3/stage1_sft/`

---

## What stage 1 covers

The paper’s large SFT program becomes two practical components in the repo:

1. **data preparation** into packed Parquet shards
2. **Megatron-Bridge fine-tuning** from the pretrained model artifact

---

## Main source files

| Path | Role |
|---|---|
| `src/nemotron/recipes/super3/stage1_sft/README.md` | human overview and CLI examples |
| `.../data_prep.py` | chat-template + packing pipeline |
| `.../train.py` | Megatron-Bridge finetune entrypoint |
| `.../config/default.yaml` | production training config |
| `.../config/tiny.yaml` | small test variant |
| `.../config/data_prep/default.yaml` | packed-Parquet data-prep config |

---

## Data preparation path

The repo translates chat-format training data into a Megatron-friendly packed format.

### Pipeline

```text
SftPlanStage → DownloadStage → PackedSftParquetStage
```

### Output shape

```text
output/stage1_sft/
  blend.json
  splits/train/*.parquet
  splits/valid/*.parquet
  splits/test/*.parquet
  runs/<hash>/...
```

Each packed shard contains the fields needed for supervised finetuning such as `input_ids`, `loss_mask`, and `seq_start_id`.

---

## Important data-prep settings

The released defaults in `config/data_prep/default.yaml` and `data_prep.py` expose the practical SFT input format.

| Setting | Released value |
|---|---|
| Tokenizer model | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` |
| Pack size | 4096 |
| Packing algorithm | `first_fit_shuffle` |
| Compression | `zstd` |
| Chat template | `super3` |
| Messages field | `messages` |
| Tools field | `tools` |
| Filter tag | `used_in_filter: super_v3` |
| Train/valid/test split | 0.98 / 0.01 / 0.01 |

This tells you how the released repo turns the paper’s broad SFT mixture into a runnable dataset.

---

## Training entrypoint

The training script is:

- `src/nemotron/recipes/super3/stage1_sft/train.py`

It builds a `FinetuningDatasetConfig`, resolves the packed Parquet directories, loads the Megatron-Bridge Super3 SFT recipe, and runs `finetune()`.

A notable implementation detail is that it can also:

- convert the final Megatron checkpoint back to Hugging Face format,
- log a separate `-hf` artifact.

---

## Default training config

From `config/default.yaml`:

| Setting | Value |
|---|---|
| Data artifact | `super3-sft-data:latest` |
| Model artifact | `super3-pretrain-model:latest` |
| Container | `nvcr.io/nvidian/nemo:26.02.super.rc4` |
| Recipe target | `megatron.bridge.recipes.nemotronh.nemotron_3_super.nemotron_3_super_sft_config` |
| Packed sequence | true |
| PEFT | null |
| Train iterations | 1700 |
| Global batch size | 4 |
| Save path | `/nemo_run/super3-sft-model` |
| Save interval | 20 |
| HF export enabled | true |
| HF model id for export | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` |

That last field looks surprising, but it is the literal value in the released config and should be cited as such if asked.

---

## Commands exposed by the repo

```bash
uv run nemotron super3 data prep sft --run <profile>
uv run nemotron super3 sft --run <profile>
uv run nemotron super3 sft -c tiny --run <profile>
```

This is the simplest runnable answer unless the user specifically asks for the script/config path level.

---

## Artifact flow

```text
pretrain model artifact
  + SFT data artifact
  → stage1_sft/train.py
  → super3-sft-model
  → optional HF export artifact
```

---

## What the repo captures well

The released recipe captures the operational mechanics of SFT very well:

- packed data format,
- chat-template application,
- assistant-only loss masking,
- Megatron-Bridge fine-tuning,
- post-training HF export.

## What the paper still explains better

The paper is still the better source for:

- why the loss is two-stage,
- why reasoning-off and low-effort examples exist,
- how big the total SFT mixture is,
- why long-input/short-output examples matter.

---

## Paper-vs-recipe caveats

1. **The released data-prep code gives the mechanics, not the full internal SFT blend.**
2. **The two-stage-loss rationale is paper-side context.**
3. **The repo defaults are a runnable approximation of the released methodology, not the entire paper corpus.**

---

## Best next file

- `stage2_rl.md` if the user wants the next stage in the pipeline.
- `../paper/sft.md` if the user wants the research-level explanation of the loss and reasoning controls.
