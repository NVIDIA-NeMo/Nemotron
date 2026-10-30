# Stage 1: SFT

Ultra3 SFT consumes packed SFT Parquet emitted by the Super3-style data-prep
pipeline (`config/default.yaml`, `config/tiny.yaml`).

## Overview

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Ray + xenna SFT packing → packed Parquet via `SftPlanStage → DownloadStage → PackedSftParquetStage`. CLI: `nemotron ultra3 data prep sft`. |
| `train.py` | Runs Megatron-Bridge `nemotron_3_ultra_sft_openmathinstruct2_packed_config`; overrides the dataset only when YAML contains `dataset:`. |
| `config/data_prep/{default,tiny}.yaml` + `data_blend_raw.json` | Data-prep configs + SFT blend. |
| `config/default.yaml`, `config/tiny.yaml` | `default`: paper-style packed SFT (pack 294,912, GBS 64, cosine LR, inferred CP8); `tiny`: 36-node 550B-A55B smoke test (seq 2048, TP2/PP36/EP4). |
| `Dockerfile` | Builds the Ultra3 SFT training image on top of `nvcr.io/nvidia/nemo:26.04.01`. |

## Container image

Ultra3 ships no released training container — the stage owns a `Dockerfile` that
builds the `nemotron_3_ultra` Megatron-Bridge branch on
`nvcr.io/nvidia/nemo:26.04.01`. Build it before running this stage.

On Slurm (our CLI builds only on Slurm):

```bash
uv run nemotron kit slurm build <profile> --recipe ultra3 --stage sft
```

Or build the Dockerfile directly with Docker on any host:

```bash
docker build -t ultra3-sft src/nemotron/recipes/ultra3/stage1_sft
```

The train configs/runspec expect the resulting squashfs at
`${build_cache_dir:-~/.cache/nemotron}/containers/ultra3-sft.sqsh`.

## Data prep — pack the open SFT blend

```bash
uv run nemotron ultra3 data prep sft -c default --run YOUR-RAY-CLUSTER --dry-run
uv run nemotron ultra3 data prep sft --run YOUR-RAY-CLUSTER --sample 10000
```

`data_blend_raw.json` reuses the open Super3 SFT data families (Ultra adapts the same
families, §3.1) with a `_missing_categories` block for internal-only domains. Applies the
chat template, role-masks losses, packs sequences, and writes packed Parquet with
per-split `blend.json`.

> Chat template: reuses the shared `nano3` template (super3 does the same; the tech report does
> not define an Ultra-specific template). `used_in_filter: nano_v3` matches the open SFT datasets.
> Still `TODO(ultra3)`: confirm the released Ultra tokenizer id, and swap to a dedicated
> `templates/ultra3.jinja` only if the post-trained Ultra tokenizer ships a different `chat_template`.

## Training on packed Parquet (default)

`config/default.yaml` and `config/tiny.yaml` expect a data artifact:

```yaml
run:
  data: ultra3-sft-data:latest

dataset:
  ultra3_packed_sft_dir: ${art:data,path}
  seq_length: ${art:data,pack_size}
  packed_sequence_specs:
    packed_sequence_size: ${art:data,pack_size}
```

For the paper-style `default`, `${art:data,pack_size}` must resolve to `294912`; matching the paper therefore requires the data-prep artifact to be packed at 294,912 tokens. `tiny` overrides this to 2048 for smoke testing only.

`train.py` builds a `FinetuningDatasetConfig` directly from this block. The contract is:

- `dataset.ultra3_packed_sft_dir` points to a directory containing `train/` and optionally `valid/`.
- `train/` must contain at least one `*.parquet` shard.
- `valid/` is optional; if no validation Parquet exists, validation is disabled for the dataset config.
- `dataset.seq_length` becomes `FinetuningDatasetConfig.seq_length`.
- `dataset.packed_sequence_specs.packed_sequence_size` becomes `PackedSequenceSpecs.packed_sequence_size`.
- Optional explicit overrides are supported under `packed_sequence_specs`: `packed_train_data_path`, `packed_val_data_path`, and `packed_metadata_path`.

```bash
uv run nemotron ultra3 sft -c tiny --run YOUR-CLUSTER --dry-run
uv run nemotron ultra3 sft --run YOUR-CLUSTER
```

## Recipe details

- MB recipe: `megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_sft_openmathinstruct2_packed_config`
- HF path: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` · Container: `~/.cache/nemotron/containers/ultra3-sft.sqsh`
- Launch: `torchrun` via runspec; `default` resources 384 nodes × 8 GPUs; `tiny` resources 36 nodes × 8 GPUs
- `default`: paper §3.1 values — packed length 294,912 (via `${art:data,pack_size}`), global batch 64, 3,200 iterations, cosine LR 1.5e-5 → 1e-6, 150 warmup iters, MTP loss scaling 0.1
- `default` parallelism: TP=2, PP=6, EP=32, ETP=1, CP=8, sequence parallel enabled. CP8 is inferred for 294,912-token packed SFT; the paper does not state SFT parallelism.
- `tiny`: smoke test only, not convergence — seq length 2048, global batch 8, 10 iters, TP=2, PP=36, EP=4, CP=1, full uniform activation recompute
