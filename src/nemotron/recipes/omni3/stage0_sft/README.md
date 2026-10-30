# Stage 0: Omni SFT

Run supervised fine-tuning for Nemotron Omni starting from the GA checkpoint.

## Overview

Omni does not ship a pre-baked training container. This stage owns:

| Component | Description |
| --- | --- |
| `Dockerfile` | Builds the Megatron-Bridge `nemotron_3_omni` environment |
| `data_prep.py` | Validates or stages a prepared Valor32k Energon dataset |
| `train.py` | Self-contained SFT entry point (loads the recipe, applies overrides, calls `finetune`) |
| `config/` | QA-guide-derived training and data-prep configs |

The build pipeline (podman → enroot → squashfs) lives in the CLI
dispatcher at `src/nemotron/cli/commands/omni3/build.py` rather than
in this folder; `nemotron omni3 build sft` invokes it.

## Quick start

```bash
# 1. Build the container
uv run nemotron omni3 build sft --run YOUR-CLUSTER

# 2. Validate the selected dataset flow (default/tiny use CORD-v2 from HF Hub)
uv run nemotron omni3 data prep sft --run YOUR-CLUSTER

# 3. Convert the GA HF checkpoint to Megatron format
uv run nemotron omni3 model import pretrain --run YOUR-CLUSTER \
  --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --megatron-path /checkpoints/nemotron_omni

# 4. Launch SFT
uv run nemotron omni3 sft --run YOUR-CLUSTER
```

## Data prep

`data_prep.py` dispatches to one of three explicit flows based on the
selected config:

| Config | Flow | Behavior |
| --- | --- | --- |
| `default` / `tiny` | HF Hub | CORD-v2 pulled by the training container on demand; manifest only. |
| `valor32k` | Valor32k self-build | Cosmos-Xenna two-pipeline build: per-video ffmpeg fan-out → per-shard tar build → driver-side Energon index + dataset.yaml. Idempotent via receipts. |
| custom (`dataset_path` only) | Generic Energon | Validates a pre-built Energon directory exists and emits a manifest. |

The Valor32k flow is structurally identical to super3/nano3 SFT data prep:
shared Cosmos-Xenna `Stage` classes plus `setup_*_run` / `finalize_*_run`
recipe glue. Code lives in:

- `nemotron.data_prep.stages.audio_extract` — `AudioExtractStage`
- `nemotron.data_prep.stages.webdataset_shard` — `WebDatasetShardStage`
- `nemotron.data_prep.recipes.sft_omni` — recipe-level setup/finalize

Valor32k cluster-local paths come from env vars
(`OMNI3_VALOR32K_VIDEOS_TAR`, `OMNI3_VALOR32K_ENERGON_PATH`,
`OMNI3_VALOR32K_RAW_DIR`) — see `config/data_prep/valor32k.yaml` for the
defaults. Worker image deps for the Valor32k flow install via the
`nemotron[audio]` extra (pulls `webdataset` and a static `imageio-ffmpeg`
binary; no `apt-get` required).

## Container image

The build dispatcher saves the image as a squashfs file at:

```text
${build_cache_dir}/containers/omni3-sft.sqsh
```

where `build_cache_dir` is the env.toml profile key (typically a
Lustre path on a slurm cluster). Pyxis mounts the squashfs file
directly at training time — no per-job `enroot import` step.

The dispatcher also bridges enroot's `nvcr.io` credentials into the
build container so `FROM nvcr.io/nvidian/...` lines in this Dockerfile
resolve without a separate podman login (see
[How container builds authenticate](../../../../docs/nemo_runspec/nemo-run.md#how-the-build-container-authenticates-with-private-registries)).

For remote (`--run`/`--batch`) builds, set the host-side cache directory
in your env.toml profile so the mount target exists on cluster nodes:

```toml
[<profile>]
build_partition = "<cpu-partition>"   # CPU-only build job
build_cache_dir = "/lustre/.../.cache/nemotron"
```

See `src/nemo_runspec/README.md` for the full list of build-context
env.toml keys.

## Training configs

The training configs port the QA guide variants:

- `default.yaml` — open CORD-v2 projector-only SFT
- `valor32k.yaml` — full Valor32k audio-visual-language SFT
- `image_text_sft.yaml` — image-text projector-only SFT
- `image_text_peft.yaml` — image-text PEFT
- `audio_text.yaml` — audio-text SFT
- `peft_valor32k.yaml` — Valor32k LoRA / PEFT
- `tiny.yaml` — small smoke-test config

Use `-c <name>` to select a variant:

```bash
uv run nemotron omni3 sft -c image_text_peft --run YOUR-CLUSTER
```

## Upstream

This recipe is the cookbook view of the upstream Megatron-Bridge omni
SFT flow. Cross-references for operators who want to dig into the
upstream source:

- **[Megatron-Bridge `nemotron_3_omni` README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_omni/examples/models/vlm/nemotron_3_omni/README.md)** — canonical SFT recipe, hyperparameters, and config tables
- **[Megatron-Bridge `nemotron_3_omni` branch root](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_omni)** — full source for the `nemotron_omni_*sft_config` / `*peft_config` recipes used by `train.py`
- **[Release blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/)** — model-level positioning, benchmarks, training-data scale
- **[Architecture deep-dive](../../../../docs/nemotron/omni3/architecture.md)** — Mamba+transformer hybrid, encoders, EVS rationale

The Dockerfile in this folder pins the `nemotron_3_omni` branches of
Megatron-Bridge and Megatron-LM (the latter as a recursive submodule
fetch). Bump those branches when the upstream merges to a versioned
tag.
