# Stage 1: Omni RL

Multimodal post-training alignment for Nemotron 3 Nano Omni via NeMo-RL. One shared container builds the full RL stack (vLLM + Megatron-Bridge + Megatron-LM + Automodel + Gym all in a single uv-managed venv); three sub-stages run on top of it.

## Overview

Three RL flows share the container, each with its own dataset, launcher, and config family:

| Sub-stage | Algorithm | Source dataset | Output artifact |
|---|---|---|---|
| `stage1_mpo/` | Mixed Preference Optimization | [`OpenGVLab/MMPR`](https://huggingface.co/datasets/OpenGVLab/MMPR) (auto-downloads) | `omni3-rl-mpo-model:latest` |
| `stage2_text_rl/` | GRPO (text-only) | [`nvidia/Nemotron-3-Nano-RL-Training-Blend`](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) (auto-downloads) | `omni3-rl-text-model:latest` |
| `stage3_vision_rl/` | GRPO (vision) | [`OpenGVLab/MMPR-Tiny`](https://huggingface.co/datasets/OpenGVLab/MMPR-Tiny) (auto-downloads) | `omni3-rl-vision-model:latest` |

The full upstream RL alignment runs **20 datasets across 25 environments / ~2.3M rollouts** (visual grounding, charts, vision-critical STEM, video understanding, ASR — see the [release blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/)). This recipe folder reproduces the **3 of 25** environments that have public datasets; the remaining 22 use internal or third-party data and aren't included.

| Component | Description |
| --- | --- |
| `Dockerfile` | Shared NeMo-RL container, mirrors the upstream `nano-v3-omni` build with `BUILD_CUSTOM_VLLM=1` and the omni vllm submodule fork |
| `data_prep.py` | Dispatcher for `-c mpo` / `-c text` / `-c vision`; auto-downloads from `source_uri` when input dirs are empty |
| `_data_prep_base.py` | Shared driver — handles config validation, run-hash caching, W&B lineage |
| `stage1_mpo/`, `stage2_text_rl/`, `stage3_vision_rl/` | Per-sub-stage launcher + config |
| `config/data_prep/` | Per-sub-stage data prep YAMLs |

The build pipeline (podman → enroot → squashfs) lives in the CLI dispatcher at `src/nemotron/cli/commands/omni3/build.py`; `nemotron omni3 build rl` invokes it.

## Quick start

```bash
# 1. Build the container (~25-40 min, downloads ~30 GB of base CUDA + builds vllm fork)
uv run nemotron omni3 build rl --run YOUR-CLUSTER

# 2. Prepare data for each sub-stage (auto-downloads from HF if not pre-staged)
uv run nemotron --batch prep omni3 data prep rl -c mpo
uv run nemotron --batch prep omni3 data prep rl -c text
uv run nemotron --batch prep omni3 data prep rl -c vision

# 3. Run the sub-stages in order
uv run nemotron omni3 rl mpo --run YOUR-CLUSTER
uv run nemotron omni3 rl text --run YOUR-CLUSTER
uv run nemotron omni3 rl vision --run YOUR-CLUSTER   # launcher pending upstream
```

Submitting the three data prep jobs in parallel is safe — sibling RL configs get unique job names from `_make_job_name` (PID + random token), so the local config-staging file and the remote Ray code dir never collide.

## Data prep

Both `vision.yaml` and `mpo.yaml` carry a `source_uri` that points at the upstream HF dataset. When the dispatcher runs:

- **Pre-staged input dir**: if `input_dir` already contains the required files, the download is skipped.
- **Auto-download**: otherwise the dispatcher calls `huggingface_hub.snapshot_download(...)` and proceeds.
- **No `source_uri` and missing files**: fails fast with the missing file list and a remediation hint.

For the full per-sub-stage data prep details (output cache shapes, helper scripts, parallel-submission semantics, artifact registration), see [`docs/nemotron/omni3/rl/data-prep.md`](../../../../docs/nemotron/omni3/rl/data-prep.md).

## Container image

The build dispatcher saves the image as a squashfs file at:

```text
${build_cache_dir}/containers/omni3-rl.sqsh
```

where `build_cache_dir` is the env.toml profile key (typically a Lustre path on a slurm cluster). Pyxis mounts the squashfs file directly at training time — no per-job `enroot import` step.

The Dockerfile mirrors NeMo-RL's `docker/Dockerfile` body (clones `NVIDIA/NeMo-RL @ nano-v3-omni` recursively, runs `tools/build-custom-vllm.sh` against the in-tree `3rdparty/vllm` submodule, then `uv sync` with the `vllm` / `mcore` / `automodel` / `--all-groups` extras). The omni vllm fork at [`aroshanghias-nvd/vllm`](https://github.com/aroshanghias-nvd/vllm) (branch `nano-v3-vl`) wires in via the upstream submodule definition — no override needed in this Dockerfile.

For remote (`--run`/`--batch`) builds, set the host-side cache directory in your env.toml profile:

```toml
[<profile>]
build_partition = "<cpu-partition>"   # CPU-only build job
build_cache_dir = "/lustre/.../.cache/nemotron"
```

See [`src/nemo_runspec/README.md`](../../../../src/nemo_runspec/README.md) for the full list of build-context env.toml keys, and [How container builds authenticate](../../../../docs/nemo_runspec/nemo-run.md#how-the-build-container-authenticates-with-private-registries) for the enroot → podman auth bridge that the dispatcher uses to pull `nvcr.io/nvidia/cuda-dl-base:...` during the build.

## Sub-stage notes

### `stage1_mpo`

Mixed Preference Optimization on the public MMPR dataset. The data prep
flow auto-downloads the full `OpenGVLab/MMPR` snapshot (~16 GB:
`images.zip` 14.2 GB + `annotations.zip` 1.5 GB + `meta.json`),
extracts both archives, and rewrites `meta.json`'s absolute petrelfs
paths into relative-to-cache paths (saved as `meta_public.json`). The
upstream prep script lives at
`scripts/prepare_public_mmpr_for_mpo.py` and is vendored into this
repo for the cookbook flow.

### `stage2_text_rl`

GRPO on `nvidia/Nemotron-3-Nano-RL-Training-Blend`. Reuses Nano3's
`run_rl_resolve_pipeline` because the source dataset, placeholder
semantics (DAPO + Skywork via `HFPlaceholderResolver`), and output
schema (`responses_create_params`-shaped JSONL) are identical.

### `stage3_vision_rl`

GRPO on the MMPR-Tiny dataset. The data prep flow is fully
functional; the training launcher (`stage3_vision_rl/train.py`)
currently raises `NotImplementedError` and declares a degenerate
resource footprint (`nodes=1, gpus_per_node=0`) so a stray submission
doesn't allocate GPUs. The launcher lands when the upstream NeMo-RL
omni fork's vision flow is published.

## Upstream

This recipe is the cookbook view of the upstream NeMo-RL omni RL flow.
Cross-references for operators who want to dig into the upstream
source:

- **[NeMo-RL `nano-v3-omni` Nemotron 3 Nano Omni guide](https://github.com/NVIDIA-NeMo/RL/blob/nano-v3-omni/docs/guides/nemotron-3-nano-omni.md)** — canonical end-to-end walkthrough (build, data prep, MPO/text/vision launchers, `.env` setup)
- **[NeMo-RL `nano-v3-omni` branch root](https://github.com/NVIDIA-NeMo/RL/tree/nano-v3-omni)** — full source for the `examples/run_vlm_mpo.py` / `examples/omni/` configs / `tools/build-custom-vllm.sh` referenced by this Dockerfile
- **[Omni vllm fork (`aroshanghias-nvd/vllm`)](https://github.com/aroshanghias-nvd/vllm/tree/nano-v3-vl)** — pulled in via NeMo-RL's `3rdparty/vllm` submodule on the `nano-v3-vl` branch
- **[Release blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/)** — model-level positioning, RL training-data scale (20 datasets / 25 envs / 2.3M rollouts)
- **[RL data prep deep-dive](../../../../docs/nemotron/omni3/rl/data-prep.md)** — auto-download semantics, helper scripts, output layouts

The Dockerfile in this folder pins the `nano-v3-omni` branch of
NeMo-RL. Bump it when the upstream merges to a versioned tag.
