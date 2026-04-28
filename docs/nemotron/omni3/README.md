# Nemotron 3 Omni Training Recipe

Multimodal post-training pipeline for **Nemotron 3 Nano Omni**, NVIDIA's
30B-A3B hybrid mixture-of-experts model. Unlike Nano3 and Super3,
Omni starts from a GA checkpoint and owns its stage-local container
builds.

## Read the blog?

Quick navigation for developers landing here from the [release blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/):

| You want… | Go to |
|---|---|
| Architecture deep-dive (Mamba+transformer, EVS, NemoClaw) | [`architecture.md`](./architecture.md) |
| Inference engines, quantization, cloud platforms, providers | [`inference.md`](./inference.md) |
| Reproduce training | [§Quick Start](#quick-start) below |
| SFT details | [`sft.md`](./sft.md) |
| RL details | [`rl.md`](./rl.md) · [`rl/data-prep.md`](./rl/data-prep.md) |
| Evaluation | [`evaluate.md`](./evaluate.md) |

## Model Overview

![Nemotron 3 Nano Omni hybrid MoE architecture: Parakeet audio encoder → audio adaptor, C-RADIOv4-H vision encoder + 3D convolution + Efficient Video Sampling → vision adaptor, text tokenizer, all feeding the unified 30B-A3B LLM decoder](../../assets/omni-3.png)

| Property | Value |
|---|---|
| Architecture | Hybrid MoE — Mamba layers (sequence/memory efficiency) + transformer layers (reasoning), with a unified text decoder ([details](./architecture.md#hybrid-moe-decoder)) |
| Total / active parameters | 30B / 3B (A3B MoE) |
| Native modalities | Text, image, video, audio |
| Max context length | 262K tokens |
| Training context schedule | 16K → 49K → 262K (progressive scaling, see [§Progressive context scaling](./architecture.md#progressive-context-scaling)) |
| Vision encoder | [C-RADIOv4-H](./architecture.md#vision-encoder--c-radiov4-h) |
| Audio encoder | [NVIDIA Parakeet (extended via Granary, Music Flamingo)](./architecture.md#audio-encoder--parakeet-extended) |
| Video pipeline | [3D convolutions + Efficient Video Sampling (EVS)](./architecture.md#video-pipeline--3d-convolutions--efficient-video-sampling-evs) |
| GA checkpoint | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) |
| License | [NVIDIA Nemotron Open Model License](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) (enterprise-friendly, on-prem and any deployment) |

For the full architectural deep-dive — including why EVS drives the throughput numbers, the perception-sub-agent framing, and the released training-data scale (127B / 124M / 20×25 / 11.4M) — see [`architecture.md`](./architecture.md).

### Capabilities (released benchmarks)

- Best-in-class on [MMlongbench-Doc](https://mmlongbench.github.io/) and [OCRBenchV2](https://github.com/Yuliang-Liu/MultimodalOCR) (document intelligence)
- Leading on [WorldSense](https://huggingface.co/datasets/honglyhly/WorldSense), [DailyOmni](https://github.com/THUNLP-MT/DailyOmni), [VoiceBench](https://huggingface.co/datasets/hlt-lab/voicebench) (video / audio understanding)
- **~9.2×** greater effective system capacity on video reasoning, **~7.4×** on multi-document workloads vs. comparable open omni models
- "Highest throughput across every task" in MediaPerf; "lowest inference cost for video-level tagging"
- See [`inference.md`](./inference.md) for engines, quantization paths, hardware support, and deployment

### Use cases

Designed as a **multimodal perception sub-agent** for agentic AI — finance, healthcare, scientific discovery, media/entertainment, ad-tech. Especially strong on long-horizon reasoning over complex documents and large video batches. The [NemoClaw sandbox](./inference.md#nemoclaw--privacy-first-video-processing) provides privacy-first video processing for compliance-bounded workloads.

### Upstream training recipes

This recipe folder is the cookbook view; upstream sources are:

| Stage | Upstream guide | Branch root |
|---|---|---|
| SFT (Megatron-Bridge) | [`nemotron_3_omni` README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_omni/examples/models/vlm/nemotron_3_omni/README.md) | [`Megatron-Bridge` `nemotron_3_omni`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_omni) |
| RL (NeMo-RL) | [`nano-v3-omni` Nemotron 3 Nano Omni guide](https://github.com/NVIDIA-NeMo/RL/blob/nano-v3-omni/docs/guides/nemotron-3-nano-omni.md) | [`NeMo-RL` `nano-v3-omni`](https://github.com/NVIDIA-NeMo/RL/tree/nano-v3-omni) |
| Evaluation | Same Megatron-Bridge path | (above) |
| Image training data | — | [`huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3`](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3) |
| Long-document SDG | — | [`src/nemotron/recipes/data/sdg/long-document/`](../../../src/nemotron/recipes/data/sdg/long-document/) (structure released; bodies port at upstream release) |

## Current Limitations

The recipe structure, CLI, Dockerfiles, and configs are all in place, but some pieces still depend on upstream work or internal-only data:

- **Vision RL training is stubbed.** `stage3_vision_rl/train.py` raises `NotImplementedError` and declares a degenerate resource footprint (`nodes=1, gpus_per_node=0`) so a stray submission doesn't allocate GPUs. `omni3 pipe` auto-detects this and skips stage 4 by default; pass `pipe.force_vision=true` once the upstream launcher lands. Data prep for MMPR-Tiny is fully functional.
- **RL container mirrors NeMo-RL's release Dockerfile.** `stage1_rl/Dockerfile` clones `NVIDIA/NeMo-RL @ nano-v3-omni` recursively (which carries the [omni vllm fork](https://github.com/aroshanghias-nvd/vllm) as a submodule) and runs the same `BUILD_CUSTOM_VLLM=1` + `uv sync` flow as the upstream `docker/Dockerfile`. `omni3 build rl` produces `omni3-rl.sqsh`.
- **SFT default uses CORD-v2 (open); Valor32k is an opt-in.** `default.yaml` pulls [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) from HuggingFace via the `vlm-hf` loader — no local shard building required. `-c valor32k` switches to the full audio-visual-language flow, but requires internal access to a prepared Valor32k-AVQA Energon dataset (the raw-shard builder is internal-only at release). `data_prep.py` validates the Energon path or emits a manifest for HF — it does not assemble shards.
- **Long-document SDG pipeline is a scaffold.** `src/nemotron/recipes/data/sdg/long-document/` has 9 numbered scripts with their argparse surfaces and a thorough README, but the script bodies raise `NotImplementedError` — port bodies from upstream at release time. See `designs/long-document-sdg-pipeline.md`.
- **Eval defaults are text-only.** `stage2_eval/config/default.yaml` inherits nano3's language-benchmark task list to track language-capability regression. Multimodal sanity checks (image / video / audio / audio-video-text) run via `nemotron omni3 model eval` — see [SFT guide §Model lifecycle](./sft.md).
- **Pinned to release branches.** The SFT Dockerfile pulls `github.com/NVIDIA-NeMo/Megatron-Bridge @ nemotron_3_omni` (and `github.com/NVIDIA/Megatron-LM @ nemotron_3_omni` as a recursive submodule fetch); the RL flow uses `github.com/NVIDIA/NeMo-RL @ nano-v3-omni`. These are the active release branches for Nemotron 3 Omni — bump to a versioned tag (or `main`) once these changes merge upstream.

The linked stage guides (SFT / RL / Evaluate) call out each stage's specific limitations.

## Quick Start

### Prerequisites

- **Slurm cluster** with GPU nodes for SFT, RL, and evaluation — see [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md)
- **[Weights & Biases](../wandb.md)** for experiment tracking and [artifact lineage](../artifacts.md)
- **Build-capable execution profile** for `nemotron omni3 build <stage>` jobs
- **GA checkpoint**: [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) (BF16 weights). FP8 and NVFP4 quantization paths are supported by the inference engines listed in [§Inference engines](#inference-engines).

### Installation

```bash
git clone https://github.com/NVIDIA/nemotron
cd nemotron
uv sync
```

### Configuration

Create an `env.toml` profile with both training and build settings:

```toml
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"

[YOUR-CLUSTER]
executor = "slurm"
account = "YOUR-ACCOUNT"
partition = "batch"
run_partition = "interactive"
# Build-context overrides — used by `omni3 build sft|rl` only. The build
# is CPU-only and writes its OCI archive to `build_cache_dir`, which
# must be a cluster-visible (e.g. Lustre) path because the dispatcher
# mounts it into the build container.
build_partition = "cpu"
build_time = "02:00:00"
build_cache_dir = "/lustre/.../users/YOUR-USER/.cache/nemotron"
nodes = 1
ntasks_per_node = 8
gpus_per_node = 8
mounts = ["/lustre:/lustre"]
```

> See [How container builds authenticate](../../nemo_runspec/nemo-run.md#how-the-build-container-authenticates-with-private-registries)
> for how the dispatcher reuses your existing
> `~/.config/enroot/.credentials` to pull `nvcr.io` images inside the
> build container — no separate podman login required.

### Run the Pipeline

<div class="termy">

```console
// Stage 0: SFT
$ uv run nemotron omni3 build sft --run YOUR-CLUSTER
$ uv run nemotron omni3 data prep sft --run YOUR-CLUSTER
$ uv run nemotron omni3 model import pretrain --run YOUR-CLUSTER \
    --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    --megatron-path /checkpoints/nemotron_omni
$ uv run nemotron omni3 sft --run YOUR-CLUSTER

// Stage 1.1: RL MPO
$ uv run nemotron omni3 build rl --run YOUR-CLUSTER
$ uv run nemotron omni3 data prep rl -c mpo --run YOUR-CLUSTER
$ uv run nemotron omni3 rl mpo --run YOUR-CLUSTER

// Stage 1.2: RL text
$ uv run nemotron omni3 data prep rl -c text --run YOUR-CLUSTER
$ uv run nemotron omni3 rl text --run YOUR-CLUSTER

// Stage 1.3: RL vision
$ uv run nemotron omni3 data prep rl -c vision --run YOUR-CLUSTER
$ uv run nemotron omni3 rl vision --run YOUR-CLUSTER

// Stage 2: Evaluation
$ uv run nemotron omni3 eval --run YOUR-CLUSTER
```

</div>

> **Note**: `omni3 build` submits a short CPU-only container build job through [NeMo-Run](../../nemo_runspec/nemo-run.md). The build produces `${build_cache_dir}/containers/omni3-{sft,rl}.sqsh` (squashfs) which pyxis mounts directly at training time — no per-job `enroot import`.

## Resources

- **Model checkpoint**: [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
- **Release blog**: [NVIDIA Nemotron 3 Nano Omni — multimodal agent reasoning](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/)
- **SFT recipe**: [Stage 0: SFT](./sft.md)
- **RL recipe**: [Stage 1: RL](./rl.md)
- **Eval recipe**: [Stage 2: Evaluation](./evaluate.md)
- **Image training data**: [`nvidia/Nemotron-Image-Training-v3`](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3)

## Training Pipeline

| Stage | Name | Purpose | Guide |
|-------|------|---------|-------|
| 0 | [SFT](./sft.md) | Fine-tune the GA checkpoint on Valor32k and related multimodal variants | [sft.md](./sft.md) |
| 1 | [RL](./rl.md) | Multi-stage Omni RL: MPO → text RL → vision RL | [rl.md](./rl.md) |
| 2 | [Evaluation](./evaluate.md) | Benchmark the final checkpoint with NeMo Evaluator | [evaluate.md](./evaluate.md) |

## Pipeline Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart LR
    sdg["data/sdg/long-document<br/>synthetic document data"] --> prep["omni3 data prep sft"]
    ga["GA HF checkpoint"] --> import["omni3 model import pretrain"]
    prep --> sft["omni3 sft"]
    import --> sft
    sft --> mpo["omni3 rl mpo"]
    mpo --> text["omni3 rl text"]
    text --> vision["omni3 rl vision"]
    vision --> eval["omni3 eval"]

    style sdg fill:#e3f2fd,stroke:#2196f3
    style sft fill:#f3e5f5,stroke:#9c27b0
    style mpo fill:#e8f5e9,stroke:#4caf50
    style text fill:#e8f5e9,stroke:#4caf50
    style vision fill:#e8f5e9,stroke:#4caf50
    style eval fill:#fff3e0,stroke:#ff9800
```

The upstream synthetic-data pipeline lives outside the family tree under `src/nemotron/recipes/data/sdg/long-document/`. That keeps the recipe split explicit:

- **`src/nemotron/recipes/data/curation/`** — filter, dedup, and curate existing corpora (for example [Nemotron-CC](../data/curation/nemotron-cc.md))
- **`src/nemotron/recipes/data/sdg/`** — generate new datasets, including the long-document SDG pipeline consumed by Omni SFT
- **`src/nemotron/recipes/omni3/`** — family-specific training, RL, and evaluation stages

## Stage Summaries

### Stage 0: SFT

Omni SFT owns its own `Dockerfile`, `data_prep.py`, and `train.py` (built via the `nemotron omni3 build sft` dispatcher). The stage ports the Valor32k flow plus LoRA/PEFT variants for image-text and audio-text tuning. The released open-data configs target shorter sequence lengths; the upstream 16K → 49K → 262K progressive schedule is documented in [`architecture.md`](./architecture.md#progressive-context-scaling).

→ [SFT Guide](./sft.md)

### Stage 1: RL

The RL stack uses one shared NeMo-RL container and three sub-stages, mirroring the upstream [`nano-v3-omni` flow](https://github.com/NVIDIA-NeMo/RL/tree/nano-v3-omni):

1. **MPO** — multimodal preference optimization on the public MMPR dataset (~83K-row preference triples per subset)
2. **Text RL** — GRPO continuation of alignment on `nvidia/Nemotron-3-Nano-RL-Training-Blend`
3. **Vision RL** — GRPO on MMPR-Tiny (data prep ready; training launcher pending upstream)

The 20 RL datasets / 25 environments / ~2.3M rollouts referenced in the [release blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/) compose the full upstream alignment corpus; this recipe surfaces the public/open-source subset.

→ [RL Guide](./rl.md) · [RL Data Prep](./rl/data-prep.md)

### Stage 2: Evaluation

Evaluation reuses the Omni SFT container to serve the Megatron checkpoint and hands the compiled config to `nemo-evaluator-launcher`.

→ [Evaluation Guide](./evaluate.md)

## Execution Options

All Omni commands support [NeMo-Run](../../nemo_runspec/nemo-run.md) execution modes:

| Option | Behavior | Use Case |
|--------|----------|----------|
| `--run <profile>` | Attached—submits job and streams logs | Interactive development |
| `--batch <profile>` | Detached—submits and exits immediately | Long-running jobs |
| `--dry-run` | Preview resolved config or build plan | Validation |

## CLI Reference

<div class="termy">

```console
$ uv run nemotron omni3 --help
Usage: nemotron omni3 [OPTIONS] COMMAND [ARGS]...

 Omni3 training recipe

╭─ Infrastructure ─────────────────────────────────────────────────────────╮
│ build      Build an Omni3 stage container via podman on the cluster.     │
╰───────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────╮
│ data       Data preparation commands                                      │
│ model      Model import/export/eval commands                              │
│ rl         Reinforcement learning sub-stages                              │
╰───────────────────────────────────────────────────────────────────────────╯
╭─ Training Stages ─────────────────────────────────────────────────────────╮
│ sft        Run omni3 supervised fine-tuning.                              │
╰───────────────────────────────────────────────────────────────────────────╯
╭─ Evaluation ──────────────────────────────────────────────────────────────╮
│ eval       Run evaluation with NeMo-Evaluator (stage2).                   │
╰───────────────────────────────────────────────────────────────────────────╯
```

</div>

## Further Reading

- [Stage 0: SFT](./sft.md)
- [Stage 1: RL](./rl.md)
- [Stage 2: Evaluation](./evaluate.md)
- [Artifact Lineage](../artifacts.md)
- [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md)
- [W&B Integration](../wandb.md)

```{toctree}
:hidden:

sft.md
rl.md
evaluate.md
```
