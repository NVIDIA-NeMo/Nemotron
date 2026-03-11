# SWE-RL (Stages 2.1–2.2)

End-to-end RL for software engineering tasks. SWE-RL is handled as a separate stage because its rollouts take substantially longer and require much longer context lengths, making it a throughput bottleneck when trained alongside the other RLVR environments.

---

## SWE Container

Both SWE stages require pre-fetched Python virtual environments that are not included in the base `nemo-rl:v0.5.0.nemotron_3_super` image. Build the SWE container once (from within the [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) repo):

```text
docker buildx build \
  -t your-registry/nemo-rl:v0.5.0.nemotron_3_super_swe \
  --push \
  -f- . <<'EOF'
FROM nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super

RUN <<'RUNEOF'
set -euxo pipefail
UV_TORCH_BACKEND=$(uv run python -c "import tomllib,pathlib; \
  indexes=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['tool']['uv']['index']; \
  print(next(i['name'].removeprefix('pytorch-') for i in indexes if i['name'].startswith('pytorch-')))") \
UV_LINK_MODE=hardlink uv run python examples/nemo_gym/prefetch_venvs.py \
    examples/configs/super/stage2_swe1.yaml \
    examples/configs/super/stage2_swe2.yaml
RUNEOF
EOF
```

Set the container image in your config or via override:

```bash
uv run nemotron super3 rl swe1 --run YOUR-CLUSTER \
    run.env.container=your-registry/nemo-rl:v0.5.0.nemotron_3_super_swe
```

---

## Stage 2.1 — SWE 1 (64 nodes)

SWE-pivot training using a single-step tool use comparison approach. The model receives a code problem and must produce a solution evaluated against ground truth.

### Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 64 (512 GPUs) |
| Generation nodes | 32 (colocated=false) |
| Prompts/step | 64 |
| Generations/prompt | 16 |
| Batch size | 1,024 |
| Max sequence length | 131,072 |
| TP / CP | 8 / 8 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Prefix caching | enabled |

### Config Files

- `stage2_swe1/config/default.yaml` — Full-scale 64-node config
- `stage2_swe1/config/small.yaml` — Reduced 8-node variant for testing

### Using nemotron CLI

```bash
uv run nemotron super3 rl swe1 --run YOUR-CLUSTER
```

> **`--run YOUR-CLUSTER`** refers to a profile defined in your `env.toml` file.
> See the [env.toml setup guide](../README.md#configuration) for details.
>
> SWE stages require the [SWE container](index.md#swe-container) with pre-fetched venvs.

### Using super_launch.sh

```bash
EXP_NAME=stage2.1-swe1 \
CONFIG_PATH=examples/configs/super/stage2_swe1.yaml \
MODEL_PATH=/path/to/rlvr3_checkpoint \
TRAIN_PATH=$DATA_DIR/swe1/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe1/val-split.jsonl \
CONTAINER=$SWE_CONTAINER \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

---

## Stage 2.2 — SWE 2 (64 nodes)

Full SWE-bench training with container-isolated sandboxes. Each rollout launches an isolated container with the target repository, runs the OpenHands agent loop to produce a code patch, and runs ground-truth tests to compute a binary reward.

### Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 64 (512 GPUs) |
| Generation nodes | 32 (colocated=false) |
| Prompts/step | 16 |
| Generations/prompt | 32 |
| Batch size | 512 |
| Max sequence length | 196,608 |
| TP / CP | 8 / 8 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Agent max turns | 200 |
| Agent concurrency | 768 |
| Agent timeout | 3,600s |
| Thinking mode | enabled |

### Config Files

- `stage2_swe2/config/default.yaml` — Full-scale 64-node config

### Infrastructure

#### Container Isolation

Each SWE task instance needs an isolated environment with its own filesystem to execute code and run tests. The upstream training uses **Apptainer** (formerly Singularity) because SLURM HPC clusters typically lack root access, which rules out Docker. Apptainer runs pre-built `.sif` images with a writable tmpfs overlay while sharing the host kernel.

> **Other environments**: If you have root access or are running outside SLURM, Docker or Podman can provide the same container isolation with stronger process and memory boundaries (cgroup isolation). The SWE-bench environment images are available as standard Docker images from R2E-Gym, SWE-Gym, and SWE-Bench Verified on HuggingFace — they only need to be converted to `.sif` format for Apptainer.

Because Apptainer shares the host kernel and memory space, additional safeguards are needed:

| Component | Description |
|-----------|-------------|
| **Memory Watchdog** | Monitors aggregate RSS of tmux process trees and proactively kills runaway processes, since Apptainer containers share host memory (unlike Docker's cgroup isolation). |
| **Command Blocklist** | Regex-based blocklist intercepts dangerous commands (`killall`, `pkill`) that could terminate training processes or vLLM servers on the same node. |

These safeguards are less critical when using Docker or Podman, which provide cgroup-based process and memory isolation by default.

#### Agent Loop

| Component | Description |
|-----------|-------------|
| **OpenHands Agent Loop** | Manages the full lifecycle: initializing runtime, presenting problems, running agent steps (up to 200 turns), extracting git patches, and running tests for binary reward. |
| **Harness Diversity** | OpenCode and Codex agent classes within OpenHands match external harness tool formats (Claude Code, Codex CLI) for training diversity. |

### Prerequisites

For SLURM clusters without root access, install [Apptainer](https://apptainer.org/docs/admin/main/installation.html) and download SIF images:

```bash
# Install Apptainer (Ubuntu/Debian)
wget https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb
sudo apt install -y ./apptainer_1.3.1_amd64.deb

# Download and convert Docker images to .sif files
./examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif --concurrency 16
```

### Using nemotron CLI

```bash
uv run nemotron super3 rl swe2 \
    --run YOUR-CLUSTER \
    run.env.sif_dir=/path/to/sif
```

> **`--run YOUR-CLUSTER`** refers to a profile defined in your `env.toml` file.
> See the [env.toml setup guide](../README.md#configuration) for details.
>
> SWE stages require the [SWE container](index.md#swe-container) with pre-fetched venvs.

### Using super_launch.sh

```bash
EXP_NAME=stage2.2-swe2 \
CONFIG_PATH=examples/configs/super/stage2_swe2.yaml \
MODEL_PATH=/path/to/swe1_checkpoint \
TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe2/val-split.jsonl \
CONTAINER=$SWE_CONTAINER \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SIF_DIR=/path/to/sif \
bash super_launch.sh
```

See the [upstream training guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/examples/nemotron_3_super/README.md) for full details on environment variables.

---

**Recipe Source**: `src/nemotron/recipes/super3/stage2_rl/stage2_swe1/` and `src/nemotron/recipes/super3/stage2_rl/stage2_swe2/`
