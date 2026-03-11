# Stage 2.2: SWE-RL (SWE-bench)

End-to-end RL for software engineering using full SWE-bench environments with container-isolated sandboxes. This stage runs the OpenHands agent loop inside isolated containers to produce code patches, evaluated by running ground-truth tests for binary rewards.

## Infrastructure

### Container Isolation

Each SWE task instance needs an isolated filesystem to execute code and run tests. The upstream training uses **Apptainer** (formerly Singularity) because SLURM HPC clusters typically lack root access, which rules out Docker. Apptainer runs pre-built `.sif` images with a writable tmpfs overlay while sharing the host kernel.

> **Other environments**: If you have root access or are not on SLURM, Docker or Podman can provide the same isolation with stronger process and memory boundaries (cgroup isolation). The SWE-bench environment images are standard Docker images from R2E-Gym, SWE-Gym, and SWE-Bench Verified on HuggingFace — they only need conversion to `.sif` format when using Apptainer.

SIF images are specified via `container_formatter` patterns that map instance IDs to container paths:
```
swebench_sweb.eval.x86_64.{instance_id}.sif
swegym_sweb.eval.x86_64.{instance_id}.sif
r2egym_{instance_id}.sif
```

### OpenHands Agent Loop

A modified version of OpenHands manages the full lifecycle of each interaction:

1. Initialize the runtime
2. Present the problem statement
3. Run the agent's step loop (up to 200 turns)
4. Extract the git patch
5. Run ground-truth tests for binary reward
6. Clean up

The agent interacts with the repository workspace through bash commands and file operations via a tmux-based session.

### Harness Diversity

To increase tool diversity during training, OpenCode and Codex agent classes are implemented within OpenHands, matching the tool input/output formats of Claude Code and Codex CLI respectively. Both agents plug into OpenHands' existing runtime and conversation memory.

### Memory Management & Command Blocklist (Apptainer-specific)

Because Apptainer shares the host kernel and memory space, additional safeguards are needed that are less critical with Docker/Podman:

- **Memory Watchdog**: Monitors aggregate RSS of tmux process trees and proactively kills runaway processes, since Apptainer containers share host memory.
- **Command Blocklist**: Regex-based blocklist intercepts dangerous commands (`killall`, `pkill`) that could terminate training processes or vLLM servers on the same node.

### Serialization

HTTP payloads between the gym and model server (carrying prompt token IDs, generated token IDs, and log probabilities) use `orjson` instead of Python's standard `json` for Rust-based serialization performance.

## Configuration

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

### Config files

- `config/default.yaml` — Full-scale 64-node config

## Prerequisites

- **NeMo-RL repo**: Clone the `super-v3` branch of [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- **SWE container**: The base nemo-rl container does not include pre-fetched venvs for SWE stages. See [stage2_swe1/README.md](../stage2_swe1/README.md#prerequisites) for build instructions.
- **Sandbox container**: Required for code execution environments
- **SIF images**: SWE-bench environment images (R2E-Gym, SWE-Gym, SWE-Bench Verified) in Apptainer `.sif` format. See below.

### Downloading SIF Images (SLURM / Apptainer)

```bash
# Install Apptainer if needed (Ubuntu/Debian)
wget https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb
sudo apt install -y ./apptainer_1.3.1_amd64.deb

# Download and convert Docker images to .sif files
./examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif --concurrency 16
```

## Usage

```bash
nemotron super3 rl swe2 \
    --run <profile> \
    run.env.sandbox.container=<sandbox-image> \
    run.env.persistent_cache=/path/to/cache \
    run.env.sif_dir=/path/to/sif
```

> **`--run <profile>`** refers to a profile defined in your `env.toml` file,
> which configures SLURM account, partition, mounts, and other cluster settings.
> See the [env.toml setup guide](../../README.md#envtoml-setup) for details.

Or via `super_launch.sh` directly:

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

See the [upstream training guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md) for full details on environment variables.

## References

- [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [NeMo Gym](https://github.com/NVIDIA/NeMo-Gym)
- [NeMo RL](https://github.com/NVIDIA/NeMo-RL)
