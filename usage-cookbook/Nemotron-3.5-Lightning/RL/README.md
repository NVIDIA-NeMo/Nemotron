# Nemotron-3.5-Lightning RL Training Cookbook

This directory contains NeMo RL training guides for
[Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/nvidia/Nemotron-3.5-Lightning-30B-A3B-BF16).

- `grpo-dapo/`: direct DAPO/GRPO convergence training with NeMo RL's native
  math environment.
- `grpo-dapo-nemo-gym/`: DAPO/GRPO training that routes rollout and reward
  handling through [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

For a non-Slurm, two-node workstation path, use the
[two-DGX Station GRPO post-training guide](../dgx-station-recipes/README.md).
It starts Ray directly in the NeMo RL containers and includes optional NFS,
ARC-AGI data preparation, a two-GPU configuration, and a smoke test.

The recipes assume a shared filesystem mounted into the training container at
`/shared`, following the convention used in the Nemotron RL cookbooks:

```text
/shared                         <- Mount point for </YOUR/SHARED/STORAGE>
|____code
|    |____RL                    <- NeMo RL root repository
|    |____Nemotron              <- Public repository containing this cookbook
|____models
|____runs
|____.cache/huggingface
```

Run the commands in this root README from the login/head node, outside a Slurm
allocation. The detailed workflow guides contain the allocation and driver
commands.

Note on prebuilt NGC container: Nemotron-3.5-Lightning requires a NemoRL container later than [v0.7](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo-rl/-/tags) which is not yet available at the time of the model release data. Follow the docker build section below to build a container from source.

## Hardware Requirements

The recipes in this directory target one DGX H100 node (eight H100 GPUs)
managed by Slurm, with a shared filesystem visible from every node and mounted
at `/shared` in the container, and an accessible Docker registry per your
Slurm setup. The separate
[DGX Station guide](../dgx-station-recipes/README.md) targets two DGX Station
GB300 systems with one B300 GPU each.

## Clone the Repositories

Clone NeMo RL and the public [Nemotron cookbook repository](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook)
onto shared storage. The guides refer to these checkouts as `${NEMO_RL}` and
`${NEMOTRON_REPO}`.

Run from the login/head node:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"

mkdir -p "${SHARED_ROOT}/code"
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL}"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "${NEMOTRON_REPO}"
cd "${NEMO_RL}"
```

If the checkout was created without `--recursive`, initialize its submodules
before building an image or launching training:

```bash
git submodule update --init --recursive
```

## Container

Build the public NeMo RL release image from the checked-out repository. Run
this from a node that support docker build:

```bash
cd "${NEMO_RL}"
export NEMO_RL_REV="$(git rev-parse --short=12 HEAD)"

docker buildx build \
  --progress=plain \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --target release \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  -t "nemo-rl:nemotron-3.5-lightning-${NEMO_RL_REV}" \
  .
```

Note: this process can takes up to ~45 mins on a DGX H100 server.

Use the container format and Slurm integration required by your site when
following the detailed workflow guides. This might entail pushing the image to a central Docker registry, or create a local .sqsh file for the NemoRL Docker image on the shared file system.


## Install Hugging Face Tools

Run from the login/head node:

```bash
python -m pip install --upgrade --user "huggingface_hub[cli]" datasets
hf auth login
```

The `datasets` package is required only for the NeMo Gym guide, which converts
DAPO-Math data into Gym JSONL files on the login/head node.

## Environment File (optional)

Store credentials in `${SHARED_ROOT}/.env`. For example:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

## Download the Model

Download the checkpoint once on shared storage.

Run from the login/head node:

```bash
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"

mkdir -p "${MODEL_DIR}" "${HF_HOME}"
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --local-dir "${MODEL_DIR}"
```

The recipes refer to the mounted model as:

```text
/shared/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

## What to Run Next

Use [grpo-dapo/grpo_training_cookbook.md](grpo-dapo/grpo_training_cookbook.md)
for direct DAPO/GRPO convergence training, or
[grpo-dapo-nemo-gym/grpo_training_cookbook_nemo_gym.md](grpo-dapo-nemo-gym/grpo_training_cookbook_nemo_gym.md)
for the NeMo Gym path.

The recipes use 160 training steps, checkpoints every 10 steps,
validates every 20 steps with 128 samples, and logs to W&B.

## Historical dataset caveat

The publisher's historical dataset-production framework produced repeated
samples in the public artifacts: DAPO-Math-17k contains each source sample
100 times, and AIME-2024 contains each source sample 32 times. These recipes
deliberately do **not** filter or
deduplicate either dataset; they consume the publisher-provided rows as-is.

Treat training-row counts as repeated sampling rather than unique-problem
counts. Likewise, repeated AIME rows are not independent validation examples,
so do not interpret the configured validation sample count as an equally sized
set of distinct problems. Deduplicate upstream if a unique-problem training or
evaluation protocol is required, and record the resulting dataset revision,
row count, and procedure with the run.
