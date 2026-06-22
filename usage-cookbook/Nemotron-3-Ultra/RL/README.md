# Nemotron 3 Ultra RL Training Cookbook

This directory contains NeMo RL training guides for the
[Nemotron 3 Ultra model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16).

- `grpo-dapo/`: direct DAPO/GRPO recipe with Megatron policy training and
  non-colocated vLLM generation.
- `grpo-dapo-nemo-gym/`: DAPO/GRPO recipe that routes rollout and reward
  handling through [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

The recipes assume a shared filesystem mounted into the training container at
`/shared`, following the convention used in the Nemotron RL cookbooks:

```text
/shared             <- Mount point for </YOUR/SHARED/NETWORK/STORAGE>
|____code
|    |____RL        <- Nemo RL root repo
|    |____Nemotron  <- Directory containing this cookbook
|____models
|____checkpoints
|____HF_HOME
```

All commands in this root README are run from the login/head node, outside any
Slurm allocation.

## Hardware Requirements

Nemotron 3 Ultra is a frontier-scale hybrid Mamba/MoE/Attention model. These
guides are intended for GB200 GPU nodes managed by a Slurm cluster, with fast
node-local and cross-node networking.

Recommended baseline:

- 18 GPU nodes total
- 4 GPUs per node
- 16 nodes for Megatron policy training
- 2 nodes for non-colocated vLLM generation
- one high-speed 16-node training segment, when your Slurm cluster supports
  topology-aware or rack-aligned allocation
- shared filesystem visible from every node and mounted into the container at
  `/shared`

The YAML profiles use Megatron TP=8, CP=8, EP=64 and vLLM TP=8. Smaller
clusters generally require changing the parallelism layout and are not covered
by this guide.

## Clone NeMo RL

Clone the Ultra-compatible NeMo RL branch onto shared storage. The later guides
refer to this checkout as `${NEMO_RL}`.

Run from the login/head node:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/NETWORK/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"

mkdir -p "${SHARED_ROOT}/code"
git clone --recursive --branch ultra-v3 https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL}"
cd "${NEMO_RL}"
```

If you cloned the repository without `--recursive`, initialize the submodules
before building the container or launching training:

```bash
git submodule update --init --recursive
```

## Container

These are the instructions to build the training container. The image targets
**aarch64 (arm64)** for GB200 NVL72 nodes and bundles a custom vLLM build
required by Ultra.

From the root of the repo:

```bash
docker buildx build \
  --progress=plain \
  -f docker/Dockerfile \
  --target release \
  -t nemo-rl-ultra:arm64 \
  --build-context nemo-rl=. \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  --build-arg BUILD_CUSTOM_VLLM_URL=https://github.com/TomerBN-Nvidia/vllm.git \
  --build-arg BUILD_CUSTOM_VLLM_REF=ultra-rl-v0.17 \
  --build-arg BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0-cp38-abi3-manylinux_2_31_aarch64.whl \
  .
```

Build args:
- `BUILD_CUSTOM_VLLM=1` with `BUILD_CUSTOM_VLLM_URL` / `BUILD_CUSTOM_VLLM_REF` -
  builds the Ultra vLLM fork; `BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION`
  points at the matching upstream aarch64 wheel so the build reuses precompiled
  kernels instead of compiling from source.
- `SKIP_SGLANG_BUILD=1` - Ultra runs on vLLM; skip the SGLang build.
- `MAX_JOBS` - parallel build jobs; tune to your machine.
- `--build-context nemo-rl=.` - build from your local checkout (otherwise the
  Dockerfile pulls `NVIDIA-NeMo/RL.git#main`).

To run on the cluster with Slurm, convert the image to a squashfs (`.sqsh`)
with [enroot](https://github.com/NVIDIA/enroot), then pass that path as
`CONTAINER` to the launcher:

```bash
enroot import -o nemo-rl-ultra.sqsh dockerd://nemo-rl-ultra:arm64
```

The container must be launched with mounts for both the shared root and any
site filesystem path needed for Slurm startup, for example:

```bash
export CONTAINER=</PATH/TO/nemo-rl-ultra.sqsh>
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"
```

## Install Hugging Face Tools

Run from the login/head node:

```bash
python -m pip install --upgrade --user "huggingface_hub[cli]" datasets
hf auth login
```

If your environment uses the older CLI entry point, `huggingface-cli login` is
equivalent to `hf auth login`.

The `datasets` package is needed by the NeMo Gym guide to convert
DAPO-Math-17K into Gym JSONL from the login/head node.

## Optional Environment File

The training launch snippets source `${NEMO_RL}/.env` when it exists. Use this
file for local credentials instead of putting secrets directly into commands.

Run from the login/head node:

```bash
cat > "${NEMO_RL}/.env" <<'EOF'
export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
EOF

chmod 600 "${NEMO_RL}/.env"
```

Set `logger.wandb_enabled=True` in the launch command or YAML when you want to
log to W&B.

## Download the Model

Download the Hugging Face checkpoint once on shared storage.

Run from the login/head node:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/NETWORK/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export HF_HOME="${SHARED_ROOT}/HF_HOME"
export HF_TOKEN=<YOUR_HF_TOKEN>

mkdir -p "${SHARED_ROOT}/models" "${HF_HOME}"

hf download nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --local-dir "${SHARED_ROOT}/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" \
  --local-dir-use-symlinks False
```

If your environment uses `huggingface-cli` instead of `hf`, the equivalent
command is also run from the login/head node:

```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --local-dir "${SHARED_ROOT}/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" \
  --local-dir-use-symlinks False
```

## Prepare a Compatibility Overlay

NeMo RL runtime expect an older format than the HF checkpoint. Therefore, we 
create a lightweight compatibility overlay before launching the training
recipes. The overlay writes a converted `config.json` and symlinks sidecar files
and weight shards from the downloaded checkpoint.

Run from the login/head node:

```bash
cd "${NEMO_RL}"

python Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/create_ultra_new_to_old_overlay.py \
  --source "${SHARED_ROOT}/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" \
  --runtime-source Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/checkpoint_compatibility \
  --output "${SHARED_ROOT}/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat" \
  --force
```

Use the overlay anywhere a recipe asks for `policy.model_name` or
`policy.tokenizer.name`:

```text
/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat
```

For vLLM chat serving through NeMo Gym, also point the reasoning parser plugin
at the overlay copy:

```text
policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser_plugin=/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat/ultra_v3_reasoning_parser.py
```

The `checkpoint_compatibility` directory contains the two small Ultra runtime
implementation files needed by runtimes that do not yet natively recognize
`model_type=nemotron_h`:

```text
checkpoint_compatibility/
|____configuration_nemotron_h.py
|____modeling_nemotron_h.py
```

These files are copied into the overlay and referenced through `auto_map`.
Modern runtimes that already support `model_type=nemotron_h` can omit
`--runtime-source`, but keeping it is harmless and makes the guide portable
across current NeMo RL containers.

## What to Run Next

Use `grpo-dapo/grpo_training_cookbook.md` for the direct DAPO/GRPO recipe, or
`grpo-dapo-nemo-gym/grpo_training_cookbook_nemo_gym.md` for the NeMo Gym path.

Both guides are written around an 18-node baseline shape: 16 nodes for Megatron
policy training and 2 nodes for non-colocated vLLM generation. The default
profile uses 4k total sequence length, 2k-token rollouts, 64 prompts x 16
generations per step, 100 training steps, and checkpoints every 10 steps.
