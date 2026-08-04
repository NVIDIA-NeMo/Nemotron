# Reinforcement Learning with Nemotron Nano 3.5

This tutorial shows how to run DAPO/GRPO reinforcement learning for
[Nemotron Nano 3.5 EA2](https://huggingface.co/nvidia/nemotron-nano-3.5-ea2)
with the `main` branch of
[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL).

Two workflows are included:

- [`grpo-dapo/`](grpo-dapo/) uses NeMo RL's native math environment for
  rollout verification and rewards.
- [`grpo-dapo-nemo-gym/`](grpo-dapo-nemo-gym/) routes rollouts and rewards
  through [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

Both workflows provide AutoModel/FSDP2 and Megatron policy-training recipes
with vLLM generation. The supplied recipes were tested on a single node with
four NVIDIA GB200 GPUs. They can be adapted to other NVIDIA GPU systems by
adjusting the parallelism shape and memory-dependent workload settings.

## Prerequisites

You will need:

- a Linux system with Docker and the NVIDIA Container Toolkit;
- an NVIDIA GPU system with enough aggregate memory for the model and
  workload;
- enough shared or local storage for the model, datasets, logs, and
  checkpoints;
- Git, Git LFS, and the Hugging Face CLI;
- access to the Nemotron Nano 3.5 EA2 model repository.

### Hardware topology

The supplied topology uses four GPUs, expert parallelism 4 for policy
training, and tensor parallelism 4 for vLLM generation. The AutoModel recipes
use FSDP2 with expert parallelism 4. The Megatron recipes use TP=1, PP=1,
CP=1, and EP=4. To use a different NVIDIA GPU system, adjust these settings
together:

- `cluster.num_nodes` and `cluster.gpus_per_node`;
- `policy.dtensor_cfg.expert_parallel_size` for AutoModel;
- the `tensor_model_parallel_size`, `pipeline_model_parallel_size`,
  `context_parallel_size`, and `expert_model_parallel_size` fields under
  `policy.megatron_cfg` for Megatron;
- `policy.generation.vllm_cfg.tensor_parallel_size`.

Choose an expert-parallel size that divides the model's 128 routed experts
and is compatible with the distributed world size. For Megatron, keep the
global batch size divisible by the resulting data-parallel size. Depending on
available GPU memory, also tune the micro/global batch sizes, sequence
lengths, rollout count, and
`policy.generation.vllm_cfg.gpu_memory_utilization`.

Set a workspace location for the tutorial:

```bash
export SHARED_ROOT=/path/to/shared-storage
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
export MODEL_DIR="${SHARED_ROOT}/models/nemotron-nano-3.5-ea2"
```

The commands mount `${SHARED_ROOT}` into the container at `/shared`. The
recipes therefore refer to the model as:

```text
/shared/models/nemotron-nano-3.5-ea2
```

## 1. Clone the Repositories

Clone the public `main` branches with their submodules:

```bash
mkdir -p "${SHARED_ROOT}/code"

git clone --branch main --recursive \
  https://github.com/NVIDIA-NeMo/RL.git \
  "${NEMO_RL}"

git clone --branch main \
  https://github.com/NVIDIA-NeMo/Nemotron.git \
  "${NEMOTRON_REPO}"
```

If NeMo RL was cloned without `--recursive`, initialize its submodules:

```bash
cd "${NEMO_RL}"
git submodule update --init --recursive
```

Because this tutorial follows `main`, pull compatible updates to the checkout
and rebuild the container together. Avoid mixing a new recipe checkout with an
older image.

## 2. Build the NeMo RL Container

The NeMo RL
[Docker guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/docker.md)
recommends the release image and supports using the local checkout as a build
context.

Build an image from the checked-out `main` branch:

```bash
cd "${NEMO_RL}"

docker buildx build \
  --progress=plain \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --target release \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  -t nemo-rl:nemotron-nano-3.5 \
  .
```

SGLang and TensorRT-LLM are not required by these recipes, so skipping them
avoids their dependency and native-build time. Do not skip vLLM, which is used
by both rollout paths, including the FlashInfer CUTLASS MoE backend.

Verify that the container can see the GPUs:

```bash
docker run --rm --gpus all \
  nemo-rl:nemotron-nano-3.5 \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## 3. Download Nemotron Nano 3.5

Authenticate with Hugging Face if required:

```bash
hf auth login
```

Download the checkpoint:

```bash
mkdir -p "${MODEL_DIR}" "${HF_HOME}"

hf download nvidia/nemotron-nano-3.5-ea2 \
  --local-dir "${MODEL_DIR}"
```

Confirm that the model, tokenizer, and reasoning parser are present:

```bash
test -f "${MODEL_DIR}/config.json"
test -f "${MODEL_DIR}/model.safetensors.index.json"
test -f "${MODEL_DIR}/tokenizer.json"
test -f "${MODEL_DIR}/ultra_v3_reasoning_parser.py"
```

The checkpoint loads directly through AutoModel and vLLM. The first Megatron
launch converts the Hugging Face checkpoint to Megatron-Bridge format and
caches it under `${HF_HOME}/nemo_rl`; later Megatron launches reuse that
cache.

## 4. Prepare Runtime Directories

Create the cache and output directories used by both recipes:

```bash
mkdir -p \
  "${HF_HOME}" \
  "${SHARED_ROOT}/data/dapo_nano_3_5_nemo_gym" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter_megatron" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter_nemo_gym" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter_megatron_nemo_gym" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter_megatron" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter_nemo_gym" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter_megatron_nemo_gym"
```

On root-squashed shared filesystems, grant write access only to these
run-specific directories according to your site's security policy. Do not
make the entire shared workspace world-writable.

If a Hugging Face token is needed during training, export `HF_TOKEN` in the
shell and pass the variable into Docker with `-e HF_TOKEN`. Never store token
values in recipes or guides.

## 5. Choose a Workflow

### Direct DAPO/GRPO

The direct path uses the DAPO math datasets and NeMo RL's math verifier.

- Guide:
  [`grpo-dapo/grpo_training_cookbook.md`](grpo-dapo/grpo_training_cookbook.md)
- AutoModel recipe:
  [`grpo-dapo/dapo_nano_3_5_starter.yaml`](grpo-dapo/dapo_nano_3_5_starter.yaml)
- Megatron recipe:
  [`grpo-dapo/dapo_nano_3_5_starter_megatron.yaml`](grpo-dapo/dapo_nano_3_5_starter_megatron.yaml)

The recipe inherits NeMo RL's public Nano FSDP2 GRPO configuration:

[`examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml)

The Megatron overlay inherits NeMo RL's public Nano Megatron GRPO
configuration:

[`examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml)

### DAPO/GRPO with NeMo Gym

The Gym path exposes the vLLM Responses API and routes each example to
`math_with_judge_simple_agent`.

- Guide:
  [`grpo-dapo-nemo-gym/grpo_training_cookbook_nemo_gym.md`](grpo-dapo-nemo-gym/grpo_training_cookbook_nemo_gym.md)
- AutoModel recipe:
  [`grpo-dapo-nemo-gym/dapo_nano_3_5_starter_nemo_gym.yaml`](grpo-dapo-nemo-gym/dapo_nano_3_5_starter_nemo_gym.yaml)
- Megatron recipe:
  [`grpo-dapo-nemo-gym/dapo_nano_3_5_starter_megatron_nemo_gym.yaml`](grpo-dapo-nemo-gym/dapo_nano_3_5_starter_megatron_nemo_gym.yaml)
- Dataset converter:
  [`grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py`](grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py)

This path uses NeMo RL's public
[`examples/nemo_gym/run_grpo_nemo_gym.py`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/nemo_gym/run_grpo_nemo_gym.py)
entry point.

## 6. Start with a Smoke Test

Run the one-step smoke command in the selected workflow guide before starting
a longer job. A smoke test checks:

- model and tokenizer loading;
- Ray worker placement;
- vLLM generation and policy weight transfer;
- reward verification;
- log-probability calculation;
- one policy optimizer step.

Smoke-test rewards and loss values are not evidence of convergence or final
model quality.

For a longer run or a different GPU topology, remove the smoke-only
command-line overrides and tune batch size, sequence length, rollout count,
and the parallelism shape described above. Keep the following relationship
valid:

```text
maximum input tokens + maximum generated tokens
    <= maximum total sequence length
```

The global training batch size must also remain compatible with the number of
prompts, generations per prompt, and distributed policy topology.

## Troubleshooting

### The model is not found inside the container

Confirm that `${SHARED_ROOT}` is mounted at `/shared` and that the checkpoint
exists at:

```text
${SHARED_ROOT}/models/nemotron-nano-3.5-ea2
```

### Ray workers do not see all GPUs

Pass the intended device set explicitly with
`-e CUDA_VISIBLE_DEVICES=0,1,2,3`, and ensure the recipe's
`cluster.gpus_per_node` matches the allocated GPUs.

### Training runs out of memory

Reduce rollout length and total sequence length first. For generation memory
pressure, reduce `policy.generation.vllm_cfg.gpu_memory_utilization`. Preserve
the FP32 Mamba cache setting unless numerical behavior has been revalidated.

### A Megatron launch was interrupted during checkpoint conversion

The first Megatron launch must finish writing
`${HF_HOME}/nemo_rl/.../iter_0000000/run_config.yaml`. If the process was
interrupted and a later launch reports that this file is missing, remove only
that incomplete converted-model cache directory and rerun. Do not remove the
source Hugging Face checkpoint.

### NeMo Gym cannot reach vLLM

Keep `policy.generation.vllm_cfg.async_engine=true` and
`policy.generation.vllm_cfg.expose_http_server=true`. When using Docker,
launch the Gym workflow with host networking or provide equivalent network
routing for its HTTP services.

## References

- [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [NeMo RL documentation](https://docs.nvidia.com/nemo/rl/latest/)
- [Nemotron Nano 3.5 EA2 model](https://huggingface.co/nvidia/nemotron-nano-3.5-ea2)
- [NVIDIA Nemotron developer repository](https://github.com/NVIDIA-NeMo/Nemotron)
- [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
