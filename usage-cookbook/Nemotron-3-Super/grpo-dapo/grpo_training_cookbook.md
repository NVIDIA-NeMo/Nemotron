# Nemotron Super V3 GRPO/DAPO Training with NeMo RL

## Overview
This guide demonstrates step-by-step RL training of the Nemotron Super V3 model on a 4-node NVIDIA H100 cluster running Slurm.

We will carry out GRPO/DAPO training of the model on the [DAPO-Math-17k dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). This is a single-domain reinforcement learning example with verifiable rewards. In this example, we start from the Nemotron Super V3 base pretrained model [https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16). The DAPO training process can help the model discover advanced math reasoning by itself, also known as the DeepSeek "aha" [moment](https://www.reddit.com/r/OpenAI/comments/1i6jsr2/deepseek_discovered_their_new_model_having_an_aha).

**Note:**
- Interactive vs. batch training: in a production setting, it is usually more convenient to submit training jobs as Slurm batch jobs. However, setting up an interactive training environment lets you iterate and debug faster. Once interactive jobs run smoothly, you can submit them as batch training jobs.

## Prerequisites

- **Compute**: 4x H100 nodes with 8 GPUs per node, for 32 H100 GPUs total. This recipe uses colocated training: vLLM rollout and Megatron policy training share the same 4-node GPU pool.

- **Storage**: A high-speed shared network file system for storing code, models, checkpoints, and other temporary assets. In this guide, we assume that the shared storage is at `</YOUR/SHARED/NETWORK/STORAGE>` on the host system, mounted as `/shared` inside the working Docker container, and accessible from all nodes. We also assume the following directory structure:

```
</YOUR/SHARED/NETWORK/STORAGE> (on host):/shared (inside container)
|______code
|        |____RL  # NeMo RL root directory
|        |____Nemotron/usage-cookbook/Nemotron-3-Super  # Repository containing this cookbook
|_______models
|        |____NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 # base model checkpoint
|_______checkpoints
|_______HF_HOME   # Hugging Face cache directory
```

Note that each model checkpoint, including model weights and optimizer state, can require up to ~1 TB of storage. You should also account for the number of checkpoints you would like to keep, such as the best 3 checkpoints in the NeMo RL training config. In addition, the base BF16 model checkpoint requires ~231 GB of storage, plus another ~231 GB for the Megatron-converted checkpoint.

- **Model**: Download the Hugging Face-format model with the [HF CLI tool](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to the shared location on the high-speed storage. 
```bash
mkdir -p </YOUR/SHARED/NETWORK/STORAGE>/models
cd </YOUR/SHARED/NETWORK/STORAGE>/models
export HF_TOKEN=<YOUR_HF_TOKEN>

hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 --local-dir NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16
```
- **Docker image**: We use the prebuilt NeMo RL Docker container from NGC at `nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super`.

Also see the latest NeMo RL Docker build [guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/docker.md) for more information about building a Docker image from source.

## Step 1. Prepare the training config file

This step defines the training recipe for a single-domain RL workload: verifiable mathematical reasoning on DAPO-style data. The policy learns from sampled solutions, and rewards are computed by math verifiers, so this setup is ideal for tasks where correctness can be programmatically checked, such as arithmetic, algebra word problems, and competition-style math.

Use the provided 4x8 H100 colocated config in this directory: [dapo17k_nemotron_super_120b_h100_4x8_colocated.yaml](dapo17k_nemotron_super_120b_h100_4x8_colocated.yaml).

Copy it to the NeMo RL repo at:

```text
</YOUR/SHARED/NETWORK/STORAGE>/code/RL/examples/configs/recipes/llm/dapo17k_nemotron_super_120b_h100_4x8_colocated.yaml
```

The config is set up for:

- 4 H100 nodes with 8 GPUs per node, for 32 GPUs total
- colocated vLLM generation and Megatron policy training
- DAPO-Math-17k training and AIME 2024 validation
- Nemotron Super 120B Base BF16 checkpoint at `/shared/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16` (path as seen inside the container)
- an explicit chat template, because the base checkpoint does not ship an instruction prompt template

All paths inside the YAML are container paths. If your shared storage is mounted somewhere other than `/shared`, update the paths in the YAML before launching.

## Step 2. Interactive Job Submission

We recommend submitting the NeMo RL job as an interactive session for quick debugging and iteration.
Submitting a NeMo RL job without an explicit `COMMAND` gives you a handle to an interactive session.

```bash
ROOT_DIR="</YOUR/SHARED/NETWORK/STORAGE>/code/RL" # NeMo RL root directory on the shared storage
RAY_SUB="${ROOT_DIR}/ray.sub" # Path to ray.sub on the host

export WANDB_API_KEY="YOUR_WANDB_KEY"
export ACCOUNT="YOUR_SLURM_ACCOUNT"
export PARTITION="YOUR_H100_PARTITION"
export CONTAINER="nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super"
export MOUNTS="</YOUR/SHARED/NETWORK/STORAGE>:/shared"

export JOB_NAME="grpo-dapo-super-h100-4x8"
export NUM_NODES="4"
export GPUS_PER_NODE="8"
export TIME_LIMIT="02:00:00"

export HF_HOME="/shared/HF_HOME"
export NCCL_DEBUG=INFO

sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --exclusive \
  --dependency=singleton \
  "${RAY_SUB}"
```
Upon successful submission, Slurm will print the `SLURM_JOB_ID`:

```text
Submitted batch job 1980204
```

Once the Ray cluster is up, a script will be created to attach to the Ray head node. Run this script to launch experiments:

```bash
bash ./1980204-attach.sh
```

See the NeMo RL [cluster guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/cluster.md) for further information.

Now that you are on the head node, launch training with:
```bash
# Path to the YAML config file inside the container.
CONFIG_PATH="/shared/code/RL/examples/configs/recipes/llm/dapo17k_nemotron_super_120b_h100_4x8_colocated.yaml"

export WANDB_API_KEY="YOUR_WANDB_KEY" # Optional for remote logging and monitoring. Set `wandb_enabled: true` in config
export HF_HOME="/shared/HF_HOME" # Your HF cache directory on the shared storage
export NRL_IGNORE_VERSION_MISMATCH=1
export NCCL_DEBUG=INFO

cd /shared/code/RL # NeMo RL repo inside the container
uv run examples/run_grpo.py \
  --config "${CONFIG_PATH}"
```

## Step 3. Batch Job Submission

Once the interactive training jobs run smoothly, you can launch batch training jobs. On a production H100 cluster, launch the 4-node colocated job from a Slurm login/head node with the following procedure.

```bash
ROOT_DIR="</YOUR/SHARED/NETWORK/STORAGE>/code/RL" # NeMo RL root directory on the shared storage
RAY_SUB="${ROOT_DIR}/ray.sub" # Path to ray.sub on the host

# Path to the YAML config file inside the container.
CONFIG_PATH="/shared/code/RL/examples/configs/recipes/llm/dapo17k_nemotron_super_120b_h100_4x8_colocated.yaml"

export WANDB_API_KEY="YOUR_WANDB_KEY"
export ACCOUNT="YOUR_SLURM_ACCOUNT"
export PARTITION="YOUR_H100_PARTITION"
export CONTAINER="nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super"
export MOUNTS="</YOUR/SHARED/NETWORK/STORAGE>:/shared"

export JOB_NAME="grpo-dapo-super-h100-4x8"
export NUM_NODES="4"
export GPUS_PER_NODE="8"
export TIME_LIMIT="02:00:00"

export HF_HOME="/shared/HF_HOME"
export NCCL_DEBUG=INFO
export NRL_IGNORE_VERSION_MISMATCH=1

export COMMAND="cd /shared/code/RL && uv run examples/run_grpo.py --config ${CONFIG_PATH}"

sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --exclusive \
  --dependency=singleton \
  "${RAY_SUB}"
```

**Tweaking training hyperparameters**

Once training verification is successful, you can tweak the configuration parameters:

- **Run length / throughput (`grpo`)**
  - `max_num_steps`, `num_prompts_per_step`, `num_generations_per_prompt`
  - These directly control training duration, sample volume, and compute cost per step.

- **Reward behavior (`grpo.reward_shaping`, `grpo.reward_scaling`)**
  - `reward_shaping.enabled`, `overlong_buffer_length`, `overlong_buffer_penalty`
  - `reward_scaling.enabled` and min/max ranges
  - Use these to penalize overly long outputs and keep reward magnitude stable.

- **Policy sequence budget (`policy`)**
  - `max_total_sequence_length`, `train_micro_batch_size`, `logprob_batch_size`
  - Increase carefully: longer contexts improve reasoning capacity but significantly raise memory and latency.

- **Distributed training topology (`policy.megatron_cfg`)**
  - `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`
  - Must match the 32-GPU colocated topology and desired DP/TP/PP/CP balance.

- **Generation backend (`policy.generation.vllm_cfg`)**
  - `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`
  - Tune for rollout speed and stability while sharing GPUs with Megatron in the colocated setup.

- **Dataset and validation (`data`, `grpo`)**
  - `data.train.dataset_name`, `data.validation.dataset_name`, `max_val_samples`, `val_period`
  - Keep validation frequent enough to catch regressions, but not so frequent that it slows training.
