# DAPO/GRPO Training for Nemotron Nano 3.5

This guide fine-tunes
[Nemotron Nano 3.5 EA2](https://huggingface.co/nvidia/nemotron-nano-3.5-ea2)
with DAPO/GRPO on the `main` branch of
[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL).

The recipes provide either AutoModel/FSDP2 or Megatron policy training with
colocated vLLM generation and NeMo RL's DAPO math verifier. Both were tested
on one node with four NVIDIA GB200 GPUs, but can be adapted to other NVIDIA
GPU systems by changing the parallelism shape and memory-dependent workload
settings.

Complete the repository, container, model, and directory setup in the
[`../README.md`](../README.md) before continuing.

## Recipe

- [`dapo_nano_3_5_starter.yaml`](dapo_nano_3_5_starter.yaml) uses
  AutoModel/FSDP2.
- [`dapo_nano_3_5_starter_megatron.yaml`](dapo_nano_3_5_starter_megatron.yaml)
  uses Megatron.

Both inherit public Nano GRPO recipes from NeMo RL and add the DAPO settings
used here:

- asymmetric policy-ratio clipping;
- token-level policy loss;
- truncated importance-sampling correction;
- reward scaling;
- overlong-response shaping;
- the DAPO math verifier.

The default workload uses:

- 2 prompts per step;
- 8 generations per prompt;
- global training batch size 16;
- 4096 total tokens;
- up to 2048 generated tokens.

AutoModel training uses FSDP2 with expert parallelism 4. Megatron training
uses TP=1, PP=1, CP=1, and EP=4 with sequence packing. Colocated vLLM
generation uses tensor parallelism 4 for both backends.

## Configure the Launch

Set the shared workspace and optional image override:

```bash
export SHARED_ROOT=/path/to/shared-storage
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export NEMO_RL_IMAGE="${NEMO_RL_IMAGE:-nemo-rl:nemotron-nano-3.5}"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"

export RECIPE_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-Nano-3.5/RL/grpo-dapo/dapo_nano_3_5_starter.yaml
```

To use Megatron policy training instead, select its recipe:

```bash
export RECIPE_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-Nano-3.5/RL/grpo-dapo/dapo_nano_3_5_starter_megatron.yaml
```

This assumes the public Nemotron repository was cloned to
`${SHARED_ROOT}/code/Nemotron`, as shown in the root tutorial.

Create the output directories:

```bash
mkdir -p \
  "${HF_HOME}" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter_megatron" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter_megatron"
```

If the storage system root-squashes container users, grant write access only
to these cache and output directories according to your site's policy.

## Run Training

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config "${RECIPE_CONTAINER}"
```

Logs and checkpoints are written to:

```text
${SHARED_ROOT}/logs/dapo_nano_3_5_starter
${SHARED_ROOT}/results/dapo_nano_3_5_starter
```

The Megatron recipe writes to the corresponding
`dapo_nano_3_5_starter_megatron` directories. Its first launch also creates a
converted checkpoint under `${HF_HOME}/nemo_rl`. This one-time conversion can
take substantially longer than subsequent launches and requires additional
storage roughly comparable to the source checkpoint.

`HF_TOKEN` is optional for public datasets. If it is needed, export it in the
shell before launching Docker; never store its value in the recipe.

## One-Step Smoke Test

Run this wiring test before a longer job:

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config "${RECIPE_CONTAINER}" \
  grpo.max_num_steps=1 \
  grpo.num_prompts_per_step=1 \
  grpo.num_generations_per_prompt=4 \
  grpo.val_period=-1 \
  grpo.reward_shaping.max_response_length=256 \
  policy.train_global_batch_size=4 \
  policy.max_total_sequence_length=1024 \
  policy.generation.max_new_tokens=256 \
  data.max_input_seq_length=768 \
  data.validation=null \
  env.math.num_workers=2 \
  checkpointing.enabled=false \
  logger.monitor_gpus=false \
  logger.tensorboard_enabled=false
```

A successful smoke test generates four responses, verifies their rewards,
computes policy log probabilities, and completes one optimizer step.
Smoke-test rewards and losses are not model-quality measurements.

## Scaling

For a different NVIDIA GPU topology, adjust `cluster.num_nodes`,
`cluster.gpus_per_node`, and
`policy.generation.vllm_cfg.tensor_parallel_size`. For AutoModel, also adjust
`policy.dtensor_cfg.expert_parallel_size`. For Megatron, adjust TP, PP, CP, and
EP under `policy.megatron_cfg`; their combined shape must fit the world size,
and the global batch size must be divisible by the resulting data-parallel
size. Choose an expert-parallel size that divides the model's 128 routed
experts. Keep the global batch size compatible with the number of prompts and
generations trained per step.

When increasing context length, preserve:

```text
maximum input tokens + maximum generated tokens
    <= maximum total sequence length
```

Longer contexts increase both policy activation memory and vLLM KV-cache
memory. If generation runs out of memory, reduce
`policy.generation.vllm_cfg.gpu_memory_utilization` or rollout length before
changing the policy parallelism.

Keep the FP32 Mamba cache and eager compilation settings unless numerical
behavior has been revalidated. If policy and rollout log probabilities exceed
the configured error threshold, inspect tokenizer, router, and generation
settings before relaxing the threshold.
