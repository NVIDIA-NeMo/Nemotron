# DAPO/GRPO with Nemotron Nano 3.5 and NeMo Gym

This guide adapts the direct Nemotron Nano 3.5 DAPO recipe to
[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). NeMo Gym manages rollout
routing and reward verification through its `math_with_judge` environment.
The judge model is disabled in this recipe, so the math verifier supplies the
reward.

The workflow supports AutoModel/FSDP2 or Megatron training with colocated
asynchronous vLLM generation. Both backends were tested on one node with four
NVIDIA GB200 GPUs. Other NVIDIA GPU systems can be used by adjusting the
inherited policy and generation parallelism shape as described in the direct
guide's
[`Scaling`](../grpo-dapo/grpo_training_cookbook.md#scaling) section.

Complete the public setup in [`../README.md`](../README.md) first.

## Assets

- [`dapo_nano_3_5_starter_nemo_gym.yaml`](dapo_nano_3_5_starter_nemo_gym.yaml):
  AutoModel/FSDP2 NeMo Gym training overlay.
- [`dapo_nano_3_5_starter_megatron_nemo_gym.yaml`](dapo_nano_3_5_starter_megatron_nemo_gym.yaml):
  Megatron NeMo Gym training overlay.
- [`prepare_hf_dapo_data_for_nemo_gym.py`](prepare_hf_dapo_data_for_nemo_gym.py):
  streaming converter for Hugging Face or local JSONL datasets.

Each Gym recipe inherits the matching adjacent direct DAPO recipe with a
relative path, so both remain portable when the Nemotron repository is cloned
to a different host directory.

## Configure the Workspace

```bash
export SHARED_ROOT=/path/to/shared-storage
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export NEMO_RL_IMAGE="${NEMO_RL_IMAGE:-nemo-rl:nemotron-nano-3.5}"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"

export GYM_ASSETS_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-Nano-3.5/RL/grpo-dapo-nemo-gym
export GYM_RECIPE_CONTAINER="${GYM_ASSETS_CONTAINER}/dapo_nano_3_5_starter_nemo_gym.yaml"
export GYM_DATA_CONTAINER=/shared/data/dapo_nano_3_5_nemo_gym
```

To use Megatron policy training instead, select its recipe:

```bash
export GYM_RECIPE_CONTAINER="${GYM_ASSETS_CONTAINER}/dapo_nano_3_5_starter_megatron_nemo_gym.yaml"
```

Create the data and output directories:

```bash
mkdir -p \
  "${HF_HOME}" \
  "${SHARED_ROOT}/data/dapo_nano_3_5_nemo_gym" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter_nemo_gym" \
  "${SHARED_ROOT}/logs/dapo_nano_3_5_starter_megatron_nemo_gym" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter_nemo_gym" \
  "${SHARED_ROOT}/results/dapo_nano_3_5_starter_megatron_nemo_gym"
```

On root-squashed storage, grant write access only to these directories
according to your site's policy.

## Prepare Gym JSONL

Convert the DAPO-Math training set:

```bash
docker run --rm \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python \
  "${GYM_ASSETS_CONTAINER}/prepare_hf_dapo_data_for_nemo_gym.py" \
  --dataset BytedTsinghua-SIA/DAPO-Math-17k \
  --cache-dir /shared/.cache/huggingface \
  --output "${GYM_DATA_CONTAINER}/train.jsonl"
```

Convert AIME-2024 for validation:

```bash
docker run --rm \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python \
  "${GYM_ASSETS_CONTAINER}/prepare_hf_dapo_data_for_nemo_gym.py" \
  --dataset BytedTsinghua-SIA/AIME-2024 \
  --cache-dir /shared/.cache/huggingface \
  --output "${GYM_DATA_CONTAINER}/validation.jsonl"
```

Create a four-row training file for the smoke test:

```bash
docker run --rm \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python \
  "${GYM_ASSETS_CONTAINER}/prepare_hf_dapo_data_for_nemo_gym.py" \
  --dataset BytedTsinghua-SIA/DAPO-Math-17k \
  --cache-dir /shared/.cache/huggingface \
  --limit 4 \
  --output "${GYM_DATA_CONTAINER}/smoke_train.jsonl"
```

Each row contains `responses_create_params`, `question`, `expected_answer`,
and an `agent_ref` that routes the example to
`math_with_judge_simple_agent`.

Validate the first converted row before launching training:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = (
    Path(os.environ["SHARED_ROOT"])
    / "data/dapo_nano_3_5_nemo_gym/smoke_train.jsonl"
)
with path.open() as stream:
    row = json.loads(stream.readline())

assert row["agent_ref"]["name"] == "math_with_judge_simple_agent"
assert {"responses_create_params", "question", "expected_answer"} <= row.keys()
print("Gym JSONL schema is valid.")
PY
```

## One-Step Smoke Test

This command starts NeMo Gym and vLLM's HTTP Responses API, collects four
rollouts, verifies rewards, computes log probabilities, and performs one
policy update:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${GYM_RECIPE_CONTAINER}" \
  grpo.num_prompts_per_step=1 \
  grpo.num_generations_per_prompt=4 \
  grpo.max_num_steps=1 \
  grpo.val_period=-1 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  policy.train_global_batch_size=4 \
  policy.max_total_sequence_length=1024 \
  policy.generation.max_new_tokens=64 \
  policy.generation.vllm_cfg.max_model_len=1024 \
  data.train.data_path="${GYM_DATA_CONTAINER}/smoke_train.jsonl" \
  logger.tensorboard_enabled=false \
  logger.monitor_gpus=false \
  checkpointing.enabled=false
```

Smoke-test rewards and losses validate wiring only; they are not model-quality
measurements.

## Run a Longer Job

After the smoke test succeeds, run with the recipe defaults:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/shared/.cache/huggingface \
  -e HF_TOKEN \
  -v "${SHARED_ROOT}:/shared" \
  -w /opt/nemo-rl \
  "${NEMO_RL_IMAGE}" \
  /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${GYM_RECIPE_CONTAINER}"
```

Checkpoints and logs are written under:

```text
${SHARED_ROOT}/results/dapo_nano_3_5_starter_nemo_gym
${SHARED_ROOT}/logs/dapo_nano_3_5_starter_nemo_gym
```

The Megatron recipe writes to the corresponding
`dapo_nano_3_5_starter_megatron_nemo_gym` directories and reuses the converted
checkpoint cache created by the direct Megatron recipe.

## Operational Notes

- Keep `async_engine` and `expose_http_server` enabled. NeMo Gym proxies the
  policy through vLLM's HTTP API.
- Keep the vLLM and Gym port ranges separate. The supplied configuration uses
  3000-4999 for NeMo RL/vLLM and 5000-5999 for Gym.
- The recipe uses the reasoning parser bundled with the model checkpoint.
- NeMo Gym consumes the validation JSONL exactly as provided; size and repeat
  it during data preparation rather than with `grpo.max_val_samples`.
- Export authentication tokens in the environment and never write their
  values into the recipe or guide.
