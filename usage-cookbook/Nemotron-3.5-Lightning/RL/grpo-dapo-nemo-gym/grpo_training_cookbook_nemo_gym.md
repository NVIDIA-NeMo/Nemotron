# DAPO/GRPO with Nemotron-3.5-Lightning and NeMo Gym

This guide runs the one-node AutoModel/FSDP2 Lightning topology with NeMo Gym
for rollout routing and `math_with_judge` reward verification. Use an attached
interactive NeMo RL allocation for data preparation and a one-step validation,
then submit the full training run through NeMo RL's `ray.sub` batch workflow.

Complete the public setup in [`../README.md`](../README.md) first.

## Dataset and training goal

This workflow converts the public `BytedTsinghua-SIA/DAPO-Math-17k` training
set and `BytedTsinghua-SIA/AIME-2024` validation set into NeMo Gym JSONL. For
each training problem, the policy generates multiple candidate solutions; NeMo
Gym routes them to the `math_with_judge` environment, where the verifiable math
reward scores them before DAPO/GRPO updates the policy. The goal is to improve
verifiable mathematical reasoning while exercising NeMo Gym's HTTP rollout and
reward-routing integration on the tested single DGX H100 node topology.

## Assets

- [`dapo_nemotron_3_5_lightning_nemo_gym.yaml`](dapo_nemotron_3_5_lightning_nemo_gym.yaml):
  self-contained one-node NeMo Gym recipe.
- The shared [Ultra NeMo Gym data-preparation
  converter](../../../Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py).
  Lightning follows the same data-preparation procedure as the Ultra guide.

## Configure the Workspace

Run from the login/head node:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export CONTAINER=<SITE_ACCESSIBLE_NEMO_RL_IMAGE>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export GPUS_PER_NODE=8
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"

mkdir -p \
  "${SHARED_ROOT}/.cache/huggingface" \
  "${SHARED_ROOT}/data/dapo_nano_3_5_nemo_gym" \
  "${SHARED_ROOT}/logs/dapo_nemotron_3_5_lightning_nemo_gym" \
  "${SHARED_ROOT}/results/dapo_nemotron_3_5_lightning_nemo_gym"
```

On root-squashed storage, grant write access only to these run-specific
directories according to your site's policy.

## Interactive NeMo RL Job: Prepare Data and Validate

Use an attached one-node allocation first. It keeps the NeMo RL container and
eight GPUs available while you prepare the JSONL files and run a one-step
smoke test. It does not automatically start training.

Run from the login/head node:

```bash
# Use the site partition that permits an attached one-node allocation. A
# partition need not literally be named "interactive"; COMMAND is unset so
# this allocation stays available for manual data preparation and validation.
export PARTITION=<SLURM_PARTITION>
unset COMMAND

cd "${NEMO_RL}"
sbatch \
  --nodes=1 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=interactive-dapo-lightning-gym \
  --time=02:00:00 \
  --gres=gpu:8 \
  --exclusive \
  --mem=0 \
  ray.sub
```

After Slurm prints the job ID and Ray creates its attach helper, enter the Ray
head from the login/head node:

```bash
bash ./<jobid>-attach.sh
```

### Prepare the Gym JSONL files

Run these commands from the attached allocation/container. They perform the
Hugging Face data conversion inside NeMo RL rather than on the login node:

```bash
cd /opt/nemo-rl

if [ -f /shared/.env ]; then set -a && source /shared/.env && set +a; fi

# Define runtime paths again inside the attached container. Do not rely on
# exports made in the login-node shell being forwarded by ray.sub or attach.sh.
export HF_HOME=/shared/.cache/huggingface
export GYM_RECIPE_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Lightning/RL/grpo-dapo-nemo-gym/dapo_nemotron_3_5_lightning_nemo_gym.yaml
export GYM_DATA_PREP_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py
export GYM_DATA_CONTAINER=/shared/data/dapo_nano_3_5_nemo_gym

/opt/nemo_rl_venv/bin/python "${GYM_DATA_PREP_CONTAINER}" \
  --dataset BytedTsinghua-SIA/DAPO-Math-17k \
  --cache-dir "${HF_HOME}" \
  --output "${GYM_DATA_CONTAINER}/train.jsonl"

/opt/nemo_rl_venv/bin/python "${GYM_DATA_PREP_CONTAINER}" \
  --dataset BytedTsinghua-SIA/AIME-2024 \
  --cache-dir "${HF_HOME}" \
  --output "${GYM_DATA_CONTAINER}/validation.jsonl"

/opt/nemo_rl_venv/bin/python "${GYM_DATA_PREP_CONTAINER}" \
  --dataset BytedTsinghua-SIA/DAPO-Math-17k \
  --cache-dir "${HF_HOME}" \
  --limit 4 \
  --output "${GYM_DATA_CONTAINER}/smoke_train.jsonl"
```

Validate the compact smoke-test file:

```bash
/opt/nemo_rl_venv/bin/python - <<'PY'
import json
with open('/shared/data/dapo_nano_3_5_nemo_gym/smoke_train.jsonl') as stream:
    row = json.loads(stream.readline())
assert row['agent_ref']['name'] == 'math_with_judge_simple_agent'
assert {'responses_create_params', 'question', 'expected_answer'} <= row.keys()
print('Gym JSONL schema is valid.')
PY
```

### Run a one-step smoke test

Still inside the attached allocation/container, run:

```bash
cd /opt/nemo-rl

/opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${GYM_RECIPE_CONTAINER}" \
  grpo.num_prompts_per_step=1 \
  grpo.num_generations_per_prompt=8 \
  grpo.max_num_steps=1 \
  grpo.val_period=-1 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  policy.train_global_batch_size=8 \
  policy.max_total_sequence_length=1024 \
  policy.generation.max_new_tokens=64 \
  policy.generation.vllm_cfg.max_model_len=1024 \
  data.train.data_path="${GYM_DATA_CONTAINER}/smoke_train.jsonl" \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=false \
  logger.monitor_gpus=false \
  checkpointing.enabled=false
```

The smoke test validates wiring only; it is not convergence evidence.

### Run the full training recipe

After the smoke test succeeds, launch the configured 160-step training run
from the same attached allocation/container. Define the output paths in this
shell, then load W&B credentials without printing them:

```bash
cd /opt/nemo-rl

export GYM_RECIPE_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Lightning/RL/grpo-dapo-nemo-gym/dapo_nemotron_3_5_lightning_nemo_gym.yaml
export GYM_RUN_DIR=/shared/results/dapo_nemotron_3_5_lightning_nemo_gym/interactive
export GYM_LOG_DIR=/shared/logs/dapo_nemotron_3_5_lightning_nemo_gym/interactive
mkdir -p "${GYM_RUN_DIR}" "${GYM_LOG_DIR}"

if [ -f /shared/.env ]; then set -a && source /shared/.env && set +a; fi

/opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${GYM_RECIPE_CONTAINER}" \
  checkpointing.checkpoint_dir="${GYM_RUN_DIR}" \
  logger.log_dir="${GYM_LOG_DIR}"
```

When the full run completes or you stop it, release the allocation from the
login/head node with `scancel <jobid>`.

## NeMo RL Batch Job: Train

After the interactive test succeeds, submit the full recipe as a NeMo RL batch
job. `ray.sub` runs `COMMAND` on the Ray head inside the eight-GPU allocation,
so no separate launcher GPU allocation is required.

Run from the login/head node:

```bash
export PARTITION=<TRAINING_PARTITION>
export RUN_NAME=dapo-lightning-nemo-gym
export GYM_RECIPE_CONTAINER=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Lightning/RL/grpo-dapo-nemo-gym/dapo_nemotron_3_5_lightning_nemo_gym.yaml
export RUN_DIR="/shared/results/dapo_nemotron_3_5_lightning_nemo_gym/${RUN_NAME}"
export LOG_DIR="/shared/logs/dapo_nemotron_3_5_lightning_nemo_gym/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

export COMMAND="cd /opt/nemo-rl && \\
if [ -f /shared/.env ]; then set -a && source /shared/.env && set +a; fi && \\
HF_HOME=/shared/.cache/huggingface \\
/opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \\
  --config ${GYM_RECIPE_CONTAINER} \\
  checkpointing.checkpoint_dir=${RUN_DIR} \\
  logger.log_dir=${LOG_DIR}"

cd "${NEMO_RL}"
sbatch \
  --nodes=1 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${RUN_NAME}" \
  --time=04:00:00 \
  --gres=gpu:8 \
  --exclusive \
  --mem=0 \
  ray.sub
```

The batch recipe runs 160 steps, saves checkpoints every 10 steps, validates
every 20 steps, and logs to W&B when `WANDB_API_KEY` is present in
`/shared/.env`. Monitor it from the login/head node:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f slurm-<jobid>.out
```

## Operational Notes

- Keep `async_engine` and `expose_http_server` enabled. NeMo Gym proxies the
  policy through vLLM's HTTP API.
- Keep the vLLM and Gym port ranges separate. The recipe uses 3000-4999 for
  NeMo RL/vLLM and 5000-5999 for Gym.
- NeMo Gym consumes validation JSONL exactly as provided; size or repeat it
  during data preparation rather than with `grpo.max_val_samples`.
- Keep authentication tokens in the environment or `/shared/.env`; never put
  their values into the recipe or guide.
