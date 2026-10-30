# One-Node DAPO/GRPO Recipe for Nemotron-3.5-Lightning

This directory contains one supported recipe:
[`dapo_nemotron_3_5_lightning.yaml`](dapo_nemotron_3_5_lightning.yaml).
It is a standalone AutoModel/FSDP2 DAPO/GRPO recipe validated with NeMo RL
nightly on one 8-GPU node. It uses EP=8 for the policy, TP=4 for colocated
vLLM, 32 prompts x 8 generations per step, 2048 generated tokens, and a
160-step convergence target.

The recipe derives from the public NeMo RL Nano FSDP2 GRPO configuration and
uses DAPOMath17K for training plus DAPOMathAIME2024 for validation. Complete
the container, model, and shared-storage setup in [`../README.md`](../README.md)
first.

Note on prebuilt NGC container: Nemotron-3.5-Lightning requires a NemoRL container later than [v0.7](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo-rl/-/tags) which is not yet available at the time of the model release data. Follow the docker build section in [../README.md](../README.md) to build a container from source.

## Dataset and training goal

The policy learns from the public `BytedTsinghua-SIA/DAPO-Math-17k` math
reasoning dataset. At each step, it samples eight candidate solutions for each
of 32 problems, scores them with the recipe's verifiable math reward, and
updates the policy with DAPO/GRPO. Validation uses the held-out
`BytedTsinghua-SIA/AIME-2024` problems. The goal is to improve reliably
verifiable mathematical reasoning while keeping rollout, policy, and
distributed-training behavior stable on one DGX H100 node.

> **Historical dataset caveat:** the publisher's public artifacts repeat
> DAPO-Math-17k samples 100 times and AIME-2024 samples 32 times. This
> cookbook does not filter either dataset. See
> [`../README.md`](../README.md#historical-dataset-caveat) for the potential
> training and validation impact.

## Required layout

Mount your shared-storage root at `/shared` in the NeMo RL container. The
direct recipe uses this layout:

```text
/shared                         <- Mount point for </YOUR/SHARED/STORAGE>
|____code
|    |____RL                    <- NeMo RL root repository
|    |____Nemotron              <- Public repository containing this cookbook
|____models
|    |____NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
|____runs
|____.cache/huggingface
```

Set the shared paths in your login/head-node shell before following the rest
of this guide.

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
```

The recipe expects the model at:

```text
/shared/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

Override `policy.model_name` and `policy.tokenizer.name` if your mounted model
uses a different location. The recipe defaults to this run directory:

```text
/shared/runs/nemotron-3.5-lightning-dapo-convergence
```

Before submitting an allocation, create the direct-workflow cache and run
directory from the login/head node:

```bash
export HF_HOME="${HF_HOME:-${SHARED_ROOT}/.cache/huggingface}"
mkdir -p \
  "${HF_HOME}" \
  "${SHARED_ROOT}/runs/nemotron-3.5-lightning-dapo-convergence"
```

On root-squashed storage, grant write access only to these run-specific
directories according to your site's policy.

For W&B logging, put `WANDB_API_KEY` in `/shared/.env` and export it in the
driver shell without printing it:

```bash
set -a
source ${SHARED_ROOT}/.env
set +a
```

To resume the same W&B run after a Slurm allocation ends, also set a stable run
ID before launching:

```bash
export WANDB_RUN_ID=<existing-run-id>
export WANDB_RESUME=allow
```

## Interactive training path

Use an attached one-node allocation to run the actual DAPO/GRPO training
recipe manually. Set `CONTAINER` to the site-accessible image built from the
public NeMo RL Dockerfile in the parent README.

Run from the login/head node:

```bash
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<INTERACTIVE_PARTITION>
export CONTAINER=<SITE_ACCESSIBLE_NEMO_RL_IMAGE>
export GPUS_PER_NODE=8
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"
unset COMMAND

cd "${NEMO_RL}"
sbatch \
  --nodes=1 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=interactive-dapo-lightning \
  --time=04:00:00 \
  --gres=gpu:8 \
  --exclusive \
  --mem=0 \
  ray.sub
```

Wait for NemoRL to write the attach helper, then attach to the allocation from the
login/head node:

```bash
bash ./<jobid>-attach.sh
```

### Smoke test

Run this one-step check from the attached allocation/container to confirm the
model, Ray, rollout, and reward path before committing the full allocation.
It disables checkpointing and external logging, so it is not convergence
evidence:

```bash
cd /opt/nemo-rl

/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config /shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Lightning/RL/grpo-dapo/dapo_nemotron_3_5_lightning.yaml \
  grpo.max_num_steps=1 grpo.num_prompts_per_step=1 grpo.num_generations_per_prompt=8 \
  grpo.val_at_start=false grpo.val_at_end=false grpo.val_period=-1 \
  policy.train_global_batch_size=8 policy.max_total_sequence_length=1024 \
  policy.generation.max_new_tokens=256 data.max_input_seq_length=768 \
  checkpointing.enabled=false logger.wandb_enabled=false logger.tensorboard_enabled=false
```

### Full run

After the smoke test succeeds, launch the actual training recipe from the same
attached allocation/container. It uses the configured 160 steps,
checkpointing, validation, and W&B logging:

```bash
cd /opt/nemo-rl

if [ -f /shared/.env ]; then set -a && source /shared/.env && set +a; fi

/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config /shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Lightning/RL/grpo-dapo/dapo_nemotron_3_5_lightning.yaml
```

When training completes or you stop it, release the allocation from the
login/head node with `scancel <jobid>`.

## Batch launch

To submit training without an interactive attachment, run the same recipe
directly through the Ray head. The `COMMAND` payload runs inside the one-node
Slurm allocation and container; it avoids consuming a separate launcher GPU
allocation.

Run from the login/head node:

```bash
export PARTITION=<TRAINING_PARTITION>
export RUN_NAME=dapo-lightning
export RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
export GPUS_PER_NODE=8
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"

export COMMAND="cd /opt/nemo-rl && \\
if [ -f /shared/.env ]; then set -a && source /shared/.env && set +a; fi && \\
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \\
  --config /shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Lightning/RL/grpo-dapo/dapo_nemotron_3_5_lightning.yaml \\
  checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints \\
  logger.log_dir=${RUN_DIR}/logs"

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

For a non-default run directory, pass both overrides together so logs and
checkpoints remain aligned:

```text
checkpointing.checkpoint_dir=/shared/runs/<run>/checkpoints
logger.log_dir=/shared/runs/<run>/logs
```

NeMo RL resumes from the latest complete checkpoint in that directory. Keep
the same W&B run ID for every continuation allocation.
