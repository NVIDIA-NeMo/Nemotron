# GRPO with Nemotron 3.5 Super VL and NeMo Gym Circle Count

This guide runs full-weight GRPO through NeMo RL's Megatron backend. NeMo Gym
generates synthetic images, routes each response through the
`circle_count_simple_agent`, and returns an exact-match reward. The recipe
evaluates the untouched checkpoint at step 0 and again at steps 5, 10, and 15.

## Tested configuration

| Component | Setting |
| --- | --- |
| NeMo RL | `super-v3.5-posttraining`, revision `eb420d15034c` |
| Model | `NVIDIA-Nemotron-3.5-Super-EA-09112026` |
| Hardware | 4 nodes × 4 GB200 GPUs |
| Megatron | TP=4, EP=16, full-weight BF16 |
| Generation | colocated vLLM, TP=4 |
| Training | 8 prompts × 2 generations, global batch 16 |
| Evaluation | 256 held-out examples, greedy decoding |

The run intentionally disables checkpoints because a full-weight artifact is
hundreds of GB. Set `checkpointing.enabled=true` and choose a checkpoint path
when a resumable training artifact is required.

## Shared-storage layout

Mount the shared root as `/shared` in the container:

```text
/shared
|____code
|    |____RL             <- NeMo RL super-v3.5-posttraining checkout
|    |____Nemotron-2     <- this repository
|____models
|    |____NVIDIA-Nemotron-3.5-Super-EA-09112026
|____runs
|____.env                <- contains WANDB_API_KEY
```

Set the host paths:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron-2"
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Super-EA-09112026"
export CONTAINER=<SITE_ACCESSIBLE_NEMO_RL_CONTAINER>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export GPUS_PER_NODE=4
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"
```

## Generate disjoint train and validation data

The NeMo RL branch includes a deterministic converter around NeMo Gym's
circle-count generator. Images are embedded as data URLs, so the JSONL files
are portable across the four nodes.

Run in the NeMo RL container or an attached NeMo RL allocation:

```bash
export DATA_DIR=/shared/runs/super35_circle_count_megatron/data
mkdir -p "${DATA_DIR}"

python /opt/nemo-rl/examples/nemo_gym/nemotron-3-super-omni/prepare_circle_count_mopd_data.py \
  --out "${DATA_DIR}/train.jsonl" \
  --num-samples 1024 \
  --seed-offset 0

python /opt/nemo-rl/examples/nemo_gym/nemotron-3-super-omni/prepare_circle_count_mopd_data.py \
  --out "${DATA_DIR}/validation.jsonl" \
  --num-samples 256 \
  --seed-offset 100000
```

Validate the row counts and ensure the splits share no generated requests:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/shared/runs/super35_circle_count_megatron/data')
rows = {}
for split, expected in [('train', 1024), ('validation', 256)]:
    values = [json.loads(line) for line in (root / f'{split}.jsonl').read_text().splitlines()]
    assert len(values) == expected
    assert all(row['agent_ref']['name'] == 'circle_count_simple_agent' for row in values)
    rows[split] = {
        hashlib.sha256(json.dumps(row['responses_create_params'], sort_keys=True).encode()).hexdigest()
        for row in values
    }
assert rows['train'].isdisjoint(rows['validation'])
print('Circle-count train and validation splits are valid and disjoint.')
PY
```

## Run the evaluated 15-step recipe

The command loads `WANDB_API_KEY` without printing it. Copy or symlink the
credential file to `${SHARED_ROOT}/.env`, or change the source path in the run
script to a protected file available on every node.

```bash
export RECIPE=/shared/code/Nemotron-2/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-circle-count-nemo-gym/super_vl_3_5_circle_count_megatron.yaml
export RUN_SCRIPT="${SHARED_ROOT}/runs/super35_circle_count_megatron/run.sh"
mkdir -p "$(dirname "${RUN_SCRIPT}")"

cat > "${RUN_SCRIPT}" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
if [[ -f /shared/.env ]]; then
  set -a
  source /shared/.env
  set +a
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be set for online experiment logging}"
cd /opt/nemo-rl
exec /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /shared/code/Nemotron-2/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-circle-count-nemo-gym/super_vl_3_5_circle_count_megatron.yaml
RUN
chmod 700 "${RUN_SCRIPT}"

cd "${NEMO_RL}"
NUM_ACTOR_NODES=4 \
GPUS_PER_NODE=4 \
CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
COMMAND="${RUN_SCRIPT}" \
sbatch \
  --nodes=4 \
  --account="${SLURM_ACCOUNT}" \
  --job-name=super35-circle-grpo \
  --partition="${PARTITION}" \
  --time=02:00:00 \
  --gres=gpu:4 \
  ray.sub
```

Monitor the submitted job from the NeMo RL checkout:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f <jobid>-logs/ray-driver.log
```

The driver log and W&B run should contain four `val:accuracy` measurements,
at global steps 0, 5, 10, and 15. A successful run exits with status 0 after
step 15.

## Interpreting the result

Accuracy is exact-match reward averaged over the fixed 256-example validation
file. Compare step 15 with step 0 from the same W&B run. The small synthetic
task is a pipeline and learning check; it is not a general visual-reasoning
benchmark.

