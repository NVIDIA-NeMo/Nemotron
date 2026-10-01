# GRPO with Nemotron 3.5 Super VL and NeMo Gym Star Count

This guide runs full-weight multimodal GRPO on a deterministic colored-star
counting task. Each square canvas is sampled from 800 x 800 through
1,200 x 1,200 pixels and contains 1–30 non-overlapping stars. The policy must
count stars of a requested color and return the answer in `\boxed{}` format.

Use [`super_vl_3_5_star_count_megatron.yaml`](super_vl_3_5_star_count_megatron.yaml)
after completing the repository, container, checkpoint, and shared-storage
setup in [`../README.md`](../README.md).

## Training goal and data

The generator creates 1,024 training examples from seeds 0–1,023 and 256
held-out examples from seeds 1,000,000–1,000,255. It uniformly samples the
integer canvas size, total star count, radius, and color assignment within the
configured bounds. Every selected color occurs at least once, so the requested
count is positive.

The pinned NeMo Gym `circle_count_simple_agent` verifier only examines the
color entries in its `circles` payload field. The star generator retains that
wire-format field while rendering and prompting exclusively with stars. This
uses the existing verifier without changing its reward semantics.

## Configuration overview

| Component | Setting |
| --- | --- |
| Backend | Megatron, full-weight BF16 |
| Hardware reference | 4 nodes x 4 GB200 GPUs |
| Training parallelism | TP=4, EP=16 |
| Generation | colocated vLLM, TP=4 |
| Rollout batch | 16 prompts x 8 generations = 128 responses |
| Policy global batch | 128 |
| Validation | 256 held-out examples, greedy decoding |
| Evaluation cadence | before RL, then every 2 steps through step 10 |
| Logging | online W&B with GPU monitoring |

The recipe preserves the Super VL settings inherited from the
`super-v3.5-posttraining` branch: trainable vision components, disabled MTP,
activation checkpointing, precision-aware distributed Adam, raw vLLM
log-probabilities, float32 Mamba state, and encoder-cache reset after each
weight refit. The recipe temporarily offloads distributed optimizer state
during each colocated vLLM refit, leaving GPU headroom for full tensor-parallel
weight gathers after a large 16 x 8 optimizer step.

## Generate and validate the dataset

Run these commands inside the NeMo RL image or an attached allocation:

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export DATA_DIR=/shared/runs/super35-star-count/data
export GENERATOR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/prepare_star_count_data.py"
mkdir -p "${DATA_DIR}"

python "${GENERATOR}" \
  --out "${DATA_DIR}/train.jsonl" \
  --num-samples 1024 \
  --seed-offset 0 \
  --canvas-size-min 800 \
  --canvas-size-max 1200 \
  --radius-min 24 \
  --radius-max 48 \
  --num-stars-min 1 \
  --num-stars-max 30

python "${GENERATOR}" \
  --out "${DATA_DIR}/validation.jsonl" \
  --num-samples 256 \
  --seed-offset 1000000 \
  --canvas-size-min 800 \
  --canvas-size-max 1200 \
  --radius-min 24 \
  --radius-max 48 \
  --num-stars-min 1 \
  --num-stars-max 30
```

Validate row counts, canvas and star bounds, routing, and split separation:

```bash
python - <<'PY'
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

root = Path('/shared/runs/super35-star-count/data')
fingerprints = {}
for split, expected in [('train', 1024), ('validation', 256)]:
    rows = [json.loads(line) for line in (root / f'{split}.jsonl').read_text().splitlines()]
    assert len(rows) == expected
    assert all(row['agent_ref']['name'] == 'circle_count_simple_agent' for row in rows)
    assert all(1 <= len(row['circles']) <= 30 for row in rows)
    for row in rows:
        image_url = row['responses_create_params']['input'][1]['content'][0]['image_url']
        image = Image.open(io.BytesIO(base64.b64decode(image_url.split(',', 1)[1])))
        assert image.width == image.height
        assert 800 <= image.width <= 1200
        prompt = row['responses_create_params']['input'][1]['content'][1]['text']
        assert 'stars' in prompt and 'circles' not in prompt
    fingerprints[split] = {
        hashlib.sha256(
            json.dumps(row['responses_create_params'], sort_keys=True).encode()
        ).hexdigest()
        for row in rows
    }

assert fingerprints['train'].isdisjoint(fingerprints['validation'])
print('Star-count train and validation splits are valid and disjoint.')
PY
```

## Launch the four-node run

Start a persistent allocation from the login node. Replace the account,
partition, and container placeholders with values for the target cluster:

```bash
export NUM_NODES=4
export GPUS_PER_NODE=4
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<NEMO_RL_CONTAINER_OR_SQUASHFS>
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${SHARED_ROOT}:/shared"
unset COMMAND

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-star-count-16x8 \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

Attach using the helper emitted by `ray.sub`:

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

Inside the attached container, configure persistent caches and launch the
recipe. Provide `WANDB_API_KEY` through the job environment or a protected
environment file.

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-star-count
export RECIPE="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/super_vl_3_5_star_count_megatron.yaml"

mkdir -p "${RUN_DIR}"/{logs,hf_modules,hf_config_locks,megatron_ckpt_cache,vllm_compile_cache}
export HF_MODULES_CACHE="${RUN_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${RUN_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${RUN_DIR}/megatron_ckpt_cache"
export VLLM_CACHE_ROOT="${RUN_DIR}/vllm_compile_cache"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${RECIPE}" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
```

## Monitor training and convergence

The run logs exact-match `val:accuracy` before training and at steps 2, 4, 6,
8, and 10. W&B also receives reward, response length, loss, gradient
norm, throughput, timing, and GPU telemetry. Use the accuracy series from one
run to assess its learning curve; the sampled GRPO rollouts make individual
steps noisy.

From the login node, monitor the scheduler and driver log:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f "${NEMO_RL}/<jobid>-logs/ray-driver.log"
```

A successful run reaches step 10, performs final validation after the last
weight refit, syncs W&B, and exits with status zero. Compare step 10 against
step 0 and inspect all intermediate validation points before drawing a
convergence conclusion.

## Reference result

A full-weight 4-node reference run completed all ten updates and all six
evaluations. Exact-match accuracy on the fixed 256-example validation split
improved from 11.33% before RL to 70.31% after the final update.

| Step | Validation accuracy |
| ---: | ---: |
| 0 | 11.33% |
| 2 | 13.67% |
| 4 | 13.67% |
| 6 | 55.47% |
| 8 | 71.09% |
| 10 | 70.31% |

The step-8 to step-10 change shows the expected noise from sampled GRPO
updates, while the complete curve shows clear learning over the baseline. The
run finished successfully in 1 hour 40 minutes. Its W&B dashboard contains the
accuracy curve, training rewards, optimization metrics, throughput, timings,
and GPU telemetry:

<https://wandb.ai/hwinf_dcm/nemotron-super-vl-35-star-count/runs/w2kbfgfr>
