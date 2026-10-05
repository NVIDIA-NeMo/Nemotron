# Train Nemotron 3.5 Super VL to Count Stars with GRPO

Counting colored stars is a compact way to exercise the complete multimodal RL
pipeline. The policy must inspect an image, identify the requested color, count
the matching objects, and return an answer that an automatic verifier can
score. This guide runs that task with full-weight GRPO, NeMo RL's Megatron
backend, dedicated vLLM generation, and NeMo Gym.

## How the task works

Each example contains a generated PNG and a question such as, “How many cyan
stars are in the image?” The expected response is a plain non-negative integer
inside `\boxed{}`, for example `\boxed{3}`. The verifier gives reward `1.0` for
an exact match and `0.0` for an incorrect or malformed answer.

The task builds on NeMo Gym's circle-count environment. The generator replaces
circles with five-point stars while retaining the environment's established
data contract and reward mechanism:

| Mechanism | Star-count behavior |
| --- | --- |
| Agent | `circle_count_simple_agent`, with one user turn and no tools |
| Request | A system message followed by a user message containing a base64 PNG and question |
| Object metadata | Each star is stored under the legacy `circles` key for compatibility. `x` and `y` are its center, `radius` is the center-to-tip distance, and `color` is its palette label. |
| Target | `target_color` identifies the color to count |
| Answer | Strict `\boxed{<digits>}` format |
| Reward | Exact comparison between the boxed integer and the number of matching metadata entries |

The `circles` field name is inherited from the circle-count environment; its
records describe the rendered stars. The verifier counts records whose `color`
matches `target_color`. It does not use `x`, `y`, or `radius` when scoring, so
the star task can reuse the existing environment without a custom NeMo Gym
service.

The deterministic generator creates 1,024 training examples with seeds
0–1,023 and 256 held-out examples with seeds 1,000,000–1,000,255. Images range
from 800 x 800 to 1,200 x 1,200 pixels and contain 1–30 non-overlapping stars
drawn from 1–4 colors. A one-star image has one color; all other images have
2–4. Every requested color appears at least once.

<table>
  <tr>
    <td width="50%"><img src="assets/star_count_sample_1.png" alt="Sparse colored-star example" width="360"><br><strong>Question:</strong> How many cyan stars are in the image?<br><strong>Answer:</strong> <code>\boxed{1}</code></td>
    <td width="50%"><img src="assets/star_count_sample_2.png" alt="Colored-star counting example" width="360"><br><strong>Question:</strong> How many red stars are in the image?<br><strong>Answer:</strong> <code>\boxed{5}</code></td>
  </tr>
  <tr>
    <td width="50%"><img src="assets/star_count_sample_3.png" alt="Orange and purple star example" width="360"><br><strong>Question:</strong> How many orange stars are in the image?<br><strong>Answer:</strong> <code>\boxed{10}</code></td>
    <td width="50%"><img src="assets/star_count_sample_4.png" alt="Crowded colored-star example" width="360"><br><strong>Question:</strong> How many red stars are in the image?<br><strong>Answer:</strong> <code>\boxed{8}</code></td>
  </tr>
</table>

## Reference configuration

The included
[`super_vl_3_5_star_count_megatron.yaml`](super_vl_3_5_star_count_megatron.yaml)
uses the following settings:

| Component | Setting |
| --- | --- |
| Compute | 5 nodes x 4 GPUs |
| Training | 4 nodes, full-weight BF16, TP=4, EP=16 |
| Generation | 1 dedicated node, vLLM TP=4 |
| GRPO batch | 16 prompts x 8 responses |
| Schedule | 10 updates; validation before RL and every 2 updates |
| Sequence limit | 4,096 total tokens; 256 generated tokens |
| Checkpoints | Disabled |

The short schedule uses the first 160 rows because data shuffling is disabled.
It is intended as a reproducible pipeline example rather than a full training
run over all 1,024 examples. TP=4 shards dense and attention tensors within a
node. EP=16 distributes the 512 routed experts across all GPUs, with 32 experts
per expert-parallel rank.

## Prerequisites

Use storage visible to every allocated node and mount its root at `/shared`
inside the container. The recipe uses this layout:

```text
</YOUR/SHARED/STORAGE> (Host):/shared (Container)
|____code
|    |____RL                    <- NeMo RL, branch super-v3.5-posttraining
|    |____Nemotron              <- this cookbook repository
|____models
|    |____NVIDIA-Nemotron-3.5-Super-VL-09212026
|____runs
|    |____super35-star-count
|         |____data            <- generated train and validation JSONL files
|         |____logs            <- training and validation logs
|____.cache
     |____huggingface          <- Hugging Face downloads
     |____megatron_ckpt        <- converted Megatron checkpoint
```

Use a NeMo RL container built from the `super-v3.5-posttraining` branch, or a
compatible prebuilt image newer than v0.7. This branch provides the Super VL
Megatron model path, vLLM integration, NeMo Gym support, and non-colocated
collective weight synchronization used by this recipe.

No suitable prebuilt image was available when this guide was published. Build
the image from the same checkout that will be mounted into the job:

```bash
# Run on a Docker-capable ARM64 build node with registry access.
cd </YOUR/SHARED/STORAGE>/code/RL
export IMAGE="<YOUR_REGISTRY>/nemo-rl:super-v3.5-posttraining-arm64"
docker buildx build --platform linux/arm64 --progress=plain --push \
  --build-context nemo-rl=. -f docker/Dockerfile --target release \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="examples/nemo_gym/prefetch_super35_all_envs.yaml" \
  -t "${IMAGE}" .
```

The model uses about 235 GiB, its converted Megatron cache about 232 GiB, and
an ARM64 squashfs image about 73 GiB. Provision at least 650 GiB for those
artifacts, logs, and working headroom. Checkpointing is disabled in this
example. If enabled, allow about 227 GiB per weights-only checkpoint or 1.4 TB
per checkpoint that includes optimizer state.

The container needs the shared filesystem at its host path for `ray.sub` and
at `/shared` for portable recipe paths. Mount the root of the site's shared
filesystem namespace, such as `/lustre`, at the same path inside the container.
Use the registry URI as `CONTAINER` when supported, or convert it to the
cluster's local container format first.

On the login or head node, define the site-specific values once:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NFS_ROOT=/lustre
export NEMO_RL="${SHARED_ROOT}/code/RL"
export CONTAINER=<NEMO_RL_CONTAINER_OR_SQUASHFS>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export GPUS_PER_NODE=4
export MOUNTS="${NFS_ROOT}:${NFS_ROOT},${SHARED_ROOT}:/shared"
```

The example assumes that `SHARED_ROOT` is below `NFS_ROOT`. Adjust `NFS_ROOT`
to the shared filesystem root used by your cluster.

## Interactive path

Use the interactive path when trying the recipe for the first time or watching
the training process directly.

### 1. Reserve five nodes — login or head node

Run from the NeMo RL repository root:

```bash
cd "${NEMO_RL}"
unset COMMAND
sbatch \
  --nodes=5 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-star-count \
  --time=04:00:00 \
  --gres=gpu:4 \
  --mem=0 \
  --exclusive \
  ray.sub
```

`--mem=0` requests all host memory on each node. Four nodes host the full-weight
Megatron policy, while the fifth node hosts the TP=4 vLLM generation engine.
The two worker groups remain resident and exchange updated weights through the
non-colocated collective path.

### 2. Attach — login or head node

After the allocation starts, use the helper created by `ray.sub`:

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

### 3. Generate data and train — attached Ray-head container

Run this block in the attached container. It contains all runtime paths and
cache settings and invokes the checked-in data generator and YAML recipe. No
additional setup script is required.

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-star-count
export DATA_DIR="${RUN_DIR}/data"
export CACHE_DIR=/shared/.cache
export EXAMPLE_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym"

mkdir -p "${DATA_DIR}" "${RUN_DIR}/logs" \
  "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt,vllm}

export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/train.jsonl" --num-samples 1024 --seed-offset 0
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/validation.jsonl" --num-samples 256 --seed-offset 1000000

uv run --no-sync python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${EXAMPLE_DIR}/super_vl_3_5_star_count_megatron.yaml" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
```

Append `logger.wandb_enabled=false` to the final command when external
experiment tracking is not desired.

## Batch path

For an unattended run, submit the same work through `COMMAND`. `ray.sub`
materializes this value inside its job log directory, so no additional launch
script is needed.

Run the following block from the NeMo RL repository root on the login or head
node. It assumes the prerequisite variables above are still defined.

```bash
cd "${NEMO_RL}"

read -r -d '' COMMAND <<'RUN' || true
set -euo pipefail
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-star-count
export DATA_DIR="${RUN_DIR}/data"
export CACHE_DIR=/shared/.cache
export EXAMPLE_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym"

mkdir -p "${DATA_DIR}" "${RUN_DIR}/logs" \
  "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt,vllm}

export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/train.jsonl" --num-samples 1024 --seed-offset 0
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/validation.jsonl" --num-samples 256 --seed-offset 1000000
exec uv run --no-sync python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${EXAMPLE_DIR}/super_vl_3_5_star_count_megatron.yaml" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
RUN
export COMMAND

sbatch \
  --nodes=5 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-star-count \
  --time=04:00:00 \
  --gres=gpu:4 \
  --mem=0 \
  --exclusive \
  ray.sub
```

Monitor a batch job from the login or head node:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f <jobid>-logs/ray-driver.log
```

## Reading the result

The driver reports held-out exact-match accuracy before RL and after steps 2,
4, 6, 8, and 10. A representative completed run produced 32, 36, 38, 65, 157,
and 181 correct answers out of 256 at those checkpoints. Accuracy increased
from 12.50% before RL to 70.70% after step 10.

This metric combines visual counting, response completion, and strict answer
formatting. Mean response length in that run fell from 249.8 to 143.3 tokens.
In a separate diagnostic run, parseable boxed answers rose from 30/256 to
255/256, and the initial policy was already correct on 29/30 parseable answers.
The result therefore demonstrates that the RL pipeline teaches the policy to
produce concise, verifiable responses; it should not be read as a pure measure
of improved visual perception.
