# Nemotron 3.5 Super VL Star-Count RL Cookbook

This directory documents multimodal RL post-training for Nemotron 3.5 Super
VL with NeMo RL's Megatron backend, colocated vLLM generation, and NeMo Gym.
The workflow applies full-weight GRPO to synthetic star-count images and
evaluates exact-match accuracy on a deterministic held-out split.

The [star-count training guide](grpo-star-count-nemo-gym/grpo_training_cookbook_nemo_gym.md)
covers dataset generation, validation, interactive and batch launch paths,
monitoring, and interpretation of the reference result. Its recipe uses a
16 x 8 rollout batch on variable 800–1,200-pixel canvases containing 1–30
colored stars.

> **Container requirement:** This recipe requires either a NeMo RL container
> built from the `super-v3.5-posttraining` source branch or a prebuilt
> [NGC NeMo RL image](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo-rl/-/tags)
> newer than v0.7. A suitable post-v0.7 prebuilt image was not yet available
> when this guide was published, so the source build below is the supported
> path for reproducing the recipe.

## Runtime and hardware requirements

Use the NeMo RL `super-v3.5-posttraining` branch. It contains the Super VL
Megatron model path, compatible vLLM integration, and NeMo Gym support used by
this cookbook.

The reference configuration uses 16 GB200 GPUs across four 4-GPU nodes. It
trains with Megatron tensor parallelism 4 and expert parallelism 16, while
colocating one tensor-parallel vLLM group on each node. Keep the following
settings aligned when adapting the topology:

```text
cluster.num_nodes=4
cluster.gpus_per_node=4
policy.megatron_cfg.tensor_model_parallel_size=4
policy.megatron_cfg.expert_model_parallel_size=16
policy.generation.vllm_cfg.tensor_parallel_size=4
```

The recipe performs full-weight BF16 updates. Approximate shared-storage usage
from the validated reference artifacts is:

| Artifact | Approximate size |
| --- | ---: |
| Hugging Face checkpoint | 235 GiB |
| Converted Megatron checkpoint cache | 232 GiB |
| ARM64 squashfs container | 73 GiB |
| Weights-only training checkpoint | 227 GiB each |
| Training checkpoint with optimizer state | 1.4 TB each |

Checkpointing is disabled in the provided smoke-test recipe. Allow at least
650 GiB for the model, one conversion cache, the container, logs, and working
headroom. If checkpointing is enabled for resumable training, add about 1.4 TB
for every retained optimizer checkpoint and configure pruning accordingly.
Exact usage varies with the model revision, save format, and filesystem.

## Shared-storage layout

Use storage visible to every allocated node and mount its root at `/shared`
inside the container. The recipe uses this layout:

```text
/shared
|____code
|    |____RL                    <- NeMo RL, branch super-v3.5-posttraining
|    |____Nemotron              <- this cookbook repository
|____models
|    |____NVIDIA-Nemotron-3.5-Super-VL-09212026
|____runs
|____.cache
     |____huggingface
```

Define the corresponding host paths before running the remaining commands:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Super-VL-09212026"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
```

## Clone NeMo RL and initialize submodules

Clone the Super VL post-training branch and initialize its submodules:

```bash
mkdir -p "${SHARED_ROOT}/code"
git clone --branch super-v3.5-posttraining --recursive \
  https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL}"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "${NEMOTRON_REPO}"

cd "${NEMO_RL}"
git submodule update --init --recursive
```

If a checkout already exists, switch it to `super-v3.5-posttraining`, update
it, and rerun the submodule command before building the image.

## Container and worker environments

Until a suitable post-v0.7 prebuilt image is available, build from the
checked-out branch so the image and mounted source use the same NeMo RL
revision. The following command creates an ARM64 release image for GB200
systems and prebuilds the NeMo Gym environments used by Super VL:

```bash
cd "${NEMO_RL}"
export NEMO_RL_REV=$(git rev-parse --short=12 HEAD)
export IMAGE="<YOUR_REGISTRY>/nemo-rl:super-v3.5-posttraining-${NEMO_RL_REV}-arm64"

docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  --push \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --target release \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="examples/nemo_gym/prefetch_super35_all_envs.yaml" \
  -t "${IMAGE}" \
  .
```

`--build-context nemo-rl=.` is required to build the current branch checkout.
Without it, the Dockerfile fetches its default remote ref. The recipe uses
vLLM, so the build skips SGLang and TensorRT-LLM.

Clusters using enroot or Pyxis can convert the registry image to a local
squashfs image:

```bash
export CONTAINER="${SHARED_ROOT}/nemo-rl-super-v3.5-posttraining-${NEMO_RL_REV}-arm64.sqsh"
enroot import -o "${CONTAINER}" "docker://${IMAGE}"
```

Use the registry URI directly when the cluster runtime supports it. Mount the
shared root at its host path for `ray.sub` and at `/shared` for portable recipe
paths:

```bash
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${SHARED_ROOT}:/shared"
```

## Obtain the checkpoint

Download a compatible Hugging Face format Nemotron 3.5 Super VL checkpoint to
the shared model directory. The example below uses the public checkpoint name
expected by the cookbook:

```bash
mkdir -p "${MODEL_DIR}" "${HF_HOME}"
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-09212026 \
  --local-dir "${MODEL_DIR}"
```

The model uses repository-provided remote code. Keep `MODEL_DIR`, the shared
Hugging Face module cache, and the mounted NeMo RL source available to every
Ray worker.

## Run the recipe

Continue with the
[star-count NeMo Gym guide](grpo-star-count-nemo-gym/grpo_training_cookbook_nemo_gym.md)
to generate the deterministic dataset and launch the four-node full-weight
training job.

## Operational notes

- The recipe evaluates before RL and every two steps through step 10.
- The training and validation files are generated from disjoint seed ranges.
- The first launch can spend several minutes converting the Hugging Face
  checkpoint into the cached Megatron representation.
- Full-weight checkpoints are large. Checkpointing is disabled in the short
  example; enable it only after provisioning an appropriate shared directory.
- Store W&B credentials in the submission environment or a protected file.
  Do not place credentials in the recipe or repository.

## Troubleshooting

| Symptom | Check and action |
| --- | --- |
| Megatron workers cannot import `transformers_modules` | Launch from the mounted NeMo RL checkout and put the shared `HF_MODULES_CACHE` on `PYTHONPATH`. |
| A Gym service environment is missing | Rebuild with `prefetch_super35_all_envs.yaml`, or allow the first job to create the environment on shared storage. |
| Model conversion repeats on every launch | Set `NRL_MEGATRON_CHECKPOINT_DIR` to a persistent shared directory. |
| vLLM runs out of memory during refit | Keep the reference TP=4 colocated layout and the recipe's memory and sequence limits. |
| Validation does not cover the complete file | Leave `grpo.max_val_samples: null`; NeMo Gym derives the validation size from the JSONL file. |
| Worker environments are stale after changing the image or branch | Remove the affected cached environment or set `NRL_FORCE_REBUILD_VENVS=true` for one launch. |
