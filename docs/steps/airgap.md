<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(steps-airgap)=
# Run Nemotron Steps in an Airgap Environment

This page describes how to run Nemotron steps in an *airgap environment*: a cluster or account boundary where training jobs cannot rely on the public internet for downloads, package indexes, or external APIs unless you explicitly provide replacements.

A build process under `deploy/nemotron-customizer/airgap/` runs on a *networked host* and produces container images and a manifest for the steps in `src/nemotron/steps/`.
You transfer those artifacts into the airgap environment, then submit jobs as usual with `nemotron steps`.
A short checklist at the end of this page covers the storage, secrets, and providers you must manage for the environment.

For definitions of *step*, *configuration*, *environment profile*, and *artifact*, see [Nemotron Steps Basics](basics.md).

## Before You Start

Confirm two things before you start the build process.

- Docker is installed and can pull `nvcr.io` and Docker Hub images on the networked host.
- The networked host has disk space under `deploy/nemotron-customizer/airgap/out/` for the image TAR files you plan to produce.
  Each image TAR file is several gigabytes.
- You [explored the Nemotron CLI](./getting-started.md).

## Procedure: Build and Run the Airgap Bundle

Follow these phases in order and on the specified host.

### Networked Host: Choose the Steps to Cover

You decide which steps the build process covers by editing the `workflow.stages` field in `airgap.yaml`.
The default value targets a small supervised fine-tuning workflow.

1. Edit `deploy/nemotron-customizer/airgap/airgap.yaml`.
2. In `workflow.stages`, add the `<step_id:config>` steps to run, such as `translate/nemo_curator:default`, and remove the rest.

   Prerequisites steps are automatically added from the `dependencies` map.
   For example, an `sft/megatron_bridge:tiny` step also produces an image that covers `data_prep/sft_packing`.

If you would rather not edit the file, leave `workflow.stages` as is and specify `--target <step_id>:<config>` one or more times in the next phase to override the file.

### Networked Host: Plan the Bundle

1. From the repository root, run a plan-only pass.

   ```console
   $ uv run python deploy/nemotron-customizer/airgap/runner.py \
       --config deploy/nemotron-customizer/airgap/airgap.yaml
   ```

   Review the `[airgap]` section at the end of the output.
   Confirm the images and steps match the steps you specified.
   The following example is for the `translate/nemo_curator` and `sft/megatron_bridge` steps.
   The `data_prep/sft_packing` step is a dependency.

   ```text
   ...
   [airgap] wrote /home/user/nemotron/deploy/nemotron-customizer/airgap/out/airgap-manifest.yaml
   [airgap] selected execution images:
     - translate/nemo_curator: nemotron-customizer-nemo-curator-airgap-4dd4fef4:latest
     - data_prep/sft_packing: nemotron-customizer-nemo-megatron-airgap-55d08ccf:latest
     - sft/megatron_bridge: nemotron-customizer-nemo-megatron-airgap-2fea0796:latest
   ```

### Networked Host: Build and Export Images

Some steps clone or expect a repository checkout at runtime by default.
The `sft/megatron_bridge`, `peft/megatron_bridge`, `byob/mcq`, and `translate/nemo_curator` configs use `auto_mount` entries that the build process embeds in the relevant execution image so airgapped jobs do not clone from GitHub.
The `optimize/modelopt/prune` and `optimize/modelopt/distill` steps expect ModelOpt examples under `/opt/Model-Optimizer` and will try to clone `https://github.com/NVIDIA/Model-Optimizer.git` if the checkout is absent; in an airgap, use an execution image or mount that already contains that checkout.

1. Confirm the target CPU architecture matches what you build on the networked host, or set `platform` on the relevant entry in `airgap.yaml` when they differ.

   ```yaml
   launcher_image:
     platform: linux/amd64

   execution_images:
     nemo-megatron:
       platform: linux/amd64
   ```

2. Run the execute pass.

   ```console
   $ uv run python deploy/nemotron-customizer/airgap/runner.py \
       --config deploy/nemotron-customizer/airgap/airgap.yaml \
       --execute
   ```

3. If a run stops halfway, rerun the command.

   If you change the workflow or image plan, delete `airgap-build-state.yaml` before rerunning.
   Otherwise, the build process reuses the stale state.

4. Collect `airgap-manifest.yaml`, launcher and execution image TAR files, and any generated requirements or repository-overlay metadata from the `deploy/nemotron-customizer/airgap/out/` directory.

### Transfer and Load Inside the Airgap Environment

1. Transfer the image TAR files and the manifest into the airgap environment by using a tool, such as `scp` or `rsync`.
2. Load images into the cluster container runtime or private registry your executor uses.
3. Stage large artifacts on executor-visible persistent storage.
   They were not embedded in the images at build time.
   These artifacts include model weights, datasets, checkpoints, tokenizer files, and customer data.
4. Reference those paths through configuration overrides and `run.env.mounts` in your environment profile.

### Inside the Airgap Environment: Submit Jobs

1. Open `airgap-manifest.yaml` and read the image tag or digest under `step_execution_images` for each step you run.
2. For the steps that embed a repository at build time, override `run.env.mounts` to an empty list at submit time.

   ```yaml
   run:
     env:
       mounts: []
   ```

   The shipped overlay configs under `deploy/nemotron-customizer/airgap/configs/` do this for `sft/megatron_bridge`.
3. Submit with `nemotron steps` and specify `run.env.container_image` to the manifest value.

   ```console
   $ nemotron steps run <step_id> \
       -c <config-or-airgap-overlay> \
       -b <airgap-profile> \
       run.env.container_image=<image-from-manifest>
   ```

4. If you operate an internal Hugging Face mirror or need the Hugging Face Hub reachable from inside the airgap environment, override the Dockerfile defaults using your profile `env_vars`.

:::{note}
The execution Dockerfiles set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `WANDB_MODE=offline` so airgap runs do not reach out to public services by default.
Those are upstream environment variable names from Hugging Face and Weights & Biases.
:::

## Cluster Checklist

Use this short list after you load the bundle and before you submit production jobs.
It covers what stays your responsibility at the cluster, regardless of how the images were built.

- Mount read-only storage for corpora, benchmarks, and pretrained checkpoints, and read-write storage for outputs.
- Replace any Hugging Face Hub identifier in your configuration with a filesystem path on a shared mount when the Hugging Face Hub is not reachable from the airgap environment.
- Update model provider configurations, such as `sdg/data_designer`, to an endpoint reachable from inside the airgap environment, such as a vLLM instance on the local network.
- Provide secrets through environment variables your executor injects, not through files committed to the repository.
- Run `nemotron steps show <step_id> --json` and confirm every `consumes` entry maps to a local path or a mounted artifact before you submit.

## Related Material

- `deploy/nemotron-customizer/airgap/README.md` for full airgap mechanics, pip cache volumes, and architecture notes.
- [Nemotron Steps Basics](basics.md) for environment profiles and mounts.
- Domain references for steps you run, for example [Model Training](../train-models/index.md), [Synthetic Data Generation](../sdg/index.md), and [Model Evaluation](../model-eval/index.md).
