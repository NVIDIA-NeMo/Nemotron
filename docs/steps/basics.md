<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(steps-basics)=
# Nemotron Steps Basics

This page defines the building blocks of the `nemotron steps` CLI.
You do not need to read it to invoke a single command, but every domain section in the documentation assumes that you already understand these definitions.

## What a Step Is

A *step* is a named unit of work with a stable identifier, such as `sft/automodel`, `byob/mcq`, or `optimize/modelopt/quantize`.
Each step packages four things:

- A description of the work the step performs.
- The artifacts the step *consumes*, such as a chat-formatted JSON Lines (JSONL) dataset or a Megatron checkpoint.
- The artifacts the step *produces*, such as a Hugging Face checkpoint or a Parquet benchmark.
- One or more named configurations that supply parameter values.

The step identifier is the only piece of information you need to invoke a job.
Steps live under `src/nemotron/steps/` in the source tree, and the CLI discovers them at startup.
Use `nemotron steps list` to see every available step, and use `nemotron steps show <id>` to inspect a single step.

## Configurations

A *configuration* is a named set of parameter values for a step, such as `tiny` or `default`.
A configuration sets values like the dataset path, the base model identifier, the batch size, the number of training steps, and the parallelism strategy.

Every step that runs at production scale ships at least two configurations:

- `tiny` runs a short job that exercises the plumbing without consuming meaningful compute.
  Use `tiny` to confirm that the step starts, that the data path is wired, and that the cluster accepts the submission.
- `default` runs a production-shape job.
  Use `default` as the starting point for real work, and override individual values from the command line when needed.

You select the configuration with the `-c` option.
You can supply ad hoc overrides as trailing `key=value` dotlist arguments on the same command line.

## Environment Profiles

An *environment profile* is a named description of an execution target.
A profile sets the container image, the node count, the resource shape, the mount points, and any cluster-specific startup commands.

You select the profile in one of three ways:

- Omit the `-r` and `-b` flags to run on the local machine.
- Pass `-r <profile-name>` to run attached, where the CLI streams logs back to your terminal until the job ends.
- Pass `-b <profile-name>` to run detached, where the CLI submits the job and returns immediately.

The CLI reads profiles from a TOML file named `env.toml` at the repository root by default.
Set the `NEMOTRON_ENV_FILE` environment variable to point at a different profile file, such as `env.lepton.toml` or `env.slurm.toml`.
See [Execution Through NeMo Run](../nemo_runspec/nemo-run.md) for the full profile model, attached and detached execution, and cluster setup.

### Generating Profile Files With the env Step

The `env/env_toml` step generates and validates profile files from compact YAML templates.
Treat it as common infrastructure that other steps rely on, not as a workload step.
Use `env/env_toml` when you need to bootstrap an `env.lepton.toml` or `env.slurm.toml` from the bundled templates, add a new profile that inherits from an existing one, or validate that a profile satisfies the resource guardrails for Curator, Data Designer, Ray, or reinforcement learning (RL) steps.

The following procedure generates a starter profile file, reviews it, and points the CLI at it.

1. Choose the bundled configuration that matches your target.
   Use `lepton` for Lepton AI, `slurm` for a Slurm cluster, or `dgxcloud` for NVIDIA DGX Cloud.
   List the available configurations with the following command.

   ```console
   $ nemotron steps show env/env_toml
   ```

2. Run the step from the repository root so the output lands beside `pyproject.toml`.
   The following command generates an `env.lepton.toml` from the `lepton` configuration.

   ```console
   $ nemotron steps run env/env_toml -c lepton
   ```

   Substitute `-c slurm` or `-c dgxcloud` for the other targets.
   The step writes to `env.lepton.toml`, `env.slurm.toml`, or `env.dgxcloud.toml` by default, and it refuses to overwrite an existing file.
   If you intentionally want to regenerate, pass `force=true` on the command line.

3. Open the generated TOML file and replace the site-specific values.
   At minimum, set the node group, the resource shape, the workspace path, and the mount points.
   For Slurm, also set the host, the user account, the partition, and the remote job directory.
   Keep secrets out of the file and reference them through `${oc.env:VAR_NAME}` placeholders.

4. Point the CLI at the generated file by exporting `NEMOTRON_ENV_FILE`.

   ```console
   $ export NEMOTRON_ENV_FILE=env.lepton.toml
   ```

   Add the export to your shell profile if you always work against the same target.
   Without the variable, the CLI looks for a plain `env.toml` at the repository root.

5. Confirm a profile resolves before you launch real work.
   Pick a step that already has a generated profile and run it with the `tiny` configuration.

   ```console
   $ nemotron steps run sft/automodel -c tiny -r lepton_sft_automodel
   ```

   A successful submission proves that the CLI loaded the profile file, matched the named profile, and accepted the resource shape.

After the file passes step (5), you can add new profiles by editing the YAML template under `src/nemotron/steps/env/env_toml/config/` and rerunning `nemotron steps run env/env_toml` with `force=true`.
Child profiles can use `extends` to inherit from an existing profile and override only the image, the mounts, the environment variables, or the resource shape that differ.

## Artifacts Between Steps

Steps chain together through *artifacts*.
An artifact is a typed payload, such as `training_jsonl`, `packed_parquet`, `binidx`, `checkpoint_hf`, `checkpoint_megatron`, or `eval_results`.
One step writes an artifact to a path you control, and a later step reads that same artifact as its input.

The chain holds as long as the consumer accepts the producer's artifact type.
When two steps speak different layouts of the same idea, a `convert/*` step bridges the gap.
For example, `convert/megatron_to_hf` turns a `checkpoint_megatron` artifact into a `checkpoint_hf` artifact for deployment.

You can list the producers and consumers of a given artifact type from the CLI:

```console
$ nemotron steps list --produces training_jsonl
$ nemotron steps list --consumes training_jsonl
```

See [Getting Started With Steps](getting-started.md) for a guided tour of these commands.

## Composing a Run

The following command composes all three concepts into one invocation.

```console
$ nemotron steps run sft/automodel -c tiny -r lepton_sft_automodel
```

The CLI reads the `sft/automodel` step, applies the `tiny` configuration, and submits the job to the cluster the `lepton_sft_automodel` profile describes.
Change the step identifier to choose a different unit of work.
Change the configuration to change parameter values without editing files.
Change the profile to move the same job to a different cluster.

## Where To Go Next

- [Getting Started With Steps](getting-started.md) walks through listing steps, inspecting inputs and outputs, and exploring the step graph from the CLI.
- [Model Training](../train-models/index.md) covers supervised fine-tuning, parameter-efficient fine-tuning, RL alignment, and post-training optimization.
- [Synthetic Data Generation](../sdg/index.md), [Translation](../translation/index.md), [Build MCQ Benchmarks](../build-benchmarks/index.md), and [Model Evaluation](../model-eval/index.md) each cover one family of steps end to end.
- [Execution Through NeMo Run](../nemo_runspec/nemo-run.md) describes profiles, attached and detached runs, and cluster setup.
