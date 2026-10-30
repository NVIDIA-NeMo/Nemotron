---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the optimize/modelopt/distill step."
topics: ["Training", "Reference", "CLI", "Optimization", "Distillation"]
tags: ["Reference", "CLI", "Steps", "Optimization", "Distillation", "Teacher-Student", "ModelOpt", "Megatron-Bridge"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# optimize/modelopt/distill

This step runs teacher-student distillation by using NVIDIA Model Optimizer through NVIDIA Megatron-Bridge.
The step can run as a standalone training job, or as a quality-recovery pass after pruning or quantization.
Real-data runs consume Megatron `bin/idx` data produced by `data_prep/pretrain_prep`.
The step produces a distilled Megatron distributed checkpoint.

## Syntax

```bash
nemotron steps run optimize/modelopt/distill \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/optimize/modelopt/distill/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Generic teacher-student distillation configuration with `Qwen/Qwen3-8B` as the teacher and `Qwen/Qwen3-4B` as the student. |
| `tiny.yaml` | Short validation run that exercises the distillation pipeline with mock data. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run optimize/modelopt/distill -c tiny
$ nemotron steps run optimize/modelopt/distill -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `checkpoint_hf` | Yes | The teacher and student Hugging Face (HF) checkpoints. |
| Consumes | `binidx` | No | Optional real distillation data from `data_prep/pretrain_prep`. This input is unnecessary when `args.use_mock_data=true`. |
| Produces | `checkpoint_megatron` | — | The distilled Megatron distributed checkpoint. |

## Step Parameters

The manifest declares five distillation parameters.
Pass them as dotlist overrides.

```{option} args.teacher_hf_path=<id-or-path>

The Hugging Face identifier or local path for the teacher checkpoint.

Example: `args.teacher_hf_path=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
```

```{option} args.student_hf_path=<id-or-path>

The Hugging Face identifier or local path for the student checkpoint.

Example: `args.student_hf_path=Qwen/Qwen3-4B`
```

```{option} args.data_paths=<list>

The Megatron data blend, expressed as the upstream command-line sequence in the form `[weight, prefix, weight, prefix, ...]`.

Example: `args.data_paths='[0.5, /lustre/data/wiki, 0.5, /lustre/data/c4]'`
```

```{option} args.use_mock_data=<bool>

When set to `true`, the step runs a validation pass with mock data instead of real Megatron `bin/idx` data.

Default: `false`.

Example: `args.use_mock_data=true`
```

```{option} extra_args=<list>

Literal upstream arguments that the step forwards to the distillation script.
Use this parameter to pass newly added Model Optimizer flags that do not yet have a dedicated `args.*` entry.

Default: `[]`.

Example: `extra_args=["--hf_export_path", "/lustre/distilled/hf"]`
```

Frequently used dotlist overrides drawn from the default configuration include the following.

```{option} args.tp_size=<n>

The tensor-parallel degree applied during distillation.

Example: `args.tp_size=4`
```

```{option} args.train_iters=<n>

The number of training iterations.

Example: `args.train_iters=2000`
```

```{option} args.seq_length=<n>

The training sequence length.

Example: `args.seq_length=4096`
```

## Strategies

The manifest records three operator strategies for `optimize/modelopt/distill`.

- When you recover quality after pruning or quantization, set the original BF16 checkpoint as the teacher and the optimized checkpoint as the student.
- When you validate the pipeline, set `args.use_mock_data=true`, `args.seq_length=512`, `args.train_iters=100`, and a small `args.eval_iters` value.
- When you need a Hugging Face checkpoint, set `args.hf_export_path` and `args.student_hf_model`, or convert a saved Megatron iteration after the run completes.

## Command Examples

Run the tiny validation configuration locally with mock data:

```console
$ nemotron steps run optimize/modelopt/distill -c tiny
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run optimize/modelopt/distill -c default --dry-run
```

Submit an attached distillation run on a Lepton profile with real data:

```console
$ nemotron steps run optimize/modelopt/distill -c default -r lepton_optimize_modelopt_distill \
    args.teacher_hf_path=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    args.student_hf_path=Qwen/Qwen3-4B \
    args.data_paths='[0.5, /lustre/data/wiki, 0.5, /lustre/data/c4]' \
    args.train_iters=2000
```

Submit a detached pipeline-validation run on a Slurm profile with mock data:

```console
$ nemotron steps run optimize/modelopt/distill -c default -b slurm_optimize_modelopt_distill \
    args.use_mock_data=true \
    args.seq_length=512 \
    args.train_iters=100
```

## Related Skill

Run the `nemotron-optimizer-distillation` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Run Post-Training Optimization](../../how-to/run-optimization.md) explains the ordering of prune and distill, hardware targets, and quality recovery.
- [optimize/modelopt/prune](prune.md) feeds pruned checkpoints into this step.

### Upstream

- [NVIDIA Model Optimizer Repository](https://github.com/NVIDIA/Model-Optimizer)
- [NVIDIA Model Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/)
- [Megatron-Bridge Distillation Guide](https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/training/distillation.html)
