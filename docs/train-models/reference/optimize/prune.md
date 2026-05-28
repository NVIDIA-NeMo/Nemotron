---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the optimize/modelopt/prune step."
topics: ["Training", "Reference", "CLI", "Optimization", "Pruning"]
tags: ["Reference", "CLI", "Steps", "Optimization", "Pruning", "ModelOpt", "Megatron-Bridge"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# optimize/modelopt/prune

This step runs structured pruning on a Hugging Face (HF) format checkpoint by using NVIDIA Model Optimizer through NVIDIA Megatron-Bridge.
The step supports pruning by a target parameter budget that Model Optimizer searches against, or pruning to an explicit architecture that you supply.
The step produces a pruned Hugging Face checkpoint that you can pass to `optimize/modelopt/distill` for quality recovery.

## Syntax

```bash
nemotron steps run optimize/modelopt/prune \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/optimize/modelopt/prune/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Generic structured-pruning configuration for `Qwen/Qwen3-8B` with two-way pipeline parallelism. |
| `tiny.yaml` | Short validation run that exercises the pruning pipeline. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run optimize/modelopt/prune -c tiny
$ nemotron steps run optimize/modelopt/prune -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `checkpoint_hf` | Yes | A Hugging Face model identifier or checkpoint to prune. |
| Produces | `checkpoint_hf` | — | The pruned Hugging Face checkpoint. |

## Step Parameters

The manifest declares four pruning parameters.
Pass them as dotlist overrides.

```{option} args.prune_target_params=<float>

The target parameter count for the Model Optimizer search.
Use scientific notation for billions of parameters.

Default: `6e9`.

Example: `args.prune_target_params=4e9`
```

```{option} args.prune_export_config=<dict>

The explicit target architecture for manual pruning, expressed as a dictionary that maps hyperparameter names such as `hidden_size`, `ffn_hidden_size`, or `num_layers` to integer values.
Set this parameter when you want a specific architecture and you set `args.prune_target_params=null`.

Example: `args.prune_export_config='{"hidden_size":4096,"ffn_hidden_size":11008,"num_layers":24}'`
```

```{option} args.hparams_to_skip=<list>

The architecture hyperparameters that the search must leave unchanged.

Example: `args.hparams_to_skip=["num_attention_heads"]`
```

```{option} extra_args=<list>

Literal upstream arguments that the step forwards to the pruning script.
Use this parameter to pass newly added Model Optimizer flags that do not yet have a dedicated `args.*` entry.

Default: `[]`.

Example: `extra_args=["--num_layers_in_first_pipeline_stage", "4"]`
```

Frequently used dotlist overrides drawn from the default configuration include the following.

```{option} args.hf_model_name_or_path=<id-or-path>

The Hugging Face identifier or local path for the checkpoint to prune.

Example: `args.hf_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct`
```

```{option} args.output_hf_path=<path>

The destination directory for the pruned Hugging Face checkpoint.

Example: `args.output_hf_path=/lustre/runs/pruned/llama-6b`
```

```{option} args.pp_size=<n>

The pipeline-parallel degree applied during pruning.

Example: `args.pp_size=4`
```

## Strategies

The manifest records three operator strategies for `optimize/modelopt/prune`.

- When you know the target model budget, set `args.prune_target_params` and leave `args.prune_export_config` unset so Model Optimizer searches candidate architectures.
- When you need a specific architecture, set `args.prune_export_config` to the target dictionary and set `args.prune_target_params=null`.
- When the layer count does not divide the pipeline-parallel size, set `args.num_layers_in_first_pipeline_stage` and `args.num_layers_in_last_pipeline_stage` to balance the partition.

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run optimize/modelopt/prune -c tiny
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run optimize/modelopt/prune -c default --dry-run
```

Submit an attached run on a Lepton profile that searches for a four-billion-parameter target:

```console
$ nemotron steps run optimize/modelopt/prune -c default -r lepton_optimize_modelopt_prune \
    args.hf_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    args.prune_target_params=4e9 \
    args.output_hf_path=/lustre/pruned/llama-4b
```

Submit a detached run on a Slurm profile that prunes to a manually specified architecture:

```console
$ nemotron steps run optimize/modelopt/prune -c default -b slurm_optimize_modelopt_prune \
    args.prune_target_params=null \
    args.prune_export_config='{"hidden_size":4096,"ffn_hidden_size":11008,"num_layers":24}' \
    args.pp_size=4
```

## Related Skill

Run the `nemotron-optimizer-pruning` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Run Post-Training Optimization](../../how-to/run-optimization.md) explains the ordering of prune and distill, hardware targets, and quality recovery.
- [optimize/modelopt/distill](distill.md) recovers quality after pruning.

### Upstream

- [NVIDIA Model Optimizer Repository](https://github.com/NVIDIA/Model-Optimizer)
- [NVIDIA Model Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Megatron-Bridge Pruning Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/pruning.html)
