---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the nemotron steps subcommand group."
topics: ["Training", "Reference", "CLI"]
tags: ["Reference", "CLI", "Command-Line", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Nemotron Steps CLI Reference

This page documents the `nemotron steps` command group, the entry point for discovering, inspecting, and running every supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), reinforcement learning (RL), and optimization step packaged under `src/nemotron/steps/`.

The command group is registered by the Nemotron CLI and exposes three subcommands:

- `nemotron steps list` enumerates discovered steps with optional filters.
- `nemotron steps show <step-id>` prints the manifest, runspec, and parameters for one step.
- `nemotron steps run <step-id>` compiles a configuration, selects an executor, and submits a job.

Each subcommand resolves a step by its identifier, such as `sft/automodel` or `rl/nemo_rl/dpo`.
The [Step Catalog](step-catalog.md) lists every identifier.

## Syntax

The `run` subcommand accepts a step identifier, a configuration name or path, an execution mode, and any number of trailing overrides:

```bash
nemotron steps run <step-id> \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

The `list` and `show` subcommands take only options and an optional step identifier:

```bash
nemotron steps list [--category <category>] [--consumes <type>] [--produces <type>] [--json]

nemotron steps show <step-id> [--json]
```

## Required Arguments

```{option} <step-id>

The step identifier under `src/nemotron/steps/`, such as `sft/automodel`, `peft/megatron_bridge`, `rl/nemo_rl/dpo`, or `optimize/modelopt/quantize`.

This argument is positional and required for `nemotron steps run` and `nemotron steps show`.

Example: `nemotron steps run sft/automodel`
```

## Optional Arguments

The following options apply to `nemotron steps run`.
Pass them between the subcommand and any trailing overrides.

```{option} -c, --config <config-name-or-path>

The configuration to compile for the run.
Bare names resolve to `config/<name>.yaml` inside the step directory.
A relative or absolute path resolves directly to that file.

Default: the configuration declared as `[config] default` in the step's runspec, typically `default`.

Example: `-c tiny` resolves to `src/nemotron/steps/sft/automodel/config/tiny.yaml`.
```

```{option} -r, --run <run-profile>

The environment profile to attach to for synchronous execution.
The profile name must match a top-level `[<run-profile>]` table in the active `env.toml` file.
The CLI streams logs to your terminal until the job finishes.

This option is mutually exclusive with `--batch`.

Example: `-r lepton_sft_automodel`
```

```{option} -b, --batch <batch-profile>

The environment profile to submit to for detached execution.
The profile name must match a top-level `[<batch-profile>]` table in the active `env.toml` file.
The CLI returns once the job is submitted.

This option is mutually exclusive with `--run`.

Example: `-b slurm_prod`
```

```{option} -d, --dry-run

Compile and print the merged configuration as a rich table, then exit without submitting the job.
Use this option to validate overrides before you commit GPU time.

Default: disabled.
```

```{option} --force-squash

Force re-creation of the squashed container image used by remote executors even when a cached image already exists.
Use this option after you change the container image referenced in your environment profile.

Default: disabled.
```

The `list` subcommand has its own option set.

```{option} --category <category>

Restrict the output to one step category, such as `sft`, `peft`, `rl`, or `optimize`.

Example: `--category sft`
```

```{option} --consumes <type>

Restrict the output to steps that declare a `[[consumes]]` entry of the given artifact type, such as `training_jsonl`, `packed_parquet`, `checkpoint_hf`, or `checkpoint_megatron`.

Example: `--consumes training_jsonl`
```

```{option} --produces <type>

Restrict the output to steps that declare a `[[produces]]` entry of the given artifact type.

Example: `--produces checkpoint_megatron`
```

```{option} --json

Emit machine-readable JSON instead of the human-formatted table.
Use this option from agents or shell pipelines.

Default: disabled.
```

## Configuration Overrides

Any token after the options that contains an equals sign and does not start with a hyphen is treated as a *dotlist override* and merged into the compiled configuration.
Any other token is preserved as a *passthrough argument* for the step's underlying script.
The split is performed by `nemo_runspec.cli_context.split_unknown_args`.

```{option} <key>.<path>=<value>

A dotlist override that sets a nested key in the compiled YAML configuration.
Use dotted paths to reach nested structures, the same way OmegaConf interprets them.

Examples:
- `step_scheduler.max_steps=10`
- `model.pretrained_model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3`
- `args.export_quant_cfg=fp8`
```

```{option} <passthrough-argument>

Any token that does not match a known option and does not contain an equals sign is forwarded verbatim to the step's underlying script.
The script applies the token only when its stripped name, with dashes converted to underscores, matches a top-level field in the compiled configuration.
Tokens that do not map to a top-level field are silently ignored, so most parameters are set through dotlist overrides instead.
```

## Configuration Resolution

The CLI resolves `--config` in the following order:

1. If the value is an absolute or relative file path that exists, the CLI loads that file.
2. Otherwise the CLI treats the value as a bare name and looks for `config/<name>.yaml` inside the step directory.
3. If `--config` is omitted, the CLI uses the runspec default declared in the step's `step.py`.

Each step ships at minimum a `config/default.yaml` file for production-shape runs and a `config/tiny.yaml` file for short validation.
Some steps ship additional variants, such as `config/fp8.yaml` for `optimize/modelopt/quantize` and `config/nemo_gym.yaml` for `rl/nemo_rl/rlvr`.
The per-step pages list every config file.

## Environment Profile and Executor Selection

The `--run` and `--batch` options select a profile from an `env.toml` file.
The file is found by the following rules:

1. If the `NEMOTRON_ENV_FILE` environment variable is set, the CLI uses that path.
2. Otherwise the CLI walks upward from the current directory until it finds a file named `env.toml`, stopping at the project root that contains `pyproject.toml`.

Each profile is a top-level TOML table.
The table's `executor` field selects `local`, `slurm`, or `lepton` execution.
The remaining fields configure the container image, mounts, resources, and environment variables for that executor.
Refer to [Environment and Executors](../how-to/env-and-executors.md) for the full schema.

If you omit both `--run` and `--batch`, the CLI submits the step on the local backend.
Remote execution requires `-r` or `-b` together with a profile that exists in the active `env.toml` file.

## Command Examples

Compile and validate the configuration without submitting the job:

```console
$ nemotron steps run sft/automodel -c tiny --dry-run
```

Submit an attached run on a Lepton profile, override the step count, and override the base model:

```console
$ nemotron steps run sft/automodel -c default -r lepton_sft_automodel \
    step_scheduler.max_steps=100 \
    model.pretrained_model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3
```

Submit a detached run on a Slurm profile and enable the mock-data path through a dotlist override:

```console
$ nemotron steps run optimize/modelopt/distill -c tiny -b slurm_optimize_modelopt_distill args.use_mock_data=true
```

Filter the catalog for steps that consume preference data:

```console
$ nemotron steps list --consumes training_jsonl --category rl
```

Print the manifest for one step as JSON for agent consumption:

```console
$ nemotron steps show sft/automodel --json
```

## Per-Category References

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Supervised Fine-Tuning Steps
:link: sft/index
:link-type: doc
The `sft/automodel` and `sft/megatron_bridge` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} Parameter-Efficient Fine-Tuning Steps
:link: peft/index
:link-type: doc
The `peft/automodel` and `peft/megatron_bridge` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} Reinforcement Learning Steps
:link: rl/index
:link-type: doc
The `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, and `rl/nemo_rl/rlhf` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} Optimization Steps
:link: optimize/index
:link-type: doc
The `optimize/modelopt/quantize`, `optimize/modelopt/prune`, and `optimize/modelopt/distill` references.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Step Catalog](step-catalog.md) lists every step identifier and its manifest path.
- [Configuration Conventions](config-conventions.md) explains the per-step `config/` layout, dotlist override rules, and the relationship to environment profiles.
- [Environment and Executors](../how-to/env-and-executors.md) explains the `env.toml` schema and profile selection.
- [Execution Through NeMo Run](../../nemo_runspec/nemo-run.md) explains attached and detached execution, image squashing, and remote job directories.
