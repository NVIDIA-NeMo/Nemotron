---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Per-step configuration conventions for Nemotron training steps."
topics: ["Training", "Reference", "Configuration"]
tags: ["Reference", "Configuration", "YAML", "Overrides"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Configuration Conventions

This page describes the per-step configuration layout, the rules the CLI uses to resolve a configuration name, and the rules it uses to merge command-line overrides.
The conventions apply to every step registered under `src/nemotron/steps/`.

Each step ships its manifest, configuration files, and entry script next to each other on disk.
The manifest, `step.toml`, remains authoritative for parameters, consumed and produced artifacts, supported models, and operator strategies.

## Per-Step Directory Layout

For a step directory `src/nemotron/steps/<category>/<name>/`, expect the following files.

| File | Purpose |
| --- | --- |
| `step.toml` | Manifest with the identifier, human-readable title, tags, `[[consumes]]` blocks, `[[produces]]` blocks, `[[parameters]]` blocks, optional `[[strategies]]`, optional `[[errors]]`, and optional `[[models]]` blocks. |
| `step.py` | Entry script that the CLI invokes after configuration compilation. The script also declares the NeMo Run specification. |
| `config/default.yaml` | Primary configuration tuned for production-shape runs. |
| `config/tiny.yaml` | Reduced configuration for short validation runs. |
| `config/<variant>.yaml` | Optional configuration variants. Examples include `config/fp8.yaml` and `config/nvfp4.yaml` for `optimize/modelopt/quantize`, and `config/nemo_gym.yaml` for `rl/nemo_rl/rlvr`. |

The per-step reference pages list every configuration file that the step ships.

## Configuration Resolution

The CLI resolves the `-c` and `--config` value in the following order.

1. When the value is an absolute or relative file path that exists on disk, the CLI loads that file directly.
2. Otherwise the CLI treats the value as a bare name and looks for `config/<name>.yaml` inside the step directory.
3. When the value is omitted, the CLI uses the runspec default declared in the step's `step.py`, typically `default`.

The following command resolves the bare name `tiny` to `src/nemotron/steps/sft/automodel/config/tiny.yaml`:

```console
$ nemotron steps run sft/automodel -c tiny
```

The following command loads an explicit path:

```console
$ nemotron steps run sft/automodel -c /lustre/configs/qwen-sft.yaml
```

## Dotlist Overrides

Any trailing argument that contains an equals sign and does not start with a hyphen is treated as a *dotlist override*.
The CLI merges dotlist overrides into the compiled YAML configuration by using OmegaConf semantics, so dotted paths reach nested structures.

```console
$ nemotron steps run sft/automodel -c default \
    step_scheduler.max_steps=200 \
    model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
```

The override syntax follows the rules in `nemo_runspec.cli_context.split_unknown_args`.
Refer to the [Nemotron Steps CLI Reference](cli-reference.md) for the full syntax and additional examples.

## Passthrough Arguments

Any trailing token that does not match a known option and does not contain an equals sign is preserved as a *passthrough argument* and forwarded to the step's entry script.
The script applies the token only when its stripped name, with dashes converted to underscores, matches a top-level field in the compiled configuration.
Tokens that do not map to a top-level field are silently ignored, so most parameters are better set through dotlist overrides.

## Environment Variables in Configurations

Configuration files use the OmegaConf `${oc.env:NAME,default}` resolver to read environment variables at compile time.
The default configurations rely on the following environment variables.

| Variable | Used By | Purpose |
| --- | --- | --- |
| `SFT_OUTPUT_DIR` | `sft/automodel` | Destination directory for SFT checkpoints. Defaults to `./output/automodel-tiny`. |
| `SFT_PACKED_DIR` | `sft/megatron_bridge`, `peft/megatron_bridge` | Source directory for packed Parquet shards produced by `data_prep/sft_packing`. |
| `RL_POLICY_MODEL` | `rl/nemo_rl/dpo`, `rl/nemo_rl/rlhf`, `rl/nemo_rl/rlvr` | Hugging Face identifier or local path for the policy. Defaults to `meta-llama/Llama-3.2-1B-Instruct`. |
| `WANDB_NAME` | All training steps | Weights and Biases run name. |
| `NEMO_RUN_DIR` | `optimize/modelopt/*` | Output root used to derive default destination paths. |

Set these variables in your shell, in the environment profile, or in an executor `env` block, depending on whether the run is local or remote.

## Environment Profiles

Per-cluster environment variables, container images, and startup commands live in `env.toml` at the repository root, not in per-step YAML files.
The active profile is selected with `--run` or `--batch`, and the file is found by the rules described in the [Nemotron Steps CLI Reference](cli-reference.md).

## Related Documentation

- [Getting Started With Training Steps](../getting-started.md) walks through a first run.
- [Nemotron Steps CLI Reference](cli-reference.md) covers `list`, `show`, and `run` syntax.
- [Step Catalog](step-catalog.md) lists every step identifier and its manifest path.
- [Execution Through NeMo Run](../../nemo_runspec/nemo-run.md) explains profiles, attached and detached execution, and clusters.
