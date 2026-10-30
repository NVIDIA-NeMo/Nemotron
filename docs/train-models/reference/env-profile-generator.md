---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the env/env_toml step that generates an environment profile file for every Nemotron training step."
topics: ["Training", "Reference", "Environment Profiles"]
tags: ["Reference", "CLI", "Environment Profile", "Lepton", "Slurm"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer"]
---

# Env Profile Generator

The `env/env_toml` step generates the *environment profile* file that every other Nemotron training step reads when it submits a job to a cluster.
The generator emits a single Tom's Obvious Minimal Language (TOML) file with one named profile per step, including data preparation, supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), reinforcement learning (RL), pretraining, optimization, evaluation, synthetic data generation, translation, and curation profiles.
You typically run this step once, before you submit your first training job to a cluster, and you re-run it only when site values such as the node group, container image, or shared mount path change.

## Syntax

```bash
nemotron steps run env/env_toml -c <config-name> [<key>=<value> ...]
```

The CLI accepts the same dotlist override syntax used by every other step.
Refer to the [Nemotron Steps CLI Reference](cli-reference.md) for the shared flag set.

## Configuration Files

The step ships three configuration files under `src/nemotron/steps/env/env_toml/config/`.

| File | Cluster Shape |
| --- | --- |
| `lepton.yaml` | Targets a DGX Cloud Lepton workspace with node-group, fileset mount, and resource-shape placeholders that you override at the command line. |
| `slurm.yaml` | Targets a Slurm cluster reached over SSH, with host, account, partition, and Lustre mount placeholders. |
| `dgxcloud.yaml` | Targets a DGX Cloud Run:AI installation with project, department, and image-pull credential placeholders. |

The default configuration is `lepton`.

## Inputs and Outputs

The step has no input artifact.
It produces one `env_toml` artifact, the generated environment profile file.

The default `output_path` differs by configuration.

| Configuration | Default `output_path` |
| --- | --- |
| `lepton` | `env.lepton.toml` |
| `slurm` | `env.slurm.toml` |
| `dgxcloud` | `env.dgxcloud.toml` |

Override `output_path` to write the file somewhere other than the repository root, for example a per-cluster directory under your quality assurance workspace.

## Parameters

The step exposes three top-level parameters.
Pass each one as a dotlist override after the configuration name.

```{option} output_path=<file-path>

Where to write the generated environment profile file.
A relative path is resolved against the current working directory.
Keep this value at the repository root for normal use so the CLI can discover the file by walking upward from any subdirectory.

Default: `env.lepton.toml`, `env.slurm.toml`, or `env.dgxcloud.toml`, taken from the selected configuration.

Example: `output_path=env.toml`
```

```{option} force=<true-or-false>

Whether to overwrite `output_path` when a file already exists at that location.
Set this value to `true` when you re-run the generator with new site overrides and you want the existing file replaced.
Leave the value at the default when you want the existing file preserved so a stale invocation cannot clobber a hand-edited profile.

Default: `false`.

Example: `force=true`
```

```{option} sections.<profile-name>.<key>=<value>

A dotlist override that reaches into one of the named profiles emitted by the generator and replaces a specific key.
Use this pattern to set site-specific values such as the node group, container image, mount path, or per-profile environment variables without editing the shipped configuration file.

The dot path always begins with `sections`, then the profile name, then the key path inside that profile.
Nested keys, such as a single environment variable, append additional dot segments.

Examples:

- `sections.lepton_base.node_group=<lepton-node-group>` replaces the node group on the base Lepton profile.
- `sections.lepton_base.env_vars.WANDB_PROJECT=<wandb-project>` sets the Weights and Biases project name for every profile that inherits from `lepton_base`.
- `sections.slurm_base.account=<slurm-account>` sets the Slurm account on the base Slurm profile.
```

The shipped configurations declare additional document-level keys such as `preamble` and `checks` that drive the generator itself.
These keys behave identically to any other dotlist override, but most users only need the three parameters described above.

## Examples

Generate a default Lepton environment profile at the repository root and overwrite any existing file at that path.

```console
$ nemotron steps run env/env_toml -c lepton output_path=env.toml force=true
```

The command writes a single `env.toml` file at the repository root with every canonical profile the Nemotron training steps expect, including data-prep, SFT, PEFT, RL, pretrain, optimize, evaluate, synthetic data generation, translation, and curation profiles.

Generate a Lepton environment profile with site-specific node-group and Weights and Biases overrides.

```console
$ nemotron steps run env/env_toml -c lepton \
    output_path=env.lepton.toml \
    force=true \
    sections.lepton_base.node_group=<lepton-node-group> \
    sections.lepton_base.env_vars.WANDB_PROJECT=<wandb-project> \
    sections.lepton_base.env_vars.WANDB_ENTITY=<wandb-entity> \
    sections.lepton_base.env_vars.WANDB_NAME=<wandb-run-name>
```

Each `sections.lepton_base.*` override replaces the matching key on the `[lepton_base]` profile in the generated file.
Every child profile that extends `lepton_base`, for example `lepton_sft_automodel` and `lepton_prep_sft_packing`, inherits the new value through the profile inheritance chain.

## Editing the Generated File

The generated file is a normal TOML document.
You can hand-edit any profile after generation, for example to add a one-off mount path or to change a container image for a single profile.
When you need only a small slice of profiles or you want to start from a minimal hand-written file rather than the full set of canonical profiles, follow the manual template walkthrough in the [Quickstart](../getting-started.md).

## Related Documentation

- [Quickstart](../getting-started.md)
- [Nemotron Steps CLI Reference](cli-reference.md)
- [Step Catalog](step-catalog.md)
- [Configuration Conventions](config-conventions.md)
