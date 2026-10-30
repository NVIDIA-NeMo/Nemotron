<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-cli-reference)=
# CLI Reference

Command-line reference for `nemotron steps run sdg/data_designer`. For pipeline overview, see {doc}`../index`.

## Syntax

```console
$ nemotron steps run sdg/data_designer \
    [-c CONFIG] \
    [--run PROFILE | --batch PROFILE] \
    [--dry-run] \
    [KEY=VALUE ...]
```

## Flags

```{option} -c, --config CONFIG

Config name (resolved from the step's `config/` directory) or an absolute/relative path to a YAML file.

Bundled names: `default`, `customer_support_tools`, `rl_pref`, `tiny`.

**Default**: `default`
```

```{option} -r, --run PROFILE

Run attached using the env.toml profile named `PROFILE`. Job output streams to the terminal. Use for short interactive runs.
```

```{option} -b, --batch PROFILE

Run detached using the env.toml profile named `PROFILE`. Job is submitted and the command returns immediately. Use for long cluster jobs.
```

```{option} -d, --dry-run

Compile the config and print the resolved job spec without executing. Useful for verifying hydra overrides before submission.
```

## Hydra Overrides

Any `KEY=VALUE` argument after the flags is passed as a hydra dotlist override and merged into the resolved config. Overrides take precedence over YAML values.

| Override | Example | Effect |
|---|---|---|
| `num_records=N` | `num_records=50` | Generate N records |
| `preview=true` | `preview=true` | Run in preview mode |
| `output_path=PATH` | `output_path=/data/out.jsonl` | Write output to PATH |
| `seed_dataset.path=PATH` | `seed_dataset.path=/data/seeds.jsonl` | Override seed file |
| `models.0.inference_parameters.temperature=T` | `models.0.inference_parameters.temperature=0.5` | Override first model's temperature |

Dotlist path follows the YAML structure. Nested keys use `.` as separator; list items use `.N` (zero-indexed).

## Examples

Preview the default config with two records:

```console
$ nemotron steps run sdg/data_designer -c default preview=true num_records=2
```

Generate 100 SFT records with a custom output path:

```console
$ nemotron steps run sdg/data_designer -c default \
    num_records=100 \
    output_path=/data/my-project/sft.jsonl
```

Dry-run a cluster submission to check the resolved config:

```console
$ nemotron steps run sdg/data_designer -c default --run my-profile --dry-run
```

Run attached on a Lepton profile with 500 records:

```console
$ nemotron steps run sdg/data_designer -c default --run lepton_sdg_data_designer num_records=500
```

Use a config at an arbitrary path:

```console
$ nemotron steps run sdg/data_designer -c /path/to/my-config.yaml preview=true num_records=2
```

## Related

- {doc}`../how-to/run` — Preview, generate, and customize output.
- {doc}`../how-to/dispatch-to-cluster` — env.toml profile setup.
- {doc}`config-schema` — YAML config field reference.
