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

(sdg-run)=
# Tips for the Data Generation Pipeline

## Preview Before Generating

Always preview before running a full generation job. Preview mode calls the same pipeline with a small record count, projects the records, and writes them to `output_path`:

```console
$ nemotron steps run sdg/data_designer -c default preview=true num_records=2
```

Use preview to verify:

- Column references in prompts (`{{ column_name }}`) resolve to the expected values.
- Seed fields, such as `{{ scenario }}`, `{{ prompt }}`, and so on, are populated from the seed file.
- The model returns text that matches the prompt's intent.
- The `output_projection` produces the schema downstream steps expect.

## Specify a Configuration File

The repository includes the following sample config files in the `src/nemotron/steps/sdg/data_designer/config` directory:

| Config | Output | Use for |
|---|---|---|
| `default.yaml` | SFT chat (`openai_messages`) | General chat SFT |
| `customer_support_tools.yaml` | Tool-call SFT (`structured_messages`) | Tool-use SFT |
| `rl_pref.yaml` | Preference pairs (`dpo_preference`) | DPO / RLHF |
| `tiny.yaml` | SFT chat, 10 records, short tokens | Fast iteration |

Specify the file in the `-c` argument:

```console
$ nemotron steps run sdg/data_designer -c customer_support_tools preview=true num_records=2
```

## Run Attached on a Cluster Profile

To dispatch to a Lepton or Slurm profile configured in `env.toml`, use `--run` (attached, streams logs) or `--batch` (detached):

```console
$ nemotron steps run sdg/data_designer -c default --run my-lepton-profile num_records=1000
```

For cluster setup, see {doc}`dispatch-to-cluster`.
