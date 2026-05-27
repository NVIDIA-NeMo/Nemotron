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

(sdg-preference-data)=
# Generate Preference Data for DPO

This example shows how to use the `rl_pref.yaml` configuration file.
The example generates _prompt_, _chosen_, and _rejected_ triples for direct preference optimization (DPO) training.
Output flows directly into `data_prep/rl_prep` and then `rl/nemo_rl/dpo`.

## How It Works

The `rl_pref.yaml` file registers two model aliases at different temperatures:
a high-temperature creative model and a low-temperature precise model.
The goal is to produce two responses per prompt that are distinct:

```{literalinclude} ../../../src/nemotron/steps/sdg/data_designer/config/rl_pref.yaml
:language: yaml
:lines: 15-
:class: scrollable
```

For each seed prompt the pipeline:

1. Generates `response_a` (high temperature) and `response_b` (low temperature) independently.
2. Asks a third LLM call (`judge` column, `llm_judge` type) to compare them and return `{"winner": "A"}` or `{"winner": "B"}`.
3. The `dpo_preference` projection maps winner → chosen / rejected and writes `{"prompt": "...", "chosen": "...", "rejected": "..."}`.

## Prerequisites

- `NVIDIA_API_KEY` set in your environment.
- A seed file with one `prompt` field per line. The bundled `rl_pref_prompt_seeds.jsonl` contains general reasoning prompts. Replace it with domain-specific prompts for targeted preference data.

## Procedure

1. Preview two records to verify the judge returns valid `winner` values:

   ```console
   $ nemotron steps run sdg/data_designer -c rl_pref preview=true num_records=2
   ```

2. Generate the dataset. The checked-in `rl_pref.yaml` default is 100 records:

   ```console
   $ nemotron steps run sdg/data_designer -c rl_pref num_records=500
   ```

   Output is written to `./output/sdg/rl_pref.jsonl`.

   Inspect the output. Each line is a preference triple:

   ```json
   {"prompt": "Explain why retrieval-augmented generation can reduce hallucinations.", "chosen": "RAG grounds the model in retrieved documents, so claims are tied to specific passages rather than purely to weights.", "rejected": "RAG is better because it uses more data and is generally smarter than standard models."}
   ```

## Adapt the Seed File

Swap `seed_dataset.path` to point at your own prompt seed file. Each line must be valid JSON with a `prompt` field:

```json
{"prompt": "Describe the tradeoffs between batch and streaming inference for real-time applications."}
```

Keep seed prompts representative of the target capability and diverse across difficulty levels.
The judge performs better when the two responses have a clear quality difference--consider widening the temperature gap between the two model aliases if the judge returns many ties or unexpected results.

## Downstream Pipeline

```text
rl_pref.jsonl  →  data_prep/rl_prep  →  rl/nemo_rl/dpo
```

`data_prep/rl_prep` tokenizes and prepares preference pairs. `rl/nemo_rl/dpo` consumes the prepared dataset. Verify the `prompt`, `chosen`, and `rejected` fields are present in every record before handing off.

## Next Steps

- **Output projection reference**: {doc}`../reference/output-projections` — `dpo_preference` schema.
- **Config schema**: {doc}`../reference/config-schema` — `llm_judge` column type and `dpo_preference` projection fields.
- **Dispatch to a cluster**: {doc}`dispatch-to-cluster`.
