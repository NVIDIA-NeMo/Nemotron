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

(sdg-config-schema)=
# Config Schema

This page provides the reference information for the YAML config file consumed by `sdg/data_designer`.

## Simple Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `output_dir` | string | no | Base output directory. Supports OmegaConf env-var interpolation. Default resolves `$SDG_OUTPUT_DIR`, then `$NEMO_RUN_DIR/sdg`, then `./output/sdg`. |
| `output_path` | string | yes | Full path for the output JSONL file. Typically `${output_dir}/my-dataset.jsonl`. |
| `num_records` | int | yes | Number of records to generate (`client.create`) or preview (`client.preview`). |
| `preview` | bool | no | When `true`, calls `client.preview()` instead of `client.create()`. Default: `false`. Prefer setting this as a CLI override (`preview=true`) rather than in the YAML. |

## seed_dataset

Optional top-level field.
When present, Data Designer samples one row per generated record from the seed file and makes the fields available to column prompts by using Jinja2.

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to a JSONL file. Each line is a JSON object. |
| `strategy` | string | no | `shuffle` (default) or `ordered`. |
| `fields` | list[string] | yes | Column names to expose. Must match keys in the seed JSONL objects. These become available as `{{ field_name }}` in prompts without being declared in `columns`. |

## models

A required top-level field.
The field specifies a list of model configurations.
Each entry defines one alias that column specs reference by name.

| Field | Type | Required | Description |
|---|---|---|---|
| `alias` | string | yes | Short name referenced by `model_alias` in column specs. |
| `model` | string | yes | Model identifier such as `nvidia/nemotron-3-nano-30b-a3b` and `openai/gpt-oss-20b`. |
| `provider` | string | no | Provider name, such as `nvidia` or `anthropic`. |
| `skip_health_check` | bool | no | Skip the startup probe against the model provider. Useful for local or offline endpoints. Default: `false`. |
| `inference_parameters.temperature` | float | no | Sampling temperature. |
| `inference_parameters.top_p` | float | no | Top-p nucleus sampling. |
| `inference_parameters.max_tokens` | int | no | Maximum output tokens per call. |

## columns

A required top-level field.
This field is an ordered list of column specs.
Each column has a `name`, a `type`, and type-specific fields.
Columns can reference earlier columns and seed fields in prompts by using Jinja2 syntax like `{{ column_name }}`.

### Categorical Columns

Samples uniformly from a fixed list of string or numeric values like the following example.

```yaml
- name: persona
  type: category
  values: [teacher, engineer, student, researcher]
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Column name. |
| `values` | yes | List of values to sample from. |

### Seed Columns

Provides a named field from the seed dataset as a column.
Use this column type when a seed field needs to appear in `metadata_fields` or must be referenced in a way that requires it to be an explicit column.

```yaml
- name: topic
  type: seed
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Must match a field name in `seed_dataset.fields`. |

Seed fields declared in `seed_dataset.fields` are available directly in prompts without this column type.
Use `seed` only when you need the field as a named column in the output schema.

### LLM Text Columns

Generates free-form text using an LLM call.
These columns can references earlier specified columns and seed fields in `prompt` by using Jinja2 syntax.

```yaml
- name: user_query
  type: llm_text
  model_alias: nvidia-text
  prompt: |
    Write a message from a {{ persona }} asking about: {{ topic }}.
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Column name. |
| `model_alias` | no | Alias from `models`. Default: `nvidia-text`. |
| `prompt` | yes | Jinja2 template. Reference any earlier column or seed field with `{{ name }}`. |

### LLM Structured Columns

This column type generates structured JSON by making an LLM call.
The column definition instructs the model to return JSON matching `output_format`.
Use this column type for multi-turn conversations, preference judges, and any output that must conform to a schema.

```yaml
- name: conversation
  type: llm_structured
  model_alias: nvidia-text
  prompt: |
    Generate a support conversation for customer {{ customer_name }}...
  output_format:
    type: object
    properties:
      messages:
        type: array
        ...
    required: [messages]
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Column name. |
| `model_alias` | no | Alias from `models`. Default: `nvidia-text`. |
| `prompt` | yes | Jinja2 template. |
| `output_format` | yes | JSON Schema dict describing the expected output structure. |

### LLM Judge Columns

This type is an alias for `llm_structured`.
This type is typically used for columns that compare or evaluate other columns.

```yaml
- name: judge
  type: llm_judge
  model_alias: nvidia-text
  prompt: |
    Compare response A and B for: {{ prompt }}
    A: {{ response_a }}
    B: {{ response_b }}
  output_format:
    type: object
    properties:
      winner:
        type: string
        enum: [A, B]
    required: [winner]
```

## output_projection

This top-level field maps raw Data Designer records into the schema expected by downstream steps.
Refer to {doc}`output-projections` for full field tables and annotated JSONL examples for each type.

| `type` | Use for | Downstream |
|---|---|---|
| `openai_messages` | Single-turn SFT chat | `data_prep/sft_packing`, AutoModel SFT |
| `dpo_preference` | Preference pairs | `data_prep/rl_prep`, `rl/nemo_rl/dpo` |
| `structured_messages` | Multi-turn with tool calls | `data_prep/sft_packing`, AutoModel SFT |

## Extending the Schema: `person` and `datetime` Samplers

The current `step.py` supports the column types above. To use Data Designer's locale-aware person sampler or datetime sampler, `step.py`'s `build_columns()` function must be extended with `person` and `datetime` branches. A reference implementation showing both additions is in:

```{literalinclude} ../_snippets/input/step-with-person-datetime.py
:language: python
:start-at: "        elif kind == \"person\":"
:end-before: "        elif kind == \"seed\":"
```

Once merged, configs can declare:

```yaml
- name: traveler
  type: person
  locale: en_US
  age_range: [22, 75]
  with_synthetic_personas: true

- name: booking_date
  type: datetime
  start: "2024-01-01"
  end: "2025-12-31"
```

Download personas for the locale before running:

```console
$ data-designer download personas --locale en_US
```

## Related Information

- {doc}`output-projections` — projection field reference and JSONL examples.
- {doc}`cli-reference` — flags and hydra override syntax.
- {doc}`../how-to/run` — preview and generate workflow.
