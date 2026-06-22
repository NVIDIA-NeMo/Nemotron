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

(sdg-create-greenteme-airlines-dataset)=
# Create a Domain Dataset for Airlines Customer Service

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: A domain-adapted SFT chat dataset modeled on fictional airlines customer-service conversations.

^^^

**In this how-to guide, you will**:

1. Create an airline-domain pipeline config.
2. Create a seed file of airline inquiry scenarios.
3. Swap the category columns for three airline-relevant dimensions.
4. Rewrite the LLM prompts for the airline domain.
5. Update the output projection and output path.
6. Run a preview to verify, then generate 100 records.

{octicon}`clock;1.5em;sd-mr-1` This guide requires between 20 and 30 minutes to complete.
:::

:::{grid-item-card}
:columns: 4

{octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**

^^^

Adapt the default SDG pipeline for Greenteme Airlines customer service with three category dimensions, run a 2-record preview, then generate 100 records and show me one output record.

:::
::::

## Prerequisites

- ✅ Completed {doc}`../getting-started` — at least one successful preview and full run of `default.yaml` so you know the pipeline works end-to-end.
- ✅ `NVIDIA_API_KEY` set in your environment.

## How This Differs From the Default Pipeline

The default pipeline mixes a single category dimension, `persona`, with seed topics.
This example adds category dimensions, `traveler_segment`, `inquiry_type`, and `channel`, on top of seed scenarios so that diversity comes from explicit, controllable values.

## Procedure

1. Create a `src/nemotron/steps/sdg/data_designer/config/greenteme.yaml` ([download](../_snippets/input/greenteme.yaml)) file like the following example:

   ```{literalinclude} ../_snippets/input/greenteme.yaml
   :language: yaml
   :class: scrollable
   ```

   The key differences from the default pipeline:
   - The variation for traveler segment,  inquiry type, and channel are all provided by category-type columns.
   - The variation for the scenarios is provided by the seed JSONL file from the next step.
   - The system-style instruction lives at the top of each prompt rather than as a separate field. The LLM text columns take a single prompt that includes the role for the LLM to assume.
   - The `output_projection` field includes the new metadata fields.

2. Create a seed file, `src/nemotron/steps/sdg/data_designer/data/greenteme_inquiry_seeds.jsonl`, ([download](../_snippets/input/greenteme_inquiry_seeds.jsonl)) like the following example:

   ```{literalinclude} ../_snippets/input/greenteme_inquiry_seeds.jsonl
   :language: json
   :class: scrollable
   ```

1. Run a preview by specifying `preview=true num_records=2` to verify the pipeline before scaling:

   ```console
   $ nemotron steps run sdg/data_designer -c greenteme preview=true num_records=2
   ```

   ````{dropdown} Example Output
   :icon: code-square

   ```{literalinclude} ../_snippets/output/greenteme_preview.jsonl
   :language: json
   ```
   ````

1. Generate the dataset by raising `num_records` after the preview output looks correct:

   ```console
   $ nemotron steps run sdg/data_designer -c greenteme num_records=100
   ```

## Going Further

**Locale-aware persona profiles.** The current YAML schema supports category, seed, and LLM column types. To replace the static `traveler_segment` category with Census-grounded persona profiles using Data Designer's [person sampler](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/person_sampling/), you can include locale, age range, and synthetic-personas integration.

**Multi-turn conversations.** The example shows a single user and assistant exchange.
For multi-turn dialogue, follow the `customer_support_tools.yaml` pattern: ask one `llm_text` column to return a JSON object with `messages` and optional `tools`, then use the `structured_messages` output projection to write training-ready JSONL.

**Dispatch to a cluster.** Generation runs locally against the NVIDIA-hosted endpoint by default. To run on Lepton or Slurm, see {doc}`dispatch-to-cluster` — env.toml profiles, container images, and the gotchas that bite first-time cluster runs.

## Schema and Downstream Use

The `openai_messages` projection emits records with a `messages` array plus the metadata fields you list. These flow directly into:

- `data_prep/sft_packing` for Megatron-Bridge-style training, or
- AutoModel SFT, which consumes the chat format directly.

For a full reference of available projection shapes, see {doc}`../reference/output-projections`.

## Next Steps

- **Generate preference pairs for DPO**: {doc}`preference-data` — the `rl_pref.yaml` pattern.
- **Generate tool-calling SFT data**: {doc}`tool-call-data` — multi-turn `messages` and `tools` with `structured_messages`.
- **CLI flags and overrides**: {doc}`../reference/cli-reference`.
- **Config schema**: {doc}`../reference/config-schema` — full reference for column types, samplers, and projections.
- **Pipeline overview**: {doc}`../index`.
