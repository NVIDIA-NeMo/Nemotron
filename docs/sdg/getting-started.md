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

(sdg-getting-started)=
# Generate Your First Synthetic Dataset

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: a small supervised fine-tuning (SFT) chat dataset in OpenAI message format.
The dataset contains five records grounded in the `sft_topic_seeds.jsonl` seed file in the repository.
Data Designer generates the records against an NVIDIA-hosted large language model (LLM) endpoint.

^^^

**In this tutorial, you will**:

1. Set up prerequisites: the repository and an NVIDIA API key.
1. Read the default pipeline configuration.
1. Run a preview to verify the pipeline and model.
1. Generate a small dataset of five records.
1. Locate and inspect the output JSON Lines (JSONL) file.

{octicon}`clock;1.5em;sd-mr-1` This tutorial requires between 5 and 10 minutes to complete.
:::

:::{grid-item-card} {octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**
:columns: 4

Run a 2-record preview of the default synthetic data generation (SDG) pipeline, then generate 5 records and show me the first output record.

:::
::::

## Prerequisites

- Run all commands from the repository root.
- An `NVIDIA_API_KEY` for the default model, `nvidia/nemotron-3-nano-30b-a3b`.
  Data generation runs against an NVIDIA-hosted endpoint, so you can complete this tutorial on any machine with network access.

## How the Default Pipeline Works

The default pipeline at `src/nemotron/steps/sdg/data_designer/config/default.yaml` combines two sources of variation for each record.
A *seed topic* is sampled from `sft_topic_seeds.jsonl`, for example a topic on safe deployment of AI assistants in enterprise support workflows.
A *persona category*, such as teacher or engineer, is sampled from a fixed category set.
Together they anchor a user prompt.
The pipeline generates a matching assistant response and projects the result into OpenAI chat-format messages.

```{literalinclude} ../../src/nemotron/steps/sdg/data_designer/config/default.yaml
:language: yaml
:lines: 15-
:class: scrollable
```

## Procedure

1. Clone the repository, if you have not already done so:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. Install the dependencies for synthetic data generation:

   ```console
   $ uv sync --extra data-sdg
   ```

1. Set your NVIDIA API key:

   ```console
   $ export NVIDIA_API_KEY="<your-api-key>"
   ```

1. Run a 2-record preview to verify the model alias, prompts, and column mappings before generating at scale.

   ```console
   $ uv run nemotron steps run sdg/data_designer -c default preview=true num_records=2
   ```

   The pipeline registers the model alias, generates two rows, and prints a summary:

   ````{dropdown} Example Output
   :icon: code-square

   ```{literalinclude} _snippets/output/preview.txt
   :language: text
   ```
   ````

1. Generate the five-record dataset:

   ```console
   $ uv run nemotron steps run sdg/data_designer -c default num_records=5
   ```

   The default output path is `./output/sdg/sft.jsonl`.
   To change the path, set the `SDG_OUTPUT_DIR` environment variable or pass `output_path=...` on the command line.

1. Inspect the output.
   Each line of `sft.jsonl` is one chat record.
   The `openai_messages` projection emits a `messages` array along with the seed `topic` and sampled `persona` as metadata for traceability.
   The following sample shows one record from the `sft.jsonl` file.

   ```{literalinclude} _snippets/output/sft_first_record.jsonl
   :language: json
   ```

## Summary

In this tutorial, you completed the following tasks:

- Ran a 2-record preview to verify the pipeline and model.
- Generated a 5-record SFT chat dataset with `default.yaml`.
- Located the OpenAI-format JSONL output.

As you scale this workflow up, keep two principles in mind:

- Run a preview first.
  The `preview=true num_records=N` form runs the same pipeline against a small record count, so you can iterate on column specifications and prompts before scaling `num_records` up.
- The output format matches the trainer.
  The `openai_messages` projection emits records ready for `data_prep/sft_packing` or AutoModel SFT.

## Next Steps

- Adapt the pipeline to a specific domain: {doc}`how-to/create-domain-dataset`.
- Preview, generate, and customize output: {doc}`how-to/run`.
- Generate preference pairs for direct preference optimization (DPO): {doc}`how-to/preference-data`.
- Dispatch to a cluster: {doc}`how-to/dispatch-to-cluster` describes env.toml profiles and container images.
- Look up flags and config fields: {doc}`reference/cli-reference`, {doc}`reference/config-schema`.
