---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run nemotron steps run curate/nemo_curator on the packaged tiny JSONL file."
topics: ["Curation", "Tutorial", "NeMo Curator"]
tags: ["Tutorial", "Curation"]
content:
  type: "Tutorial"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(getting-started-curate)=
# Getting Started With Data Curation

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: a filtered set of JSONL shards from the packaged
tiny fixture.
The run reads the fixture, passes it through the NeMo Curator pipeline with all
optional filters disabled, and writes output shards to a local directory.

^^^

**In this tutorial, you will**:

1. Clone the repository and install dependencies.
1. Configure the Ray runtime environment.
1. Inspect the packaged fixture.
1. Run the curation step with the tiny configuration.
1. Inspect the output shards.

{octicon}`clock;1.5em;sd-mr-1` This tutorial requires approximately 5 minutes to complete.
:::

:::{grid-item-card} {octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**
:columns: 4

Run the `curate/nemo_curator` step on the packaged tiny JSONL fixture, then show me
the names and record counts of the output shards.

:::
::::

## Prerequisites

- The `uv` tool is available in your shell.

## Procedure

1. Clone the repository, if you haven't already:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. Install the dependencies for curating data:

   ```console
   $ uv sync --extra curate
   ```

1. Set the Ray runtime environment variable so Ray workers reuse the synchronized
   project environment:

   ```console
   $ export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
   ```

1. Inspect the packaged fixture at
   `src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl`.
   Each record contains a `text` field.

   ```{literalinclude} ../../src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl
   :language: json
   :class: scrollable
   ```

1. Run the curation step.
   The `tiny` configuration disables optional language, word-count, and domain
   filters, making this run a baseline validation of the reader, writer, and Ray
   startup.
   The checked-in `tiny.yaml` uses container paths, so override `input_glob` and
   `output_dir` with host paths:

   ```console
   $ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
       input_glob="${PWD}/src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl" \
       output_dir="${PWD}/output/curate-tiny"
   ```

1. Inspect the output directory:

   ```console
   $ find output/curate-tiny -type f
   ```

   Open a shard and confirm that records contain the configured `text_field`.
   NeMo Curator assigns the exact shard name.

## Summary

In this tutorial, you completed the following tasks:

- Cloned the repository and installed the project dependencies.
- Configured the Ray runtime environment.
- Ran the `curate/nemo_curator` step with the packaged tiny JSONL fixture.
- Inspected the output shards and verified that records contain the configured
  text field.

The `tiny` configuration disables all optional filters.
To add language, word-count, or domain filters to a production corpus, refer to
the how-to guides below.

## Next Steps

- Point the step at your own JSONL corpus: {doc}`how-to/run-local-jsonl`.
- Download a Hugging Face dataset before curation:
  {doc}`how-to/use-huggingface-snapshot`.
- Add language, word-count, or domain filters: {doc}`how-to/enable-filters`.
- Look up all configuration fields and CLI flags: {doc}`reference/curate-config`
  and {doc}`reference/cli-curate`.
