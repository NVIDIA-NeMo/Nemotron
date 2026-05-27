<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(getting-started-byob-mcq)=
# Getting Started with Building MCQ Benchmarks

<!-- Tutorial: end-to-end `tiny` BYOB run; about 10 to 15 minutes including model waits; requires Nemotron clone, uv, BYOB extra, and NVIDIA_API_KEY. -->

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: A small multiple-choice question (MCQ) benchmark for math questions from the sample `tiny` configuration.
You'll run the `nemotron steps run byob/mcq` command and it will use an NVIDIA-hosted model endpoint for inference.

^^^

**In this tutorial, you will**:

1. Install Python dependencies.
2. Run the `tiny` configuration from the repository root.
3. Locate outputs and scan the main Parquet artifacts.
4. Confirm `benchmark.parquet` columns against the output reference.

{octicon}`clock;1.5em;sd-mr-1` This tutorial requires between 10 and 15 minutes to complete.
:::

:::{grid-item-card}
:columns: 4

{octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**

^^^

Help me create an MCQ benchmark using the `tiny` configuration from the Nemotron repository clone, write outputs under `./output`, then show me which Parquet files to open first.

:::
::::

## Start Here

- Run all commands from the repository root so the `input_dir` path in the procedure resolves.
- The sample configuration uses the `cais/mmlu` dataset from Hugging Face for few-shot examples of multiple-choice questions.
- The sample configuration uses the `src/nemotron/steps/byob/data/tiny_input/maths/tiny.txt` file for input.

  ```{literalinclude} ../../src/nemotron/steps/byob/data/tiny_input/maths/tiny.txt
  ```

## Prerequisites

- You have a host with access to https://integrate.api.nvidia.com.
- The `uv` tool available in your shell.
- `NVIDIA_API_KEY` exported in the same shell session before you run the procedure.
  The default model for the configuration is `openai/gpt-oss-120b`.

## Procedure

1. Clone the repository:

   ```console
   git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. From the repository root, add the dependencies for building benchmarks:

   ```console
   uv sync --extra byob
   ```

1. Run generation with host paths.
   The `tiny` configuration sets `stage: all`.
   This stage setting chains the data preparation and then generation for MCQ.

   ```console
   uv run nemotron steps run byob/mcq \
     -c tiny \
     family=mcq \
     stage=all \
     input_dir="./src/nemotron/steps/byob/data/tiny_input" \
     output_dir=./byob-output
   ```

   When the `-c` / `--config` argument is not a path, the command resolves the config file name in the `src/nemotron/steps/byob/mcq/config/` directory.

   When the command finishes, list the `./byob-output/byob_mcq_tiny/` directory.
   The `expt_name` field in the `src/nemotron/steps/byob/mcq/config/tiny.yaml` file specifies that directory.
   Look for the following files:

   - `stage_cache/*.parquet`, one file per intermediate stage, described in {doc}`reference/output-files`
   - `benchmark_raw.parquet`, the full row set before optional removals
   - `benchmark.parquet`, the final `mcq` schema for downstream use

1. Open `benchmark.parquet` with Pandas, Polars, or another Parquet-aware tool and confirm columns match {doc}`reference/output-files`.

## Next Steps

For background on how stages connect, read {doc}`explanation/pipeline-overview`.
For task-focused changes, continue with:

- Swap in your own corpus and mapping: {doc}`how-to/prepare-data`.
- Tune endpoints and keys: {doc}`how-to/custom-model-endpoints`.
