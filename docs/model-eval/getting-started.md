<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-getting-started)=
# Getting Started With Model Evaluation

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: one NeMo Evaluator Launcher result directory for a single hosted chat smoke-test task, written by the `eval/model_eval` step.

^^^

**In this tutorial, you will**:

1. Discover the `eval/model_eval` step from the local catalog.
1. Inspect the hosted-endpoint sample config, `tiny_chat.yaml`.
1. Run a one-sample hosted chat evaluation.
1. List the result files on disk.

{octicon}`clock;1.5em;sd-mr-1` This tutorial requires between 15 and 30 minutes to complete, depending on endpoint latency.
:::

:::{grid-item-card} {octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**
:columns: 4

Run a one-sample hosted chat evaluation with `eval/model_eval` and `tiny_chat.yaml`, then show me the launcher config and result files.
:::
::::

## Prerequisites

- Run all commands from the repository root.
- Install the evaluator extra:

  ```console
  $ uv sync --extra evaluator
  ```

- A reachable OpenAI-compatible chat-completions endpoint.
- A model identifier advertised by that endpoint.
- A bearer token exported as the environment variable referenced by `target.api_endpoint.api_key_name`.

## About The Sample Configuration

The hosted chat sample file is at `src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml`.
It sets `deployment.type: none`, points NeMo Evaluator Launcher at `target.api_endpoint`, and runs the chat-compatible `mmlu_instruct` task with `limit_samples: 1`.

```{literalinclude} ../../src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml
:language: yaml
```

## Procedure

1. Clone the repository, if you haven't already:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. Synchronize dependencies:

   ```console
   $ uv sync --extra evaluator
   ```

1. Export the endpoint values.
   `EVAL_ROOT` is a directory you choose; it is the parent of the per-run `output_dir`.

   ```console
   $ export NVIDIA_API_KEY="<your-api-key>"
   $ export NEMO_EVALUATOR_MODEL_URL="<full-chat-completions-url>"
   $ export NEMO_EVALUATOR_MODEL_ID="<model-identifier-from-the-endpoint>"
   $ export NEMO_EVALUATOR_ENDPOINT_TYPE=chat
   $ export EVAL_ROOT="$(pwd)/output/eval-getting-started"
   ```

1. Confirm that the local catalog exposes `eval/model_eval`.

   ```console
   $ uv run --no-sync nemotron steps show eval/model_eval
   ```

1. Run the hosted chat smoke test.

   ```console
   $ uv run --no-sync nemotron steps run eval/model_eval \
       -c tiny_chat \
       output_dir="$EVAL_ROOT/results-tiny-chat" \
       target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
       target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID" \
       target.api_endpoint.api_key_name=NVIDIA_API_KEY \
       target.api_endpoint.type=chat \
       evaluation.nemo_evaluator_config.config.params.limit_samples=1
   ```

   The step writes the launcher config path to stdout.
   If NeMo Evaluator Launcher returns an invocation id, the step also prints `status_command` and `logs_command` values that you can run to inspect the job.
   Treat those commands as part of the run: wait until the launcher reports a
   terminal status before expecting final metric artifacts.

   To inspect the merged Nemotron job config without invoking the launcher, add `--dry-run`.
   To pass NeMo Evaluator Launcher's own dry-run flag, use the config override `dry_run=true`.

1. List the files written under the output directory after the launcher job
   reaches a terminal status.

   ```console
   $ find "$EVAL_ROOT/results-tiny-chat" -maxdepth 5 -type f | sort
   ```

   The exact file names are owned by NeMo Evaluator Launcher and can vary by task version.

## Next Steps

- Run the standard checkpoint-evaluation config: {doc}`how-to/evaluate-deployed-checkpoint`.
- Look up the full YAML schema: {doc}`reference/config-schema`.
- Drive the step from a coding agent: {doc}`using-skills`.
- Run hosted evaluations with custom task settings: {doc}`how-to/run-hosted-evaluation`.
