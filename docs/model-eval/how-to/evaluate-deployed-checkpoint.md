<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-evaluate-deployed-checkpoint)=
# Evaluate A Deployed Checkpoint

This guide covers the two supported evaluation shapes in the current `eval/model_eval` step.

- Use `tiny_chat.yaml` when the model is already hosted behind an OpenAI-compatible chat endpoint.
- Use `default.yaml` when NeMo Evaluator Launcher should deploy a Megatron Bridge checkpoint and run the configured tasks.

## Prerequisites

- `uv sync --extra evaluator` has completed.
- For hosted evaluation, you have the endpoint URL, model id, and API key environment variable.
- For checkpoint evaluation, you have a concrete Megatron Bridge `iter_*` checkpoint path.

## Hosted Endpoint Path

For an existing endpoint, use the same command shape as {doc}`run-hosted-evaluation`.

```bash
export NVIDIA_API_KEY="<your-api-key>"
export NEMO_EVALUATOR_MODEL_URL="https://<endpoint>/v1/chat/completions"
export NEMO_EVALUATOR_MODEL_ID="<served-model-name>"

uv run --no-sync nemotron steps run eval/model_eval \
  -c tiny_chat \
  output_dir=./output/eval-hosted \
  target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
  target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID" \
  target.api_endpoint.api_key_name=NVIDIA_API_KEY \
  target.api_endpoint.type=chat
```

## Launcher-Managed Checkpoint Path

For Megatron Bridge checkpoints, use `default.yaml`.
Point `deployment.checkpoint_path` at the concrete iteration directory.

```bash
uv run --no-sync nemotron steps run eval/model_eval \
  -c default \
  output_dir=./output/eval-megatron \
  deployment.checkpoint_path=/path/to/run/iter_0001000 \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

The default config uses the launcher `deployment` block to serve the checkpoint and then runs the configured `tasks`.
If you change tasks to a log-probability task, keep `evaluation.nemo_evaluator_config.config.params.extra.tokenizer` aligned with the deployed checkpoint.

## Endpoint And Task Pairing

The endpoint type must match the task family.
Chat and instruction tasks need `target.api_endpoint.type=chat`.
Log-probability tasks such as HellaSwag need a completions endpoint with logprobs support and a matching tokenizer.
For the matching rule, refer to {doc}`../explanation/endpoint-types-and-benchmarks`.

## What To Check After Submission

The step prints the launcher config path.
When NeMo Evaluator Launcher returns an invocation id, it also prints:

```text
status_command: nemo-evaluator-launcher status <invocation-id>
logs_command: nemo-evaluator-launcher logs <invocation-id>
```

Run those commands to monitor the job, then inspect `output_dir` after completion.

## Related

- {doc}`run-hosted-evaluation` for the hosted endpoint walk-through.
- {doc}`../explanation/index` for endpoint type, tokenizer alignment, and task-family concepts.
- {ref}`model-eval-comparing-runs` for before-and-after evaluation framing.
- {doc}`../../deployment-guides` for broader deployment options.
- {doc}`../reference/config-schema` for the YAML field reference.
