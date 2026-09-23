<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-evaluate-deployed-checkpoint)=
# Evaluate A Deployed Checkpoint

This guide covers the two launcher-mode evaluation shapes in the `eval/model_eval` step.

- Use `tiny_chat.yaml` when the model is already hosted behind an OpenAI-compatible chat endpoint and the launcher's local executor is acceptable.
- Use `default.yaml` when NeMo Evaluator Launcher should deploy a Megatron Bridge checkpoint and run the configured tasks on a local machine or a Slurm cluster.

To evaluate an endpoint you host from a Nemotron-scheduled job on any backend, including Run:ai and DGX Cloud, use direct mode instead: {doc}`run-direct-mode-evaluation`.
Direct mode does not deploy checkpoints.

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
Select the launcher executor with `run.env.launcher_executor=local` or `run.env.launcher_executor=slurm`; Slurm also needs `run.env.host`, `run.env.user`, `run.env.account`, and `run.env.partition`.
Keep `evaluation.nemo_evaluator_config.config.params.extra.tokenizer` aligned with the deployed checkpoint; `default.yaml` sets it to `${deployment.checkpoint_path}/tokenizer`.

:::{note}
Launcher-managed deployment on Lepton is experimental.
The launcher's Lepton executor rejects the `generic` deployment type that `default.yaml` uses, and it does not remove the endpoints it creates after a run.
On Lepton, serve the model yourself and use direct mode.
:::

## Endpoint And Task Pairing

The endpoint type must match the task family.
Chat and instruction tasks need `target.api_endpoint.type=chat`.
Log-probability tasks such as HellaSwag need a completions endpoint with logprobs support.
If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, set the evaluator to use the generated tokenizer.
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

- {doc}`run-hosted-evaluation` for the launcher-mode hosted endpoint walk-through.
- {doc}`run-direct-mode-evaluation` for evaluating an endpoint you host from a Nemotron-scheduled job.
- {doc}`../explanation/index` for endpoint type, tokenizer alignment, and task-family concepts.
- {ref}`model-eval-comparing-runs` for before-and-after evaluation framing.
- {doc}`../../deployment-guides` for broader deployment options.
- {doc}`../reference/config-schema` for the YAML field reference.
