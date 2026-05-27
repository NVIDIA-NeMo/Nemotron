<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-using-skills)=
# Use The Model Evaluation Skill With Confidence

This page is for users who plan to drive `eval/model_eval` from a coding agent.
The goal is a clear handoff between what you decide and what the agent edits or runs.

## What The Agent Needs

For the hosted chat smoke test, provide:

- The endpoint URL, including the `/v1/chat/completions` path.
- The model id advertised by the endpoint.
- The API key environment variable name, usually `NVIDIA_API_KEY`.
- The output directory to use.

For checkpoint evaluation with `default.yaml`, also provide:

- The concrete Megatron Bridge `iter_*` checkpoint path.
- The tokenizer path or Hugging Face tokenizer id if the selected task needs logprobs.

## Recommended First Run

Ask the agent to start with `tiny_chat.yaml` unless you explicitly need launcher-managed checkpoint deployment.
The first command should look like:

```bash
uv run --no-sync nemotron steps run eval/model_eval \
  -c tiny_chat \
  output_dir=./output/eval-tiny-chat \
  target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
  target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID" \
  target.api_endpoint.api_key_name=NVIDIA_API_KEY \
  target.api_endpoint.type=chat
```

## A Reusable Opening Brief

```text
Context: [one sentence on the model and what you want to score]
Goal for this session: [for example, a hosted chat smoke test that writes files on disk]
Endpoint URL: [full URL with path, or "I do not have this yet, please ask"]
Model identifier: [as the endpoint advertises it]
API key environment variable: [name only, for example NVIDIA_API_KEY]
Checkpoint path: [only if using default.yaml launcher deployment]
Tokenizer: [only if using log-probability tasks]
Hard limits: [for example, do not change endpoint type, do not fabricate values]
Please: Use `eval/model_eval` defaults from the repo unless something blocks that.
```

The agent should ask for missing fields instead of guessing.

## What Success Looks Like

A reasonable first success is the hosted chat smoke test described in {doc}`getting-started`.
The session reaches that point when:

- The agent issues one `nemotron steps run eval/model_eval -c tiny_chat ...` command.
- The command prints a `launcher_config` path.
- The output directory contains files after NeMo Evaluator Launcher completes.

If the launcher fails, the agent should report the error and the relevant config fields.
It should not retry with fabricated endpoint, model, or credential values.

## Next Steps

- Run the tutorial: {doc}`getting-started`.
- Pick a deployment path: {doc}`how-to/evaluate-deployed-checkpoint`.
- Run a hosted evaluation: {doc}`how-to/run-hosted-evaluation`.
- Look up flags and YAML fields: {doc}`reference/cli-reference` and {doc}`reference/config-schema`.
