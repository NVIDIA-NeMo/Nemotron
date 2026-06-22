<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-troubleshooting)=
# Troubleshooting

This page maps common `eval/model_eval` failures to the config fields that usually need correction.
Nemotron builds a launcher config and calls NeMo Evaluator Launcher; task execution, endpoint checks, and result writing are owned by the launcher.

## Evaluator Extra Missing

Symptom:

```text
Error: nemo-evaluator-launcher is required for evaluation
Install with: uv sync --extra evaluator
```

Recovery:

```bash
uv sync --extra evaluator
```

Then rerun the same `nemotron steps run eval/model_eval` command with `uv run --no-sync`.

## Hosted Endpoint Fails

Most hosted failures come from one of these fields:

| Field | What To Check |
| --- | --- |
| `target.api_endpoint.url` | Full endpoint URL, including `/v1/chat/completions` or `/v1/completions`. |
| `target.api_endpoint.model_id` | Exact model id returned by the endpoint's models API or UI. |
| `target.api_endpoint.api_key_name` | Environment variable name, not the secret value. |
| `target.api_endpoint.type` | `chat` for chat tasks, `completions` for completions/logprob tasks. |

For hosted smoke tests, start with `tiny_chat.yaml` and `target.api_endpoint.type=chat`.

## Wrong Task For Endpoint Type

Chat tasks need a chat endpoint.
Log-probability tasks generally need a completions endpoint with logprobs support and a tokenizer.

If the launcher fails after endpoint setup, check:

```text
tasks
target.api_endpoint.type
evaluation.nemo_evaluator_config.config.params.extra.tokenizer
```

Use exact task IDs from:

```bash
nemo-evaluator-launcher ls tasks
```

## Bad Checkpoint Path

When using `default.yaml`, point `deployment.checkpoint_path` at a concrete Megatron Bridge `iter_*` directory.
Do not point it only at the parent training output directory.

```bash
deployment.checkpoint_path=/path/to/run/iter_0001000
```

For log-probability tasks, also verify:

```bash
evaluation.nemo_evaluator_config.config.params.extra.tokenizer=/path/to/run/iter_0001000/tokenizer
```

## Launcher Job State

The step prints launcher follow-up commands when the launcher returns an invocation id.

```text
status_command: nemo-evaluator-launcher status <id>
logs_command: nemo-evaluator-launcher logs <id>
```

Run those commands before changing config.
The launcher logs usually distinguish endpoint/authentication failures from task-schema failures.

## Related Pages

- {doc}`config-schema` for field names and config shape.
- {doc}`output-artifacts` for launcher config and result paths.
- {doc}`../explanation/tokenizer-alignment` for tokenizer alignment.
- {doc}`../explanation/endpoint-types-and-benchmarks` for endpoint/task pairing.
- `src/nemotron/steps/eval/model_eval/step.toml` for the documented error names.
