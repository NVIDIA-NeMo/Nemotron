<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-run-hosted-evaluation)=
# Run A Hosted Evaluation

This guide runs `eval/model_eval` against an already-running OpenAI-compatible endpoint.
For the out-of-the-box hosted smoke test, use `tiny_chat.yaml`.

## Prerequisites

- The Nemotron repository is synced with `uv sync --extra evaluator`.
- A reachable endpoint URL.
- The endpoint's advertised model id.
- A credential exported as an environment variable.

## Choose The Starting Config

| Config | Use |
| --- | --- |
| `tiny_chat.yaml` | Hosted chat smoke test. Runs `mmlu_instruct` with `limit_samples: 1`. |
| `default.yaml` | Launcher-managed checkpoint deployment and evaluation. Use this when the launcher should deploy a Megatron Bridge checkpoint. |

Hosted chat QA should start with `tiny_chat.yaml`.

## Set Endpoint Values

```bash
export NVIDIA_API_KEY="<your-api-key>"
export NEMO_EVALUATOR_MODEL_URL="<full-chat-completions-url>"
export NEMO_EVALUATOR_MODEL_ID="<endpoint-model-id>"
export NEMO_EVALUATOR_ENDPOINT_TYPE=chat
```

The config stores the API key environment variable name, not the secret value.

## Run And Stream Output

```bash
uv run --no-sync nemotron steps run eval/model_eval \
  -c tiny_chat \
  output_dir=./output/eval-tiny-chat \
  target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
  target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID" \
  target.api_endpoint.api_key_name=NVIDIA_API_KEY \
  target.api_endpoint.type=chat \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

The step saves the launcher config and then calls NeMo Evaluator Launcher.
If the launcher returns an invocation id, the command prints the status and log commands to run next.

To compile the Nemotron job without invoking the launcher:

```bash
uv run --no-sync nemotron steps run eval/model_eval -d -c tiny_chat \
  target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
  target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID"
```

To invoke the launcher in its own dry-run mode:

```bash
uv run --no-sync nemotron steps run eval/model_eval -c tiny_chat dry_run=true
```

## Validate Output Artifacts

After the run completes, list files under `output_dir`.

```bash
find ./output/eval-tiny-chat -maxdepth 5 -type f | sort
```

The exact file set is owned by NeMo Evaluator Launcher and can vary by task version.

## Related

- {doc}`discover-the-step` for discovery commands.
- {doc}`evaluate-deployed-checkpoint` for the launcher-managed checkpoint path.
- {doc}`../reference/cli-reference` for the full flag and override surface.
- {doc}`../reference/config-schema` for the YAML field reference.
