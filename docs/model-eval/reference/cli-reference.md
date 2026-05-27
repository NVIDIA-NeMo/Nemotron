<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-cli-reference)=
# CLI Reference

This page documents the CLI surface for `nemotron steps run eval/model_eval`.
The flags are shared by every Nemotron step.
The override examples are specific to the `eval/model_eval` YAML schema.

## Syntax

```bash
uv run nemotron steps run eval/model_eval [FLAGS] [HYDRA_OVERRIDES...]
```

Run the command from the repository root after `uv sync --extra evaluator`.
Pass the configuration name with `-c`, per-step overrides as `key=value` dotlists, and optional execution flags.

## Flags

| Flag | Long form | Purpose |
| --- | --- | --- |
| `-c` | `--config` | Config name inside `src/nemotron/steps/eval/model_eval/config/`, such as `default` or `tiny_chat`. Accepts a path to a YAML file. |
| `-r` | `--run` | Attached execution by using an environment profile defined in `env.toml`. |
| `-b` | `--batch` | Detached execution by using an environment profile defined in `env.toml`. |
| `-d` | `--dry-run` | Compile the Nemotron job config and exit without dispatching. |
| | `--force-squash` | Force re-squash of the container image when the selected backend builds one. |

Invoking the command without `-c` resolves the runspec default, `default.yaml`.

## Common Overrides

| Override | Purpose |
| --- | --- |
| `output_dir=<path>` | Base output directory. The runtime also writes this into `execution.output_dir` before calling NeMo Evaluator Launcher. |
| `dry_run=true` | Pass dry-run mode to NeMo Evaluator Launcher. This is different from CLI `--dry-run`, which only compiles the Nemotron job. |
| `task_filters=[<task>,...]` | Optional subset of configured task names passed to NeMo Evaluator Launcher. |
| `target.api_endpoint.url=<url>` | OpenAI-compatible endpoint URL for hosted evaluation when `deployment.type=none`. |
| `target.api_endpoint.model_id=<id>` | Exact model id advertised by the hosted endpoint. |
| `target.api_endpoint.api_key_name=<env-var-name>` | Name of the environment variable holding the bearer token. This is the variable name, not the secret. |
| `target.api_endpoint.type=<chat|completions>` | Endpoint type expected by the selected task. |
| `evaluation.nemo_evaluator_config.config.params.limit_samples=<int>` | Per-task sample cap for smoke tests. |
| `evaluation.nemo_evaluator_config.config.params.parallelism=<int>` | Concurrent requests issued by the evaluator where supported. |
| `evaluation.nemo_evaluator_config.config.params.request_timeout=<int>` | Per-request timeout in seconds. |
| `evaluation.nemo_evaluator_config.config.params.extra.tokenizer=<path-or-id>` | Tokenizer used by log-probability tasks such as HellaSwag. |
| `deployment.checkpoint_path=<iter_* path>` | Megatron Bridge checkpoint path used by `default.yaml` launcher deployment. |
| `deployment.image=<container>` | Container image used by the launcher deployment in `default.yaml`. |

## Discovery Commands

```bash
uv run --no-sync nemotron steps list --category eval --json
uv run --no-sync nemotron steps show eval/model_eval --json
```

`nemotron steps show eval/model_eval --json` prints the full step contract, including `consumes`, `produces`, `parameters`, `strategies`, and `errors`.

## Examples

### Hosted Chat Smoke Test

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY}"
: "${NEMO_EVALUATOR_MODEL_URL:?Set the chat-completions endpoint URL}"
: "${NEMO_EVALUATOR_MODEL_ID:?Set the endpoint model id}"

uv run --no-sync nemotron steps run eval/model_eval \
  -c tiny_chat \
  output_dir=./output/eval-tiny-chat \
  target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
  target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID" \
  target.api_endpoint.api_key_name=NVIDIA_API_KEY \
  target.api_endpoint.type=chat \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

### Megatron Checkpoint Evaluation Config

Use `default.yaml` when NeMo Evaluator Launcher should deploy a Megatron Bridge checkpoint and then run the configured tasks.

```bash
uv run --no-sync nemotron steps run eval/model_eval \
  -c default \
  output_dir=./output/eval-megatron \
  deployment.checkpoint_path=/path/to/checkpoint/iter_0001000 \
  evaluation.nemo_evaluator_config.config.params.limit_samples=1
```

### Compile Without Dispatching

```bash
uv run --no-sync nemotron steps run eval/model_eval -d -c tiny_chat \
  target.api_endpoint.url="$NEMO_EVALUATOR_MODEL_URL" \
  target.api_endpoint.model_id="$NEMO_EVALUATOR_MODEL_ID"
```

### Launcher Dry Run

```bash
uv run --no-sync nemotron steps run eval/model_eval -c tiny_chat dry_run=true
```

## Related

- {doc}`config-schema` for the YAML schema accepted by `-c` and dotlist overrides.
- {doc}`output-artifacts` for the on-disk layout produced under `output_dir`.
- {doc}`../how-to/run-hosted-evaluation` for a procedural walk-through.
- {doc}`../how-to/discover-the-step` for the step-contract discovery commands.
