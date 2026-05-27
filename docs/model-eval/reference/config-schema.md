<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-config-schema)=
# Configuration Reference

This page documents the YAML schema consumed by `nemotron steps run eval/model_eval`.
The step is a thin wrapper around NeMo Evaluator Launcher: it loads a YAML config, applies Hydra-style overrides, removes Nemotron-only keys, saves a launcher config, and calls `nemo_evaluator_launcher.api.functional.run_eval`.

## Sample Configs

| Config | Purpose |
| --- | --- |
| `tiny_chat.yaml` | Hosted chat endpoint smoke test. Uses `deployment.type: none`, `target.api_endpoint.*`, and one configured task, `mmlu_instruct`. |
| `default.yaml` | Megatron Bridge checkpoint evaluation through NeMo Evaluator Launcher. Uses launcher-managed `execution`, `deployment`, `evaluation`, and `tasks` sections. |

## Top-Level Keys

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml
:language: yaml
:class: scrollable
```

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/default.yaml
:language: yaml
:class: scrollable
```

| Key | Used By | Purpose |
| --- | --- | --- |
| `dry_run` | Nemotron runtime | Passed to NeMo Evaluator Launcher as `run_eval(..., dry_run=...)`. |
| `output_dir` | Nemotron runtime | Copied into `execution.output_dir` before launcher dispatch. |
| `task_filters` | Nemotron runtime | Optional task-name subset passed to launcher. |
| `run` | Nemotron runtime | Nemotron-side artifact, environment, and W&B interpolation. Removed before launcher dispatch. |
| `execution` | NeMo Evaluator Launcher | Where and how launcher execution runs. |
| `deployment` | NeMo Evaluator Launcher | How the evaluated model is deployed, or `type: none` for an existing endpoint. |
| `target` | NeMo Evaluator Launcher | Existing API endpoint metadata for hosted evaluation. |
| `evaluation` | NeMo Evaluator Launcher | Evaluator config, generation params, logging, caching, and adapter settings. |
| `tasks` | NeMo Evaluator Launcher | Exact task entries to run. Each entry has a `name`. |
| `export` | NeMo Evaluator Launcher | Optional export settings, such as W&B export. |

## Hosted Endpoint Fields

Use these fields with `tiny_chat.yaml` or any config that sets `deployment.type: none`.

| Field | Purpose |
| --- | --- |
| `target.api_endpoint.model_id` | Exact model id advertised by the endpoint. |
| `target.api_endpoint.url` | Full OpenAI-compatible endpoint URL, including `/v1/chat/completions` or `/v1/completions`. |
| `target.api_endpoint.api_key_name` | Environment variable name that holds the bearer token. Never put the secret value in config. |
| `target.api_endpoint.type` | Endpoint type, usually `chat` for hosted chat smoke tests. |

The `tiny_chat.yaml` file reads these values from `NEMO_EVALUATOR_MODEL_ID`, `NEMO_EVALUATOR_MODEL_URL`, `NEMO_EVALUATOR_API_KEY_NAME`, and `NEMO_EVALUATOR_ENDPOINT_TYPE`.

## Evaluation Params

Generation and evaluator controls live under:

```text
evaluation.nemo_evaluator_config.config.params
```

Common fields are:

| Field | Purpose |
| --- | --- |
| `temperature` | Sampling temperature for generation tasks. |
| `top_p` | Top-p nucleus sampling. |
| `max_new_tokens` | Maximum generated tokens for chat/instruction tasks. |
| `max_retries` | Request retry count. |
| `parallelism` | Request concurrency where supported. |
| `request_timeout` | Per-request timeout in seconds. |
| `limit_samples` | Optional per-task sample cap. Use `1` for smoke tests. |
| `extra.tokenizer` | Tokenizer path or Hugging Face id required by log-probability tasks. |
| `extra.tokenizer_backend` | Tokenizer backend, usually `huggingface`. |

## Tasks

Tasks are NeMo Evaluator Launcher task entries.
Use exact task IDs from the installed launcher, for example:

```bash
nemo-evaluator-launcher ls tasks
nemo-evaluator-launcher ls task mmlu_instruct
```

The sample configs define these starting points.

| Config | Tasks |
| --- | --- |
| `tiny_chat.yaml` | `mmlu_instruct` |
| `default.yaml` | `adlr_mmlu`, `hellaswag` |

Do not prepend a harness name unless the launcher lists that exact dotted task id.

## Checkpoint Deployment Fields

The `default.yaml` config uses launcher-managed deployment for a Megatron Bridge checkpoint.
The most common override is:

```bash
deployment.checkpoint_path=/path/to/iter_0001000
```

Use the concrete `iter_*` checkpoint directory, not just the parent training output directory.
For log-probability tasks, keep the tokenizer aligned with the deployed checkpoint through `evaluation.nemo_evaluator_config.config.params.extra.tokenizer`.

## Validation Behavior

Nemotron does not implement a separate benchmark loop for this step.
It validates only enough to build the launcher config and import NeMo Evaluator Launcher.
Endpoint checks, task validation, result writing, and launcher invocation state are owned by NeMo Evaluator Launcher.

## Related

- {doc}`cli-reference` for command-line flags and Hydra override syntax.
- {doc}`benchmarks-catalog` for task identifiers grouped by endpoint family.
- {doc}`output-artifacts` for the `eval_results` contract and the on-disk layout.
- {doc}`troubleshooting` for common launcher and config failures.
- `src/nemotron/steps/eval/model_eval/step.toml` for the full step contract.
