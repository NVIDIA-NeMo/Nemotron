<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-config-schema)=
# Configuration Reference

This page documents the YAML schema consumed by `nemotron steps run eval/model_eval`.
The step loads a YAML config, applies Hydra-style overrides, and then branches on `mode`.
In launcher mode it removes Nemotron-only keys, saves a launcher config, and calls `nemo_evaluator_launcher.api.functional.run_eval`.
In direct mode it resolves the endpoint and task list from the environment and runs `nemo-evaluator run_eval` per task in the current job.

Keys are grouped below as shared, launcher-only, or direct-only.

## Sample Configs

| Config | Mode | Purpose |
| --- | --- | --- |
| `tiny_chat.yaml` | launcher | Hosted chat endpoint verification run. Uses `deployment.type: none`, `target.api_endpoint.*`, and one configured task, `mmlu_instruct`. |
| `default.yaml` | launcher | Megatron Bridge checkpoint evaluation through NeMo Evaluator Launcher. Uses launcher-managed `execution`, `deployment`, `evaluation`, and `tasks` sections. |
| `direct.yaml` | direct | Base config for direct mode. Reads the endpoint from `EVAL_*` environment variables and ships two default tasks, `adlr_arc_challenge_llama_25_shot` and `hellaswag`. |
| `base_en.yaml` | direct | `defaults: direct.yaml` plus the `base_en` task list. |
| `mmlu_prox.yaml` | direct | `defaults: direct.yaml` plus `mmlu_prox_completions`, with `config.params.task` pinned from `MMLU_PROX_LANG`. |
| `milu.yaml` | direct | `defaults: direct.yaml` plus the MILU tasks; overrides `harness_image` with the sovereign image. |
| `instruct_en.yaml` | direct | `defaults: direct.yaml`, `target.api_endpoint.type: chat`, deterministic generation parameters, and the `instruct_en` task list. |
| `mmlu_prox_chat.yaml` | direct | `defaults: direct.yaml`, `target.api_endpoint.type: chat`, deterministic generation parameters, and `mmlu_prox_chat` pinned from `MMLU_PROX_LANG`. |

The suite configs use `defaults: direct.yaml` to inherit every key in the base file and override only what differs.

## Top-Level Keys

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/direct.yaml
:language: yaml
:class: scrollable
```

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml
:language: yaml
:class: scrollable
```

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/default.yaml
:language: yaml
:class: scrollable
```

### Shared Keys

| Key | Purpose |
| --- | --- |
| `mode` | `launcher` (default) or `direct`. Selects which runtime path the step follows. |
| `dry_run` | In launcher mode, passed to NeMo Evaluator Launcher as `run_eval(..., dry_run=...)`. In direct mode, prints each resolved harness command and writes `summary.dry-run.json` and `run_manifest.dry-run.json` without evaluating. |
| `output_dir` | Where results are written. In launcher mode, copied into `execution.output_dir`. In direct mode, the root that receives `summary.json`, `run_manifest.json`, and one directory per task; must be durable storage. |
| `task_filters` | Task names to run; also settable per invocation with repeatable `-t`. In launcher mode, a subset of the configured tasks. In direct mode, used exactly as supplied, so a name absent from `evaluation.tasks` still runs. |
| `run` | Nemotron-side execution environment, artifact interpolation, and W&B settings. Removed before launcher dispatch. |
| `target.api_endpoint.*` | The endpoint the harness calls. Refer to [Endpoint Fields](#endpoint-fields). |
| `evaluation.nemo_evaluator_config.config.params.*` | Generation and request parameters. Refer to [Evaluation Params](#evaluation-params). |
| `evaluation.tasks` | Task entries to run. Each entry has a `name` and may carry its own `nemo_evaluator_config`. |

### Launcher-Only Keys

| Key | Purpose |
| --- | --- |
| `run.env.executor` | The outer scheduler that runs this step; overridden by `--run` and `--batch`. |
| `run.env.launcher_executor` | The inner scheduler that NeMo Evaluator Launcher submits to: `local`, `slurm`, or `lepton`. Feeds `execution.type`. |
| `run.env.container_image` | Image used by the `deployment` block. |
| `run.env.host`, `run.env.user`, `run.env.account`, `run.env.partition`, `run.env.remote_job_dir`, `run.env.time` | Slurm connection and allocation values interpolated into `execution.*`. |
| `run.wandb.entity`, `run.wandb.project` | W&B identity, read only by the `export.wandb` block. Commented out by default. |
| `execution.type` | Launcher executor type. |
| `execution.output_dir` | Populated from `output_dir`. |
| `execution.walltime`, `execution.num_nodes`, `execution.deployment.n_tasks`, `execution.mounts.*` | Allocation and mount settings for the launcher executor. |
| `execution.auto_export` | Optional list of exporters to run after evaluation; commented out by default. |
| `deployment.type` | `generic` to have the launcher deploy the checkpoint with `deployment.command`, or `none` for an existing endpoint. |
| `deployment.image`, `deployment.checkpoint_path`, `deployment.port`, `deployment.served_model_name`, `deployment.health_check_path`, `deployment.command`, `deployment.endpoints.*`, `deployment.env_vars.*`, `deployment.multiple_instances` | How the launcher serves a Megatron Bridge checkpoint. `checkpoint_path` must be a concrete `iter_*` directory. |
| `export.wandb.*` | W&B export target. Its presence is treated as a request for W&B credentials, so leave the block commented out unless you intend to export. |

### Direct-Only Keys

| Key | Default in `direct.yaml` | Purpose |
| --- | --- | --- |
| `harness_image` | `${oc.env:EVAL_HARNESS_IMAGE,nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03}` | The container image that ships the harness. Becomes `run.env.container_image` and is recorded as `harness.image` in `run_manifest.json`, together with whether it was pinned by digest. |
| `overwrite` | `false` | A non-empty task directory is refused by default. Set `true` to replace those directories; the plan is validated before anything is deleted, and a dry run never deletes. |
| `run.env.env_vars.EVAL_HARNESS_IMAGE` | `${run.env.container_image}` | Forwards the resolved image into the job so the manifest records it. |
| `evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config.*` | see below | Adapter proxy between the harness and the endpoint. Refer to [Adapter Configuration](#adapter-configuration). |

## Endpoint Fields

| Field | Purpose |
| --- | --- |
| `target.api_endpoint.model_id` | Exact model id advertised by the endpoint. Must equal the server's `--served-model-name`. |
| `target.api_endpoint.url` | Full OpenAI-compatible endpoint URL, including `/v1/chat/completions` or `/v1/completions`. |
| `target.api_endpoint.api_key_name` | Environment variable name that holds the bearer token. Never put the secret value in config. |
| `target.api_endpoint.type` | `chat` or `completions`. Direct mode rejects other values. |

Each config reads these values from a different set of environment variables.

| Field | `tiny_chat.yaml` | `direct.yaml` and suites | Direct default |
| --- | --- | --- | --- |
| `model_id` | `NEMO_EVALUATOR_MODEL_ID` | `EVAL_MODEL_HANDLE` | none; required |
| `url` | `NEMO_EVALUATOR_MODEL_URL` | `EVAL_ENDPOINT_URL` | none; required |
| `api_key_name` | `NEMO_EVALUATOR_API_KEY_NAME` (default `NVIDIA_API_KEY`) | `EVAL_API_KEY_NAME` | `ENDPOINT_TOKEN` |
| `type` | `NEMO_EVALUATOR_ENDPOINT_TYPE` (default `chat`) | `EVAL_ENDPOINT_TYPE` | `completions`; the chat suites set `chat` in YAML |

Direct mode resolves `${oc.env:...}` inside the job, so the `*_eval_direct` profile must forward every variable the config reads.
The shipped profiles forward `EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE`, `EVAL_RESULTS_DIR`, `EVAL_ENDPOINT_TYPE`, `EVAL_API_KEY_NAME`, `EVAL_TOKENIZER`, `EVAL_LIMIT_SAMPLES`, and `ENDPOINT_TOKEN`.
`EVAL_HARNESS_IMAGE` is the exception: `direct.yaml` resolves it into `run.env.container_image` and forwards the resolved value itself, so the profiles intentionally omit it.

## Evaluation Params

Generation and evaluator controls live under:

```text
evaluation.nemo_evaluator_config.config.params
```

| Field | `direct.yaml` default | Purpose |
| --- | --- | --- |
| `task` | unset; `mmlu_prox_${MMLU_PROX_LANG}` in the `mmlu_prox*` suites | Harness-level subset for a multi-subset task. Global to the run unless set per task. |
| `temperature` | unset; `0.0` in the chat suites | Sampling temperature for generation tasks. |
| `top_p` | unset; `1.0` in the chat suites | Top-p nucleus sampling. |
| `max_new_tokens` | unset; `2048` in the chat suites | Maximum generated tokens for chat/instruction tasks. |
| `max_retries` | `5` | Request retry count. |
| `parallelism` | `16` | Request concurrency where supported. |
| `request_timeout` | `3600` | Per-request timeout in seconds. |
| `limit_samples` | `${oc.decode:${oc.env:EVAL_LIMIT_SAMPLES,null}}` | Deterministic first-N sample cap. A limited run is a subset, not a full-split score. |
| `extra.tokenizer` | `${oc.env:EVAL_TOKENIZER,null}` | Loaded automatically when the served model name is its Hugging Face model ID. Set it for an alias, a custom tokenizer path, or Tokenizer Extension; use the generated tokenizer for Tokenizer Extension. |
| `extra.tokenizer_backend` | `huggingface` | Tokenizer backend. |
| `extra.tokenized_requests` | `false` | Passed through to the lm-evaluation-harness `tokenized_requests` option. Every shipped config sets `false`. |
| `extra.args` | unset | Harness-specific command-line arguments, for example `'--confirm_run_unsafe_code'` for code benchmarks. |

Direct mode accepts only `task`, `limit_samples`, `parallelism`, `request_timeout`, `max_retries`, `temperature`, `top_p`, `max_new_tokens`, and `extra` at the top level of `params`.
Any other key stops the run with `unsupported nemo_evaluator_config params`.
Harness-specific arguments belong under `extra`.

Per-task values in `evaluation.tasks[].nemo_evaluator_config.config.params` override the global values; the merged result per task is recorded in `run_manifest.json`.

## Adapter Configuration

Direct mode forwards `evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config` to the harness, which runs an adapter proxy between itself and the endpoint.

| Field | `direct.yaml` default | Purpose |
| --- | --- | --- |
| `use_caching` | `true` | Cache requests and responses on disk. Leave enabled: without it the harness holds every request and response for a task in memory. |
| `output_dir` | `null` | Cache and log location. `null` places it under the task's output directory as `adapter/`. |
| `use_request_logging` | `true` | Log a sample of requests. |
| `max_logged_requests` | `10` | Number of requests to log. |
| `use_response_logging` | `true` | Log a sample of responses. |
| `max_logged_responses` | `10` | Number of responses to log. |
| `use_progress_tracking` | `false` | Progress reporting from the adapter. |

`default.yaml` sets the same fields for launcher mode with `output_dir: /results` and `caching_dir: /results/cache`.
The adapter configuration is recorded in `run_manifest.json`, so a cached run and an uncached run are distinguishable afterwards.

## Tasks

Tasks are entries under `evaluation.tasks`.
In launcher mode, use exact task ids from the launcher registry.
In direct mode, the harness image is authoritative.

```bash
uv run nemo-evaluator-launcher ls tasks
uv run nemo-evaluator-launcher ls task mmlu_instruct --json
docker run --rm <harness-image> nemo-evaluator ls
```

The sample configs define these starting points.

| Config | Tasks |
| --- | --- |
| `tiny_chat.yaml` | `mmlu_instruct` |
| `default.yaml` | `adlr_mmlu`, `hellaswag` |
| `direct.yaml` | `adlr_arc_challenge_llama_25_shot`, `hellaswag` |
| `base_en.yaml` | `adlr_arc_challenge_llama_25_shot`, `hellaswag`, `gsm8k` |
| `mmlu_prox.yaml` | `mmlu_prox_completions` |
| `milu.yaml` | `milu_${MILU_LANG}` (default `milu_Hindi`), `milu_English` |
| `instruct_en.yaml` | `ifeval`, `mmlu_instruct`, `gsm8k_cot_instruct`; `humaneval_instruct` commented out |
| `mmlu_prox_chat.yaml` | `mmlu_prox_chat` |

Do not prepend a harness name unless the launcher lists that exact dotted task id.

A task entry can pin a multi-subset task without affecting other tasks in the same run.

```yaml
evaluation:
  tasks:
    - name: mmlu_prox_completions
      nemo_evaluator_config: {config: {params: {task: mmlu_prox_hi}}}
```

## Checkpoint Deployment Fields

The `default.yaml` config uses launcher-managed deployment for a Megatron Bridge checkpoint.
The most common override is:

```bash
deployment.checkpoint_path=/path/to/iter_0001000
```

Use the concrete `iter_*` checkpoint directory, not only the parent training output directory.
Keep the tokenizer aligned with the deployed checkpoint through `evaluation.nemo_evaluator_config.config.params.extra.tokenizer`; `default.yaml` sets it to `${deployment.checkpoint_path}/tokenizer`.

Direct mode has no deployment fields.
It requires `target.api_endpoint.url` and `target.api_endpoint.model_id` and exits with `direct mode needs target.api_endpoint.url and .model_id` when either is unset.

## Validation Behavior

In launcher mode, Nemotron validates only enough to build the launcher config and import NeMo Evaluator Launcher, then runs the launcher's own schema checks so that `dry_run=true` fails in the same way as a real run.
Endpoint checks, task validation, result writing, and launcher invocation state are owned by NeMo Evaluator Launcher.

In direct mode, Nemotron validates before any task starts: the endpoint URL and model id are present, the endpoint type is `chat` or `completions`, every task name is a plain directory name, every `params` key is in the supported set, and no target task directory is non-empty unless `overwrite=true`.
A failed preflight leaves no new output directory behind.

## Related

- {doc}`cli-reference` for command-line flags, `-t`, environment variables, and Hydra override syntax.
- {doc}`benchmarks-catalog` for suites and task identifiers grouped by endpoint family.
- {doc}`output-artifacts` for the `eval_results` contract and the on-disk layout.
- {doc}`troubleshooting` for common launcher and config failures.
- `src/nemotron/steps/eval/model_eval/step.toml` for the full step contract.
