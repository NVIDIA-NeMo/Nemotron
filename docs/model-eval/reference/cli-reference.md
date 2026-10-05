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
uv run nemotron steps run eval/model_eval [FLAGS] [-t TASK ...] [HYDRA_OVERRIDES...]
```

Run the command from the repository root after `uv sync --extra evaluator`.
Pass the configuration name with `-c`, task names with repeatable `-t`, per-step overrides as `key=value` dotlists, and optional execution flags.

The config selected with `-c` decides the execution mode: `default` and `tiny_chat` run in launcher mode; `direct` and the suite configs (`base_en`, `instruct_en`, `mmlu_prox`, `mmlu_prox_chat`, `milu`) run in direct mode.
Refer to {ref}`model-eval-choose-a-mode`.

## Flags

| Flag | Long form | Purpose |
| --- | --- | --- |
| `-c` | `--config` | Config name inside `src/nemotron/steps/eval/model_eval/config/`, such as `default`, `tiny_chat`, `direct`, or a suite name. Accepts a path to a YAML file. |
| `-t` | `--task` | Task name to run. Repeatable. Direct mode runs exactly the supplied list, including names absent from `evaluation.tasks`; launcher mode passes the list to NeMo Evaluator Launcher as `task_filters`. Any other passthrough argument without `=` is rejected. |
| `-r` | `--run` | Attached execution by using an environment profile defined in `env.toml`. |
| `-b` | `--batch` | Detached execution by using an environment profile defined in `env.toml`. Direct mode uses `<backend>_eval_direct` profiles: `slurm_eval_direct`, `lepton_eval_direct`, `dgxcloud_eval_direct`. |
| `-d` | `--dry-run` | Compile the Nemotron job config and exit without dispatching. |
| | `--force-squash` | Force re-squash of the container image when the selected backend builds one. |

Invoking the command without `-c` resolves the runspec default, `default.yaml`.

## Direct-Mode Environment Variables

`direct.yaml` and the suite configs read these variables with `${oc.env:...}` when the config is resolved inside the job.
The `*_eval_direct` profiles forward the endpoint, tokenizer, result, limit, and credential variables plus `ENDPOINT_TOKEN` from the submitting shell into the job.
`direct.yaml` resolves and forwards `EVAL_HARNESS_IMAGE` itself.

| Variable | Required | Purpose |
| --- | --- | --- |
| `EVAL_ENDPOINT_URL` | yes | Full OpenAI-compatible URL, `.../v1/completions` or `.../v1/chat/completions`. |
| `EVAL_MODEL_HANDLE` | yes | Model id; must equal the server's `--served-model-name`. |
| `EVAL_TOKENIZER` | when the model name is an alias, the tokenizer is at a custom path, or Tokenizer Extension was used | Hugging Face repo id or local path. Tokenizer Extension must use the generated tokenizer. |
| `EVAL_RESULTS_DIR` | yes | Durable output root; becomes `output_dir`. |
| `EVAL_API_KEY_NAME` | no; default `ENDPOINT_TOKEN` | Name of the variable that holds the bearer token, not the token. `null` disables authentication. |
| `ENDPOINT_TOKEN` | when the endpoint is authenticated | The token, under the default `EVAL_API_KEY_NAME`. |
| `EVAL_ENDPOINT_TYPE` | no; default `completions` | `completions` or `chat`. The chat suites set `chat` in YAML. |
| `EVAL_HARNESS_IMAGE` | no; default `nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03`, or the sovereign image for `milu` | Harness image; pin by digest for published results. |
| `EVAL_LIMIT_SAMPLES` | no; default unset | Deterministic first-N sample cap per task. |
| `MMLU_PROX_LANG` | no; default `en` | Language subset for the `mmlu_prox` and `mmlu_prox_chat` suites. Not forwarded by the shipped profiles; the `config.params.task` override is the portable alternative. |
| `MILU_LANG` | no; default `Hindi` | Target language for the `milu` suite. Not forwarded by the shipped profiles; `-t milu_<Language>` is the portable alternative. |

## Common Overrides

| Override | Purpose |
| --- | --- |
| `output_dir=<path>` | Base output directory. In launcher mode the runtime also writes this into `execution.output_dir`; in direct mode it defaults to `EVAL_RESULTS_DIR`. |
| `dry_run=true` | Step-level dry run. Launcher mode passes it to NeMo Evaluator Launcher; direct mode prints each harness command and writes `summary.dry-run.json` and `run_manifest.dry-run.json`. Different from CLI `--dry-run`, which only compiles the Nemotron job. |
| `overwrite=true` | Direct mode only. Replace non-empty task directories instead of refusing. |
| `task_filters=[<task>,...]` | Task names as a list; equivalent to repeated `-t`. |
| `evaluation.nemo_evaluator_config.config.params.task=<subset>` | Pin a multi-subset task, for example `mmlu_prox_hi`. Global to the run. |
| `target.api_endpoint.url=<url>` | OpenAI-compatible endpoint URL for hosted evaluation when `deployment.type=none`. |
| `target.api_endpoint.model_id=<id>` | Exact model id advertised by the hosted endpoint. |
| `target.api_endpoint.api_key_name=<env-var-name>` | Name of the environment variable holding the bearer token. This is the variable name, not the secret. |
| `target.api_endpoint.type=<chat|completions>` | Endpoint type expected by the selected task. |
| `evaluation.nemo_evaluator_config.config.params.limit_samples=<int>` | Per-task sample cap for verification runs. |
| `evaluation.nemo_evaluator_config.config.params.parallelism=<int>` | Concurrent requests issued by the evaluator where supported. |
| `evaluation.nemo_evaluator_config.config.params.request_timeout=<int>` | Per-request timeout in seconds. |
| `evaluation.nemo_evaluator_config.config.params.extra.tokenizer=<path-or-id>` | Explicit tokenizer for a model alias, custom tokenizer path, or tokenizer-extended model. |
| `harness_image=<image>` | Direct mode only. Harness image; overrides `EVAL_HARNESS_IMAGE`. |
| `deployment.checkpoint_path=<iter_* path>` | Megatron Bridge checkpoint path used by `default.yaml` launcher deployment. |
| `deployment.image=<container>` | Container image used by the launcher deployment in `default.yaml`. |

## Discovery Commands

```bash
uv run --no-sync nemotron steps list --category eval --json
uv run --no-sync nemotron steps show eval/model_eval --json
```

`nemotron steps show eval/model_eval --json` prints the full step contract, including `consumes`, `produces`, `parameters`, `strategies`, and `errors`.

## Examples

### Direct-Mode Verification Run

```bash
: "${EVAL_ENDPOINT_URL:?Set the completions endpoint URL}"
: "${EVAL_MODEL_HANDLE:?Set the served model name}"
: "${EVAL_RESULTS_DIR:?Set a durable results directory}"
: "${ENDPOINT_TOKEN:?Set the endpoint token}"

EVAL_LIMIT_SAMPLES=5 uv run nemotron steps run eval/model_eval \
  -c direct --batch slurm_eval_direct \
  -t hellaswag
```

### Direct-Mode Suite With A Pinned Subset

```bash
uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
  -t mmlu_prox_completions \
  evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_hi
```

### Direct-Mode Dry Run

```bash
uv run nemotron steps run eval/model_eval -c base_en --batch slurm_eval_direct dry_run=true
```

### Hosted Chat Verification Run (Launcher Mode)

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
- {doc}`../how-to/run-direct-mode-evaluation` for the direct-mode procedure.
- {doc}`../how-to/run-hosted-evaluation` for the launcher-mode procedure.
- {doc}`../how-to/discover-the-step` for the step-contract discovery commands.
