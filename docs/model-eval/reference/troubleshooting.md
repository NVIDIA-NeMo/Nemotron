<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-troubleshooting)=
# Troubleshooting

This page maps common `eval/model_eval` failures to the config fields that usually need correction.
In launcher mode, Nemotron builds a launcher config and calls NeMo Evaluator Launcher; task execution, endpoint checks, and result writing are owned by the launcher.
In direct mode, Nemotron runs the harness itself, so most failures surface in `<output_dir>/<task>/harness.log` and are summarized in `failures.txt`.

The first sections apply to both modes; direct-mode-specific symptoms follow under [Direct Mode](#direct-mode).

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

For hosted verification runs in launcher mode, start with `tiny_chat.yaml` and `target.api_endpoint.type=chat`.
In direct mode these fields are set from `EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE`, `EVAL_API_KEY_NAME`, and `EVAL_ENDPOINT_TYPE`.

## Wrong Task For Endpoint Type

Chat tasks need a chat endpoint.
Completions tasks need a completions endpoint, with logprobs support for multiple-choice tasks.
If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, use the generated tokenizer.

If the harness fails after endpoint setup, check:

```text
tasks
target.api_endpoint.type
evaluation.nemo_evaluator_config.config.params.extra.tokenizer
```

Use exact task IDs from the launcher registry in launcher mode, or from the harness image in direct mode:

```bash
uv run nemo-evaluator-launcher ls tasks
docker run --rm <harness-image> nemo-evaluator ls
```

## Bad Checkpoint Path

When using `default.yaml`, point `deployment.checkpoint_path` at a concrete Megatron Bridge `iter_*` directory.
Do not point it only at the parent training output directory.

```bash
deployment.checkpoint_path=/path/to/run/iter_0001000
```

Also verify the tokenizer setting:

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

## Direct Mode

Start with the two run-level files, then the failed task's log:

```bash
cat "$EVAL_RESULTS_DIR/summary.json"
cat "$EVAL_RESULTS_DIR/failures.txt"
tail -n 80 "$EVAL_RESULTS_DIR/<task>/harness.log"
```

`summary.json` records only `ok` or `failed(<code>)`; the cause is in `harness.log`.
`<output_dir>/<task>/eval_factory_metrics.json` reports `response_stats.status_codes` and `successful_count`, which separate authentication failures from timeouts.

### Every Request Returns 404

Cause: `EVAL_MODEL_HANDLE` does not equal the model server's `--served-model-name`.
Fix: set both from one variable.

### `RepositoryNotFoundError` Naming Your Served Model

Symptom: a Hugging Face Hub traceback reports that a repository named after your served model does not exist, and the step printed `Warning: no tokenizer configured`.

Cause: `EVAL_TOKENIZER` is unset, so lm-evaluation-harness falls back to loading the served model name as a Hugging Face repository, but that name does not resolve.
Fix: set `EVAL_TOKENIZER` to the served model's tokenizer repository id or a local snapshot path.
A tokenizer-extended checkpoint must use its own tokenizer, not the base model's.

### `Tokenizer class TokenizersBackend does not exist`

Current `convert/megatron_to_hf` exports normalize supported internal tokenizer class names to `PreTrainedTokenizerFast` automatically when `tokenizer.json` is present.
If an older or externally produced export still declares `"tokenizer_class": "TokenizersBackend"`, stock `transformers` cannot import it.
Copy `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` to a side directory, set `tokenizer_class` to `PreTrainedTokenizerFast`, remove `auto_map`, and point `EVAL_TOKENIZER` at that directory.
Refer to {doc}`../explanation/tokenizer-alignment`.

### Every Request Returns 401

Check, in order:

1. `EVAL_API_KEY_NAME` names the variable that holds the token, not the token itself; the default name is `ENDPOINT_TOKEN`.
1. The named variable is set in the submitting shell. The `*_eval_direct` profiles forward `ENDPOINT_TOKEN`; a differently named variable is not forwarded unless you add it to the profile's `env_vars`.
1. For an endpoint without authentication, set `EVAL_API_KEY_NAME=null`. Pointing the name at an unset variable is an error, not a no-op.

The sovereign container's MILU harness reads its token from `OPENAI_API_KEY` and ignores `--api_key_name`; direct mode exports `OPENAI_API_KEY` from the configured variable automatically.

### Run Never Finishes On `mmlu_prox_completions` Or `mmlu_prox_chat`

Cause: the task expanded to all 29 language subsets because `config.params.task` was not pinned.
The shipped `mmlu_prox` and `mmlu_prox_chat` suites pin it; a bare `-t mmlu_prox_completions` does not.
Fix: add `evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_<lang>`, one language per run.
The same applies to `global_mmlu` and `global_mmlu_full`.

### Task Not Found In The Harness Image

Symptom: the harness reports an unknown task even though `nemo-evaluator-launcher ls tasks` lists it, or the reverse.

Cause: in direct mode, the task set is whatever `harness_image` ships, and task sets differ between image tags.
For example, `sovereign:26.05` ships no `milu_*` tasks while `sovereign:latest` does, and both report the same version string.
Fix: list the image's tasks with `docker run --rm <harness-image> nemo-evaluator ls`, then pin the resolved digest in `EVAL_HARNESS_IMAGE` for repeatable runs.

### `unsupported nemo_evaluator_config params`

Symptom:

```text
unsupported nemo_evaluator_config params: [...]
Supported: [...]
Pass harness-specific arguments under `extra:`.
```

Cause: a top-level key under `config.params` is not in the direct-mode allowlist (`task`, `limit_samples`, `parallelism`, `request_timeout`, `max_retries`, `temperature`, `top_p`, `max_new_tokens`, `extra`).
Fix: move harness-specific arguments under `extra:`.

(model-eval-troubleshooting-profile-not-found)=
### `--batch` Profile Not Found

Cause: `env.toml` is generated and then ignored by Git, so a file created before the `*_eval_direct` profiles were added does not contain them.
Fix: generate a separate TOML file for your backend, then copy its generated `[<backend>_eval_direct]` table into your existing `env.toml`, preserving your site-specific values.

```console
$ profile_tmp_dir="$(mktemp -d)"
$ uv run nemotron steps run env/env_toml -c slurm \
    output_path="$profile_tmp_dir/env.slurm.toml"
```

Replace `slurm` with `lepton` or `dgxcloud` as needed, then copy the generated `[<backend>_eval_direct]` table up to, but not including, the next table header (or through end of file).
Do not copy the YAML template directly; it is generator input, not a runtime profile.
Confirm the result with `grep '^\[<backend>_eval_direct\]' env.toml`, replacing `<backend>` with the selected backend.
A CPU-only profile that inherits a GPU base may also inherit a shared-memory request that no CPU shape can satisfy, and the scheduler rejects the submission before a job exists; neither config generation nor `--dry-run` catches that.

### Output Directory Is Claimed Or Not Empty

Symptom:

```text
<output_dir> is claimed by another run (pid ... on ..., started ...)
```

Cause: `.nemotron-eval.lock` exists because a run is in progress, or a preempted job left its claim behind.
Fix: use a different `output_dir`, or delete the lock file if the run is definitely gone; the message reports the claim's age.
`overwrite=true` replaces non-empty task directories, but does not override a live claim.

### Harness Exits Client-Side While The Endpoint Returns 200

Cause: `adapter_config.use_caching` was disabled, so the harness held every request and response in memory; around 30 000 requests it exits on a large CPU shape while the endpoint is healthy.
Fix: keep `use_caching: true` (the `direct.yaml` default), and use `EVAL_LIMIT_SAMPLES` for large splits.
Caching removes the memory ceiling; it has not been shown to make an unbounded full split finish.

### `TimeoutError` Late In A Long Generative Task

Cause: `request_timeout` is a deadline measured from request creation, and lm-evaluation-harness creates every request up front behind a concurrency semaphore, so queued requests consume the clock while waiting.
Fix: raise `request_timeout` and `parallelism` together; lowering concurrency makes it worse.
For MCQA tasks on a base model, also set `config.params.max_new_tokens` to a small value; MILU's default of 1024 lets a model without a stop token generate to the cap.
Large MILU splits can still abort this way; smaller splits and `limit_samples` runs complete normally.

## Related Pages

- {doc}`config-schema` for field names and config shape.
- {doc}`output-artifacts` for launcher config and result paths.
- {doc}`../how-to/run-direct-mode-evaluation` for the direct-mode procedure.
- {doc}`../explanation/tokenizer-alignment` for tokenizer alignment.
- {doc}`../explanation/endpoint-types-and-benchmarks` for endpoint/task pairing.
- `src/nemotron/steps/eval/model_eval/step.toml` for the documented error names.
