<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-run-direct-mode-evaluation)=
# Run A Direct-Mode Evaluation

This guide runs `eval/model_eval` in *direct mode*: the evaluation harness runs inside the step's own job, scheduled by a Nemotron `--batch` profile, against an OpenAI-compatible endpoint that you host.
Direct mode never creates or tears down an endpoint.
Use it when the model is already served, when the backend is one that NeMo Evaluator Launcher has no executor for (Run:ai and DGX Cloud), or when a benchmark exists in a harness image but not in the launcher registry.
For the decision between the two modes, refer to {ref}`model-eval-choose-a-mode`.

## Prerequisites

- The Nemotron repository is synced with `uv sync --extra evaluator`.
- A running OpenAI-compatible endpoint that exposes `/v1/completions` or `/v1/chat/completions`, is reachable from the cluster where the evaluation job runs, and has a context length at least as long as the longest few-shot prompt a task builds.
- The value that the server was started with as `--served-model-name`.
- An explicit tokenizer ID, or a local tokenizer path visible inside the evaluation job, if the served model name is an alias, the tokenizer is at a custom path, or you used Tokenizer Extension.
- A directory on shared storage for results.
- An `env.toml` that contains the `<backend>_eval_direct` profile for your backend: `slurm_eval_direct`, `lepton_eval_direct`, or `dgxcloud_eval_direct`.
  For Slurm, check with `grep '^\[slurm_eval_direct\]' env.toml`; substitute your backend elsewhere in this example.
  If the profile is missing, generate a separate TOML file:

  ```console
  $ profile_tmp_dir="$(mktemp -d)"
  $ uv run nemotron steps run env/env_toml -c slurm \
      output_path="$profile_tmp_dir/env.slurm.toml"
  ```

  Replace `slurm` with `lepton` or `dgxcloud` as needed, then copy the generated `[<backend>_eval_direct]` table up to, but not including, the next table header (or through end of file) into your existing `env.toml`, preserving its site-specific values.
  Do not copy the YAML template directly; it is generator input, not a runtime profile.
  Confirm the result, for example:

  ```console
  $ grep '^\[slurm_eval_direct\]' env.toml
  [slurm_eval_direct]
  ```

## Set The Endpoint Values

Direct mode reads its endpoint from environment variables, which the `*_eval_direct` profiles forward into the submitted job.
Exporting a variable in your shell affects submission only; the profile is what carries it to the job.

```bash
export EVAL_ENDPOINT_URL="https://<host>/v1/completions"
export EVAL_MODEL_HANDLE="<served-model-name>"
export EVAL_RESULTS_DIR="/mnt/shared/eval/<run-name>"
export EVAL_API_KEY_NAME=ENDPOINT_TOKEN
export ENDPOINT_TOKEN="<endpoint-token>"
```

If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, use the generated tokenizer:

```bash
export EVAL_TOKENIZER="<generated-tokenizer-repo-id-or-path>"
```

| Variable | Requirement |
| --- | --- |
| `EVAL_ENDPOINT_URL` | Required. The full URL, including the `/v1/completions` or `/v1/chat/completions` path. |
| `EVAL_MODEL_HANDLE` | Required. Must equal the server's `--served-model-name`; otherwise every request returns HTTP 404. |
| `EVAL_TOKENIZER` | Loaded automatically when the served model name is its Hugging Face model ID. Set it for an alias, a custom tokenizer path, or Tokenizer Extension; use the generated tokenizer for Tokenizer Extension. |
| `EVAL_RESULTS_DIR` | Required. Storage that outlives the job, not container disk. |
| `EVAL_API_KEY_NAME` | The *name* of the variable that holds the token, never the token itself. Defaults to `ENDPOINT_TOKEN`. NeMo Evaluator fails if the named variable is unset, so for an endpoint without authentication set `EVAL_API_KEY_NAME=null` explicitly. |
| `EVAL_ENDPOINT_TYPE` | `completions` (default) or `chat`. The chat suites set `chat` in their config, so this variable is needed only with `-c direct`. |

## Select The Harness Image

The harness image decides which tasks exist.
Set `EVAL_HARNESS_IMAGE` in the submitting shell; do not add it to the `*_eval_direct` profile.
`direct.yaml` defaults `EVAL_HARNESS_IMAGE` to `nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03`; the `milu` suite overrides it with the sovereign image.
`direct.yaml` resolves the selected image into `run.env.container_image` and forwards that resolved value into the job, so the `*_eval_direct` profiles intentionally omit `EVAL_HARNESS_IMAGE`.
Task sets differ between tags of the same image family, so list the tasks in the image you intend to use.

```bash
docker run --rm <harness-image> nemo-evaluator ls
```

For any number you publish, pin the image by digest and record it.
`run_manifest.json` records whether the image was digest-pinned.

```bash
docker buildx imagetools inspect <image>:<tag> --format '{{.Manifest.Digest}}'
export EVAL_HARNESS_IMAGE="<image>@sha256:<digest>"
```

## Run A Verification Run

Confirm the endpoint, credential, model handle, and any explicit tokenizer on a small sample before a full run.
`EVAL_LIMIT_SAMPLES` selects a deterministic first-N subset.

```bash
EVAL_LIMIT_SAMPLES=5 uv run nemotron steps run eval/model_eval \
  -c direct --batch slurm_eval_direct \
  -t hellaswag
```

`-t <task>` is repeatable and, in direct mode, the list is used exactly as supplied: a name that is not in `evaluation.tasks` still runs, so any task the harness image ships can be requested without editing a config.

To see the resolved harness commands without evaluating, add `dry_run=true`.
This writes `summary.dry-run.json` and `run_manifest.dry-run.json` and never overwrites a real result.
`-d` (`--dry-run`) is different: it compiles the Nemotron job and returns before the step runs.
Neither option proves that the scheduler accepts the submission; only a real submission does.

## Run The Evaluation

Direct mode runs one job per endpoint with every requested task inside it, so a sweep loops over checkpoints, not over benchmarks.
Use a shipped suite with `-c <suite>`, or `-c direct` with explicit `-t` flags.

```bash
export EVAL_HARNESS_IMAGE="nvcr.io/nvidia/eval-factory/lm-evaluation-harness@sha256:<digest>"
export EVAL_ENDPOINT_TYPE=completions
export EVAL_API_KEY_NAME=ENDPOINT_TOKEN
export ENDPOINT_TOKEN="<endpoint-token>"
# Required here because run-a and run-b are served-model aliases.
export EVAL_TOKENIZER="<hf-repo-id-or-path>"

for CKPT in run-a run-b; do
  export EVAL_MODEL_HANDLE="$CKPT"
  export EVAL_ENDPOINT_URL="https://$CKPT.example.com/v1/completions"
  export EVAL_RESULTS_DIR="/mnt/shared/eval/$CKPT"
  uv run nemotron steps run eval/model_eval -c base_en --batch slurm_eval_direct
done
```

Replace `slurm_eval_direct` with `lepton_eval_direct` or `dgxcloud_eval_direct` for those backends.
The shipped profiles are CPU-only because the GPUs are wherever the model is served.
For suite selection, language pinning, and multi-subset tasks, refer to {doc}`run-a-benchmark-suite`.

Direct mode refuses to write into a task directory that already contains results.
Use a new `EVAL_RESULTS_DIR` per run, or pass `overwrite=true` to replace the directory.

## Inspect The Results

```bash
cat "$EVAL_RESULTS_DIR/summary.json"
jq '.harness, .tasks' "$EVAL_RESULTS_DIR/run_manifest.json"
```

- `summary.json` reports `ok` or `failed(<exit code>)` per task.
- `run_manifest.json` records the model, the redacted endpoint, the harness image and whether it was digest-pinned, the merged per-task parameters, the adapter configuration, and the Nemotron version and Git SHA.
- `failures.txt` exists only when a task failed and holds the tail of each failed task's `harness.log`.
- `<task>/results_*.json` holds the scores; `<task>/samples_*.jsonl` holds per-sample records; `<task>/adapter/` holds the proxy cache and request/response logs.

A number is citable only when `image_pinned_by_digest` is `true` and `limit_samples` is unset.
For the full layout, refer to {doc}`../reference/output-artifacts`.

## Related

- {doc}`run-a-benchmark-suite` for choosing among the shipped suites.
- {doc}`run-hosted-evaluation` for the launcher-mode hosted flow.
- {doc}`../explanation/tokenizer-alignment` for when to set `EVAL_TOKENIZER`.
- {doc}`../reference/cli-reference` for the environment-variable table and flag surface.
- {doc}`../reference/troubleshooting` for HTTP 404, 401, and tokenizer failures.
