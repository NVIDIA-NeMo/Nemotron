<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-output-artifacts)=
# Output Artifacts

This page describes the artifacts produced by `eval/model_eval`.
Launcher mode delegates the file set to NeMo Evaluator Launcher; direct mode writes a fixed set of Nemotron-owned files around the harness output.

## The `eval_results` Contract

`step.toml` declares a single produced artifact.

| Field | Value |
| --- | --- |
| `type` | `eval_results` |
| `description` | Benchmark metrics, artifacts, and evaluation summaries produced by NeMo Evaluator. |

The contract is intentionally loose.
Nemotron does not normalize evaluator scores.
In launcher mode, NeMo Evaluator Launcher owns the exact file set and directory shape under the configured `output_dir`.
In direct mode, Nemotron owns the run-level files described in [Direct-Mode Layout](#direct-mode-layout); the harness owns the per-task files beneath them.

## Direct-Mode Layout

`output_dir` defaults to `EVAL_RESULTS_DIR`.
Every task writes into its own subdirectory named after the task.

```text
<output_dir>/
├── summary.json          per-task outcome: "ok", "failed(<exit code>)", or "dry-run"
├── run_manifest.json     what ran: model, redacted endpoint, harness image and
│                         whether it was digest-pinned, merged per-task params
│                         and adapter config, Nemotron version and Git SHA
├── failures.txt          written only on failure: the tail of each failed
│                         task's harness.log
├── .nemotron-eval.lock   present only while a run is in progress
└── <task>/
    ├── harness.log       the harness's own stdout and stderr
    ├── results_*.json    scores
    ├── samples_*.jsonl   per-sample requests and responses
    ├── run_config.yml    the config the harness resolved
    └── adapter/          proxy cache and request/response logs
```

The step prints the locations as it goes:

```text
eval_manifest: <output_dir>/run_manifest.json
[<task>] <harness command with the endpoint URL redacted>
eval_results: <output_dir>
eval_summary: <output_dir>/summary.json
  <task>: ok
```

### `run_manifest.json`

The manifest is written before the first task starts and is meant to be publishable alongside results.
Credentials are redacted: the endpoint URL loses any embedded user information and query string, credential-valued parameters are replaced, and `api_key_name` records the variable name only.

| Key | Content |
| --- | --- |
| `schema_version`, `generated_at`, `mode`, `dry_run` | Manifest schema version, UTC timestamp, always `"direct"`, and whether this was a `dry_run=true` run. |
| `model.id`, `model.endpoint_url`, `model.endpoint_type`, `model.api_key_name` | Served model name, redacted URL, `chat` or `completions`, credential variable name. |
| `harness.image`, `harness.image_pinned_by_digest`, `harness.packages` | Harness image reference, whether it ends in `@sha256:<64 hex digits>`, and installed harness package versions. |
| `code.nemotron_version`, `code.git_sha`, `code.git_dirty` | Nemotron package version, Git SHA, and whether the working tree had uncommitted changes; `null` when unavailable. |
| `tasks.<task>.output_dir`, `.params`, `.adapter_config`, `.command` | Per task: the output directory, the fully merged global and per-task parameters (which is where a pinned `task` subset appears), the adapter configuration, and the redacted harness command. |
| `output_dir`, `summary_path` | The run root and the summary file it will write. |

Before publishing a number, confirm `harness.image_pinned_by_digest` is `true` and `tasks.<task>.params.limit_samples` is `null`; a mutable tag or a sample cap makes the run unreproducible or partial.

### `summary.json`

A flat map from task name to outcome.
It is rewritten after every task, so a job that is preempted midway still leaves the outcomes of the tasks that finished.
The step exits non-zero when any task is `failed(...)`.

```json
{
  "adlr_arc_challenge_llama_25_shot": "ok",
  "hellaswag": "failed(1)"
}
```

### `failures.txt`

Written only when at least one task fails; it contains the last lines of each failed task's `harness.log` under a `===== <task> =====` header.
A clean run removes any `failures.txt` left by an earlier run in the same directory.

### Dry Runs And Locks

`dry_run=true` writes `summary.dry-run.json` and `run_manifest.dry-run.json` instead, creates the task directories, and takes no lock, so it never clobbers or blocks a real run.

A real run claims `output_dir` by creating `.nemotron-eval.lock` atomically and removes it on completion.
A second run against a claimed directory exits with a message identifying the holder's PID, host, and start time; `overwrite=true` does not override a live claim.
If the holder is definitely gone, for example after a preempted pod, delete the lock file or choose another `output_dir`.

## Launcher Config

Before calling the launcher, the step saves the resolved launcher config and prints:

```text
launcher_config: <path>
```

If the launcher returns an invocation id, the step also prints:

```text
launcher_invocation_id: <id>
status_command: nemo-evaluator-launcher status <id>
logs_command: nemo-evaluator-launcher logs <id>
```

Those commands are the source of truth for job state and logs after submission.

## Launcher-Mode Layout

The base output directory is `output_dir`, copied into `execution.output_dir`.
The exact files inside that directory depend on the configured launcher tasks.

For the hosted chat verification run, inspect:

```bash
find ./output/eval-tiny-chat -maxdepth 5 -type f | sort
```

For checkpoint evaluation, inspect the output directory you supplied:

```bash
find ./output/eval-megatron -maxdepth 5 -type f | sort
```

(model-eval-comparing-runs)=
## Comparing Runs

Evaluation results carry meaning when paired with another evaluation.
A trained checkpoint is scored against a baseline, a new prompt format is scored against an older one, and a quantized export is scored against the unquantized weights.
The comparison is honest when the surrounding configuration is held constant.

Apply the following practices before treating any single evaluation as a result.

- Run a lightweight baseline before the training, conversion, or quantization step you are measuring.
- Snapshot the exact evaluation config, including config file name, `output_dir`, endpoint fields, task list, tokenizer, harness image, and generation parameters. In direct mode, `run_manifest.json` records all of these.
- Place a date or run identifier in `output_dir` so baseline and post-change directories live side by side.
- Keep endpoint type, task versions, tokenizer, and generation parameters identical between runs.
- Rerun the baseline task set first before exploring new tasks.

## Related

- {doc}`config-schema` for the YAML keys that influence what is written.
- {doc}`benchmarks-catalog` for task identifiers.
- {doc}`../how-to/run-direct-mode-evaluation` for reading the direct-mode artifacts after a run.
- `src/nemotron/steps/eval/model_eval/step.toml` for the full step contract.
