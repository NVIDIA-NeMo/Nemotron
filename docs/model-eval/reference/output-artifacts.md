<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-output-artifacts)=
# Output Artifacts

This page describes the artifact produced by `eval/model_eval`.

## The `eval_results` Contract

`step.toml` declares a single produced artifact.

| Field | Value |
| --- | --- |
| `type` | `eval_results` |
| `description` | Benchmark metrics, artifacts, and evaluation summaries produced by NeMo Evaluator. |

The contract is intentionally loose.
Nemotron does not normalize evaluator outputs.
NeMo Evaluator Launcher owns the exact file set and directory shape under the configured `output_dir`.

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

## Directory Layout

The base output directory is `output_dir`, copied into `execution.output_dir`.
The exact files inside that directory depend on the configured launcher tasks.

For the hosted chat smoke test, inspect:

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
- Snapshot the exact evaluation config, including config file name, `output_dir`, endpoint fields, task list, tokenizer, and generation parameters.
- Place a date or run identifier in `output_dir` so baseline and post-change directories live side by side.
- Keep endpoint type, task versions, tokenizer, and generation parameters identical between runs.
- Rerun the baseline task set first before exploring new tasks.

## Related

- {doc}`config-schema` for the YAML keys that influence what is written.
- {doc}`benchmarks-catalog` for task identifiers.
- `src/nemotron/steps/eval/model_eval/step.toml` for the full step contract.
