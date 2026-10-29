---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How nemotron eval/model_eval runs in launcher mode and direct mode, and how each writes eval_results."
topics: ["Model Evaluation", "Pipeline"]
tags: ["Explanation", "Architecture"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(model-eval-pipeline-overview)=
# Pipeline Overview

The `eval/model_eval` step does not implement benchmark scoring itself.
Scoring belongs to a NeMo Evaluator harness, such as lm-evaluation-harness, that runs inside a harness container image.
The step decides how that harness is reached and who schedules it, and it does so in one of two execution modes selected by the `mode` key.

- *Launcher mode* (`mode: launcher`, the default) hands the resolved config to NeMo Evaluator Launcher, which owns scheduling, optional model deployment, and result files.
- *Direct mode* (`mode: direct`) runs the harness in the step's own job, so scheduling comes from the Nemotron `--run` or `--batch` profile and the launcher's executor layer is not used.

## Architecture

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart LR
    subgraph launcher_mode["Launcher mode (-c default)"]
        ckpt["Megatron Bridge iter_* checkpoint"] --> deploy["deployment.* block"]
        deploy --> launcher["NeMo Evaluator Launcher run_eval"]
        launcher --> lres["eval_results under output_dir"]
    end
    subgraph direct_mode["Direct mode (-c direct)"]
        hosted["Endpoint you host"] --> env["EVAL_ENDPOINT_URL / EVAL_MODEL_HANDLE"]
        env --> harness["nemo-evaluator run_eval per task<br/>inside the --batch job"]
        harness --> dres["summary.json, run_manifest.json,<br/>one directory per task"]
    end
```

## Runtime Flow

Both modes share the first three steps.

1. `step.py` calls `run_model_eval` from `runtime.py`.
1. The runtime loads the selected config from `config/<name>.yaml` or a user-supplied YAML path.
1. Hydra-style dotlist overrides are merged into the config, and `-t <task>` flags are collected as task filters.

In launcher mode the runtime continues as follows.

1. Nemotron-only keys are removed before launcher dispatch: `dry_run`, `output_dir`, `task_filters`, and `run`.
1. `output_dir` is copied into `execution.output_dir`.
1. W&B environment mappings are injected, and the launcher's own schema checks run on the result so that `dry_run=true` fails in the same way as a real run.
1. The resolved launcher config is saved and printed as `launcher_config`.
1. `nemo_evaluator_launcher.api.functional.run_eval` is called with the launcher config and optional task filters.

In direct mode the runtime continues as follows.

1. The config is resolved inside the job, including `${oc.env:...}` lookups for `EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE`, and the other `EVAL_*` variables.
   The run fails if `target.api_endpoint.url` or `target.api_endpoint.model_id` is unset, or if `target.api_endpoint.type` is not `chat` or `completions`.
1. The task list is built from `evaluation.tasks`, or used exactly as supplied when `-t` flags are present, and each name is validated as a plain directory name.
1. A warning is printed if no client-side tokenizer is configured.
1. Each task directory under `output_dir` is checked; a non-empty task directory is refused unless `overwrite=true`.
   The output root is then claimed with a `.nemotron-eval.lock` file so that two runs cannot write into the same directory.
1. `run_manifest.json` is written before the first task starts.
1. For each task, `config.params.*` and `target.api_endpoint.adapter_config.*` are translated into `nemo-evaluator run_eval --overrides`, and the harness runs with its output captured to `<task>/harness.log`.
   An unrecognized top-level parameter is rejected rather than dropped.
1. `summary.json` records `ok` or `failed(<exit code>)` per task, and `failures.txt` collects the tail of each failed task's log.

## Input Artifacts

The step declares optional `checkpoint_megatron` input.
Hosted endpoint runs, in either mode, do not consume a checkpoint artifact.
Launcher-managed checkpoint runs usually pass a concrete Megatron Bridge `iter_*` directory through `deployment.checkpoint_path`.
Direct mode never deploys a checkpoint: it requires an endpoint URL and a model handle.

## Output Artifact

The step produces `eval_results`.
In launcher mode the exact directory layout is owned by NeMo Evaluator Launcher and the selected task implementations.
In direct mode the step itself writes `summary.json`, `run_manifest.json`, `failures.txt` on failure, and one directory per task; the files inside each task directory are owned by the harness.
For result inspection guidance, refer to {doc}`../reference/output-artifacts`.

## What Is Owned Where

| Owned by Nemotron | Owned by NeMo Evaluator Launcher (launcher mode) | Owned by the harness image (direct mode) |
| --- | --- | --- |
| Step discovery, config loading, dotlist overrides, `-t` task filters, and mode selection. | Scheduling through `execution.type`, deployment orchestration, endpoint probing, and result files. | Task definitions, prompt formatting, and the files inside each `<task>/` directory. |
| `dry_run`, `output_dir`, `task_filters`, `overwrite`, `harness_image`, and `run` preprocessing. | `execution`, `deployment`, `target`, `evaluation`, `tasks`, and `export` semantics after dispatch. | The set of task names that `nemo-evaluator ls` reports for that image tag or digest. |
| Direct-mode preflight, `run_manifest.json`, `summary.json`, `failures.txt`, and the translation of `config.params.*` and `adapter_config.*` into harness overrides. | Accepted task identifiers in `nemo-evaluator-launcher ls tasks`. | The adapter proxy between the harness and the endpoint, configured through `adapter_config`. |

## Why Two Modes Exist

NeMo Evaluator Launcher ships executors for `local`, `slurm`, and `lepton` only; there is no Run:ai executor, and the Lepton executor is experimental for this step's deployment type.
Direct mode removes the launcher's executor layer from the path.
Because the harness runs in the step's own job, the backend is whichever runspec profile submits the step, and that is the same code path every other Nemotron step uses.
The cost is that direct mode does not serve the model: you host the endpoint and supply its URL.

## Related Pages

- {ref}`model-eval-choose-a-mode` for the decision table between the two modes.
- {doc}`endpoint-types-and-benchmarks` for endpoint/task pairing.
- {doc}`tokenizer-alignment` for when to set an explicit tokenizer.
- {doc}`../how-to/run-direct-mode-evaluation` for the direct-mode procedure.
- {doc}`../reference/output-artifacts` for result inspection.
- {doc}`../how-to/discover-the-step` for reading the step contract before configuring a run.
