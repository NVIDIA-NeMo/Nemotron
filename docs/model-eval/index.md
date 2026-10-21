<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-index)=
# About Model Evaluation

The `eval/model_eval` Nemotron step scores a model on NeMo Evaluator benchmark tasks and writes an `eval_results` artifact to disk.
The model is always reached over HTTP through an OpenAI-compatible endpoint.
The step runs in one of two *execution modes*: *launcher mode*, which hands the run to NeMo Evaluator Launcher and can deploy a Megatron Bridge checkpoint for you, and *direct mode*, which runs the evaluation harness inside the step's own job against an endpoint that you host.
The {ref}`model-eval-choose-a-mode` section explains how to pick one.

:::{tip}
New to model evaluation or the Nemotron CLI?
Read {doc}`using-skills` for a short guide to productive agent sessions, then start the {doc}`getting-started` tutorial to run one benchmark on one sample against a hosted endpoint.
:::

## When To Use

Use `eval/model_eval` when the work matches one of the following.

- Score a trained checkpoint with NeMo Evaluator Launcher tasks.
- Score a model that you already serve, on any Nemotron backend, with one of the shipped benchmark suites such as `base_en` or `instruct_en`.
- Compare a new training run against a baseline by running the same task set against both, with generation parameters and endpoint type held constant.
- Perform a verification run against a hosted endpoint, to confirm the URL, credential, and model id before scaling up.
- Pair this step with a baseline evaluation before training to capture before-and-after measurements around a training change, by following {ref}`model-eval-comparing-runs`.

(model-eval-choose-a-mode)=
## Choose A Mode

Both modes evaluate over HTTP and write the same `eval_results` artifact.
They differ in who creates the endpoint and who schedules the evaluation job.

| Question | Launcher mode | Direct mode |
| --- | --- | --- |
| How do I select it? | `-c default` (the runspec default; `mode: launcher`) | `-c direct`, or a suite config that inherits from `direct.yaml` (`mode: direct`) |
| Who serves the model? | NeMo Evaluator Launcher can deploy a Megatron Bridge checkpoint (`deployment.*`), or you point it at an existing endpoint (`deployment.type: none`). | You. Direct mode never creates or tears down an endpoint. |
| Who schedules the evaluation job? | The launcher's own executor, selected with `run.env.launcher_executor` (`local`, `slurm`, or `lepton`). | The Nemotron `--batch` profile, so every Nemotron backend works: `slurm_eval_direct`, `lepton_eval_direct`, `dgxcloud_eval_direct`. |
| Where do task names come from? | The launcher registry: `nemo-evaluator-launcher ls tasks`. | The selected harness image: `nemo-evaluator ls` inside that image. Harness-only tasks, such as MILU, are addressable. |
| How is it configured? | YAML plus dotlist overrides. | Environment variables (`EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE`, `EVAL_TOKENIZER`, `EVAL_RESULTS_DIR`, and related) plus `-t <task>` and dotlist overrides. |
| What is recorded? | The saved launcher config and the launcher invocation id. | `summary.json`, `run_manifest.json` with harness image, digest pinning, and merged per-task parameters, and `failures.txt` when a task fails. |
| Does it export to W&B? | Optional, through `execution.auto_export` and `export.wandb`. | No. Direct mode writes files and needs no tracker credentials. |

Choose **launcher mode** when you want the tool to deploy a Megatron Bridge checkpoint on a local machine or a Slurm cluster, or when you want the launcher's status and log commands.
Start with {doc}`getting-started` and {doc}`how-to/evaluate-deployed-checkpoint`.

Choose **direct mode** when the model is already served, when you run on a backend the launcher has no executor for (Run:ai and DGX Cloud), or when the benchmark exists in a harness image but not in the launcher registry.
Start with {doc}`how-to/run-direct-mode-evaluation` and {doc}`how-to/run-a-benchmark-suite`.

:::{note}
Launcher-managed Lepton deployment is experimental: the launcher's Lepton executor does not accept the `generic` deployment type that `default.yaml` uses, and it does not remove created endpoints after a run.
On Lepton, host the endpoint yourself and evaluate it with direct mode.
:::

## Pipeline At A Glance

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart LR
    ckpt["Megatron Bridge checkpoint"] -->|launcher mode deploys| deploy["OpenAI-compatible<br/>endpoint"]
    hosted["Endpoint you host"] --> deploy
    deploy --> step["eval/model_eval<br/>launcher mode or direct mode"]
    step --> results["eval_results<br/>per-task subdirs, summary.json,<br/>run_manifest.json"]
```

In launcher mode, NeMo Evaluator Launcher owns task execution and result files under `output_dir`.
In direct mode, the step runs `nemo-evaluator run_eval` once per task and writes `summary.json`, `run_manifest.json`, and one directory per task under `output_dir`.
For the contract and the on-disk layout, refer to {doc}`reference/output-artifacts`.

## How It Works

The runner reads a single YAML document and applies command-line overrides.
It then branches on the `mode` key.

- In launcher mode, it removes Nemotron-only keys, saves the resolved launcher config, and calls `nemo_evaluator_launcher.api.functional.run_eval`.
- In direct mode, it resolves the endpoint and task list from the environment, writes `run_manifest.json`, and runs `nemo-evaluator run_eval` for each task inside the current job.

The endpoint type must match the benchmark family.
Chat and instruction benchmarks need a *chat* endpoint.
*Log-probability* tasks, such as HellaSwag, need a *completions* endpoint with `logprobs` support.
If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, set the evaluator to use the generated tokenizer.

The hosted verification config is `tiny_chat.yaml`.
The launcher checkpoint-evaluation config is `default.yaml`.
The direct-mode base config is `direct.yaml`, and the shipped suites `base_en`, `mmlu_prox`, `milu`, `instruct_en`, and `mmlu_prox_chat` each add a task list on top of it.
Generation settings live under `evaluation.nemo_evaluator_config.config.params`.

For the full concept set behind these design rules, refer to {doc}`explanation/index`.

## Documentation

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Getting Started
:link: getting-started
:link-type: doc
Run one benchmark on one sample against a hosted endpoint, end to end.
+++
{bdg-success}`15-30 min` {bdg-secondary}`tutorial`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` Use The Model Evaluation Skill With Confidence
:link: using-skills
:link-type: doc
Run a productive agent session: opening brief, four required inputs, and how `SKILL.md` keeps the session focused.
+++
{bdg-success}`10 min read` {bdg-secondary}`newcomer`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` How-To Guides
:link: how-to/index
:link-type: doc
Discover the step, run a direct-mode evaluation against an endpoint you host, run a benchmark suite, run a launcher-mode hosted evaluation, and evaluate a deployed checkpoint.
+++
{bdg-success}`5 guides` {bdg-secondary}`task-focused`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
YAML schema, command-line flags, output artifact layout, benchmark catalog, and troubleshooting.
+++
{bdg-success}`5 references` {bdg-secondary}`lookup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Concepts
:link: explanation/index
:link-type: doc
Architecture, endpoint and benchmark families, and tokenizer alignment.
+++
{bdg-success}`3 pages` {bdg-secondary}`explanation`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Getting Started

| Guide | What You Will Do | Time |
|---|---|---|
| {doc}`getting-started` | Run a one-sample evaluation against a hosted endpoint | 15-30 min |
| {doc}`using-skills` | Drive `eval/model_eval` from a coding agent | 10 min read |

```

```{tab-item} How-To Guides

| Guide | What You Will Do |
|---|---|
| {doc}`how-to/discover-the-step` | List the step, read its contract, and decide whether it applies |
| {doc}`how-to/run-direct-mode-evaluation` | Run the harness in the step's own job against an endpoint you host |
| {doc}`how-to/run-a-benchmark-suite` | Select a shipped suite, pin a language or subset, and select tasks with `-t` |
| {doc}`how-to/run-hosted-evaluation` | Run launcher-mode benchmarks against an already-running endpoint |
| {doc}`how-to/evaluate-deployed-checkpoint` | Pick a launcher deployment path, then point the step at the endpoint |

```

```{tab-item} Reference

| Reference | What You Will Find |
|---|---|
| {doc}`reference/config-schema` | YAML field reference for `direct.yaml`, `default.yaml`, `tiny_chat.yaml`, and the suite configs |
| {doc}`reference/cli-reference` | Flags, `-t` task selection, `EVAL_*` environment variables, and Hydra overrides |
| {doc}`reference/output-artifacts` | `eval_results` contract, `summary.json`, `run_manifest.json`, and on-disk layout |
| {doc}`reference/benchmarks-catalog` | Shipped suites and task identifiers grouped by family |
| {doc}`reference/troubleshooting` | Failure modes for both modes, with cause and recovery |

```

```{tab-item} Concepts

| Concept | What You Will Learn |
|---|---|
| {doc}`explanation/index` | Map of the concept pages and how they relate |
| {doc}`explanation/pipeline-overview` | Artifact flow through `eval/model_eval` in launcher mode and direct mode |
| {doc}`explanation/endpoint-types-and-benchmarks` | Chat versus completions endpoints, and which suites and benchmark families match each one |
| {doc}`explanation/tokenizer-alignment` | When to set an explicit tokenizer and why tokenizer-extended models need the generated tokenizer |

```

````

## Before You Start

- The Nemotron repository is synced and `uv sync --extra evaluator` is complete.
- A bearer token is exported as the environment variable named in `target.api_endpoint.api_key_name` (launcher mode) or `EVAL_API_KEY_NAME` (direct mode).
  Hosted verification runs usually use `NVIDIA_API_KEY`; direct mode defaults to `ENDPOINT_TOKEN`.
- A reachable evaluation endpoint URL and a model identifier the endpoint advertises.
  In direct mode the identifier must equal the server's `--served-model-name`.
- An explicit tokenizer if the served model name is an alias, the tokenizer is at a custom path, or you used Tokenizer Extension.
  Tokenizer Extension must use the generated tokenizer.
- For direct mode on a cluster, an `env.toml` that contains the `<backend>_eval_direct` profile.
  Refer to {ref}`model-eval-troubleshooting-profile-not-found` if the profile is missing.

## Limitations And Considerations

- Cost: every benchmark sample issues at least one request to the endpoint, and hosted endpoints incur per-token cost.
- Rate limits: hosted endpoints throttle concurrent requests, so set `evaluation.nemo_evaluator_config.config.params.parallelism` to a value the endpoint can serve.
- Deployment: `tiny_chat.yaml` and `direct.yaml` target an already-deployed endpoint; `default.yaml` uses launcher-managed deployment for a Megatron Bridge checkpoint.
  Direct mode does not serve checkpoints and does not iterate over checkpoints; the loop over checkpoints is yours.
- Multi-subset tasks: `mmlu_prox_completions`, `mmlu_prox_chat`, `global_mmlu`, and `global_mmlu_full` cover every language by default.
  Pin one subset per run, as described in {doc}`how-to/run-a-benchmark-suite`.
- Sample limits: a run with `limit_samples` set scores a deterministic first-N subset, not the full split, and should be reported as such.
- Comparability: scores are comparable when the endpoint type, task version, tokenizer, and generation parameters are held constant across runs.
  The {ref}`model-eval-comparing-runs` section explains the framing.

## Related Documentation

- The full `step.toml` contract: `src/nemotron/steps/eval/model_eval/step.toml` in the repository.
- The step's own operator notes: [`src/nemotron/steps/eval/model_eval/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/eval/model_eval/README.md) and [`REFERENCE.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/eval/model_eval/REFERENCE.md).
- The before-and-after evaluation framing: {ref}`model-eval-comparing-runs`.
- Upstream NeMo Evaluator quick-start: <https://docs.nvidia.com/nemo/evaluator/nightly/get-started/quickstart>.
