<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-index)=
# About Model Evaluation

The `eval/model_eval` Nemotron step is a wrapper around NeMo Evaluator Launcher.
It runs launcher tasks against either an existing OpenAI-compatible endpoint or a launcher-managed Megatron Bridge checkpoint deployment, then writes an `eval_results` artifact to disk.

:::{tip}
New to model evaluation or the Nemotron CLI?
Read {doc}`using-skills` for a short guide to productive agent sessions, then start the {doc}`getting-started` tutorial to run one benchmark on one sample against a hosted endpoint.
:::

## When To Use

Use `eval/model_eval` when the work matches one of the following.

- Score a trained checkpoint with NeMo Evaluator Launcher tasks.
- Compare a new training run against a baseline by running the same task set against both, with generation parameters and endpoint type held constant.
- Perform a sample run against a hosted endpoint, to confirm the URL, credential, and model id before scaling up.
- Pair this step with a baseline evaluation before training to capture before-and-after measurements around a training change, by following {ref}`model-eval-comparing-runs`.

## Pipeline At A Glance

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart LR
    ckpt["Hugging Face or<br/>Megatron Bridge checkpoint"] --> deploy["OpenAI-compatible<br/>endpoint"]
    hosted["Hosted endpoint"] --> deploy
    deploy --> step["eval/model_eval<br/>(NeMo Evaluator)"]
    step --> results["eval_results<br/>per-benchmark subdirs"]
```

NeMo Evaluator Launcher owns task execution and result files under `output_dir`.
For the contract and the on-disk layout, refer to {doc}`reference/output-artifacts`.

## How It Works

The runner reads a single YAML document, applies command-line overrides, removes Nemotron-only keys, saves the resolved launcher config, and calls `nemo_evaluator_launcher.api.functional.run_eval`.

The endpoint type must match the benchmark family.
Chat and instruction benchmarks need a *chat* endpoint.
*Log-probability* tasks, such as HellaSwag, need a *completions* endpoint with `logprobs` support and a tokenizer that matches the served model.

The hosted smoke-test config is `tiny_chat.yaml`.
The checkpoint-evaluation config is `default.yaml`.
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
Discover the step, run a hosted evaluation, and evaluate a deployed checkpoint.
+++
{bdg-success}`3 guides` {bdg-secondary}`task-focused`
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
| {doc}`how-to/run-hosted-evaluation` | Run benchmarks against an already-running endpoint |
| {doc}`how-to/evaluate-deployed-checkpoint` | Pick a deployment path, then point the step at the endpoint |

```

```{tab-item} Reference

| Reference | What You Will Find |
|---|---|
| {doc}`reference/config-schema` | YAML field reference for `default.yaml` and `tiny_chat.yaml` |
| {doc}`reference/cli-reference` | Flags and Hydra overrides for `nemotron steps run eval/model_eval` |
| {doc}`reference/output-artifacts` | `eval_results` contract and on-disk layout |
| {doc}`reference/benchmarks-catalog` | NeMo Evaluator Launcher task identifiers grouped by family |
| {doc}`reference/troubleshooting` | Named error modes from `step.toml`, with cause and recovery |

```

```{tab-item} Concepts

| Concept | What You Will Learn |
|---|---|
| {doc}`explanation/index` | Map of the concept pages and how they relate |
| {doc}`explanation/pipeline-overview` | Artifact flow from checkpoint through `eval/model_eval` into `eval_results` |
| {doc}`explanation/endpoint-types-and-benchmarks` | Chat versus completions endpoints, and which benchmark families match each one |
| {doc}`explanation/tokenizer-alignment` | Why log-probability benchmarks need a tokenizer that matches the served model |

```

````

## Before You Start

- The Nemotron repository is synced and `uv sync` is complete.
- A bearer token is exported as the environment variable named in `target.api_endpoint.api_key_name`.
  Hosted smoke tests usually use `NVIDIA_API_KEY`.
- A reachable evaluation endpoint URL and a model identifier the endpoint advertises.
- A tokenizer that matches the served model when running log-probability tasks.
  The hosted chat smoke test does not require a tokenizer override.

## Limitations And Considerations

- Cost: every benchmark sample issues at least one request to the endpoint, and hosted endpoints incur per-token cost.
- Rate limits: hosted endpoints throttle concurrent requests, so set `evaluation.nemo_evaluator_config.config.params.parallelism` to a value the endpoint can serve.
- Deployment: `tiny_chat.yaml` targets an already-deployed endpoint; `default.yaml` uses launcher-managed deployment for a Megatron Bridge checkpoint.
- Comparability: scores are comparable when the endpoint type, task version, tokenizer, and generation parameters are held constant across runs.
  The {ref}`model-eval-comparing-runs` section explains the framing.

## Related Documentation

- The full `step.toml` contract: `src/nemotron/steps/eval/model_eval/step.toml` in the repository.
- The before-and-after evaluation framing: {ref}`model-eval-comparing-runs`.
- Upstream NeMo Evaluator quick-start: <https://docs.nvidia.com/nemo/evaluator/nightly/get-started/quickstart>.
