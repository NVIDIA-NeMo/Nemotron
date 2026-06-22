<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-discover-the-step)=
# Discover The Model Evaluation Step

This guide shows how to find `eval/model_eval` in the step catalog, how to read its contract, and how to decide whether it applies.

## Prerequisites

- The Nemotron repository is synced.
- A local checkout is sufficient; discovery reads local `step.toml` files only.

## List Eval-Category Steps

```bash
uv run --no-sync nemotron steps list --category eval --json
```

The response includes `eval/model_eval`, the step that wraps NeMo Evaluator Launcher.

## Inspect The Step Contract

```bash
uv run --no-sync nemotron steps show eval/model_eval --json
```

The response contains the fields declared in `src/nemotron/steps/eval/model_eval/step.toml`.

| Field | What It Tells You |
| --- | --- |
| `consumes` | Optional input artifact type. This step accepts `checkpoint_megatron`. |
| `produces` | Output artifact type. This step produces `eval_results`. |
| `parameters` | Documented knobs such as `target.api_endpoint.*`, `deployment.checkpoint_path`, `task_filters`, and launcher params. |
| `strategies` | Rules for hosted smoke tests, checkpoint evaluation, endpoint/task pairing, and task-name selection. |
| `errors` | Named failure modes and recovery guidance. |
| `reference` | Upstream NeMo Evaluator Launcher references. |

## Read The Sample Files

The step provides two config files under `src/nemotron/steps/eval/model_eval/config/`.

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml
:language: yaml
```

`tiny_chat.yaml` is the hosted chat smoke-test config.
It sets `deployment.type: none`, reads `target.api_endpoint.*` from environment variables, and runs `mmlu_instruct` with `limit_samples: 1`.

```{literalinclude} ../../../src/nemotron/steps/eval/model_eval/config/default.yaml
:language: yaml
```

`default.yaml` is the Megatron Bridge checkpoint evaluation config.
It uses NeMo Evaluator Launcher deployment and evaluates the configured `tasks` entries.

## Decide Whether It Applies

`eval/model_eval` applies when the following statements are true.

- The model is already available as an OpenAI-compatible endpoint, or NeMo Evaluator Launcher can deploy the checkpoint from the selected config.
- The tasks you need are implemented by the installed NeMo Evaluator Launcher stack.
- The endpoint type matches the selected task family.

`eval/model_eval` is not the right step when the evaluation needs a custom scorer that NeMo Evaluator Launcher does not implement.
Write a dedicated evaluation step in that case, modeled on the contract layout under `src/nemotron/steps/`.

## Related

- `src/nemotron/steps/eval/model_eval/step.toml` for the full step contract.
- {doc}`run-hosted-evaluation` for the first procedural walk-through after discovery.
- {doc}`../reference/config-schema` for field-by-field YAML reference.
- {doc}`../reference/cli-reference` for the flag and override surface.
