<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-benchmarks-catalog)=
# Tasks Catalog

This page catalogs task identifiers used by `eval/model_eval`.
NeMo Evaluator Launcher owns the authoritative task list.
Use this page as a quick map, then verify exact names with the installed launcher.

```bash
nemo-evaluator-launcher ls tasks
nemo-evaluator-launcher ls task <task-id>
```

## Naming Rule

Use the exact task id listed by NeMo Evaluator Launcher.
Do not prepend a harness name unless the launcher lists that exact dotted id.

## Repository Starting Points

| Config | Task entries | When to use |
| --- | --- | --- |
| `tiny_chat.yaml` | `mmlu_instruct` | Hosted chat smoke test. |
| `default.yaml` | `adlr_mmlu`, `hellaswag` | Launcher-managed Megatron checkpoint evaluation. |

## Chat And Instruction Tasks

These tasks use a chat endpoint.
The hosted smoke-test config uses this family.

| Identifier | Notes |
| --- | --- |
| `mmlu_instruct` | Chat/instruction smoke task used by `tiny_chat.yaml`. |
| `adlr_mmlu` | Configured by `default.yaml`; verify endpoint requirements in the installed launcher. |

## Log-Probability Tasks

These tasks generally need a completions endpoint with logprobs support and a tokenizer that matches the served model.

| Identifier | Notes |
| --- | --- |
| `hellaswag` | Configured by `default.yaml`; requires endpoint/tokenizer compatibility for meaningful scores. |

Configure tokenizer values under:

```text
evaluation.nemo_evaluator_config.config.params.extra.tokenizer
evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend
```

## Choosing Tasks

Ask three questions before changing the task list.

1. Does the installed launcher list the task id exactly?
1. Does the endpoint type match the task family?
1. Is this a smoke test or a production comparison?

For production comparisons, keep the same task list, endpoint type, tokenizer, and generation parameters across baseline and post-training runs.

## Related

- {doc}`config-schema` for the `tasks` section and evaluator params.
- {doc}`output-artifacts` for result layout expectations.
- {doc}`../explanation/endpoint-types-and-benchmarks` for endpoint/task pairing.
- {ref}`model-eval-comparing-runs` for before-and-after evaluation framing.
