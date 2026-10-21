---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Why eval/model_eval couples endpoint type to NeMo Evaluator Launcher task family."
topics: ["Model Evaluation", "Endpoints"]
tags: ["Explanation", "Model Evaluation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(model-eval-endpoint-types-and-benchmarks)=
# Endpoint Types And Task Families

`eval/model_eval` reaches the model through an OpenAI-compatible endpoint in both execution modes.
The endpoint type must match the selected task family.

## Endpoint Fields

Hosted endpoint runs use:

```text
target.api_endpoint.url
target.api_endpoint.model_id
target.api_endpoint.api_key_name
target.api_endpoint.type
```

The `type` value is `chat` or `completions`; direct mode rejects any other value.
The URL path should agree with that value: `/v1/chat/completions` for `chat`, `/v1/completions` for `completions`.

In direct mode, `direct.yaml` reads these fields from the environment.

| Field | Environment variable | Default |
| --- | --- | --- |
| `target.api_endpoint.url` | `EVAL_ENDPOINT_URL` | none; required |
| `target.api_endpoint.model_id` | `EVAL_MODEL_HANDLE` | none; required, and must equal the server's `--served-model-name` |
| `target.api_endpoint.type` | `EVAL_ENDPOINT_TYPE` | `completions` |
| `target.api_endpoint.api_key_name` | `EVAL_API_KEY_NAME` | `ENDPOINT_TOKEN` |

The chat suites, `instruct_en` and `mmlu_prox_chat`, set `target.api_endpoint.type: chat` in their config, so `EVAL_ENDPOINT_TYPE` is not needed for them.

## Task Families

- Chat and instruction tasks issue chat-completions requests and score generated answers.
  They suit instruct models.
- Log-probability tasks need a completions endpoint with logprobs support.
  They suit base models, including continued-pretraining checkpoints, and use few-shot prompts.

If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, set the evaluator to use the generated tokenizer.
{doc}`tokenizer-alignment` explains why.

## Decision Table

| Task family | Required endpoint type | Shipped suites | Extra requirements |
| --- | --- | --- | --- |
| Hosted chat verification | `chat` | `tiny_chat` (launcher mode) | A chat-completions URL and a valid API key. |
| Instruction/chat tasks | `chat` | `instruct_en`, `mmlu_prox_chat` | Generation parameters appropriate for the model and task; a reasoning parser on the server for reasoning models. |
| Log-probability tasks | `completions` | `base_en`, `mmlu_prox`, `milu` | A completions endpoint with logprobs support. If you used Tokenizer Extension, use the generated tokenizer. |

The repository verification config, `tiny_chat.yaml`, uses `mmlu_instruct` with `target.api_endpoint.type=chat`.
The launcher checkpoint config, `default.yaml`, includes `adlr_mmlu` and `hellaswag` for Megatron checkpoint evaluation; verify endpoint and tokenizer requirements before changing those tasks.
The direct-mode suites are listed with their tasks in {doc}`../reference/benchmarks-catalog`.

## The Same Benchmark, Two Task Names

Some benchmarks exist as a separate task for each endpoint type.
`mmlu_prox_completions` and `mmlu_prox_chat` target the same benchmark, but the first is a completions task and the second is a chat task.
They are different tasks with different prompt formats, and scores should not be compared across endpoint types.
Use `nemo-evaluator-launcher ls task <name> --json` and read `supported_endpoint_types` before selecting a task; using the wrong endpoint type fails, or scores something other than the intended benchmark.

## Reasoning Models

For a reasoning model, the endpoint must be served with a reasoning parser, and with a tool-call parser for tool-use benchmarks.
Without one, the reasoning trace is returned as ordinary content and generative benchmarks such as IFEval score that trace as the answer.
The endpoint returns HTTP 200 throughout, so the failure appears as a plausible but low score rather than as an error.
Log-likelihood tasks are unaffected.
Read one raw completion from the endpoint before trusting a verification run.

## Related Pages

- {doc}`tokenizer-alignment` for the tokenizer side of both task families.
- {doc}`pipeline-overview` for where endpoint config enters the run.
- {doc}`../how-to/run-a-benchmark-suite` for selecting a shipped suite.
- {doc}`../how-to/evaluate-deployed-checkpoint` for choosing the hosted or checkpoint path in launcher mode.
- {doc}`../reference/benchmarks-catalog` for suites and task identifiers.
