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

`eval/model_eval` passes endpoint and task configuration to NeMo Evaluator Launcher.
The endpoint type must match the selected task family.

## Endpoint Fields

Hosted endpoint runs use:

```text
target.api_endpoint.url
target.api_endpoint.model_id
target.api_endpoint.api_key_name
target.api_endpoint.type
```

The `type` value is usually `chat` or `completions`.
The URL path should agree with that value.

## Task Families

- Chat and instruction tasks issue chat-completions requests and score generated answers.
- Log-probability tasks need a completions endpoint with logprobs support and a tokenizer that matches the served model.

## Decision Table

| Task family | Required endpoint type | Extra requirements |
| --- | --- | --- |
| Hosted chat smoke tests | `chat` | A chat-completions URL and a valid API key. |
| Instruction/chat tasks | `chat` | Generation parameters appropriate for the model and task. |
| Log-probability tasks | `completions` | A completions endpoint with logprobs support and a matching tokenizer. |

The repository smoke-test config, `tiny_chat.yaml`, uses `mmlu_instruct` with `target.api_endpoint.type=chat`.
The checkpoint config, `default.yaml`, includes launcher tasks for Megatron checkpoint evaluation; verify endpoint and tokenizer requirements before changing those tasks.

## Related Pages

- {doc}`tokenizer-alignment` for the tokenizer side of log-probability tasks.
- {doc}`pipeline-overview` for where endpoint config enters the run.
- {doc}`../how-to/evaluate-deployed-checkpoint` for choosing the hosted or checkpoint path.
- {doc}`../reference/benchmarks-catalog` for task identifiers.
