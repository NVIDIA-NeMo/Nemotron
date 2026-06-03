---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Why log-probability tasks in eval/model_eval require a tokenizer that matches the served model."
topics: ["Model Evaluation", "Tokenizer"]
tags: ["Explanation", "Model Evaluation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(model-eval-tokenizer-alignment)=
# Tokenizer Alignment

The hosted chat smoke path in `tiny_chat.yaml` does not require a tokenizer override.
Tokenizer alignment becomes important when you run log-probability tasks such as HellaSwag or other tasks that score candidate continuations by token likelihood.

## Where The Tokenizer Lives

Tokenizer settings live under:

```text
evaluation.nemo_evaluator_config.config.params.extra.tokenizer
evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend
```

The `default.yaml` config sets the tokenizer to `${deployment.checkpoint_path}/tokenizer`, which matches the Megatron Bridge checkpoint path when that checkpoint contains a tokenizer subdirectory.

## Why It Must Match

Log-probability tasks ask the endpoint to score candidate token sequences.
If the evaluator tokenizes candidates differently from the served model, the model scores the wrong token ids and the metric is not meaningful.

## Accepted Shapes

Use one of these tokenizer values when a selected task requires local tokenization:

- A Hugging Face model id.
- A filesystem path containing tokenizer files.
- The `tokenizer/` subdirectory inside a Megatron Bridge `iter_*` checkpoint.

Use `huggingface` for `evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend` unless the selected launcher task explicitly requires another backend.

## Related Pages

- {doc}`endpoint-types-and-benchmarks` for endpoint/task pairing.
- {doc}`pipeline-overview` for runtime flow.
- {doc}`../reference/config-schema` for field-by-field documentation.
- {doc}`../reference/troubleshooting` for common tokenizer and checkpoint-path failures.
