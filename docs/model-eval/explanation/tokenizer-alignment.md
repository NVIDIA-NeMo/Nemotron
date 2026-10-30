---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "When to set an explicit tokenizer for eval/model_eval and why tokenizer-extended models must use the generated tokenizer."
topics: ["Model Evaluation", "Tokenizer"]
tags: ["Explanation", "Model Evaluation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(model-eval-tokenizer-alignment)=
# Tokenizer Alignment

If the served model name is its Hugging Face model ID, the evaluator loads that tokenizer automatically.
If you used Tokenizer Extension, use the generated tokenizer.

## Where The Tokenizer Lives

Tokenizer settings live under:

```text
evaluation.nemo_evaluator_config.config.params.extra.tokenizer
evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend
```

The `default.yaml` config sets the tokenizer to `${deployment.checkpoint_path}/tokenizer`, which matches the Megatron Bridge checkpoint path when that checkpoint contains a tokenizer subdirectory.

The `direct.yaml` config, and every suite that inherits from it, reads the tokenizer from the `EVAL_TOKENIZER` environment variable and sets `tokenizer_backend: huggingface` and `tokenized_requests: false`.
Direct mode prints a warning before the first task if no tokenizer is configured.

## When To Set It Explicitly

With `extra.tokenizer` unset, the harness falls back to the endpoint's served model name.
That works when the name identifies the unmodified model and its tokenizer.
Set an explicit tokenizer when the served model name is an alias, when the tokenizer is stored at a custom path, or when you used Tokenizer Extension.
Otherwise, the harness can fail with `RepositoryNotFoundError` while trying to load the served model name as a Hugging Face repository.
Some chat tasks also load a tokenizer, so the same guidance applies to chat endpoints.

## Why It Must Match

Log-probability tasks ask the endpoint to score candidate token sequences that the harness has already tokenized.
If the evaluator tokenizes candidates differently from the served model, the model scores the wrong token ids and the metric is not meaningful.

A tokenizer-extended checkpoint must use its own extended tokenizer, not the base model's.
The base tokenizer can tokenize affected text differently from the served model, and the run can complete with scores that are silently wrong.
The `tokenizer_extension` step category produces such checkpoints together with their extended tokenizer.

## Accepted Shapes

When you set a tokenizer explicitly, use one of these values:

- A Hugging Face model id.
- A filesystem path containing tokenizer files.
- The `tokenizer/` subdirectory inside a Megatron Bridge `iter_*` checkpoint.

Use `huggingface` for `evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend` unless the selected launcher task explicitly requires another backend.

## Tokenizer Exports From Tokenizer Extension

Current `convert/megatron_to_hf` exports normalize supported internal tokenizer class names to `PreTrainedTokenizerFast` automatically when `tokenizer.json` is present.
If an older or externally produced export still declares `"tokenizer_class": "TokenizersBackend"`, lm-evaluation-harness fails with `Tokenizer class TokenizersBackend does not exist`.
Copy `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` to a side directory, set `tokenizer_class` to `PreTrainedTokenizerFast`, remove `auto_map`, and point `EVAL_TOKENIZER` at that directory.
The vocabulary is unchanged; only the loader class differs.

## Related Pages

- {doc}`endpoint-types-and-benchmarks` for endpoint/task pairing.
- {doc}`pipeline-overview` for runtime flow.
- {doc}`../how-to/run-direct-mode-evaluation` for setting `EVAL_TOKENIZER`.
- {doc}`../reference/config-schema` for field-by-field documentation.
- {doc}`../reference/troubleshooting` for common tokenizer and checkpoint-path failures.
