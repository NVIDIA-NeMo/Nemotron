---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "YAML reference for tokenizer_extension/eval_init aligned with config/default.yaml."
topics: ["Tokenizer Extension", "Configuration"]
tags: ["Reference", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Eval Init YAML Reference

The `tokenizer_extension/eval_init` step ships `src/nemotron/steps/tokenizer_extension/eval_init/config/default.yaml` as its starter configuration.
This page lists the keys you can override with `nemotron steps run tokenizer_extension/eval_init key=value` dotlists, with the full file inlined below.
The step translates these keys into arguments for `bpb.py`.

## Default Configuration File

```{literalinclude} ../../../src/nemotron/steps/tokenizer_extension/eval_init/config/default.yaml
:language: yaml
:class: scrollable
```

## Keys Grouped by Concern

### Models

| Key | Default | Description |
|-----|---------|-------------|
| `models` | `[./output/resized_checkpoint]` | One or more checkpoint paths to score. Required; the `empty_models` error is raised when the list is empty. |
| `base_model` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | Unextended reference model, scored first. The report includes the delta of each model against it. |
| `trust_remote_code` | `true` | Passed when loading models. The step default is `false`. |
| `dtype` | `bfloat16` | Weight dtype. |
| `device` | unset | Optional device override. |

### Corpus

Set either `data_file` or `corpus.hf_dataset`; the step raises when neither is set.

| Key | Default | Description |
|-----|---------|-------------|
| `data_file` | `./data/eval/target_val.jsonl` | Local JSON Lines file with `text_field`, or a text file with one document per line. `corpus.path` is accepted as an alias. |
| `text_field` | `text` | Text column for JSON Lines input. `corpus.text_field` is accepted as an alias. |
| `hf_dataset` | unset | Streamed Hugging Face dataset, used when `data_file` is unset. `corpus.hf_dataset` is accepted as an alias. |
| `hf_config` | unset | Dataset configuration for `hf_dataset`. |
| `hf_split` | `train` | Split for `hf_dataset`. |
| `skip_docs` | `0` | Leading documents to skip for `hf_dataset`. |

### Budget and Windowing

| Key | Default | Description |
|-----|---------|-------------|
| `max_docs` | `-1` | Document cap; `-1` scores the whole corpus. Tokenizer-independent, so use it for comparisons. |
| `max_tokens` | `-1` | Token cap; `-1` disables it. Not comparable across tokenizers. |
| `allow_token_cap_comparison` | `false` | Permit scoring more than one model under `max_tokens` with no `max_docs`. Exists to reproduce a historical run. |
| `max_length` | `2048` | Sliding-window length in tokens. |
| `stride` | `512` | Sliding-window stride in tokens. |
| `output_json` | `./output/eval/bpb_add_subword.json` | Report path. |

## Related Pages

- Procedure: {doc}`../how-to/evaluate-a-tokenizer`
- Metric definitions: {doc}`../explanation/evaluation-metrics`
- Report fields: {doc}`outputs`
