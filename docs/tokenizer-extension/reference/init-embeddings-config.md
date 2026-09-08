---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "YAML reference for tokenizer_extension/init_embeddings aligned with config/default.yaml."
topics: ["Tokenizer Extension", "Configuration"]
tags: ["Reference", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Init Embeddings YAML Reference

The `tokenizer_extension/init_embeddings` step ships `src/nemotron/steps/tokenizer_extension/init_embeddings/config/default.yaml` as its starter configuration.
This page lists the keys you can override with `nemotron steps run tokenizer_extension/init_embeddings key=value` dotlists, with the full file inlined below.

## Default Configuration File

```{literalinclude} ../../../src/nemotron/steps/tokenizer_extension/init_embeddings/config/default.yaml
:language: yaml
:class: scrollable
```

## Keys Grouped by Concern

### Model, Tokenizer, and Arm

| Key | Default | Description |
|-----|---------|-------------|
| `base_model` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | Base checkpoint whose embeddings are resized. Its embedding row count must equal the base tokenizer size. |
| `trust_remote_code` | `true` | Passed when loading the model. The step default is `false`; `default.yaml` opts in because the base model requires it. |
| `dtype` | `bfloat16` | Weight dtype for loading and saving. |
| `language` | `hindi` | Registered language profile that selects the target script, the auxiliary encoder, and the fastText vectors. |
| `arm` | `add` | `add` for an `add/` or `expand/` tokenizer; `replace` for a `replace/` tokenizer with `id_remap.json`. |
| `extended_tokenizer` | `./output/tokenizer_extension/add` | Tokenizer directory written by `extend`. |
| `output_dir` | `./output/resized_checkpoint` | Resized checkpoint directory; the value for `hf_model_path` in `pretrain/megatron_bridge`. |
| `method` | `subword` | `baseline`, `subword`, or `focus`. See {doc}`../explanation/embedding-initialization`. |

### `baseline`

| Key | Default | Description |
|-----|---------|-------------|
| `baseline.mode` | `mean_target` | `hf_default`, `mean_all`, or `mean_target`. `mean_hindi` is accepted as an alias for `mean_target`. |
| `baseline.norm_correction` | `true` | Rescale new rows to the median norm of the existing rows. |
| `baseline.num_samples` | `10` | Number of new tokens whose initialization is reported in the log. |

### `subword`

| Key | Default | Description |
|-----|---------|-------------|
| `subword.input_averaging` | `uniform` | `uniform`, `char_weighted`, `max_char`, `bert_weighted`, or `gemma_weighted` for input embedding rows. |
| `subword.output_averaging` | `uniform` | Same choices for language-model-head rows. |
| `subword.input_norm_correction` | `true` | Rescale new input rows to the median norm. |
| `subword.output_norm_correction` | `false` | Rescale new output rows. Enabling it risks language-model-head over-confidence and loss divergence. |
| `subword.input_target_norm` | `true` | Use the median norm of the target-script rows rather than of all rows. `input_hindi_norm` is accepted as an alias. |
| `subword.output_target_norm` | `false` | Same for output rows. `output_hindi_norm` is accepted as an alias. |
| `subword.temperature` | `0.1` | Softmax temperature; used only by `bert_weighted` and `gemma_weighted`. |
| `subword.bert_model` | unset | Encoder for `bert_weighted`. Leave unset so that `language` selects one that covers the language. |
| `subword.gemma_model` | `google/gemma-2-27b` | Decoder whose input embedding table `gemma_weighted` uses. |
| `subword.num_samples` | `10` | Number of new tokens reported in the log. |

### `focus`

| Key | Default | Description |
|-----|---------|-------------|
| `focus.fasttext_model` | `null` | Path to a fastText `.bin`; required for `method: focus`. |
| `focus.candidate_pool` | `target` | `target` restricts neighbors to the base model's target-script rows; `all` uses every base row. `hindi` is accepted as an alias for `target`. |
| `focus.sparsemax_temperature` | `0.05` | Divisor applied to cosine similarities before Sparsemax. |
| `focus.num_samples` | `10` | Number of new tokens reported in the log. |
| `focus.top_contributors` | `5` | Number of contributing base tokens listed per reported token. |

## Related Pages

- Procedure: {doc}`../how-to/initialize-embeddings`
- Output layout: {doc}`outputs`
- Errors: {doc}`troubleshooting`
