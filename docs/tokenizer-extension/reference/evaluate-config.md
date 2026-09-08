---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "YAML reference for tokenizer_extension/evaluate aligned with config/default.yaml."
topics: ["Tokenizer Extension", "Configuration"]
tags: ["Reference", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Evaluate YAML Reference

The `tokenizer_extension/evaluate` step ships `src/nemotron/steps/tokenizer_extension/evaluate/config/default.yaml` as its starter configuration.
This page lists the keys you can override with `nemotron steps run tokenizer_extension/evaluate key=value` dotlists, with the full file inlined below.

## Default Configuration File

```{literalinclude} ../../../src/nemotron/steps/tokenizer_extension/evaluate/config/default.yaml
:language: yaml
:class: scrollable
```

## Keys Grouped by Concern

### Tokenizer and Output

| Key | Default | Description |
|-----|---------|-------------|
| `tokenizer` | `./output/tokenizer_extension/replace` | Hugging Face identifier or local tokenizer directory to score. |
| `label` | `replace-30k` | Free-text tag recorded in the output JSON. Defaults to the `tokenizer` value when unset. |
| `trust_remote_code` | `true` | Passed to `AutoTokenizer.from_pretrained`. |
| `batch_size` | `1000` | Documents tokenized per batch. |
| `output` | `./output/eval/fertility_replace.json` | Report path. |

### Corpus

Set either `corpus.hf_dataset` or `corpus.path`; the `corpus_source_unset` error is raised when neither is set.
Keep the evaluation corpus distinct from the corpus that trained the extension.

| Key | Default | Description |
|-----|---------|-------------|
| `corpus.hf_dataset` | `ai4bharat/samanantar` | Dataset identifier. |
| `corpus.hf_config` | `hi` | Dataset configuration. `corpus.hf_name` is accepted as an alias. |
| `corpus.hf_split` | `train` | Split to read. |
| `corpus.path` | `null` | Local Parquet directory or glob, or JSON Lines path. |
| `corpus.glob` | `"*.parquet"` | File pattern applied when `corpus.path` is a directory. |
| `corpus.text_field` | `tgt` | Text column. The `wrong_text_field` error names the correct column for common datasets: `tgt` for Samanantar, `text` for Sangraha. |
| `corpus.num_docs` | `0` | Documents to score; `0` scores the full corpus. |
| `corpus.skip_docs` | `0` | Leading documents to skip, to avoid overlap with the training slice when both share a source. |

## Related Pages

- Procedure: {doc}`../how-to/evaluate-a-tokenizer`
- Report fields: {doc}`outputs`
