---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "YAML reference for tokenizer_extension/extend aligned with config/default.yaml."
topics: ["Tokenizer Extension", "Configuration"]
tags: ["Reference", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Extend YAML Reference

The `tokenizer_extension/extend` step ships `src/nemotron/steps/tokenizer_extension/extend/config/default.yaml` as its starter configuration.
This page lists the keys you can override with `nemotron steps run tokenizer_extension/extend key=value` dotlists, with the full file inlined below.

## Default Configuration File

```{literalinclude} ../../../src/nemotron/steps/tokenizer_extension/extend/config/default.yaml
:language: yaml
:class: scrollable
```

## Keys Grouped by Concern

### Base Tokenizer and Method

| Key | Default | Description |
|-----|---------|-------------|
| `model_id` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Hugging Face identifier or local path whose tokenizer is extended. Only the tokenizer is read. |
| `trust_remote_code` | `true` | Passed to `AutoTokenizer.from_pretrained`. Set `false` for repositories you do not trust. |
| `method` | `replace` | `add`, `replace`, or `expand`. One arm per job. See {doc}`../explanation/extension-methods`. |
| `extension_size` | `30000` | Number of new tokens to splice into the vocabulary. |
| `batch_size` | `1000` | Documents per batch fed to the BPE trainer. |
| `output_dir` | `./output/tokenizer_extension` | Parent directory; the step writes to `output_dir/<method>/`. |

### Language and Overrides

| Key | Default | Description |
|-----|---------|-------------|
| `language` | `hindi` | Registered language profile that selects the corpus normalizer and the Replace prune script. Absent, the Devanagari defaults apply. |
| `remove_script` | unset | Comma-separated `SCRIPT_UNICODE_RANGES` keys whose residual tokens `replace` prunes. Overrides the profile when set. |
| `script_normalizer` | unset | `none`, `nfkc`, or `devanagari`. Overrides the profile when set. |

### Corpus

The corpus block accepts either a Hugging Face dataset or a local path; the `corpus_source_unset` error is raised when neither is set.

| Key | Default | Description |
|-----|---------|-------------|
| `corpus.hf_dataset` | `ai4bharat/sangraha` | Dataset identifier passed as `path` to `load_dataset`. Set `null` to use `corpus.path`. |
| `corpus.hf_name` | `verified` | Dataset configuration or subset passed as `name`. |
| `corpus.hf_split` | `hin` | Split to read. |
| `corpus.hf_data_dir` | `null` | Optional `data_dir` argument. |
| `corpus.hf_data_files` | `null` | Optional `data_files` argument. |
| `corpus.hf_revision` | `null` | Optional dataset revision. |
| `corpus.streaming` | `false` | `false` downloads and caches the dataset so later jobs reuse it; `true` streams. |
| `corpus.path` | `null` | Local Parquet directory or glob, or a JSON Lines path. Leave `corpus.hf_dataset` null when set. |
| `corpus.glob` | `"*.parquet"` | File pattern applied when `corpus.path` is a directory. |
| `corpus.text_field` | `text` | Column that holds the document text. |
| `corpus.samples` | `1000000` | Maximum documents used for training. |
| `corpus.max_doc_chars` | `20000` | Documents are truncated to this many characters. Lower it on a memory-limited node. |
| `corpus.min_frequency` | `0` | Minimum word frequency for a merge; `0` keeps every merge. |
| `corpus.diversify` | `true` | For a local directory, sample evenly across shards. |

## Related Pages

- Procedure: {doc}`../how-to/extend-a-tokenizer`
- Output layout: {doc}`outputs`
- Errors: {doc}`troubleshooting`
