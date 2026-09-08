---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Directories and JSON reports written by the tokenizer_extension steps, with field definitions."
topics: ["Tokenizer Extension", "Reference"]
tags: ["Reference", "Outputs"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Output Files

Each step writes to a path you set in its configuration.
Field names below trace to the writer in each step's executor.

## `extend`

Writes `output_dir/<method>/`, where `<method>` is `add`, `replace`, or `expand`.

| File | Written by | Content |
|------|-----------|---------|
| Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, and related files) | `save_pretrained` | The extended tokenizer, loadable with `AutoTokenizer`. |
| `summary.json` | every method | Run record; fields below. |
| `id_remap.json` | `replace` only | Map from old token ID (string key) to new token ID for surviving tokens. Consumed by `init_embeddings` with `arm=replace`. |
| `removed_tokens.txt` | `replace` only | One pruned token per line. |

`summary.json` fields:

| Field | Meaning |
|-------|---------|
| `model_id` | Base tokenizer identifier. |
| `method` | `add`, `replace`, or `expand`. |
| `extension_size` | Requested budget. |
| `corpus` | The `corpus` block as configured. |
| `base_vocab_size` | Token count of the base tokenizer. |
| `final_vocab_size` | Token count of the written tokenizer. |
| `new_candidates` | Novel tokens produced by BPE training before the splice. |
| `tokens_requested` | Same value as `extension_size`. |
| `tokens_spliced` | Tokens actually added. Constructive merging may add intermediate tokens, so this may differ from `tokens_requested`; compare arms only at equal `tokens_spliced`. |
| `removed` | `replace` only: number of pruned tokens. |
| `output` | Output directory. |
| `timings_sec` | `train`, `build`, and `total` wall-clock seconds. |

## `init_embeddings`

Writes `output_dir/` as a resized Hugging Face checkpoint: model weights saved with `save_pretrained` in the configured `dtype`, plus a copy of the extended tokenizer.
The directory is the `hf_model_path` input for `pretrain/megatron_bridge`.
Per-token initialization samples are printed to the log rather than written to a file.

## `evaluate`

Writes the JSON report at `output`.

| Field | Meaning |
|-------|---------|
| `label` | The configured label, or the `tokenizer` value. |
| `tokenizer` | Tokenizer identifier or path. |
| `vocab_size` | Token count. |
| `fix_mistral_regex` | Whether the Mistral pre-tokenizer regex correction was applied. |
| `eval_corpus` | The `corpus` block as configured. |
| `fertility_definition` | `sum(tokens)/sum(words)`. |
| `totals` | `docs`, `words`, `tokens`, `chars`. |
| `metrics.fertility` | Tokens per whitespace-delimited word; lower is better. |
| `metrics.chars_per_token` | Characters per token; higher is better. |
| `metrics.unique_tokens_used` | Distinct token IDs observed. |
| `metrics.vocab_coverage` | `unique_tokens_used / vocab_size`. |
| `elapsed_sec` | Wall-clock seconds. |

## `eval_init`

Writes the JSON report at `output_json`.

| Field | Meaning |
|-------|---------|
| `data_file`, `max_docs`, `max_tokens`, `max_length`, `stride`, `dtype` | The effective evaluation settings. |
| `results.<label>.loss` | Mean negative log-likelihood per token. |
| `results.<label>.perplexity` | `exp(loss)`; not comparable across tokenizers. |
| `results.<label>.bpb` | Bits per byte; comparable across tokenizers on identical text. |
| `results.<label>.num_tokens`, `num_bytes`, `num_docs` | Scored volume. |

The base model appears under its own label; the console output also prints each model's BPB delta against it.
