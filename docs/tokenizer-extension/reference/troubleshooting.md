---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Named errors declared in the tokenizer_extension step.toml files, with causes and recoveries."
topics: ["Tokenizer Extension", "Troubleshooting"]
tags: ["Reference", "Troubleshooting"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Troubleshooting

The errors below are declared in each step's `step.toml`.
The steps raise hard errors rather than substituting a fallback, so a run that completes produced what the configuration asked for.

## `extend`

| Error | Cause | Recovery |
|-------|-------|----------|
| `corpus_source_unset` | Neither `corpus.hf_dataset` nor `corpus.path` is set. | Set one of them, with `corpus.hf_name`/`corpus.hf_split` or `corpus.glob` as appropriate. |
| `rank_dead_tokens` | A spliced token is unreachable at its merge rank, which indicates a corrupt merge set. | Retrain with `corpus.min_frequency` greater than `0`, or with fewer or cleaner documents. |
| `oom_during_training` | The BPE trainer exhausted host memory; it is CPU- and RAM-bound. | Lower `corpus.samples`, set `corpus.max_doc_chars`, and raise `corpus.min_frequency`. |

A `language` value that is not registered raises a `ValueError` listing the registered names.
A Devanagari-normalized language raises an `ImportError` when `indic-nlp-library` is not installed.

## `init_embeddings`

| Error | Cause | Recovery |
|-------|-------|----------|
| `unknown_method` | `method` is not `baseline`, `subword`, or `focus`. | Set one of the three values. |
| `replace_tokenizer_rejected` | `extended_tokenizer` contains `id_remap.json`, so it is a Replace tokenizer, but `arm=add` was set. | Set `arm=replace`. Use `arm=add` only for an `add/` or `expand/` tokenizer. |
| `row_count_mismatch` | The model's embedding row count differs from the base tokenizer size. | Confirm that `base_model` is the checkpoint whose tokenizer was extended. |
| `missing_fasttext` | `method=focus` without `focus.fasttext_model`, or the `fasttext` package is not installed. | Set `focus.fasttext_model` to a fastText `.bin` and install the `tokenizer-extension` extra. |

Unsupported method and arm combinations, such as `baseline.mode=mean_target` or `subword.*_averaging=gemma_weighted` with `arm=replace`, also fail explicitly.
See the support table in {doc}`../explanation/embedding-initialization`.

## `evaluate`

| Error | Cause | Recovery |
|-------|-------|----------|
| `corpus_source_unset` | Neither `corpus.hf_dataset` nor `corpus.path` is set. | Set one of them. |
| `wrong_text_field` | `corpus.text_field` does not name the text column. | Set the actual column: `tgt` for Samanantar, `text` for Wikipedia or Sangraha. |

A corpus that yields no documents raises a `RuntimeError` rather than reporting a fertility of `0.0`.

## `eval_init`

| Error | Cause | Recovery |
|-------|-------|----------|
| `missing_data_file` | No local corpus was given and no `hf_dataset` was set. | Set `data_file` to a local `.jsonl` or `.txt` corpus. |
| `empty_models` | `models` is empty. | Set at least one checkpoint path. |

Scoring more than one model with `max_tokens` set and `max_docs` unset raises a `ValueError` because the token cap is tokenizer-dependent.
Set `max_docs`, or set `allow_token_cap_comparison=true` only to reproduce a historical run.

## Related Pages

- {doc}`extend-config`
- {doc}`init-embeddings-config`
- {doc}`evaluate-config`
- {doc}`eval-init-config`
