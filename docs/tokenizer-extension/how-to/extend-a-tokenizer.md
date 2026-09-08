---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run tokenizer_extension/extend with method add, replace, or expand on a Hugging Face or local corpus and verify the spliced token count."
topics: ["Tokenizer Extension", "BPE"]
tags: ["How-To", "Tokenizer Extension"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Extend a Tokenizer

Use `tokenizer_extension/extend` to train a BPE on a target-language corpus and splice the new tokens into the base tokenizer.
The step runs on CPU.

## Before You Begin

- Choose a registered `language` value; see {doc}`add-a-language` if yours is not registered.
- Install `indic-nlp-library` when the language uses the Devanagari normalizer (`hindi`, `marathi`, `nepali`, `sanskrit`):

  ```console
  $ uv pip install -e '.[tokenizer-extension]'
  ```

- Decide the method; see {doc}`../explanation/extension-methods`.

## Select the Corpus

Provide either a Hugging Face dataset or a local path, plus the text column.
The keys live under `corpus` in `extend/config/default.yaml`.

Hugging Face dataset:

```console
$ uv run nemotron steps run tokenizer_extension/extend -c default \
    language=vietnamese method=add extension_size=30000 \
    corpus.hf_dataset=<org/dataset> corpus.hf_name=<config> corpus.hf_split=<split> \
    corpus.text_field=text
```

`corpus.hf_name` is the dataset configuration or subset passed as `name` to `load_dataset`; `hf_data_dir`, `hf_data_files`, and `hf_revision` are optional.
Set `corpus.streaming=true` to stream instead of downloading and caching the dataset.

Local Parquet or JSON Lines:

```console
$ uv run nemotron steps run tokenizer_extension/extend -c default \
    language=vietnamese method=add extension_size=30000 \
    corpus.hf_dataset=null corpus.path=<dir-or-glob> corpus.glob='*.parquet' \
    corpus.text_field=text
```

Leave `corpus.hf_dataset` null when `corpus.path` is set.
For a local directory, keep `corpus.diversify: true` so that sampling covers every shard.

## Choose the Method

```console
$ uv run nemotron steps run tokenizer_extension/extend -c default language=<lang> method=add
$ uv run nemotron steps run tokenizer_extension/extend -c default language=<lang> method=replace
$ uv run nemotron steps run tokenizer_extension/extend -c default language=<lang> method=expand
```

Each command writes a separate subdirectory of `output_dir`: `add/`, `replace/`, or `expand/`.
Run the arms you want to compare as separate jobs with the same `extension_size`.

Do not set `remove_script` or `script_normalizer` unless you intend to override the language profile.
An explicit value takes precedence over `language`; for example, `language=vietnamese remove_script=devanagari` prunes Devanagari rows from a Vietnamese run.

## Bound Memory on a Small Node

The BPE trainer is CPU- and RAM-bound.
The `step.toml` strategies recommend:

```console
$ uv run nemotron steps run tokenizer_extension/extend -c default language=<lang> method=add \
    corpus.samples=200000 corpus.max_doc_chars=600 corpus.min_frequency=3
```

`corpus.min_frequency` ignores merges below that word frequency; `0` keeps every merge.

## Verify the Result

Open `output_dir/<method>/summary.json` and compare `tokens_spliced` with `extension_size`.

- Equal values mean the requested budget was reached.
- A smaller `tokens_spliced` means the corpus was too small for the budget; widen the corpus or lower `extension_size`. The step logs a warning because fertility and BPB comparisons between arms are valid only at a matched `tokens_spliced`.

The `replace/` directory also contains `id_remap.json`, which the `init_embeddings` step needs for `arm=replace`.

## Preview the Configuration

```console
$ uv run nemotron steps run tokenizer_extension/extend -d -c default language=<lang> method=add
```

## Next Steps

- Initialize the embedding rows: {doc}`initialize-embeddings`
- Field reference: {doc}`../reference/extend-config`
