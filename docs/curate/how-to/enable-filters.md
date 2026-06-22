---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Enable language, word-count, and domain filters in curate/nemo_curator."
topics: ["Curation", "How-To", "Filters"]
tags: ["How-To", "Curation", "Language ID", "Domain Classification"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Enable Curation Filters

Start with no filters, confirm JSONL input and output, then add one filter family at a time.

## Language Filtering

Set `language_codes` to uppercase language codes and provide a FastText language identification model.

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/data/my_corpus/**/*.jsonl" \
    output_dir="${PWD}/output/curated-en" \
    language_codes=[EN] \
    models.fasttext_langid="${PWD}/cache/models/fasttext/lid.176.bin" \
    quality_filters.min_langid_score=0.3
```

Set `language_codes=[]` to skip FastText language identification entirely.

## Word-Count Filtering

Set both `quality_filters.min_words` and `quality_filters.max_words`.
The step raises an error if only one of those keys is present.

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/data/my_corpus/**/*.jsonl" \
    output_dir="${PWD}/output/curated-word-count" \
    quality_filters.min_words=50 \
    quality_filters.max_words=5000
```

Set `quality_filters={}` to skip word-count filtering.

## Domain Filtering

Set `domains` to the domains you want to keep.
The step uses NeMo Curator's multilingual domain classifier.

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/data/my_corpus/**/*.jsonl" \
    output_dir="${PWD}/output/curated-domain" \
    domains=[STEM] \
    models.hf_cache_dir="${PWD}/cache/huggingface"
```

```{tip}
Keep the first domain-filtered run small.
The classifier may download or cache model assets on first use.
```

## Filter Order

The step applies filters in this order:

1. FastText language identification and language filtering, when `language_codes` is non-empty.
2. Word-count filtering, when `quality_filters.min_words` and `quality_filters.max_words` are both set.
3. Multilingual domain classification, when `domains` is non-empty.

When output is unexpectedly small, disable later filters first, then relax thresholds.
