---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Curate JSONL text with nemotron steps run curate/nemo_curator and NeMo Curator reader, filter, classifier, and writer stages."
topics: ["Curation", "NeMo Curator", "JSONL"]
tags: ["Curation", "Documentation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(curate-index)=
# About Data Curation With NeMo Curator

The `nemotron steps run curate/nemo_curator` command reads JSONL data, optionally materializes a Hugging Face dataset snapshot, applies lightweight NeMo Curator filters, and writes filtered JSONL shards for downstream translation or training data preparation.

Use this step when you already have JSONL records and need a small, repeatable curation pass before a later step such as `translate/nemo_curator`, `data_prep/pretrain_prep`, or `data_prep/sft_packing`.

## When to Use

Use `curate/nemo_curator` when you need:

- A local JSONL reader and writer path using NeMo Curator.
- Optional FastText language identification and language filtering.
- Optional word-count filtering.
- Optional multilingual domain classification and filtering.
- Optional Hugging Face dataset snapshot download before the Curator reader runs.

```{note}
This step is intentionally lightweight.
It does not crawl web pages, extract Common Crawl WARC files, or run large deduplication workflows.
Use a dedicated Curator recipe for those jobs before this step, or add a separate step when that behavior is needed.
```

## Pipeline Summary

```{mermaid}
flowchart LR
    A[Optional Hugging Face snapshot] --> B[JSONL files]
    C[Local JSONL files] --> B
    B --> D[JsonlReader]
    D --> E{Language filter enabled?}
    E -->|yes| F[FastText language ID]
    E -->|no| G{Word-count filter enabled?}
    F --> G
    G -->|yes| H[WordCountFilter]
    G -->|no| I{Domain filter enabled?}
    H --> I
    I -->|yes| J[MultilingualDomainClassifier]
    I -->|no| K[JsonlWriter]
    J --> K
    K --> L[Filtered JSONL shards]
```

## Documentation Series

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorial
:link: getting-started
:link-type: doc
Install the Nemotron CLI, run a local tiny JSONL initial curation validation, and inspect output shards.
+++
{bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-To Guides
:link: how-to/index
:link-type: doc
Run local JSONL curation, download a Hugging Face snapshot, and enable optional filters.
+++
{bdg-secondary}`task-based`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
YAML parameters, CLI syntax, input/output format, and troubleshooting.
+++
{bdg-secondary}`lookup`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Tutorial

| Guide | What you do |
| --- | --- |
| {doc}`getting-started` | Run `curate/nemo_curator` on the packaged tiny JSONL fixture |

```

```{tab-item} How-To Guides

| Guide | Focus |
| --- | --- |
| {doc}`how-to/run-local-jsonl` | Local JSONL reader/writer path |
| {doc}`how-to/use-huggingface-snapshot` | `dataset` block and Hugging Face snapshot download |
| {doc}`how-to/enable-filters` | Language, word-count, and domain filters |

```

```{tab-item} Reference

| Guide | Content |
| --- | --- |
| {doc}`reference/curate-config` | YAML field reference |
| {doc}`reference/cli-curate` | `nemotron steps run curate/nemo_curator` syntax |
| {doc}`reference/io-format` | Input and output shapes |
| {doc}`reference/troubleshooting` | Common failures and fixes |

```

````

## What You Need

- JSONL input with one text field, usually named `text`.
- Optional model assets when filters are enabled, such as a FastText language identification model for `language_codes`.
- A writable output directory for JSONL shards.

## Quick Paths

1. First local run: {doc}`getting-started`
2. Local corpus setup: {doc}`how-to/run-local-jsonl`
3. Hugging Face snapshot setup: {doc}`how-to/use-huggingface-snapshot`
4. Filter setup: {doc}`how-to/enable-filters`
5. Lookup flags: {doc}`reference/cli-curate`
