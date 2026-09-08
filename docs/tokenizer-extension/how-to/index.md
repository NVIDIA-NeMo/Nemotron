---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Task guides for the tokenizer_extension steps: extension methods, embedding initialization, evaluation, language registration, and cluster execution."
topics: ["Tokenizer Extension", "How-To"]
tags: ["How-To", "Tokenizer Extension"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# How-To Guides

This section has task-focused procedures for the `tokenizer_extension` steps.
Start with {doc}`../getting-started` if you have not run the steps yet.

```{toctree}
:maxdepth: 1
:hidden:

extend-a-tokenizer
initialize-embeddings
evaluate-a-tokenizer
add-a-language
run-on-a-cluster
```

## Build and Initialize

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Extend a tokenizer
:link: extend-a-tokenizer
:link-type: doc
Select `method: add`, `replace`, or `expand`, point `corpus` at your data, and verify `tokens_spliced`.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Initialize embeddings
:link: initialize-embeddings
:link-type: doc
Match `arm` to the extension method, choose an engine, and produce the checkpoint for continued pretraining.
:::

::::

## Measure

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluate a tokenizer
:link: evaluate-a-tokenizer
:link-type: doc
Compare fertility across tokenizers and BPB across resized checkpoints on identical text.
:::

::::

## Adapt and Scale

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Add a language
:link: add-a-language
:link-type: doc
Register a `LanguageProfile` and, if needed, Unicode ranges so that one `language` key configures every step.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Run on a cluster
:link: run-on-a-cluster
:link-type: doc
Use the generated `*_tokenizer_*` environment profiles and install the optional dependencies in the container.
:::

::::
