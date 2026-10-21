---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Concepts behind the tokenizer_extension steps: pipeline architecture, extension methods, embedding initialization, and evaluation metrics."
topics: ["Tokenizer Extension", "Concepts"]
tags: ["Explanation", "Tokenizer Extension"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Concepts

These pages explain how the `tokenizer_extension` steps work and why they make the choices they do.
Read them when you need to decide between methods rather than run a fixed procedure.

```{toctree}
:maxdepth: 1
:hidden:

pipeline-overview
extension-methods
embedding-initialization
evaluation-metrics
```

| Page | Topic |
|------|-------|
| {doc}`pipeline-overview` | The four steps, the artifacts they exchange, and the hand-off to continued pretraining |
| {doc}`extension-methods` | Add, Replace, and Expand, and why the splice into the merge table is constructive |
| {doc}`embedding-initialization` | The baseline, subword, and FOCUS engines, norm correction, and Add versus Replace surgery |
| {doc}`evaluation-metrics` | Fertility and bits per byte, and why per-token perplexity does not compare across vocabularies |

For measured results in Hindi, Malayalam, and Vietnamese, read the [Tokenizer Extension Guidebook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/guidebook/README.md) in the repository.
