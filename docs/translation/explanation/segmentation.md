---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Segmentation modes for nemotron steps run translate/nemo_curator: coarse and fine modes, plus min_segment_chars."
topics: ["Translation", "Segmentation"]
tags: ["Explanation", "Translation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Segmentation

This page helps you pick `segmentation_mode` values that keep translations faithful without chopping structured payloads.

## Controls

| YAML key | Meaning |
|----------|---------|
| `segmentation_mode` | `coarse` is the default. It balances throughput while respecting code fences and markup boundaries. `fine` emits smaller linguistic segments when coarse spans truncate meaning. |
| `min_segment_chars` | Skips extremely short fragments that harm throughput or confuse backends. The default is `0`, which disables filtering. |

## When Coarse Is Enough

Start with `segmentation_mode: coarse` when records contain JSON-like strings, tool arguments, or fenced code blocks. Coarse mode tracks reconstruction metadata so `TranslationStage` can rebuild the original nesting after translating natural-language spans.

## When to Switch to Fine

Move to `segmentation_mode: fine` when you observe clipped sentences, missing trailing punctuation, or uneven translations inside long paragraphs, and you have confirmed the issue disappears after finer segmentation.

Fine mode increases API calls or neural machine translation (NMT) batches. Budget concurrency by using `max_concurrent_requests` or backend-specific knobs.

## Interaction With FAITH

FAITH scoring is part of Curator's translation stage and follows the translated segment pairs produced by the stage, which keeps thresholds interpretable on long documents.

## Practical Workflow

1. Run with `coarse` first.
2. Inspect a handful of difficult rows such as legal text, mixed Markdown, or multilingual snippets.
3. If boundaries look wrong, toggle `fine` on a slice before scaling up.

## Related Pages

- Step-by-step fine mode: {doc}`../how-to/use-fine-segmentation`
- Pipeline context: {doc}`pipeline-overview`
- FAITH alignment with segments: {doc}`faith-evaluation`
