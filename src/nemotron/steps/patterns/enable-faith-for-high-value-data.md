---
id: enable-faith-for-high-value-data
title: "Enable FAITH for high-value translated data"
tags: [translate, faith, quality]
triggers:
  - "Translated data will be used for governance, audit, or high-value model training."
  - "The user needs quality scores or threshold filtering for translated corpus rows."
  - "Translation quality must gate SFT, CPT, or customer-facing training data."
steps: [translate/translation]
confidence: high
---

## When to apply

Use this when translated data quality needs evidence, not just output files. FAITH is useful for governance review, high-value SFT data, or any translation path where bad rows should be visible or filtered.

## What to do

Enable `faith_eval.enabled=true`. Prefer `faith_eval.segment_level=true` for long corpora because scoring follows the same segment boundaries used by translation.

Ask whether low-scoring rows should be filtered or only annotated. Set `faith_eval.filter_enabled` accordingly.

Configure an LLM client for FAITH even when the translation backend is `nmt`, `google`, or `aws`.

Keep `output_mode=both` so downstream users can audit translated fields, metadata, and scores.

## Exceptions

Do not enable FAITH by default when throughput is the priority and the user only needs a quick bulk translation.

Do not silently drop rows. Filtering changes corpus size and should be explicit in the plan.
