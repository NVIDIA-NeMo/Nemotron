---
id: prefer-nmt-for-large-corpora
title: "Prefer NMT for large plain-text corpora"
tags: [translate, nmt, throughput]
triggers:
  - "The corpus is large, mostly plain text, and a local NMT service is available."
  - "Translation throughput or cost matters more than nuanced instruction following."
  - "The user mentions an IndicTrans, NMT, or local translation server."
steps: [translate/translation]
confidence: high
---

## When to apply

Use this when translation volume is high and the data shape is simple. A local NMT service can be cheaper and faster than per-segment LLM translation for large plain-text corpora.

## What to do

Set `backend=nmt` and configure `nmt.server_url`. Confirm that the service accepts `POST /translate` with `texts`, `src_lang`, and `tgt_lang`, and returns one translation per input text.

Tune `nmt.batch_size`, `nmt.timeout`, and `nmt.max_concurrent_requests` only after the basic smoke run works.

If FAITH is enabled, still configure the LLM `server` block because FAITH scoring uses an LLM even when translation uses NMT.

## Exceptions

Do not prefer NMT for tool-calling chat, schema-sensitive payloads, or cases where JSON/code preservation is the primary risk.

If no local NMT server is reachable, choose another backend rather than generating a fake server stub.
