---
id: prefer-llm-for-structured-chat
title: "Prefer LLM translation for structured chat data"
tags: [translate, chat, structured-data]
triggers:
  - "The input is OpenAI-style chat data, tool-calling transcripts, or nested message records."
  - "The translated output must preserve JSON, code blocks, markup, or message structure."
  - "The user wants to translate messages.*.content or another wildcard field path."
steps: [translate/translation]
confidence: high
---

## When to apply

Use this when formatting fidelity matters more than raw throughput. Chat transcripts, tool payloads, and mixed natural-language plus structured content need a backend that can follow preservation instructions.

## What to do

Prefer `backend=llm`. Set `text_field=messages.*.content` for OpenAI-style chat data and enable `reconstruct_messages=true` when the user needs a translated message list for inspection.

Ask for the exact OpenAI-compatible endpoint, model name, and API key environment variable. Verify hosted model names before real runs because hosted catalogs can retire models.

Keep secrets out of checked-in config. Use environment variables for keys.

## Exceptions

If the data is mostly plain text and throughput or cost dominates, consider `prefer-nmt-for-large-corpora` instead.

If the user has a managed cloud translation requirement, use `google` or `aws` and document the structure-preservation risk.
