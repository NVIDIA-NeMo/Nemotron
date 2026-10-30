---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Configure text_field, output_mode, reconstruct_messages, and writer formats."
topics: ["Translation", "Configuration"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Configure Fields and Output

Use this guide when you need to align `text_field`, `output_mode`, reconstruction flags, and formats with your dataset schema before or after a first run.

For a guided first invocation, see {doc}`../getting-started`.

## Choosing `text_field`

- Chat corpora in OpenAI layout typically use `messages.*.content` so every message `content` entry is translated consistently.
- Plain documents might use a single column such as `article_body`. Omit wildcards when the schema is flat.

## Outputs

| YAML key | Behavior |
|----------|----------|
| `output_field` / `translation_column` | Control column names used when emitting translated strings. The default is `translated_text`. |
| `output_mode` | `replaced` overwrites source strings, `raw` preserves originals plus metadata, and `both` keeps audit trails. The starter default is `both`. |
| `merge_scores` | Keeps FAITH outputs adjacent to translations when scoring runs. |
| `reconstruct_messages`, `messages_field`, `messages_content_field` | Enable faithful reconstructions of chat arrays. These default to `true` for standard `messages` and `content` layouts. |

## Formats

Set `input_format` when automatic probing cannot distinguish ambiguous globs. Align `output_format` with downstream packing expectations; values are `jsonl` or `parquet`.

## CLI Overrides

You can override any YAML key with dotlists:

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  text_field=messages.*.content \
  output_mode=both \
  reconstruct_messages=true \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=fr
```

## Related Pages

- Schema patterns: {doc}`../reference/io-format`
- Segmentation interactions: {doc}`use-fine-segmentation`
